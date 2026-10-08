"""ToolObserver optimizer on the ``api.Optimizer`` interface.

Port of the ToolObserver framework (Hallinan et al., 2026 — arXiv:2602.15197,
"OpaqueToolsBench", CAIS 2026): improve tool *descriptions* by observing real
agent trajectories. Its distinguishing
algorithm — vs. the one-shot ``llm`` optimizer — is **minibatch analysis +
consensus merge**: split the baseline transcripts into mini-batches, have an
editor LLM propose an improved tool file from each batch independently, then a
second pass synthesizes one final edit, keeping patterns that recur across
batches and discarding single-batch noise.

Blind by construction: it analyzes the baseline transcripts only — it never runs
candidates or reads the test reward (``wants_train_eval`` stays False), so it
sits inside the anti-reward-hacking boundary like ``llm``/``draft``. This ports
the paper's *Reflection* phase (mini-batch analysis + consensus merge) only; the
paper's *Exploration* phase (running the agent to gather fresh trajectories) and
its outer K-iteration loop are omitted to stay blind.

Batch construction is model-token-aware: it keeps complete trajectories while a
request fits, splits the batch when it does not, and structurally fits only a
singleton trajectory that is too large by itself. Opaque trajectories are never
shortened.

Naturally paired with a descriptions-only ``ToolTarget`` (the editable file is
the tool descriptions), but works on any single editable file (it rewrites the
whole file, focusing edits on the descriptions). The optimizer's own model is
its config; the injected client owns provider routing.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent_tool_opt_core.api import (
    Candidate,
    Optimizer,
    RunResult,
    TaskRun,
    ToolSet,
    Validator,
)
from agent_tool_opt_core.llm_client import LLMClient, LiteLLMClient
from agent_tool_opt_core.optimizers._common import (
    BASE_OBJECTIVE,
    DEFAULT_OPTIMIZER_MODEL,
    DEFAULT_OUTPUT_RESERVE_TOKENS,
    OptimizerLLMFailure,
    call_llm_with_retry,
    ensure_request_fits,
    fit_trajectory,
    model_context_window,
    render_trajectory,
    request_token_count,
    single_editable_target,
    strip_code_fences,
)

logger = logging.getLogger(__name__)

_BATCH_SYSTEM = (
    "You improve an agent's tools by analyzing its usage trajectories. Rewrite the "
    "tool file, informed by what you observe across the tasks."
)

_BATCH_USER = """\
## Current tool file
```
{source}
```

## Observed trajectories (batch {idx}/{total})
{trajectories}

## Task
Analyze how the agent used the tools that appear above, then return the FULL tool
file with improved tool descriptions/docstrings so the agent would succeed on more
tasks. Keep all code, imports, class names, and signatures identical.

{rules}

Output ONLY the complete modified file content."""

_MERGE_SYSTEM = (
    "You synthesize several independent analyses of different trajectory batches "
    "into one definitive set of tool descriptions. Keep improvements that recur "
    "across batches; discard single-batch noise."
)

_MERGE_USER = """\
## Original tool file
```
{source}
```

## {n} independently-proposed improved files (one per trajectory batch)
{proposals}

## Task
Synthesize ONE final tool file. Keep description improvements that appear across
multiple batches (reliable signal); drop changes seen in only one batch (likely
noise). Preserve all code, imports, class names, and signatures identical to the
original.

{rules}

Output ONLY the complete modified file content."""


class ToolObserverOptimizer(Optimizer):
    id = "toolobserver"
    wants_train_eval = False  # blind: transcripts only, never probes reward

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPTIMIZER_MODEL,
        batch_size: int = 10,
        require_validation: bool = True,
        max_retries: int = 2,
        context_window_size: int | None = None,
        output_reserve_tokens: int = DEFAULT_OUTPUT_RESERVE_TOKENS,
        llm: LLMClient | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.require_validation = require_validation
        self.max_retries = max_retries
        self.context_window_size = context_window_size
        self.output_reserve_tokens = output_reserve_tokens
        self.llm = llm or LiteLLMClient()

    def propose(
        self,
        tools: ToolSet,
        run: RunResult,
        validate: Validator,
        scratch: Path,
        train_eval=None,
    ) -> Candidate:
        if not tools.allowlist:
            return Candidate({})
        target = single_editable_target(tools)
        source = tools.files.get(target, "")
        if not run.runs:
            return Candidate({})  # nothing observed → no edit

        batches = self._token_batches(run, source, tools.language_rules)
        client = self.llm
        proposals = []
        for i, batch in enumerate(batches):
            edit = self._analyze_batch(
                client, source, batch, i + 1, len(batches), tools.language_rules
            )
            if edit:
                proposals.append(edit)
        if not proposals:
            return Candidate({})

        if len(proposals) == 1:
            final = proposals[0]
        else:
            final = self._merge(client, source, proposals, tools.language_rules)

        cand = Candidate({target: final})
        if not self.require_validation or validate.validate(cand).ok:
            return cand
        # one corrective retry: re-merge against the original with the failure note
        retry = self._merge(
            client, source, proposals, tools.language_rules, corrective=True
        )
        cand = Candidate({target: retry})
        if validate.validate(cand).ok:
            return cand
        raise OptimizerLLMFailure(
            "toolobserver optimizer: the corrective re-merge still did not pass "
            "validation"
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _segment(r: TaskRun, body: str | None = None) -> str:
        outcome = "PASS" if r.reward >= 1.0 else "FAIL"
        rendered = render_trajectory(r.trajectory) if body is None else body
        return f"### task {r.task_id} [{outcome}] (reward={r.reward})\n{rendered}"

    def _segments(self, run: RunResult) -> list[str]:
        """Per-task transcript blocks (PASS/FAIL + reward + rendered trajectory)."""
        return [self._segment(r) for r in run.runs]

    def _batch(self, segments: list[str], size: int) -> list[list[str]]:
        if size <= 0:
            return [segments]
        return [segments[i : i + size] for i in range(0, len(segments), size)]

    def _batch_messages(self, source: str, batch: list[str], rules: str):
        # Four digits make this at least as expensive as ordinary idx/total labels,
        # so the final prompt cannot overflow merely because batch counts were added.
        user = _BATCH_USER.format(
            source=source,
            idx=9999,
            total=9999,
            trajectories="\n\n".join(batch),
            rules=rules,
        )
        return [
            {"role": "system", "content": _BATCH_SYSTEM + "\n\n" + BASE_OBJECTIVE},
            {"role": "user", "content": user},
        ]

    def _batch_fits(self, source: str, batch: list[str], rules: str) -> bool:
        try:
            ensure_request_fits(
                self._batch_messages(source, batch, rules),
                self.model,
                output_reserve_tokens=self.output_reserve_tokens,
                context_window_size=self.context_window_size,
            )
            return True
        except OptimizerLLMFailure:
            return False

    def _fit_segment(self, r: TaskRun, source: str, rules: str) -> str:
        header_only = self._segment(r, body="")
        fixed = request_token_count(
            self._batch_messages(source, [header_only], rules), self.model
        )
        budget = (
            model_context_window(self.model, self.context_window_size)
            - self.output_reserve_tokens
            - fixed
            - 32  # small tokenizer-boundary/framing margin
        )
        body = fit_trajectory(
            r.trajectory,
            token_budget=budget,
            model=self.model,
            optimizer=self.id,
            task_id=r.task_id,
        )
        return self._segment(r, body=body)

    def _token_batches(
        self, run: RunResult, source: str, rules: str
    ) -> list[list[str]]:
        """Greedily pack complete trajectories, fitting only a too-large singleton."""
        limit = self.batch_size if self.batch_size > 0 else len(run.runs)
        batches: list[list[str]] = []
        current: list[str] = []
        for r in run.runs:
            segment = self._segment(r)
            if current and (
                len(current) >= limit
                or not self._batch_fits(source, [*current, segment], rules)
            ):
                batches.append(current)
                current = []
            if not current and not self._batch_fits(source, [segment], rules):
                segment = self._fit_segment(r, source, rules)
                # Give a precise whole-request failure if the fitted singleton
                # still cannot fit because of fixed source/instruction overhead.
                ensure_request_fits(
                    self._batch_messages(source, [segment], rules),
                    self.model,
                    output_reserve_tokens=self.output_reserve_tokens,
                    context_window_size=self.context_window_size,
                )
            current.append(segment)
        if current:
            batches.append(current)
        count_only_batches = max(
            1, (len(run.runs) + max(limit, 1) - 1) // max(limit, 1)
        )
        if len(batches) > count_only_batches:
            logger.warning(
                "toolobserver split %d trajectories into %d token-aware batches "
                "(configured batch_size=%d, model=%s)",
                len(run.runs),
                len(batches),
                self.batch_size,
                self.model,
            )
        return batches

    def _complete(self, client, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        ensure_request_fits(
            messages,
            self.model,
            output_reserve_tokens=self.output_reserve_tokens,
            context_window_size=self.context_window_size,
        )
        completion = call_llm_with_retry(
            lambda: client.complete(
                model=self.model,
                messages=messages,
            ),
            model=self.model,
        )
        return strip_code_fences(completion.text)

    def _analyze_batch(self, client, source, batch, idx, total, rules) -> str | None:
        user = _BATCH_USER.format(
            source=source, idx=idx, total=total,
            trajectories="\n\n".join(batch), rules=rules,
        )  # fmt: skip
        edit = self._complete(client, _BATCH_SYSTEM + "\n\n" + BASE_OBJECTIVE, user)
        return edit or None

    def _merge(self, client, source, proposals, rules, *, corrective=False) -> str:
        blocks = "\n\n".join(
            f"### proposal {i + 1}\n```\n{p}\n```" for i, p in enumerate(proposals)
        )
        user = _MERGE_USER.format(
            source=source, n=len(proposals), proposals=blocks, rules=rules
        )
        if corrective:
            user += (
                "\n\nYour previous synthesis did not pass validation. Re-read the "
                "original and return the full corrected file content."
            )
        return self._complete(client, _MERGE_SYSTEM + "\n\n" + BASE_OBJECTIVE, user)
