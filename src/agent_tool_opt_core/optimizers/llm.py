"""One-shot LLM optimizer on the ``api.Optimizer`` interface.

Reads the editable tool source + the baseline transcripts, asks an LLM
to rewrite it, and returns the edit as a ``Candidate``. The
optimizer's own model is its config — distinct from the subject ``Agent``
whose transcripts it trains on. Targets a single editable file (e.g. tau2's
``tools.py``); the injected client owns provider routing.

Carries the same orthogonal ablation knobs as the pi optimizer (assembled by
``optimizers/catalog.py``): ``use_transcripts`` (feed the baseline run vs.
optimize blind), composable insight ``methods`` (``method_addendum`` +
``context_builder``, injected as prompt text), and ``require_validation`` (gate
the rewrite on the language ``Validator``, retrying on failure). Edit *scope*
(full vs docstrings-only) is owned by the ToolTarget, not here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from agent_tool_opt_core.api import Candidate, Optimizer, RunResult, ToolSet, Validator
from agent_tool_opt_core.llm_client import LLMClient, LiteLLMClient
from agent_tool_opt_core.optimizers._common import (
    BASE_OBJECTIVE,
    DEFAULT_OPTIMIZER_MODEL,
    DEFAULT_OUTPUT_RESERVE_TOKENS,
    OptimizerLLMFailure,
    call_llm_with_retry,
    ensure_request_fits,
    fit_transcripts,
    model_context_window,
    request_token_count,
    single_editable_target,
    strip_code_fences,
)

_SYSTEM = BASE_OBJECTIVE + "\n\nOutput ONLY the complete modified file content."

_RETHINK = (
    "Your previous output did not pass validation. Re-read the source, fix the "
    "underlying issue, and return the full corrected file content."
)

_COLD = "(no transcripts provided — optimize from the tool source alone)"


class LLMOptimizer(Optimizer):
    id = "llm"

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPTIMIZER_MODEL,
        method_addendum: str = "",
        context_builder: Callable[[RunResult], dict[str, str]] | None = None,
        use_transcripts: bool = True,
        require_validation: bool = True,
        max_retries: int = 3,
        context_window_size: int | None = None,
        output_reserve_tokens: int = DEFAULT_OUTPUT_RESERVE_TOKENS,
        llm: LLMClient | None = None,
    ) -> None:
        self.model = model
        self.method = method_addendum
        self.context_builder = context_builder
        self.use_transcripts = use_transcripts
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
        train_eval=None,  # blind optimizer: no train evaluator used
    ) -> Candidate:
        if not tools.allowlist:
            return Candidate({})
        target = single_editable_target(tools)
        source = tools.files.get(target, "")
        system = _SYSTEM + "\n\n" + tools.language_rules
        if self.method:
            system += "\n\n" + self.method
        if self.use_transcripts:
            context_parts = []
            if self.context_builder is not None:
                context_parts = [
                    f"## {name}\n{body}"
                    for name, body in self.context_builder(run).items()
                    if not name.startswith("transcripts/")
                ]
            # Charge source, instructions and method context first; the remaining
            # model-aware budget belongs to the transcript set as a whole.
            fixed_user = (
                self._user_prompt(
                    source, run, transcripts="", context_parts=context_parts
                )
                + "\n\n"
                + _RETHINK
            )
            window = model_context_window(self.model, self.context_window_size)
            fixed = request_token_count(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": fixed_user},
                ],
                self.model,
            )
            transcripts = fit_transcripts(
                run,
                token_budget=window - self.output_reserve_tokens - fixed,
                model=self.model,
                optimizer=self.id,
            )
            user = self._user_prompt(
                source, run, transcripts=transcripts, context_parts=context_parts
            )
        else:
            user = self._user_prompt(source, run)

        candidate = Candidate({})
        for attempt in range(1, self.max_retries + 1):
            content = user if attempt == 1 else user + "\n\n" + _RETHINK
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ]
            ensure_request_fits(
                messages,
                self.model,
                output_reserve_tokens=self.output_reserve_tokens,
                context_window_size=self.context_window_size,
            )
            completion = call_llm_with_retry(
                lambda: self.llm.complete(
                    model=self.model,
                    messages=messages,
                ),
                model=self.model,
            )
            new = strip_code_fences(completion.text)
            candidate = Candidate({target: new})
            if not self.require_validation or validate.validate(candidate).ok:
                return candidate
        raise OptimizerLLMFailure(
            f"llm optimizer exhausted {self.max_retries} attempts without "
            f"producing a candidate that passes validation"
        )

    def _user_prompt(
        self,
        source: str,
        run: RunResult,
        *,
        transcripts: str | None = None,
        context_parts: list[str] | None = None,
    ) -> str:
        parts = [f"## Current tool source\n```\n{source}\n```"]
        if self.use_transcripts:
            parts.append(
                "## Baseline transcripts (the agent's runs)\n"
                + (transcripts if transcripts is not None else "")
            )
            # Inject digested context (e.g. reward-shaping's INDEX), but not raw
            # transcript files. It is built once so request sizing and sending
            # cannot observe different context-builder results.
            parts.extend(context_parts or [])
        else:
            parts.append(_COLD)
        parts.append("Return the full modified file content.")
        return "\n\n".join(parts)
