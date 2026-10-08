"""GEPA optimizer on the ``api.Optimizer`` interface (tools_code mode).

Port of GEPA (arXiv:2507.19457, "Reflective Prompt Evolution…", ICLR 2026) as a
clean core optimizer. Unlike the blind ``llm``/``draft``/``toolobserver``
optimizers, GEPA is a *search* optimizer: it must score candidate tools to drive
selection. It
declares ``wants_train_eval = True`` and the driver hands it a ``TrainEvaluator``
bound to the **train** split only — so GEPA may search on train but can never
see the test split (the anti-reward-hacking "train jail").

The loop (one ``propose`` call, *softly* bounded by ``max_eval_budget``
task-evals — the budget is checked at the top of each iteration, so a single
accepted iteration may overshoot by one minibatch + one full train eval):

  seed = current tool source; its per-task train scores come *free* from the
  baseline ``run`` (no eval spent). Then repeatedly:
    1. select a parent from the Pareto frontier (frequency-weighted over the
       non-dominated candidates — diversify across task specialists);
    2. sample a train minibatch; evaluate the parent on it (with transcripts);
    3. reflect on the parent's failures → propose a new full source (LLM);
    4. evaluate the proposal on the same minibatch; ACCEPT iff it strictly
       improves the minibatch score, then full-evaluate on all train tasks and
       add it to the Pareto pool.
  Return the best candidate (highest average train score), or no edit if none
  beat the seed.

tools_code mode (full-file) is the core mode — it fits the single-file
``Candidate`` contract apples-to-apples with the ``llm`` optimizer. The model is
the optimizer's config; the injected client owns provider routing.

``--methods generalization`` swaps GEPA's reflection prompt for a transfer-first
one (``_REFLECT_USER_GENERALIZE``): it optimizes for unseen tasks instead of folding
in observed-task facts, so the two framings can be compared head-to-head.

Simplification vs. the reference implementation, kept deliberately: the reflection
LM sees the transcript + scalar reward rather than a richer textual feedback signal
(µ_f) — a future refinement, not the metric. (Selection does prune globally
dominated candidates before frequency-weighting, per Alg 2.) Reflection overflow is
handled paper-style by reducing the already-sampled minibatch before structurally
fitting a single trajectory's bulky values; opaque trajectories are never shortened.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from pathlib import Path

from agent_tool_opt_core.api import (
    Candidate,
    Optimizer,
    RunResult,
    ToolSet,
    TrainEvaluator,
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
    fit_transcripts,
    model_context_window,
    request_token_count,
    single_editable_target,
    strip_code_fences,
    summarize_transcripts,
)

logger = logging.getLogger(__name__)

# Reflection ("meta") prompt, adapted from GEPA's Appendix C / the reference repo's
# ``InstructionProposalSignature`` to our tool-file target: infer the task from the
# traces, fold in domain-specific knowledge, and capture any generalizable strategy.
_REFLECT_SYSTEM = (
    "You improve an assistant's tools by reflecting on how it used them across a "
    "sample of tasks. Rewrite the tool file so the assistant does better."
)

_REFLECT_USER = """\
## The assistant's current tool file
```
{source}
```

## Tasks the assistant attempted (its transcript + reward for each; 1.0 = success)
{transcripts}

## Your task
Write a new version of the tool file. Read the tasks carefully and infer a detailed
description of what the assistant is being asked to do. Read the transcripts and
their rewards, and identify any niche, domain-specific knowledge about the tasks that
would help the assistant and may not be obvious from the tools alone — fold it into
the tool documentation. If a transcript reveals a generalizable strategy for using
the tools well, capture that too.

{rules}

Output ONLY the complete modified file content."""

# Reflection prompt used when ``--methods generalization`` is applied to gepa: same
# reflective rewrite, but it optimizes for transfer to *unseen* tasks instead of
# folding in observed-task facts — the deliberate opposite of GEPA's encode-facts step.
_REFLECT_USER_GENERALIZE = """\
## The assistant's current tool file
```
{source}
```

## Tasks the assistant attempted (its transcript + reward for each; 1.0 = success)
{transcripts}

## Your task
Write a new version of the tool file. Read the tasks carefully and infer a detailed
description of what the assistant is being asked to do. Prefer changes that would
help on *unseen* tasks of this kind, not only the specific tasks shown: fix the
general shortcoming a transcript reveals rather than its one-off details, and if a
transcript reveals a generalizable strategy for using the tools well, capture that.
Do not fold in knowledge that only fits the observed tasks.

{rules}

Output ONLY the complete modified file content."""


class _Pareto:
    """Compact GEPA candidate pool + per-task Pareto frontier (see gepa.state)."""

    def __init__(self, seed_source: str, seed_scores: dict[str, float]):
        self.candidates: list[str] = [seed_source]
        self.val_scores: list[dict[str, float]] = [dict(seed_scores)]
        self.front: dict[str, set[int]] = {t: {0} for t in seed_scores}
        self.best: dict[str, float] = dict(seed_scores)

    def _avgs(self) -> list[float]:
        return [sum(v.values()) / len(v) if v else 0.0 for v in self.val_scores]

    def add(self, source: str, scores: dict[str, float]) -> int:
        idx = len(self.candidates)
        self.candidates.append(source)
        self.val_scores.append(dict(scores))
        for tid, s in scores.items():
            prev = self.best.get(tid, float("-inf"))
            if s > prev:
                self.best[tid] = s
                self.front[tid] = {idx}
            elif s == prev:
                self.front.setdefault(tid, set()).add(idx)
        return idx

    def _nondominated(self) -> set[int]:
        """Candidate indices not strictly dominated by any other across all tasks
        (GEPA Alg 2's prune step): j dominates i iff j ≥ i on every task and > on
        at least one. A candidate beaten everywhere gets no sampling weight."""
        scores = self.val_scores
        tasks = set().union(*(s.keys() for s in scores)) if scores else set()
        dominated: set[int] = set()
        for i, si in enumerate(scores):
            for j, sj in enumerate(scores):
                if i == j:
                    continue
                ge = all(sj.get(t, 0.0) >= si.get(t, 0.0) for t in tasks)
                gt = any(sj.get(t, 0.0) > si.get(t, 0.0) for t in tasks)
                if ge and gt:
                    dominated.add(i)
                    break
        return set(range(len(scores))) - dominated

    def select(self, rng: random.Random) -> int:
        """Frequency-weighted pick over non-dominated frontier candidates: prune
        globally dominated candidates first (GEPA Alg 2), then each task's frontier
        votes for its top scorer(s)."""
        nondom = self._nondominated()
        avgs = self._avgs()
        freq: Counter[int] = Counter()
        for progs in self.front.values():
            live = [i for i in progs if i in nondom]
            if not live:
                continue
            mx = max(avgs[i] for i in live)
            for i in live:
                if avgs[i] >= mx:
                    freq[i] += 1
        if not freq:
            return self.best_idx()
        cands = list(freq)
        return rng.choices(cands, weights=[freq[c] for c in cands], k=1)[0]

    def best_idx(self) -> int:
        avgs = self._avgs()
        return max(range(len(avgs)), key=lambda i: avgs[i])

    def best_avg(self) -> float:
        avgs = self._avgs()
        return max(avgs) if avgs else 0.0


class GEPAOptimizer(Optimizer):
    id = "gepa"
    wants_train_eval = True  # search optimizer: scores candidates on TRAIN

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPTIMIZER_MODEL,
        max_eval_budget: int = 200,
        minibatch_size: int = 3,
        seed: int = 0,
        require_validation: bool = True,
        generalize: bool = False,
        context_window_size: int | None = None,
        output_reserve_tokens: int = DEFAULT_OUTPUT_RESERVE_TOKENS,
        llm: LLMClient | None = None,
    ) -> None:
        self.model = model
        self.max_eval_budget = max_eval_budget
        self.minibatch_size = minibatch_size
        self.seed = seed
        self.require_validation = require_validation
        self.context_window_size = context_window_size
        self.output_reserve_tokens = output_reserve_tokens
        self.llm = llm or LiteLLMClient()
        # ``--methods generalization`` maps here: swap the reflection prompt for a
        # transfer-first one (GEPA's faithful prompt otherwise folds in observed
        # facts). Composed by the catalog; see build_optimizer.
        self.generalize = generalize
        self.reflect_user = _REFLECT_USER_GENERALIZE if generalize else _REFLECT_USER

    def propose(
        self,
        tools: ToolSet,
        run: RunResult,
        validate: Validator,
        scratch: Path,
        train_eval: TrainEvaluator | None = None,
    ) -> Candidate:
        if not tools.allowlist:
            return Candidate({})
        target = single_editable_target(tools)
        seed_src = tools.files.get(target, "")
        client = self.llm

        # Without a train evaluator GEPA can't search — degrade to a single
        # reflective edit over the baseline run (still useful, just not Pareto).
        if train_eval is None:
            new = self._reflect(client, seed_src, run, tools.language_rules)
            cand = (
                Candidate({target: new}) if new and new != seed_src else Candidate({})
            )
            if (
                cand.files
                and self.require_validation
                and not validate.validate(cand).ok
            ):
                return Candidate({})
            return cand

        # Unique task ids: minibatches are sampled without replacement, so any
        # duplicate here (e.g. a per-trial baseline run) would draw the same task
        # twice and be rejected downstream.
        train_ids = list(dict.fromkeys(train_eval.train_tasks))
        # Seed per-task scores come free from the baseline run (no eval spent).
        seed_scores = {
            r.task_id: r.reward for r in run.runs if r.task_id in set(train_ids)
        }
        seed_scores |= {t: seed_scores.get(t, 0.0) for t in train_ids}
        pool = _Pareto(seed_src, seed_scores)
        seed_avg = pool.best_avg()
        rng = random.Random(self.seed)
        evals = 0
        iteration = 0
        rejects: Counter[str] = Counter()

        while evals < self.max_eval_budget:
            iteration += 1
            parent_src = pool.candidates[pool.select(rng)]
            k = min(self.minibatch_size, len(train_ids))
            batch = rng.sample(train_ids, k)

            before = train_eval.evaluate(Candidate({target: parent_src}), batch)
            evals += len(batch)
            before_sum = sum(r.reward for r in before.runs)
            if all(r.reward >= 1.0 for r in before.runs):
                rejects["perfect_minibatch"] += 1
                continue  # nothing to learn from a perfect minibatch

            new_src = self._reflect(client, parent_src, before, tools.language_rules)
            if not new_src or new_src == parent_src:
                rejects["no_or_unchanged_reflection"] += 1
                continue
            cand = Candidate({target: new_src})
            if self.require_validation and not validate.validate(cand).ok:
                rejects["invalid_candidate"] += 1
                continue

            after = train_eval.evaluate(cand, batch)
            evals += len(batch)
            if sum(r.reward for r in after.runs) <= before_sum:
                rejects["no_minibatch_improvement"] += 1
                continue  # no strict improvement on the minibatch → reject

            full = train_eval.evaluate(cand, train_ids)  # promote: score on all train
            evals += len(train_ids)
            pool.add(new_src, {r.task_id: r.reward for r in full.runs})
            logger.info(
                "gepa iteration %d: promoted candidate %d (train avg %.3f)",
                iteration, len(pool.candidates) - 1, pool.best_avg(),
            )  # fmt: skip

        logger.info(
            "gepa search done: %d iterations, %d evals, %d candidates promoted, "
            "rejections=%s",
            iteration, evals, len(pool.candidates) - 1, dict(rejects),
        )  # fmt: skip
        if pool.best_avg() > seed_avg:
            return Candidate({target: pool.candidates[pool.best_idx()]})
        return Candidate({})  # nothing beat the seed → keep baseline

    def _reflect(self, client, source: str, run: RunResult, rules: str) -> str:
        """One reflective proposal: rewrite the file from the run's transcripts."""
        records = tuple(run.failing() or run.runs)

        def messages_for(transcripts: str) -> list[dict[str, str]]:
            user = self.reflect_user.format(
                source=source, transcripts=transcripts, rules=rules
            )
            return [
                {
                    "role": "system",
                    "content": _REFLECT_SYSTEM + "\n\n" + BASE_OBJECTIVE,
                },
                {"role": "user", "content": user},
            ]

        messages = None
        if not records:
            messages = messages_for("(none)")
            ensure_request_fits(
                messages,
                self.model,
                output_reserve_tokens=self.output_reserve_tokens,
                context_window_size=self.context_window_size,
            )
        else:
            # GEPA's paper controls reflection context with small minibatches.
            # Keep the largest lossless prefix of this already-randomized batch.
            for n in range(len(records), 0, -1):
                subset = RunResult(
                    run.benchmark, run.agent, records[:n], split=run.split
                )
                transcripts = summarize_transcripts(subset, max_passing=n)
                candidate_messages = messages_for(transcripts)
                try:
                    ensure_request_fits(
                        candidate_messages,
                        self.model,
                        output_reserve_tokens=self.output_reserve_tokens,
                        context_window_size=self.context_window_size,
                    )
                except OptimizerLLMFailure:
                    continue
                messages = candidate_messages
                if n < len(records):
                    logger.warning(
                        "gepa reflection subset reduced from %d to %d trajectories "
                        "to fit model=%s",
                        len(records),
                        n,
                        self.model,
                    )
                break

            if messages is None:
                # A single full structured trajectory still does not fit. Fit
                # only its bulky values; an opaque trajectory fails explicitly.
                subset = RunResult(
                    run.benchmark, run.agent, records[:1], split=run.split
                )
                fixed_messages = messages_for("")
                transcript_budget = (
                    model_context_window(self.model, self.context_window_size)
                    - self.output_reserve_tokens
                    - request_token_count(fixed_messages, self.model)
                )
                transcripts = fit_transcripts(
                    subset,
                    token_budget=transcript_budget,
                    model=self.model,
                    max_passing=1,
                    optimizer=self.id,
                )
                messages = messages_for(transcripts)
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
