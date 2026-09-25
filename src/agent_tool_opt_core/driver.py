"""The single-run optimization driver — three composable phases.

Benchmark/agent-agnostic; operates purely on the ``api`` roles. The
``Experiment`` layer (experiment.py) composes these across a matrix and
shares the (expensive) baseline phase.

Boundary: a blind ``Optimizer`` is handed ``validate`` (the cheap language
gate) and nothing else — never ``apply``/``evaluate`` — so it cannot see the
reward. Search/select optimizers (``wants_train_eval``) additionally receive a
``TrainEvaluator`` that scores candidates on the *train* split only, never test
(the "train jail": an optimizer may see the train signal, never test).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_tool_opt_core.costs import active_costs

from agent_tool_opt_core.api import (
    Agent,
    Benchmark,
    Candidate,
    Metrics,
    Optimizer,
    RunResult,
    TaskId,
    ToolTarget,
    TrainEvaluator,
)


def collect_baseline(
    benchmark: Benchmark,
    agent: Agent,
    tasks: list[TaskId],
    tool_target: ToolTarget,
    *,
    split: str = "train",
) -> RunResult:
    """Run ``agent`` on ``tasks`` with the unmodified tools (shared baseline).

    Stamps the run's ``split`` so the leakage gate in ``propose`` can enforce
    that an optimizer only ever sees the *train* split.
    """
    return replace(benchmark.evaluate(agent, tasks, tool_target.extract()), split=split)


def propose(
    optimizer: Optimizer,
    tool_target: ToolTarget,
    baseline: RunResult,
    scratch: Path,
    train_eval: TrainEvaluator | None = None,
) -> Candidate:
    """Ask ``optimizer`` for a candidate from the baseline transcripts.

    The optimizer receives the tool source + the run + a ``Validator`` and
    nothing else — and only ever the *train*-split run (never test): this is the
    train/test leakage gate, the data-side twin of the Validator-only (no-reward)
    boundary. The gate is fail-closed — the run must be explicitly stamped
    ``split="train"`` (an unstamped ``None`` is rejected too, so a forgotten
    stamp fails loudly rather than leaking). A search optimizer may also receive
    ``train_eval`` (train-only scoring); it is passed only when
    ``optimizer.wants_train_eval`` (else ``None``).
    """
    if baseline.split != "train":
        raise ValueError(
            "Optimizer must receive the train-split baseline only (train/test "
            f"leakage gate): got split={baseline.split!r}. Stamp the run "
            'split="train" (e.g. via driver.collect_baseline) before propose().'
        )
    return optimizer.propose(
        tool_target.extract(), baseline, tool_target.validator(), scratch, train_eval
    )


def evaluate_candidate(
    benchmark: Benchmark,
    agent: Agent,
    tool_target: ToolTarget,
    candidate: Candidate,
    tasks: list[TaskId],
) -> RunResult:
    """Apply ``candidate``, run ``agent`` on ``tasks``, then always restore."""
    try:
        tool_target.apply(candidate)
        return benchmark.evaluate(agent, tasks, tool_target.effective_toolset())
    finally:
        tool_target.restore()  # always restore, even if apply/evaluate raised


class _TrainEvaluator(TrainEvaluator):
    """Driver-provided train-only scorer (see ``api.TrainEvaluator``).

    Bound to the train task pool; reuses ``evaluate_candidate`` (apply →
    evaluate → restore) and stamps ``split="train"`` so a search/select
    optimizer can score candidates on train without ever touching test.
    """

    def __init__(
        self,
        benchmark: Benchmark,
        agent: Agent,
        tool_target: ToolTarget,
        train_tasks: list[TaskId],
    ) -> None:
        self._benchmark = benchmark
        self._agent = agent
        self._tool_target = tool_target
        self._train = tuple(train_tasks)

    @property
    def train_tasks(self) -> tuple[TaskId, ...]:
        return self._train

    def evaluate(
        self, candidate: Candidate, tasks: list[TaskId] | None = None
    ) -> RunResult:
        ids = list(tasks) if tasks is not None else list(self._train)
        unknown = set(ids) - set(self._train)
        if unknown:
            raise ValueError(
                f"TrainEvaluator: tasks outside train split: {sorted(unknown)}"
            )
        costs = active_costs()
        try:
            run = evaluate_candidate(
                self._benchmark, self._agent, self._tool_target, candidate, ids
            )
        except Exception as exc:
            if costs is not None:
                partial = getattr(exc, "partial_run", None)
                if partial is not None:
                    costs.search.add_run(partial)
                costs.search.add(None, reason="search_evaluation_failed")
            raise
        if costs is not None:
            costs.search.add_run(run)
            if not run.runs:
                costs.search.add(None, reason="search_results_missing")
        return replace(run, split="train")


def make_train_evaluator(
    benchmark: Benchmark,
    agent: Agent,
    tool_target: ToolTarget,
    train_tasks: list[TaskId],
) -> TrainEvaluator:
    """Public constructor for a train-only evaluator (see ``api.TrainEvaluator``).

    For harnesses that drive ``propose`` directly (``Experiment``, the tau2
    ``run_local``) and must hand a search optimizer (``wants_train_eval``) its
    train-split scorer. ``driver.optimize`` builds one itself.
    """
    return _TrainEvaluator(benchmark, agent, tool_target, train_tasks)


def optimize(
    benchmark: Benchmark,
    agent: Agent,
    tool_target: ToolTarget,
    optimizer: Optimizer,
    *,
    train: list[TaskId],
    test: list[TaskId],
    scratch: Path,
    n_iters: int = 1,
) -> tuple[Candidate | None, Metrics]:
    """The 1x1x1 case: baseline → propose → (validate) → evaluate → keep best.

    Returns ``(best_candidate, test_metrics)``. ``best_candidate`` is None
    if no proposed candidate beat the baseline (then the baseline is kept).
    """
    validator = tool_target.validator()
    # Search/select optimizers (GEPA, k-fold) get a train-only evaluator; blind
    # optimizers get None (cannot probe reward). Bound to train → never test.
    train_eval = (
        _TrainEvaluator(benchmark, agent, tool_target, train)
        if getattr(optimizer, "wants_train_eval", False)
        else None
    )
    baseline = collect_baseline(benchmark, agent, train, tool_target)
    best_candidate: Candidate | None = None
    best_train = benchmark.score(baseline)

    for i in range(n_iters):
        # propose() always sees the shared train *baseline* — never a prior
        # candidate's run — so multi-iteration proposals stay independent of one
        # another and the (train-stamped) leakage gate is always satisfied.
        candidate = propose(
            optimizer, tool_target, baseline, scratch / f"iter_{i}", train_eval
        )
        if not validator.validate(candidate).ok:
            continue  # cheap gate; driver double-checks the optimizer
        cand_run = evaluate_candidate(benchmark, agent, tool_target, candidate, train)
        m = benchmark.score(cand_run)
        if m.better_than(best_train):
            best_candidate, best_train = candidate, m

    if best_candidate is not None:
        test_run = evaluate_candidate(
            benchmark, agent, tool_target, best_candidate, test
        )
    else:
        test_run = collect_baseline(benchmark, agent, test, tool_target, split="test")
    return best_candidate, benchmark.score(test_run)
