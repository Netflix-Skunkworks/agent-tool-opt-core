"""Tests for the 3-phase driver and the Experiment matrix engine.

Uses small in-test doubles (a deterministic benchmark, two agents of
different skill, an in-process ToolTarget, and identity/marker optimizers)
to prove the load-bearing properties: the driver applies+restores, the
optimizer only ever gets a Validator, the Experiment shares one baseline
across optimizers, evaluates cross-agent, and reports paired deltas.
"""

from __future__ import annotations

import json

from agent_tool_opt_core.api import (
    Agent,
    Benchmark,
    Candidate,
    Optimizer,
    RunResult,
    TaskRun,
    ToolSet,
    ToolTarget,
    Validator,
)
from agent_tool_opt_core.driver import (
    collect_baseline,
    evaluate_candidate,
    optimize,
    propose,
)
from agent_tool_opt_core.experiment import BASELINE, Experiment, ExperimentSpec
from agent_tool_opt_core.optimizers._common import (
    OptimizerInfrastructureFailure,
    OptimizerLLMFailure,
)
from agent_tool_opt_core.orchestration import ProposalRetryPolicy

MARKER = "# OPTIMIZED"
WEAK_TOOL = "weather.py"


# --- doubles ---------------------------------------------------------------


class FakeAgent(Agent):
    def __init__(self, id: str, skill: float) -> None:
        self.id = id
        self.skill = skill


class FakeBenchmark(Benchmark):
    """A task passes iff the weak tool was optimized OR the agent is strong."""

    def __init__(self, name: str = "fake", n: int = 2) -> None:
        self.name = name
        self._n = n

    def tasks(self, split: str) -> list[str]:
        return [f"{split}-{i}" for i in range(self._n)]

    def evaluate(self, agent: Agent, tasks, tools: ToolSet) -> RunResult:
        optimized = MARKER in tools.files.get(WEAK_TOOL, "")
        strong = getattr(agent, "skill", 1.0) >= 0.8
        reward = 1.0 if (optimized or strong) else 0.0
        return RunResult(
            self.name,
            agent.id,
            tuple(TaskRun(t, reward) for t in tasks),
        )


class InProcessToolTarget(ToolTarget):
    kind = "py"
    language_rules = "edit weather.py; keep it importable"

    def __init__(self) -> None:
        self._baseline = {WEAK_TOOL: "def weather(): ..."}
        self._live = dict(self._baseline)

    def extract(self) -> ToolSet:
        return ToolSet(dict(self._baseline), (WEAK_TOOL,), self.language_rules)

    def effective_toolset(self) -> ToolSet:
        return ToolSet(dict(self._live), (WEAK_TOOL,), self.language_rules)

    def apply(self, c: Candidate) -> None:
        self._live.update(c.files)

    def restore(self) -> None:
        self._live = dict(self._baseline)


class IdentityOptimizer(Optimizer):
    id = "identity"

    def propose(self, tools, run, validate, scratch, train_eval=None):
        return Candidate({})


class MarkerOptimizer(Optimizer):
    """Edits the failing tool to add the marker; self-checks via the validator."""

    id = "marker"

    def __init__(self) -> None:
        self.seen_validate = None

    def propose(
        self,
        tools: ToolSet,
        run: RunResult,
        validate: Validator,
        scratch,
        train_eval=None,
    ):
        self.seen_validate = validate
        edited = {
            rel: tools.files[rel] + "\n" + MARKER
            for rel in tools.allowlist
            if rel in tools.files
        }
        cand = Candidate(edited)
        assert validate.validate(cand).ok  # inner gate, Pi-style
        return cand


def _agents():
    return FakeAgent("weak", 0.5), FakeAgent("strong", 0.9)


# --- driver ----------------------------------------------------------------


def test_collect_baseline_uses_unmodified_tools():
    weak, _ = _agents()
    b, tt = FakeBenchmark(), InProcessToolTarget()
    run = collect_baseline(b, weak, b.tasks("train"), tt)
    assert b.score(run).pass_rate == 0.0  # weak agent fails on baseline tools


def test_evaluate_candidate_applies_then_restores(tmp_path):
    weak, _ = _agents()
    b, tt = FakeBenchmark(), InProcessToolTarget()
    cand = MarkerOptimizer().propose(
        tt.extract(),
        collect_baseline(b, weak, b.tasks("train"), tt),
        tt.validator(),
        tmp_path,
    )
    run = evaluate_candidate(b, weak, tt, cand, b.tasks("test"))
    assert b.score(run).pass_rate == 1.0  # weak agent now passes
    # restored: the live toolset no longer carries the marker
    assert MARKER not in tt.effective_toolset().files[WEAK_TOOL]


def test_optimizer_only_receives_a_validator(tmp_path):
    weak, _ = _agents()
    b, tt = FakeBenchmark(), InProcessToolTarget()
    opt = MarkerOptimizer()
    propose(opt, tt, collect_baseline(b, weak, b.tasks("train"), tt), tmp_path)
    assert isinstance(opt.seen_validate, Validator)
    assert not hasattr(opt.seen_validate, "apply")
    assert not hasattr(opt.seen_validate, "evaluate")


def test_propose_rejects_test_split_run(tmp_path):
    """Leakage gate: the optimizer may train on a 'train' run, never a 'test'
    one (the data-side twin of the Validator-only boundary)."""
    import pytest
    from dataclasses import replace

    weak, _ = _agents()
    b, tt = FakeBenchmark(), InProcessToolTarget()
    train = collect_baseline(b, weak, b.tasks("train"), tt)
    assert train.split == "train"
    propose(IdentityOptimizer(), tt, train, tmp_path)  # train accepted
    with pytest.raises(ValueError, match="train/test"):
        propose(IdentityOptimizer(), tt, replace(train, split="test"), tmp_path)


def test_optimize_keeps_best_and_reports_test_metrics(tmp_path):
    weak, _ = _agents()
    b, tt = FakeBenchmark(), InProcessToolTarget()
    cand, metrics = optimize(
        b,
        weak,
        tt,
        MarkerOptimizer(),
        train=b.tasks("train"),
        test=b.tasks("test"),
        scratch=tmp_path,
    )
    assert cand is not None and metrics.pass_rate == 1.0


def test_optimize_identity_keeps_baseline(tmp_path):
    weak, _ = _agents()
    b, tt = FakeBenchmark(), InProcessToolTarget()
    cand, metrics = optimize(
        b,
        weak,
        tt,
        IdentityOptimizer(),
        train=b.tasks("train"),
        test=b.tasks("test"),
        scratch=tmp_path,
    )
    assert cand is None and metrics.pass_rate == 0.0  # no improvement, baseline kept


# --- experiment ------------------------------------------------------------


def test_experiment_shares_one_baseline_across_optimizers(tmp_path):
    weak, strong = _agents()
    spec = ExperimentSpec(
        benchmarks=[FakeBenchmark("b1"), FakeBenchmark("b2")],
        subject_agents=[weak],
        optimizers=[IdentityOptimizer(), MarkerOptimizer()],
        tool_target_for=lambda b: InProcessToolTarget(),
        eval_agents=[weak, strong],
    )
    experiment = Experiment()
    table = experiment.run(spec, tmp_path)
    # ONE baseline per (benchmark, subject_agent) — NOT per optimizer.
    assert table.baseline_runs == 2  # 2 benchmarks x 1 subject agent


class _TrainEvalSpy(Optimizer):
    id = "te-spy"
    wants_train_eval = True

    def __init__(self):
        self.got = "unset"

    def propose(self, tools, run, validate, scratch, train_eval=None):
        self.got = train_eval
        return Candidate({})


def test_experiment_passes_train_eval_to_search_optimizers(tmp_path):
    weak, _ = _agents()
    spy = _TrainEvalSpy()
    spec = ExperimentSpec(
        benchmarks=[FakeBenchmark("b1")],
        subject_agents=[weak],
        optimizers=[spy, IdentityOptimizer()],  # search opt + a blind one
        tool_target_for=lambda b: InProcessToolTarget(),
    )
    Experiment().run(spec, tmp_path)
    # the search optimizer got a train-only evaluator (GEPA-in-sweeps fix); a
    # blind optimizer would have received None.
    assert spy.got is not None and hasattr(spy.got, "evaluate")
    assert spy.got.train_tasks  # bound to the train split


def test_experiment_cross_agent_matrix_and_deltas(tmp_path):
    weak, strong = _agents()
    spec = ExperimentSpec(
        benchmarks=[FakeBenchmark("b1")],
        subject_agents=[weak],
        optimizers=[IdentityOptimizer(), MarkerOptimizer()],
        tool_target_for=lambda b: InProcessToolTarget(),
        eval_agents=[weak, strong],
    )
    exp = Experiment()
    table = exp.run(spec, tmp_path)

    # a row for every (optimizer ∪ baseline) x eval_agent
    optimizers = {r.optimizer for r in table.rows}
    assert optimizers == {BASELINE, "identity", "marker"}
    eval_agents = {r.eval_agent for r in table.rows}
    assert eval_agents == {"weak", "strong"}

    deltas = {
        (d["optimizer"], d["eval_agent"]): d["delta_reward"] for d in exp.analyze(table)
    }
    # marker lifts the WEAK agent (0 -> 1); strong already passed (delta 0); identity never helps
    assert deltas[("marker", "weak")] == 1.0
    assert deltas[("marker", "strong")] == 0.0
    assert deltas[("identity", "weak")] == 0.0


class _FailingOptimizer(Optimizer):
    def __init__(self, failure):
        self.failure = failure
        self.id = f"failing-{type(failure).__name__}"
        self.calls = 0

    def propose(self, tools, run, validate, scratch, train_eval=None):
        self.calls += 1
        raise self.failure


class _OuterInvalidOptimizer(Optimizer):
    id = "outer-invalid"

    def propose(self, tools, run, validate, scratch, train_eval=None):
        return Candidate({"outside.py": "x"})


def test_experiment_scores_candidate_failure_as_zero_without_evaluation(tmp_path):
    weak, _ = _agents()
    optimizer = _FailingOptimizer(OptimizerLLMFailure("context window"))
    spec = ExperimentSpec(
        benchmarks=[FakeBenchmark()],
        subject_agents=[weak],
        optimizers=[optimizer],
        tool_target_for=lambda b: InProcessToolTarget(),
    )

    experiment = Experiment()
    table = experiment.run(spec, tmp_path)
    row = next(row for row in table.rows if row.optimizer == optimizer.id)

    assert row.status == "candidate_failed"
    assert row.avg_reward == 0.0
    assert row.proposal["failure_kind"] == "candidate"
    assert optimizer.calls == 1
    assert experiment.analyze(table)[0]["delta_reward"] == 0.0


def test_experiment_excludes_persistent_infrastructure_failure(tmp_path):
    weak, _ = _agents()
    optimizer = _FailingOptimizer(OptimizerInfrastructureFailure("gateway 503"))
    spec = ExperimentSpec(
        benchmarks=[FakeBenchmark()],
        subject_agents=[weak],
        optimizers=[optimizer],
        tool_target_for=lambda b: InProcessToolTarget(),
        proposal_retry=ProposalRetryPolicy(max_attempts=2, base_delay=0, jitter=0),
    )
    experiment = Experiment()

    table = experiment.run(spec, tmp_path)
    row = next(row for row in table.rows if row.optimizer == optimizer.id)

    assert row.status == "incomplete"
    assert row.avg_reward is None
    assert row.proposal["failure_kind"] == "infrastructure"
    assert optimizer.calls == 2
    assert experiment.analyze(table) == []


def test_experiment_updates_proposal_artifact_after_outer_validation(tmp_path):
    weak, _ = _agents()
    spec = ExperimentSpec(
        benchmarks=[FakeBenchmark()],
        subject_agents=[weak],
        optimizers=[_OuterInvalidOptimizer()],
        tool_target_for=lambda b: InProcessToolTarget(),
    )

    table = Experiment().run(spec, tmp_path)

    row = next(row for row in table.rows if row.optimizer == "outer-invalid")
    persisted = json.loads(
        (tmp_path / "fake__weak__outer-invalid" / "proposal.json").read_text()
    )
    assert row.status == "candidate_failed"
    assert row.proposal == persisted
    assert persisted["attempt_count"] == 1
    assert persisted["exception_type"] == "ValidationFailure"
