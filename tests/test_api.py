"""Tests for the v2 tool-optimization API (``agent_tool_opt_core.api``).

These lock the invariants the refactor relies on: a single ``Candidate``
artifact, an optimizer that only ever receives a (picklable, validate-only)
``Validator``, and ToolTargets whose apply/restore round-trips.
"""

from __future__ import annotations

import pickle

import pytest

from agent_tool_opt_core.api import (
    Agent,
    Benchmark,
    Candidate,
    Metrics,
    Optimizer,
    RunResult,
    TaskRun,
    ToolSet,
    ToolTarget,
    Validator,
    score,
)


# ---------------------------------------------------------------------------
# Validator — reconstructable across remote workers + validate-only + gate
# ---------------------------------------------------------------------------


def test_validator_is_picklable_and_equal_after_roundtrip():
    v = Validator("py", ("tools.py", "helper.py"))
    v2 = pickle.loads(pickle.dumps(v))
    assert v2 == v
    assert v2.validate(Candidate({"tools.py": "x = 1"})).ok


def test_validator_rejects_off_allowlist():
    res = Validator("py", ("tools.py",)).validate(Candidate({"evil.py": "x = 1"}))
    assert not res.ok
    assert "allowlist" in res.log


def test_validator_rejects_empty_file():
    assert not Validator("py", ("tools.py",)).validate(Candidate({"tools.py": "  "})).ok


def test_validator_accepts_clean_candidate():
    res = Validator("py", ("tools.py",)).validate(
        Candidate({"tools.py": "def f():\n    return 1\n"})
    )
    assert res.ok


def test_validator_is_validate_only():
    """The only capability an optimizer gets — never apply/evaluate."""
    v = Validator("py", ("tools.py",))
    assert not hasattr(v, "apply")
    assert not hasattr(v, "evaluate")


# ---------------------------------------------------------------------------
# score() + Metrics + RunResult views
# ---------------------------------------------------------------------------


def test_score_avg_and_pass_rate():
    run = RunResult(
        "b", "a", (TaskRun("t1", 1.0), TaskRun("t2", 0.0), TaskRun("t3", 1.0))
    )
    m = score(run)
    assert m.n == 3
    assert m.avg_reward == pytest.approx(2 / 3)
    assert m.pass_rate == pytest.approx(2 / 3)


def test_score_empty_run_is_zero():
    assert score(RunResult("b", "a", ())) == Metrics(0.0, 0.0, 0)


def test_metrics_better_than():
    assert Metrics(0.8, 0.8, 5).better_than(Metrics(0.5, 0.5, 5))
    assert not Metrics(0.5, 0.5, 5).better_than(Metrics(0.5, 0.5, 5))
    # ties broken by pass_rate
    assert Metrics(0.5, 0.9, 5).better_than(Metrics(0.5, 0.4, 5))


def test_runresult_transcripts_and_failing():
    runs = (TaskRun("t1", 1.0), TaskRun("t2", 0.0))
    rr = RunResult("b", "a", runs)
    assert rr.transcripts == runs
    assert rr.failing() == (TaskRun("t2", 0.0),)


# ---------------------------------------------------------------------------
# Roles are implementable; ToolTarget contract
# ---------------------------------------------------------------------------


class _InProcTarget(ToolTarget):
    """Minimal in-process ToolTarget (tau2-style state mutation)."""

    kind = "py"
    language_rules = "edit tools.py"

    def __init__(self) -> None:
        self._baseline = {"tools.py": "def f(): pass"}
        self._live = dict(self._baseline)

    def extract(self) -> ToolSet:
        return ToolSet(dict(self._baseline), ("tools.py",), self.language_rules)

    def effective_toolset(self) -> ToolSet:
        return ToolSet(dict(self._live), ("tools.py",), self.language_rules)

    def apply(self, c: Candidate) -> None:
        self._live.update(c.files)

    def restore(self) -> None:
        self._live = dict(self._baseline)


def test_tooltarget_validator_derives_kind_and_allowlist():
    v = _InProcTarget().validator()
    assert v.kind == "py"
    assert v.allowlist == ("tools.py",)


def test_tooltarget_apply_restore_roundtrip():
    tt = _InProcTarget()
    base = tt.extract().files["tools.py"]
    tt.apply(Candidate({"tools.py": "EDITED"}))
    assert tt.effective_toolset().files["tools.py"] == "EDITED"
    tt.restore()
    assert tt.effective_toolset().files["tools.py"] == base


def test_abstract_tooltarget_cannot_instantiate():
    class Bad(ToolTarget):
        kind = "py"
        language_rules = ""

    with pytest.raises(TypeError):
        Bad()


def test_minimal_benchmark_agent_optimizer_compose(tmp_path):
    class _Subject(Agent):
        id = "subject"

    class _Bench(Benchmark):
        name = "bench"

        def tasks(self, split):
            return ["t1", "t2"]

        def evaluate(self, agent, tasks, tools):
            # every task passes when the tool was optimized
            opt = "OPTIMIZED" in tools.files.get("tools.py", "")
            return RunResult(
                self.name,
                agent.id,
                tuple(TaskRun(t, 1.0 if opt else 0.0) for t in tasks),
            )

    class _Opt(Optimizer):
        id = "marker"

        def propose(self, tools, run, validate, scratch):
            cand = Candidate({"tools.py": tools.files["tools.py"] + "\n# OPTIMIZED"})
            assert validate.validate(cand).ok  # optimizer self-checks via validator
            return cand

    a, b, o, tt = _Subject(), _Bench(), _Opt(), _InProcTarget()
    baseline = b.evaluate(a, b.tasks("train"), tt.extract())
    assert b.score(baseline).pass_rate == 0.0

    cand = o.propose(tt.extract(), baseline, tt.validator(), tmp_path)
    tt.apply(cand)
    after = b.evaluate(a, b.tasks("train"), tt.effective_toolset())
    assert b.score(after).pass_rate == 1.0
