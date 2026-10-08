"""Tests for the train-only evaluator handed to search/select optimizers."""

from __future__ import annotations

import pytest

from agent_tool_opt_core import driver
from agent_tool_opt_core.api import (
    Agent,
    Benchmark,
    Candidate,
    Optimizer,
    RunResult,
    TaskRun,
    ToolSet,
    ToolTarget,
)


class _Agent(Agent):
    id = "subj"


class _Bench(Benchmark):
    name = "fake"

    def tasks(self, split):
        return {"train": ["t1", "t2"], "test": ["e1"]}[split]

    def evaluate(self, agent, tasks, tools):
        # reward 1.0 iff the candidate marker is in the *effective* toolset
        ok = "OPT" in tools.files.get("tools.py", "")
        return RunResult(
            self.name, agent.id, tuple(TaskRun(t, 1.0 if ok else 0.0) for t in tasks)
        )


class _TT(ToolTarget):
    kind = "py"
    language_rules = "rules"

    def __init__(self):
        self._live = "base"

    def extract(self):
        return ToolSet({"tools.py": "base"}, ("tools.py",), self.language_rules)

    def effective_toolset(self):
        return ToolSet({"tools.py": self._live}, ("tools.py",), self.language_rules)

    def apply(self, c):
        self._live = c.files.get("tools.py", self._live)

    def restore(self):
        self._live = "base"


def test_train_evaluator_scores_train_and_is_test_blind():
    tt = _TT()
    te = driver._TrainEvaluator(_Bench(), _Agent(), tt, ["t1", "t2"])
    assert te.train_tasks == ("t1", "t2")

    run = te.evaluate(Candidate({"tools.py": "OPT"}))
    assert run.split == "train"  # always stamped train
    assert len(run.runs) == 2 and all(r.reward == 1.0 for r in run.runs)
    assert tt.effective_toolset().files["tools.py"] == "base"  # restored after eval

    sub = te.evaluate(Candidate({"tools.py": "OPT"}), ["t1"])
    assert [r.task_id for r in sub.runs] == ["t1"]  # minibatch subset honored

    with pytest.raises(ValueError, match="outside train split"):
        te.evaluate(Candidate({"tools.py": "OPT"}), ["e1"])  # test id rejected


class _Spy(Optimizer):
    id = "spy"

    def __init__(self, wants: bool):
        self.wants_train_eval = wants
        self.seen = "unset"

    def propose(self, tools, run, validate, scratch, train_eval=None):
        self.seen = train_eval
        return Candidate({})


def test_driver_passes_train_eval_only_to_opt_in_optimizers(tmp_path):
    b, a, tt = _Bench(), _Agent(), _TT()

    search = _Spy(wants=True)
    driver.optimize(
        b, a, tt, search, train=["t1"], test=["e1"], scratch=tmp_path, n_iters=1
    )
    assert search.seen is not None and hasattr(search.seen, "evaluate")

    blind = _Spy(wants=False)
    driver.optimize(
        b, a, tt, blind, train=["t1"], test=["e1"], scratch=tmp_path, n_iters=1
    )
    assert blind.seen is None  # blind optimizer cannot probe reward


def test_search_cost_counts_each_evaluation_and_preserves_partial_failure(monkeypatch):
    from agent_tool_opt_core.costs import collect_optimizer_costs

    benchmark, target = _Bench(), _TT()
    complete = RunResult("fake", "subj", (TaskRun("t1", 1, cost_usd=2),))
    monkeypatch.setattr(benchmark, "evaluate", lambda *args: complete)
    evaluator = driver.make_train_evaluator(benchmark, _Agent(), target, ["t1"])
    with collect_optimizer_costs() as costs:
        evaluator.evaluate(Candidate({"tools.py": "OPT"}))
        evaluator.evaluate(Candidate({"tools.py": "OPT"}))
        assert costs.search.cost_usd == 4
        error = RuntimeError("later simulation failed")
        error.partial_run = RunResult(
            "fake", "subj", (TaskRun("t1", 0, known_cost_usd=1),)
        )

        def fail(*args):
            raise error

        monkeypatch.setattr(benchmark, "evaluate", fail)
        with pytest.raises(RuntimeError):
            evaluator.evaluate(Candidate({"tools.py": "OPT"}))
        assert costs.search.cost_usd is None
        assert costs.search.known_usd == 5
        assert costs.llm.cost_usd == 0
    assert target._live == "base"
