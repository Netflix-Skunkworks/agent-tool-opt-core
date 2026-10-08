"""Tests for the core GEPA optimizer (LLM mocked; fake train evaluator)."""

from __future__ import annotations

import random

from agent_tool_opt_core.api import RunResult, TaskRun, ToolSet, Validator
from agent_tool_opt_core.optimizers import gepa as gepa_mod
from agent_tool_opt_core.optimizers.catalog import build_optimizer
from agent_tool_opt_core.optimizers.gepa import GEPAOptimizer
from agent_tool_opt_core.testing import FakeLLMClient


def _toolset():
    return ToolSet(
        {"tools.py": "def t():\n    'doc'\n    pass\n"},
        ("tools.py",),
        "keep importable",
    )


def _failing_run(ids=("t1", "t2")):
    return RunResult(
        "b",
        "a",
        tuple(TaskRun(t, 0.0, "-> get_x()\nerr") for t in ids),
    )


class _FakeTrainEval:
    """Scores a candidate 1.0 iff its source contains FIXED; records the task
    ids it was asked to score (to assert test-blindness)."""

    def __init__(self, ids):
        self._ids = tuple(ids)
        self.seen_tasks = []

    @property
    def train_tasks(self):
        return self._ids

    def evaluate(self, candidate, tasks=None):
        ids = list(tasks) if tasks is not None else list(self._ids)
        self.seen_tasks.append(tuple(ids))
        fixed = "FIXED" in candidate.files.get("tools.py", "")
        return RunResult(
            "b",
            "a",
            tuple(TaskRun(t, 1.0 if fixed else 0.0, "x") for t in ids),
            split="train",
        )


_FIXED = "def t():\n    'FIXED'\n    pass\n"


def test_build_and_wants_train_eval():
    opt = build_optimizer("gepa")
    assert isinstance(opt, GEPAOptimizer)
    assert opt.wants_train_eval is True  # search optimizer → gets a train evaluator


def test_gepa_finds_improvement_and_is_test_blind(tmp_path):
    te = _FakeTrainEval(("t1", "t2"))
    opt = GEPAOptimizer(
        max_eval_budget=6,
        minibatch_size=2,
        require_validation=False,
        llm=FakeLLMClient([_FIXED]),
    )
    cand = opt.propose(
        _toolset(), _failing_run(), Validator("py", ("tools.py",)), tmp_path, te
    )
    assert "FIXED" in cand.files["tools.py"]  # Pareto search promoted the fix
    assert te.seen_tasks  # it did score on train
    assert all(set(t) <= {"t1", "t2"} for t in te.seen_tasks)  # never left train


def test_gepa_degrades_to_oneshot_without_train_eval(tmp_path):
    opt = GEPAOptimizer(require_validation=False, llm=FakeLLMClient([_FIXED]))
    cand = opt.propose(
        _toolset(), _failing_run(), Validator("py", ("tools.py",)), tmp_path, None
    )
    assert "FIXED" in cand.files["tools.py"]  # one-shot reflective fallback


def test_gepa_reduces_reflection_subset_before_fitting(tmp_path):
    fake = FakeLLMClient([_FIXED])
    run = RunResult(
        "b",
        "a",
        tuple(TaskRun(f"t{i}", 0.0, "x" * 10_000) for i in range(3)),
    )
    opt = GEPAOptimizer(
        model="gpt-4o",
        context_window_size=2500,
        output_reserve_tokens=500,
        require_validation=False,
        llm=fake,
    )
    opt.propose(_toolset(), run, Validator("py", ("tools.py",)), tmp_path, None)
    user = fake.calls[0]["messages"][1]["content"]
    shown = sum(f"### task t{i}" in user for i in range(3))
    assert 0 < shown < 3
    assert "middle-elided" not in user  # subset reduction came first


def test_gepa_fits_an_oversized_single_structured_trajectory(tmp_path):
    fake = FakeLLMClient([_FIXED])
    trajectory = {
        "steps": [{"tool": "search", "result": "head " + "x" * 50_000 + " tail"}]
    }
    run = RunResult("b", "a", (TaskRun("t1", 0.0, trajectory),))
    opt = GEPAOptimizer(
        model="gpt-4o",
        context_window_size=2000,
        output_reserve_tokens=500,
        require_validation=False,
        llm=fake,
    )
    opt.propose(_toolset(), run, Validator("py", ("tools.py",)), tmp_path, None)
    user = fake.calls[0]["messages"][1]["content"]
    assert "middle-elided" in user
    assert '"tool": "search"' in user


def test_gepa_generalization_method_reflects_for_transfer(tmp_path):
    """`--methods generalization` makes GEPA reflect for transfer to unseen tasks
    (swapped prompt), not GEPA's encode-domain-facts step."""
    fake = FakeLLMClient([_FIXED])
    opt = build_optimizer("gepa", methods=["generalization"], llm=fake)
    assert isinstance(opt, GEPAOptimizer)
    assert opt.generalize is True
    assert opt.wants_train_eval is True  # still a search optimizer
    assert build_optimizer("gepa").reflect_user is gepa_mod._REFLECT_USER  # faithful

    opt.propose(
        _toolset(), _failing_run(), Validator("py", ("tools.py",)), tmp_path, None
    )
    user = fake.calls[0]["messages"][1]["content"]
    assert "unseen" in user  # transfer-first framing reached the LLM
    assert "domain-specific" not in user  # not GEPA's encode-facts step


def test_pareto_prunes_globally_dominated_candidates():
    pool = gepa_mod._Pareto("seed-src", {"t1": 0.0, "t2": 0.0})
    pool.add("better-src", {"t1": 1.0, "t2": 1.0})  # idx 1 dominates the seed (idx 0)
    nd = pool._nondominated()
    assert 1 in nd and 0 not in nd  # a candidate beaten on every task is pruned
    rng = random.Random(0)
    assert all(
        pool.select(rng) == 1 for _ in range(20)
    )  # never picks the dominated seed
