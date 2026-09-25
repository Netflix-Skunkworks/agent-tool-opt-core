"""Tests for the provider-neutral ToolObserver optimizer."""

from __future__ import annotations

import pytest

from agent_tool_opt_core.api import (
    RunResult,
    TaskRun,
    ToolSet,
    ValidationResult,
    Validator,
)
from agent_tool_opt_core.optimizers._common import OptimizerLLMFailure
from agent_tool_opt_core.optimizers.catalog import build_optimizer
from agent_tool_opt_core.optimizers.toolobserver import ToolObserverOptimizer
from agent_tool_opt_core.testing import FakeLLMClient


def _toolset():
    return ToolSet(
        {"tools.py": "def t():\n    'doc'\n    pass\n"},
        ("tools.py",),
        "keep importable",
    )


def _run():
    return RunResult(
        "b",
        "gpt",
        (
            TaskRun("t1", 0.0, trajectory="-> get_x()\nerr"),
            TaskRun("t2", 1.0, trajectory="-> get_y()\nok"),
        ),
    )


def test_build_and_blind():
    opt = build_optimizer("toolobserver")
    assert isinstance(opt, ToolObserverOptimizer)
    assert opt.wants_train_eval is False  # never probes reward


def test_batch_then_consensus(tmp_path):
    # batch_size=1 + 2 tasks => 2 batches => consensus merge (3rd call).
    fake = FakeLLMClient([
        "def t():\n    'doc v1'\n    pass\n",
        "def t():\n    'doc v2'\n    pass\n",
        "def t():\n    'doc merged'\n    pass\n",
    ])  # fmt: skip
    opt = ToolObserverOptimizer(batch_size=1, require_validation=False, llm=fake)
    cand = opt.propose(_toolset(), _run(), Validator("py", ("tools.py",)), tmp_path)
    assert cand.files["tools.py"] == "def t():\n    'doc merged'\n    pass\n"
    assert len(fake.calls) == 3  # 2 batch analyses + 1 consensus merge


def test_single_batch_skips_merge(tmp_path):
    fake = FakeLLMClient(["def t():\n    'doc improved'\n    pass\n"])
    opt = ToolObserverOptimizer(batch_size=10, require_validation=False, llm=fake)
    cand = opt.propose(_toolset(), _run(), Validator("py", ("tools.py",)), tmp_path)
    assert "doc improved" in cand.files["tools.py"]
    assert len(fake.calls) == 1  # single batch => no consensus merge


def test_token_aware_batches_split_before_fitting(tmp_path):
    run = RunResult(
        "b",
        "gpt",
        tuple(TaskRun(f"t{i}", 0.0, "x" * 10_000) for i in range(3)),
    )
    fake = FakeLLMClient(
        [
            "def t():\n    'v1'\n    pass\n",
            "def t():\n    'v2'\n    pass\n",
            "def t():\n    'v3'\n    pass\n",
            "def t():\n    'merged'\n    pass\n",
        ]
    )
    opt = ToolObserverOptimizer(
        model="gpt-4o",
        batch_size=10,
        context_window_size=2500,
        output_reserve_tokens=500,
        require_validation=False,
        llm=fake,
    )
    opt.propose(_toolset(), run, Validator("py", ("tools.py",)), tmp_path)
    assert len(fake.calls) == 4  # 3 token-sized batches + consensus
    for call in fake.calls[:3]:
        assert call["messages"][1]["content"].count("### task") == 1


def test_oversized_single_structured_trajectory_is_fitted(tmp_path):
    trajectory = {
        "steps": [{"tool": "search", "result": "head " + "x" * 50_000 + " tail"}]
    }
    run = RunResult("b", "gpt", (TaskRun("t1", 0.0, trajectory),))
    fake = FakeLLMClient(["def t():\n    'improved'\n    pass\n"])
    opt = ToolObserverOptimizer(
        model="gpt-4o",
        context_window_size=2000,
        output_reserve_tokens=500,
        require_validation=False,
        llm=fake,
    )
    opt.propose(_toolset(), run, Validator("py", ("tools.py",)), tmp_path)
    user = fake.calls[0]["messages"][1]["content"]
    assert "middle-elided" in user
    assert '"tool": "search"' in user


class _AlwaysInvalid(Validator):
    def validate(self, c):
        return ValidationResult(False, "not valid python")


def test_corrective_remerge_is_validated_and_raises_on_failure(tmp_path):
    # The corrective re-merge used to be returned WITHOUT re-validating it, so a
    # candidate that does not even parse reached the eval and was scored as a real
    # reward number — the same failure shape as the llm optimizer's bug.
    fake = FakeLLMClient([
        "def t():\n    'v1'\n    pass\n",  # batch 1
        "def t():\n    'v2'\n    pass\n",  # batch 2
        "still not valid python (",  # merge
        "STILL not valid python (",  # corrective re-merge
    ])  # fmt: skip
    opt = ToolObserverOptimizer(batch_size=1, llm=fake)
    with pytest.raises(OptimizerLLMFailure):
        opt.propose(_toolset(), _run(), _AlwaysInvalid("py", ("tools.py",)), tmp_path)
    assert len(fake.calls) == 4  # the corrective retry still happens
