"""Tests for the provider-neutral DRAFT optimizer implementation."""

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
from agent_tool_opt_core.optimizers.draft import DRAFTOptimizer
from agent_tool_opt_core.testing import FakeLLMClient


def _patch_two_stage(mocker, analysis, rewrite, calls):
    fake = FakeLLMClient([analysis, rewrite])
    complete = fake.complete

    def record(**kw):
        calls.append(kw)
        return complete(**kw)

    fake.complete = record
    mocker.patch(
        "agent_tool_opt_core.optimizers.draft.LiteLLMClient",
        return_value=fake,
    )


def _toolset():
    return ToolSet({"tools.py": "def t(): pass\n"}, ("tools.py",), "keep importable")


def _run():
    return RunResult(
        "b", "gpt", (TaskRun("t1", 0.0, trajectory="tried get_x, failed"),)
    )


def test_draft_is_analyze_then_rewrite(mocker, tmp_path):
    calls = []
    _patch_two_stage(
        mocker, "issue: get_x doc is vague", "def t():\n    return 2\n", calls
    )
    cand = DRAFTOptimizer(model="m", context_window_size=100_000).propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
    )
    # two LLM calls: analyze, then rewrite
    assert len(calls) == 2
    # the analysis is fed into the rewrite prompt
    assert "issue: get_x doc is vague" in calls[1]["messages"][1]["content"]
    # candidate is the rewritten source
    assert cand.files == {"tools.py": "def t():\n    return 2\n"}


def test_draft_strips_fences(mocker, tmp_path):
    _patch_two_stage(mocker, "analysis", "```python\nx = 1\n```", [])
    cand = DRAFTOptimizer().propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
    )
    assert cand.files["tools.py"] == "x = 1\n"


def test_draft_fits_oversized_structured_transcripts(mocker, tmp_path):
    calls = []
    _patch_two_stage(mocker, "analysis", "def t():\n    return 2\n", calls)
    trajectory = {
        "steps": [
            {"tool": f"tool_{i}", "result": "head " + "x" * 5000 + " tail"}
            for i in range(5)
        ]
    }
    run = RunResult("b", "gpt", (TaskRun("t1", 0.0, trajectory),))
    DRAFTOptimizer(
        model="gpt-4o", context_window_size=3000, output_reserve_tokens=500
    ).propose(_toolset(), run, Validator("py", ("tools.py",)), tmp_path)
    prompt = calls[0]["messages"][1]["content"]
    assert "middle-elided" in prompt
    assert all(f'"tool": "tool_{i}"' in prompt for i in range(5))


class _AlwaysInvalid(Validator):
    def validate(self, c):
        return ValidationResult(False, "not valid python")


def test_draft_raises_instead_of_returning_an_invalid_rewrite(mocker, tmp_path):
    # draft is single-shot and used to return the rewrite unvalidated, so an
    # unparseable file reached the eval and was scored as a real reward number
    # (the same failure the llm optimizer's retry-exhaustion raise fixed).
    _patch_two_stage(mocker, "analysis", "not valid python (", [])
    with pytest.raises(OptimizerLLMFailure):
        DRAFTOptimizer().propose(
            _toolset(), _run(), _AlwaysInvalid("py", ("tools.py",)), tmp_path
        )


def test_draft_can_opt_out_of_validation(mocker, tmp_path):
    # The ablation arm keeps the old behaviour explicitly rather than by default.
    _patch_two_stage(mocker, "analysis", "whatever", [])
    cand = DRAFTOptimizer(require_validation=False).propose(
        _toolset(), _run(), _AlwaysInvalid("py", ("tools.py",)), tmp_path
    )
    assert cand.files == {"tools.py": "whatever\n"}
