"""Tests for the core LLM optimizer (bug #11: invalid-Python candidates)."""

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
from agent_tool_opt_core.optimizers.llm import LLMOptimizer
from agent_tool_opt_core.testing import FakeLLMClient


class _AlwaysInvalid(Validator):
    """Stands in for a real language gate that never accepts (e.g. a
    candidate that fails to parse as Python)."""

    def validate(self, c):
        return ValidationResult(False, "not valid python")


def _toolset():
    return ToolSet(
        {"tools.py": "def t():\n    'doc'\n    pass\n"},
        ("tools.py",),
        "keep importable",
    )


def _run():
    return RunResult("b", "a", (TaskRun("t1", 0.0, "-> get_x()\nerr"),))


def test_llm_raises_on_retry_exhaustion(tmp_path):
    # When retries exhaust, the optimizer must raise rather than return the last
    # invalid candidate.
    opt = LLMOptimizer(
        max_retries=2,
        llm=FakeLLMClient(["not valid python (", "not valid python ("]),
    )
    with pytest.raises(OptimizerLLMFailure):
        opt.propose(_toolset(), _run(), _AlwaysInvalid("py", ("tools.py",)), tmp_path)


def test_llm_accepts_fenced_candidate_with_leading_prose(tmp_path):
    reply = (
        "Here's the updated file:\n\n```python\ndef t():\n    'FIXED'\n    pass\n```"
    )
    opt = LLMOptimizer(max_retries=1, llm=FakeLLMClient([reply]))
    cand = opt.propose(_toolset(), _run(), Validator("py", ("tools.py",)), tmp_path)
    assert cand.files["tools.py"] == "def t():\n    'FIXED'\n    pass\n"
