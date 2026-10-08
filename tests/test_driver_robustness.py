"""Driver robustness: the ToolTarget is always restored, even if apply raises."""

from __future__ import annotations

import pytest

from agent_tool_opt_core.api import Candidate, RunResult, ToolSet, ToolTarget
from agent_tool_opt_core.driver import evaluate_candidate


class _BoomTarget(ToolTarget):
    kind = "py"
    language_rules = "r"

    def __init__(self) -> None:
        self.restored = False

    def extract(self) -> ToolSet:
        return ToolSet({"t.py": "x"}, ("t.py",), self.language_rules)

    def effective_toolset(self) -> ToolSet:
        return self.extract()

    def apply(self, c: Candidate) -> None:
        raise RuntimeError("boom on apply")

    def restore(self) -> None:
        self.restored = True


class _Bench:
    name = "b"

    def evaluate(self, agent, tasks, tools):  # never reached (apply raises first)
        return RunResult("b", "a", ())


def test_evaluate_candidate_restores_when_apply_raises():
    tt = _BoomTarget()
    with pytest.raises(RuntimeError, match="boom"):
        evaluate_candidate(_Bench(), object(), tt, Candidate({"t.py": "y"}), ["t1"])
    assert tt.restored is True  # finally ran restore despite the apply failure
