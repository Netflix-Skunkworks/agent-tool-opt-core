"""Tests for the provider-neutral LLM optimizer implementation."""

from __future__ import annotations

import pytest

from agent_tool_opt_core.api import RunResult, TaskRun, ToolSet, Validator
from agent_tool_opt_core.optimizers._common import OptimizerLLMFailure
from agent_tool_opt_core.optimizers.llm import LLMOptimizer
from agent_tool_opt_core.testing import FakeLLMClient


def _patch(mocker, content, sink):
    fake = FakeLLMClient([content])
    complete = fake.complete

    def record(**kw):
        sink.update(kw)
        return complete(**kw)

    fake.complete = record
    mocker.patch(
        "agent_tool_opt_core.optimizers.llm.LiteLLMClient",
        return_value=fake,
    )


def _toolset():
    return ToolSet(
        {"tools.py": "def t(): pass\n"}, ("tools.py",), "edit tools.py; keep importable"
    )


def _run():
    return RunResult(
        "b",
        "gpt",
        (TaskRun("t1", 0.0, trajectory="agent tried get_x and failed"),),
    )


def test_llm_propose_returns_candidate_edit(mocker, tmp_path):
    sink = {}
    _patch(mocker, "def t():\n    return 2\n", sink)
    cand = LLMOptimizer(model="m", context_window_size=100_000).propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
    )
    assert cand.files == {"tools.py": "def t():\n    return 2\n"}
    assert sink["model"] == "m"
    msgs = sink["messages"]
    assert "def t(): pass" in msgs[1]["content"]  # current source embedded
    assert "get_x" in msgs[1]["content"]  # transcript embedded


def test_llm_strips_code_fences(mocker, tmp_path):
    _patch(mocker, "```python\ndef t(): return 1\n```", {})
    cand = LLMOptimizer().propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
    )
    assert cand.files["tools.py"] == "def t(): return 1\n"


def test_llm_empty_allowlist_is_noop(mocker, tmp_path):
    _patch(mocker, "x", {})
    cand = LLMOptimizer().propose(
        ToolSet({}, (), "rules"), _run(), Validator("py", ()), tmp_path
    )
    assert cand.files == {}


def test_llm_fits_structured_transcripts_to_total_request_budget(mocker, tmp_path):
    sink = {}
    _patch(mocker, "def t():\n    return 2\n", sink)
    steps = [
        {
            "tool": f"tool_{i}",
            "arguments": {"id": f"R{i}"},
            "result": "head " + "x" * 5000 + " tail",
        }
        for i in range(5)
    ]
    run = RunResult("b", "g", (TaskRun("t", 0.0, {"steps": steps}),))
    LLMOptimizer(
        model="m",
        context_window_size=4000,
        output_reserve_tokens=1000,
    ).propose(_toolset(), run, Validator("py", ("tools.py",)), tmp_path)
    user = sink["messages"][1]["content"]
    assert "middle-elided" in user
    assert all(f'"tool": "tool_{i}"' in user for i in range(5))


def test_llm_fails_when_opaque_evidence_cannot_fit(mocker, tmp_path):
    sink = {}
    _patch(mocker, "unused", sink)
    run = RunResult("b", "g", (TaskRun("t", 0.0, "x" * 100_000),))
    with pytest.raises(OptimizerLLMFailure, match="skeletons"):
        LLMOptimizer(
            model="m",
            context_window_size=5000,
            output_reserve_tokens=1000,
        ).propose(_toolset(), run, Validator("py", ("tools.py",)), tmp_path)
    assert sink == {}  # failed before making a gateway call


# ---- ablation parity (same knobs as the pi optimizer) ----------------------


def _patch_seq(mocker, outs, calls):
    replies = [outs[min(index, len(outs) - 1)] for index in range(max(len(outs), 3))]
    fake = FakeLLMClient(replies)
    complete = fake.complete

    def record(**kw):
        calls["n"] += 1
        return complete(**kw)

    fake.complete = record
    mocker.patch(
        "agent_tool_opt_core.optimizers.llm.LiteLLMClient",
        return_value=fake,
    )


def test_llm_blind_omits_transcripts(mocker, tmp_path):
    sink = {}
    _patch(mocker, "def t():\n    return 2\n", sink)
    LLMOptimizer(use_transcripts=False).propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
    )
    user = sink["messages"][1]["content"]
    assert "get_x" not in user  # transcript omitted
    assert "from the tool source alone" in user  # cold prompt


def test_llm_injects_method_objective_and_digest_context(mocker, tmp_path):
    sink = {}
    _patch(mocker, "def t():\n    return 2\n", sink)
    LLMOptimizer(
        method_addendum="REWARD ADDENDUM",
        context_builder=lambda run: {
            "TOOLS_RISK.md": "risk table",
            "transcripts/failed/x.json": "BULK RAW",
        },
    ).propose(_toolset(), _run(), Validator("py", ("tools.py",)), tmp_path)
    system, user = sink["messages"][0]["content"], sink["messages"][1]["content"]
    assert "REWARD ADDENDUM" in system  # method objective -> system prompt
    assert "risk table" in user  # digest context injected as text
    assert "BULK RAW" not in user  # raw transcripts/ skipped in the prompt


def test_llm_retries_until_valid(mocker, tmp_path):
    calls = {"n": 0}
    _patch_seq(
        mocker, ["   ", "def t():\n    return 9\n"], calls
    )  # invalid, then valid
    cand = LLMOptimizer(max_retries=3).propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
    )
    assert calls["n"] == 2  # retried once
    assert cand.files == {"tools.py": "def t():\n    return 9\n"}


def test_llm_no_validation_accepts_first(mocker, tmp_path):
    calls = {"n": 0}
    _patch_seq(mocker, ["   "], calls)  # invalid output
    LLMOptimizer(require_validation=False, max_retries=3).propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
    )
    assert calls["n"] == 1  # accepted first despite invalid -> no retry
