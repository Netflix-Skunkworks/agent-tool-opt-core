"""Tests for shared optimizer helpers + the single-file guard."""

from __future__ import annotations

import ast

import pytest

from agent_tool_opt_core.api import RunResult, TaskRun, ToolSet, Validator
from agent_tool_opt_core.optimizers._common import (
    OptimizerInfrastructureFailure,
    OptimizerLLMFailure,
    call_llm_with_retry,
    single_editable_target,
    strip_code_fences,
    summarize_transcripts,
)
from agent_tool_opt_core.optimizers.draft import DRAFTOptimizer
from agent_tool_opt_core.optimizers.llm import LLMOptimizer


def test_strip_code_fences():
    assert strip_code_fences("```python\nx = 1\n```") == "x = 1\n"
    assert strip_code_fences("```\nx = 1\n```") == "x = 1\n"
    assert strip_code_fences("x = 1") == "x = 1\n"


def test_strip_code_fences_with_leading_prose():
    text = "Here is the updated file:\n\n```python\nx = 1\n```\n\nLet me know!"
    assert strip_code_fences(text) == "x = 1\n"


def test_strip_code_fences_picks_largest_block():
    text = "```python\ny = 2\n```\n\nfull file:\n\n```python\nx = 1\ndef f():\n    pass\n```"
    assert strip_code_fences(text) == "x = 1\ndef f():\n    pass\n"


def test_strip_code_fences_handles_an_unclosed_fence():
    assert strip_code_fences("```python\ndef t():\n    return 1\n") == (
        "def t():\n    return 1\n"
    )
    # The candidate gets written out as a .py file, so the real contract is
    # "parses", not "equals some exact string". ast.parse raises SyntaxError —
    # failing the test — if a leftover ``` marker or the prose survived the strip.
    ast.parse(strip_code_fences("Here you go:\n```python\nx = 1\ny = 2\n"))


def test_summarize_renders_raw_trajectories():
    run = RunResult(
        "b",
        "g",
        (
            TaskRun("t1", 0.0, trajectory="plain string traj"),
            # a structured (raw) trajectory is JSON-serialized verbatim
            TaskRun("t2", 0.0, trajectory={"role": "user", "content": "hi"}),
        ),
    )
    out = summarize_transcripts(run)
    assert "plain string traj" in out  # string passed through
    assert '"content": "hi"' in out  # dict serialized raw (no schema imposed)
    assert "### task t1 [FAIL] (reward=0.0)" in out


def test_summarize_shows_failures_first_then_capped_successes():
    # Regression context: the optimizer must see the runs it could break, not
    # only the ones it is fixing — failures first, then a capped success sample.
    runs = [TaskRun(f"f{i}", 0.0, trajectory=f"fail-{i}") for i in range(2)]
    runs += [TaskRun(f"p{i}", 1.0, trajectory=f"pass-{i}") for i in range(5)]
    out = summarize_transcripts(RunResult("b", "g", tuple(runs)), max_passing=3)
    for i in range(2):
        assert f"### task f{i} [FAIL] (reward=0.0)" in out  # every failure shown
    assert out.count("[PASS]") == 3  # successes capped at max_passing
    assert out.index("[FAIL]") < out.index("[PASS]")  # failures precede successes


def test_summarize_selects_one_representative_per_task_across_trials():
    run = RunResult(
        "b",
        "g",
        (
            TaskRun("t1", 0.0, trajectory="first failure"),
            TaskRun("t1", 0.0, trajectory="duplicate failure"),
            TaskRun("t2", 1.0, trajectory="first success"),
            TaskRun("t2", 1.0, trajectory="duplicate success"),
        ),
    )
    out = summarize_transcripts(run)
    assert out.count("### task t1 [FAIL]") == 1
    assert out.count("### task t2 [PASS]") == 1
    assert "first failure" in out and "duplicate failure" not in out
    assert "first success" in out and "duplicate success" not in out


def test_summarize_is_lossless_even_for_late_failures():
    steps = [
        {
            "tool_calls": [{"function_name": f"tool_{i}", "arguments": {"n": i}}],
            "result": {"content": "body " * 400, "is_error": i == 7},
        }
        for i in range(8)
    ]
    run = RunResult("b", "g", (TaskRun("t1", 0.0, trajectory={"steps": steps}),))
    out = summarize_transcripts(run)
    assert "tool_7" in out and '"is_error": true' in out
    assert "body " * 400 in out


def test_summarize_does_not_shorten_a_transcript():
    run = RunResult("b", "g", (TaskRun("t1", 0.0, trajectory={"log": "x" * 20_000}),))
    out = summarize_transcripts(run)
    assert "x" * 20_000 in out
    assert "elided" not in out


def test_single_editable_target_ok():
    ts = ToolSet({"tools.py": "x"}, ("tools.py",), "r")
    assert single_editable_target(ts) == "tools.py"


def test_single_editable_target_rejects_multifile():
    ts = ToolSet({"a.ts": "x", "b.ts": "y"}, ("a.ts", "b.ts"), "r")
    with pytest.raises(ValueError, match="single file"):
        single_editable_target(ts)


def test_llm_and_draft_reject_multifile_target(tmp_path):
    # The guard fires before any LLM call.
    ts = ToolSet({"a.ts": "x", "b.ts": "y"}, ("a.ts", "b.ts"), "rules")
    run = RunResult("b", "g", ())
    v = Validator("ts", ("a.ts", "b.ts"))
    with pytest.raises(ValueError, match="single file"):
        LLMOptimizer().propose(ts, run, v, tmp_path)
    with pytest.raises(ValueError, match="single file"):
        DRAFTOptimizer().propose(ts, run, v, tmp_path)


class _ProviderError(Exception):
    def __init__(self, message, *, status_code=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}


@pytest.mark.parametrize(
    "status_code", [408, 503], ids=["request_timeout", "service_unavailable"]
)
def test_transient_call_exhaustion_is_infrastructure_failure(monkeypatch, status_code):
    waits = []
    monkeypatch.setattr(
        "agent_tool_opt_core.optimizers._common.time.sleep", waits.append
    )
    monkeypatch.setattr(
        "agent_tool_opt_core.optimizers._common.random.uniform", lambda low, high: 0
    )

    def unavailable():
        raise _ProviderError("gateway unavailable", status_code=status_code)

    with pytest.raises(OptimizerInfrastructureFailure) as raised:
        call_llm_with_retry(unavailable, max_retries=2, base_delay=1)

    assert isinstance(raised.value.__cause__, _ProviderError)
    assert waits == [1]


@pytest.mark.parametrize(
    "status_code", [400, 503], ids=["bad_request", "service_unavailable"]
)
def test_context_failure_stays_a_candidate_failure(status_code):
    def too_large():
        raise _ProviderError("maximum context length exceeded", status_code=status_code)

    with pytest.raises(OptimizerLLMFailure):
        call_llm_with_retry(too_large)


def test_provider_retry_after_is_a_minimum_wait(monkeypatch):
    calls = 0
    waits = []
    monkeypatch.setattr(
        "agent_tool_opt_core.optimizers._common.time.sleep", waits.append
    )
    monkeypatch.setattr(
        "agent_tool_opt_core.optimizers._common.random.uniform", lambda low, high: 0
    )

    def recovers():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _ProviderError(
                "limited", status_code=429, headers={"Retry-After": "7"}
            )
        return "ok"

    assert call_llm_with_retry(recovers, max_retries=2, base_delay=1) == "ok"
    assert waits == [7]
