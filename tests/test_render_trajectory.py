"""Lossless trajectory rendering and explicit structural fitting policy."""

from __future__ import annotations

import json

import pytest

from agent_tool_opt_core.api import RunResult, TaskRun
from agent_tool_opt_core.optimizers._common import (
    OptimizerLLMFailure,
    ensure_request_fits,
    fit_trajectory,
    fit_transcripts,
    model_context_window,
    render_trajectory,
    request_token_count,
    text_token_count,
    transcript_workspace_files,
)


def test_string_passthrough():
    assert render_trajectory("raw agent stdout\nline 2") == "raw agent stdout\nline 2"


def test_structured_is_json_serialized_verbatim():
    traj = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    assert json.loads(render_trajectory(traj)) == traj


def test_none_is_empty():
    assert render_trajectory(None) == ""


def test_same_processing_regardless_of_origin():
    for traj in (
        [{"role": "user", "content": "x"}],
        {"final_answer": "y", "answer_steps": []},
        {"steps": [{"tool_calls": [{"function_name": "bash"}]}]},
    ):
        assert json.loads(render_trajectory(traj)) == traj


def _episode(n_steps: int = 12, fail_at: int = 9, bulk: int = 1) -> dict:
    steps = []
    for i in range(n_steps):
        failed = i == fail_at
        steps.append(
            {
                "source": "assistant",
                "message": f"Thinking about step {i}. " + ("filler. " * 120 * bulk),
                "tool_calls": [
                    {
                        "function_name": (
                            "get_reservation_details" if failed else "search"
                        ),
                        "arguments": {"reservation_id": f"R{i:04d}"},
                    }
                ],
                "observation": {
                    "content": ("ToolError: bad argument. " if failed else "ok. ")
                    + ("payload " * 150 * bulk),
                    "is_error": failed,
                },
            }
        )
    return {"steps": steps, "stop_reason": "done"}


def test_fitter_keeps_late_failure_and_full_structure(caplog):
    traj = _episode()
    assert "get_reservation_details" not in render_trajectory(traj)[:4000]
    fitted = fit_trajectory(
        traj,
        token_budget=5000,
        optimizer="llm",
        task_id="late-failure",
    )
    assert text_token_count(fitted, "claude-opus-4-7") <= 5000
    for i in range(12):
        assert f'"reservation_id": "R{i:04d}"' in fitted
    assert "get_reservation_details" in fitted
    assert '"is_error": true' in fitted
    assert "ToolError" in fitted
    assert "chars elided" in fitted
    assert "optimizer=llm task=late-failure" in caplog.text
    assert "structured-middle-elision" in caplog.text


def test_lossless_renderer_has_no_budget_argument():
    with pytest.raises(TypeError):
        render_trajectory(_episode(), budget=4000)  # type: ignore[call-arg]


def test_opaque_trajectory_fails_instead_of_prefix_truncating():
    text = "HEAD\n" + "x" * 20_000 + "\nTAIL"
    with pytest.raises(OptimizerLLMFailure, match="opaque trajectory"):
        fit_trajectory(text, token_budget=500)


def test_nonpositive_trajectory_budget_is_an_optimizer_failure():
    with pytest.raises(OptimizerLLMFailure, match="no request budget remains"):
        fit_trajectory({"steps": []}, token_budget=0)


def test_unshrinkable_structure_fails_instead_of_prefix_truncating():
    traj = [{"a": i} for i in range(4000)]
    with pytest.raises(OptimizerLLMFailure, match="structure"):
        fit_trajectory(traj, token_budget=500)


def test_total_transcript_budget_not_per_task_budget():
    run = RunResult(
        "b",
        "a",
        tuple(TaskRun(str(i), 0.0, _episode(n_steps=3, fail_at=2)) for i in range(3)),
    )
    fitted = fit_transcripts(run, token_budget=8000)
    assert text_token_count(fitted, "claude-opus-4-7") <= 8000
    assert fitted.count("### task") == 3
    assert fitted.count("get_reservation_details") == 3


def test_total_fitter_does_not_abbreviate_opaque_transcripts():
    run = RunResult("b", "a", (TaskRun("t", 0.0, "x" * 20_000),))
    with pytest.raises(OptimizerLLMFailure, match="skeletons"):
        fit_transcripts(run, token_budget=500)


def test_request_preflight_reserves_output_space():
    messages = [{"role": "user", "content": "x" * 1000}]
    used = request_token_count(messages, "gpt-4o")
    assert (
        ensure_request_fits(
            messages,
            "gpt-4o",
            context_window_size=used + 100,
            output_reserve_tokens=100,
        )
        == used
    )
    with pytest.raises(OptimizerLLMFailure, match="reserved output"):
        ensure_request_fits(
            messages,
            "gpt-4o",
            context_window_size=used + 99,
            output_reserve_tokens=100,
        )


def test_unknown_model_requires_explicit_context_window():
    with pytest.raises(ValueError, match="unknown context window"):
        ensure_request_fits([], "unknown-model")


def test_configured_custom_models_resolve_from_model_registry():
    assert model_context_window("claude-opus-4-6") == 1_000_000
    assert model_context_window("claude-haiku-4-5") == 200_000
    assert model_context_window("gpt-4o") == 128_000


def test_model_token_count_is_not_a_utf8_byte_count():
    text = "ordinary English evidence " * 1000
    assert text_token_count(text, "claude-opus-4-7") < len(text.encode()) // 2


def test_claude_count_has_headroom_over_litellm_approximation():
    text = "structured trajectory evidence " * 1000
    claude = text_token_count(text, "claude-opus-4-7")
    fallback = text_token_count(text, "unknown-model")
    assert claude >= fallback * 1.5


def test_workspace_files_are_full_indexed_and_safe():
    run = RunResult(
        "b",
        "a",
        (
            TaskRun("../unsafe", 0.0, {"body": "x" * 10_000}),
            TaskRun("ok", 1.0, "complete stdout"),
        ),
    )
    files = transcript_workspace_files(run)
    assert set(files) == {
        "BASELINE_TRANSCRIPTS_INDEX.md",
        "baseline_transcripts/0000.json",
        "baseline_transcripts/0001.txt",
    }
    assert "../unsafe" in files["BASELINE_TRANSCRIPTS_INDEX.md"]
    assert "x" * 10_000 in files["baseline_transcripts/0000.json"]
    assert files["baseline_transcripts/0001.txt"] == "complete stdout"
