"""Reward-shaping context engine — raw evidence only (no interpretation/priors)."""

from __future__ import annotations

from agent_tool_opt_core.api import RunResult, TaskRun
from agent_tool_opt_core.optimizers.reward_shaping import (
    REWARD_SHAPING_OBJECTIVE,
    build_reward_shaping_context,
)


def _run():
    return RunResult(
        "bench",
        "gpt",
        (
            TaskRun("f1", 0.0, trajectory=[{"role": "user", "content": "do it"}]),
            TaskRun("p1", 1.0, trajectory="raw stdout for p1"),
        ),
    )


def test_materializes_raw_transcripts_split_by_outcome():
    ctx = build_reward_shaping_context(_run())
    assert set(ctx) == {
        "transcripts/failed/f1.json",
        "transcripts/passed/p1.json",
        "INDEX.md",
    }
    # stored verbatim: a string trajectory is passed through as-is
    assert ctx["transcripts/passed/p1.json"] == "raw stdout for p1"
    # a structured trajectory is JSON-serialized raw (no schema imposed by us)
    assert '"role": "user"' in ctx["transcripts/failed/f1.json"]


def test_index_is_a_neutral_inventory():
    idx = build_reward_shaping_context(_run())["INDEX.md"]
    assert "transcripts/failed/f1.json" in idx
    assert "transcripts/passed/p1.json" in idx
    assert "reward=" in idx
    assert "TOOLS_RISK" not in idx and "purpose" not in idx.lower()


def test_no_derived_artifacts_or_strategy():
    ctx = build_reward_shaping_context(_run())
    assert "TOOLS_RISK.md" not in ctx  # no risk labels / prescriptions
    obj = REWARD_SHAPING_OBJECTIVE.lower()  # a neutral pointer, not a strategy
    for banned in ("protected", "target", "additive", "lever", "facts", "refus"):
        assert banned not in obj, banned
