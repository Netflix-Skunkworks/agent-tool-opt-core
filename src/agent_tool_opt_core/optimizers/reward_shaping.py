"""Reward shaping — an evidence method for the optimizer (benchmark-agnostic).

The headline method gives the optimizer the *full baseline run on disk*: every
task's RAW transcript, organized only by outcome (passed/failed, by reward), plus
a neutral INDEX. It adds no interpretation, labels, or strategy — just the raw
evidence — so the optimizer reasons from what the agent actually did rather than
from any prior we injected. ``REWARD_SHAPING_OBJECTIVE`` only points at the files;
``build_reward_shaping_context(run)`` materializes them.
"""

from __future__ import annotations

import json

from agent_tool_opt_core.api import RunResult, TaskRun

REWARD_SHAPING_OBJECTIVE = """\
The full baseline run is on disk as evidence: each task's transcript under
``transcripts/passed/<id>.json`` or ``transcripts/failed/<id>.json`` (split by
reward), listed in ``INDEX.md``. Read them to ground your edits in what the agent
actually did."""


def _dump(trajectory: object) -> str:
    """The benchmark's raw transcript, verbatim (string) or JSON-serialized."""
    if isinstance(trajectory, str):
        return trajectory
    if isinstance(trajectory, (dict, list)):
        return json.dumps(trajectory, indent=2, default=str, ensure_ascii=False)
    return str(trajectory) if trajectory is not None else ""


def _index_md(failed: list[TaskRun], passed: list[TaskRun]) -> str:
    """A neutral inventory: which transcripts exist, by outcome and reward."""
    lines = ["# Transcripts\n"]
    for split, runs in (("failed", failed), ("passed", passed)):
        lines.append(f"\n## {split}\n")
        if not runs:
            lines.append("_(none)_\n")
            continue
        for r in sorted(runs, key=lambda r: str(r.task_id)):
            lines.append(
                f"- `transcripts/{split}/{r.task_id}.json` (reward={r.reward})\n"
            )
    return "".join(lines)


def build_reward_shaping_context(run: RunResult) -> dict[str, str]:
    """Read-only evidence files from the baseline run: every task's raw transcript
    (split passed/failed by reward) + a neutral INDEX. No interpretation added."""
    failed = [r for r in run.runs if r.reward < 1.0]
    passed = [r for r in run.runs if r.reward >= 1.0]
    ctx: dict[str, str] = {}
    for split, runs in (("failed", failed), ("passed", passed)):
        for r in runs:
            ctx[f"transcripts/{split}/{r.task_id}.json"] = _dump(r.trajectory)
    ctx["INDEX.md"] = _index_md(failed, passed)
    return ctx
