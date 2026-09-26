"""Public five-phase benchmark loop shared by the TauBench and Harbor CLIs.

This is the experiment setup used by the original adapters: baseline train,
baseline test, train-only optimization, optimized train, optimized test. Each
phase is saved as a separate local artifact. A missing or invalid candidate is
reported, never silently scored as an optimized run.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path, PurePosixPath

from agent_tool_opt_core.api import (
    Agent,
    Benchmark,
    Candidate,
    Optimizer,
    RunResult,
    ToolTarget,
)
from agent_tool_opt_core.costs import collect_optimizer_costs
from agent_tool_opt_core.driver import (
    collect_baseline,
    evaluate_candidate,
    make_train_evaluator,
    propose,
)


def _write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _phase(benchmark: Benchmark, run: RunResult) -> dict:
    measured = benchmark.score(run)
    return {
        "benchmark": run.benchmark,
        "agent": run.agent,
        "split": run.split,
        "avg_reward": measured.avg_reward,
        "pass_rate": measured.pass_rate,
        "n": measured.n,
        "cost_usd": run.total_cost_usd,
        "known_cost_usd": run.known_cost_usd,
        "unknown_cost_count": sum(task.cost_usd is None for task in run.runs),
        "runs": [
            {
                "task_id": task.task_id,
                "reward": task.reward,
                "trajectory": task.trajectory,
                "cost_usd": task.cost_usd,
                "known_cost_usd": task.known_cost_usd,
            }
            for task in run.runs
        ],
    }


def _phase_metrics(artifact: dict) -> dict:
    return {key: value for key, value in artifact.items() if key != "runs"}


def _cost(cost) -> dict:
    return {
        "cost_usd": cost.cost_usd,
        "known_cost_usd": cost.known_usd,
        "unknown_count": cost.unknown_count,
        "sources": sorted(cost.sources),
        "reasons": sorted(cost.reasons),
    }


def _comparison(baseline: dict, optimized: dict) -> dict:
    cost_a = baseline["cost_usd"]
    cost_b = optimized["cost_usd"]
    return {
        "avg_reward_delta": optimized["avg_reward"] - baseline["avg_reward"],
        "pass_rate_delta": optimized["pass_rate"] - baseline["pass_rate"],
        "cost_usd_delta": (
            cost_b - cost_a if cost_a is not None and cost_b is not None else None
        ),
    }


def _save_candidate(
    root: Path, candidate: Candidate, allowlist: tuple[str, ...]
) -> None:
    destination = root / "candidate"
    destination.mkdir()
    for name, content in candidate.files.items():
        rel = PurePosixPath(name)
        if name not in allowlist or rel.is_absolute() or ".." in rel.parts:
            raise ValueError("candidate path is outside the editable allowlist")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def run_five_phases(
    benchmark: Benchmark,
    agent: Agent,
    tool_target: ToolTarget,
    optimizer: Optimizer,
    *,
    train: list[str],
    test: list[str],
    output_dir: Path,
) -> dict:
    """Run and save paired A/B phases; only train evidence reaches optimizer."""
    if (
        not train
        or not test
        or len(train) != len(set(train))
        or len(test) != len(set(test))
    ):
        raise ValueError("train and test must contain distinct task IDs")
    if set(train) & set(test):
        raise ValueError("train and test task IDs overlap")
    if train != benchmark.tasks("train") or test != benchmark.tasks("test"):
        raise ValueError("task IDs must match the benchmark's frozen splits")
    if tool_target.effective_toolset() != tool_target.extract():
        raise ValueError("tool target must start at baseline")

    root = Path(output_dir).expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    summary: dict = {
        "benchmark": benchmark.name,
        "agent": agent.id,
        "optimizer": optimizer.id,
        "train_tasks": train,
        "test_tasks": test,
        "phases": {},
    }

    baseline_runs = {}
    baseline_artifacts = {}
    for split, tasks in (("train", train), ("test", test)):
        phase_name = f"baseline_{split}"
        try:
            run = collect_baseline(benchmark, agent, tasks, tool_target, split=split)
        except Exception as exc:
            summary["optimization"] = {
                "status": "baseline_failed",
                "failed_phase": phase_name,
                "error_type": type(exc).__name__,
            }
            _write_json(root / "summary.json", summary)
            raise
        artifact = _phase(benchmark, run)
        baseline_runs[split] = run
        baseline_artifacts[split] = artifact
        summary["phases"][phase_name] = _phase_metrics(artifact)
        if split == "train":
            _write_json(root / "baseline_train.json", artifact)

    train_eval = (
        make_train_evaluator(benchmark, agent, tool_target, train)
        if optimizer.wants_train_eval
        else None
    )
    try:
        with collect_optimizer_costs() as costs:
            candidate = propose(
                optimizer,
                tool_target,
                baseline_runs["train"],
                root / "optimize",
                train_eval,
            )
    except Exception as exc:
        summary["optimizer_cost"] = {
            "llm": _cost(costs.llm),
            "train_search": _cost(costs.search),
        }
        summary["optimization"] = {
            "status": "optimizer_failed",
            "error_type": type(exc).__name__,
        }
        _write_json(root / "baseline_test.json", baseline_artifacts["test"])
        _write_json(root / "summary.json", summary)
        raise
    summary["optimizer_cost"] = {
        "llm": _cost(costs.llm),
        "train_search": _cost(costs.search),
    }
    # Keep held-out transcripts out of the optimizer's on-disk workspace until
    # after propose() returns. The optimizer receives only baseline_train.
    _write_json(root / "baseline_test.json", baseline_artifacts["test"])
    validation = tool_target.validator().validate(candidate)
    original = tool_target.extract()
    changed = {
        name: text
        for name, text in candidate.files.items()
        if original.files.get(name) != text
    }
    if not validation.ok or not changed:
        summary["optimization"] = {
            "status": "invalid_candidate" if not validation.ok else "no_edit",
            "validation": validation.log,
        }
        _write_json(root / "summary.json", summary)
        return summary

    candidate = Candidate(changed)
    _save_candidate(root, candidate, original.allowlist)
    summary["optimization"] = {
        "status": "candidate_applied",
        "changed_files": sorted(changed),
        "validation": validation.log,
    }
    for split, tasks in (("train", train), ("test", test)):
        phase_name = f"optimized_{split}"
        try:
            run = replace(
                evaluate_candidate(benchmark, agent, tool_target, candidate, tasks),
                split=split,
            )
        except Exception as exc:
            summary["optimization"].update(
                status="evaluation_failed",
                failed_phase=phase_name,
                error_type=type(exc).__name__,
            )
            _write_json(root / "summary.json", summary)
            raise
        artifact = _phase(benchmark, run)
        summary["phases"][phase_name] = _phase_metrics(artifact)
        _write_json(root / f"{phase_name}.json", artifact)
    summary["comparison"] = {
        split: _comparison(
            summary["phases"][f"baseline_{split}"],
            summary["phases"][f"optimized_{split}"],
        )
        for split in ("train", "test")
    }
    _write_json(root / "summary.json", summary)
    return summary
