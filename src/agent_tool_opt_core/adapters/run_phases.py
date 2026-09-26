"""Public five-phase benchmark loop shared by the TauBench and Harbor CLIs.

This is the experiment setup used by the original adapters: baseline train,
baseline test, train-only optimization, optimized train, optimized test. Each
phase is saved as a separate local artifact. A missing or invalid candidate is
reported, never silently scored as an optimized run.
"""

from __future__ import annotations

import json
import hashlib
import html
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path, PurePosixPath

from agent_tool_opt_core.api import (
    Agent,
    Benchmark,
    Candidate,
    Optimizer,
    RunResult,
    TaskRun,
    ToolSet,
    ToolTarget,
)
from agent_tool_opt_core.costs import collect_optimizer_costs, usd
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


def _toolset_digest(toolset: ToolSet) -> str:
    payload = json.dumps(
        {
            "files": toolset.files,
            "allowlist": toolset.allowlist,
            "context": toolset.context,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_reused_baseline(
    source: Path,
    benchmark: Benchmark,
    agent: Agent,
    setup: dict,
    train: list[str],
    test: list[str],
) -> dict[str, RunResult]:
    source = source.expanduser().resolve(strict=True)
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    expected = {
        "benchmark": benchmark.name,
        "agent": agent.id,
        "train_tasks": train,
        "test_tasks": test,
        "setup": setup,
    }
    if not isinstance(summary, dict) or any(
        summary.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(
            "reused baseline has incompatible benchmark, agent, tasks, or tool snapshot"
        )
    loaded = {}
    for split, tasks in (("train", train), ("test", test)):
        artifact = json.loads(
            (source / f"baseline_{split}.json").read_text(encoding="utf-8")
        )
        if (
            not isinstance(artifact, dict)
            or artifact.get("benchmark") != benchmark.name
            or artifact.get("agent") != agent.id
            or artifact.get("split") != split
            or not isinstance(artifact.get("runs"), list)
        ):
            raise ValueError(f"reused baseline_{split} artifact is invalid")
        runs = []
        for item in artifact["runs"]:
            if not isinstance(item, dict) or item.get("task_id") not in tasks:
                raise ValueError(f"reused baseline_{split} has unexpected task")
            reward = item.get("reward")
            cost = item.get("cost_usd")
            known = item.get("known_cost_usd")
            if (
                isinstance(reward, bool)
                or not isinstance(reward, (int, float))
                or not math.isfinite(reward)
                or (cost is not None and usd(cost) is None)
                or usd(known) is None
            ):
                raise ValueError(f"reused baseline_{split} has invalid reward or cost")
            runs.append(
                TaskRun(
                    item["task_id"], float(reward), item.get("trajectory"), cost, known
                )
            )
        expected_count = getattr(benchmark, "runs_per_task", 1)
        counts = Counter(run.task_id for run in runs)
        if any(counts[task] != expected_count for task in tasks):
            raise ValueError(f"reused baseline_{split} has incomplete task trials")
        run = RunResult(benchmark.name, agent.id, tuple(runs), split=split)
        if _phase_metrics(_phase(benchmark, run)) != _phase_metrics(artifact):
            raise ValueError(f"reused baseline_{split} metrics do not match its runs")
        loaded[split] = run
    return loaded


def _render_report(summary: dict) -> str:
    def cell(value: object) -> str:
        if value is None:
            return "unknown"
        if isinstance(value, float):
            return f"{value:.4f}"
        return html.escape(str(value), quote=True)

    rows = []
    for name, phase in summary.get("phases", {}).items():
        rows.append((name, phase))
    for candidate in summary.get("candidates", []):
        for name, phase in candidate.get("phases", {}).items():
            rows.append((f"candidate {candidate['index']}: {name}", phase))
    body = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{cell(value)}</td>"
            for value in (
                name,
                phase.get("avg_reward"),
                phase.get("pass_rate"),
                phase.get("n"),
                phase.get("cost_usd"),
                phase.get("unknown_cost_count"),
            )
        )
        + "</tr>"
        for name, phase in rows
    )
    candidates = summary.get("candidates")
    if candidates is None:
        candidates = [summary] if "optimizer_cost" in summary else []
    optimizer_rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{cell(value)}</td>"
            for value in (
                candidate.get("index", 0),
                candidate.get("optimization", {}).get("status"),
                candidate.get("optimizer_cost", {}).get("llm", {}).get("cost_usd"),
                candidate.get("optimizer_cost", {}).get("llm", {}).get("unknown_count"),
                candidate.get("optimizer_cost", {})
                .get("train_search", {})
                .get("cost_usd"),
            )
        )
        + "</tr>"
        for candidate in candidates
    )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>Benchmark A/B report</title><style>body{font-family:system-ui;max-width:"
        "70rem;margin:2rem auto;padding:0 1rem}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #bbb;padding:.5rem;text-align:left}</style>"
        f"<h1>{cell(summary.get('benchmark'))} A/B report</h1>"
        f"<p>Agent: {cell(summary.get('agent'))}; optimizer: {cell(summary.get('optimizer'))}</p>"
        "<table><thead><tr><th>Phase</th><th>Average reward</th><th>Pass rate</th>"
        "<th>Runs</th><th>Model cost USD</th><th>Unknown costs</th></tr></thead>"
        f"<tbody>{body}</tbody></table><h2>Optimization</h2>"
        "<table><thead><tr><th>Candidate</th><th>Status</th><th>Optimizer LLM USD</th>"
        "<th>Unknown LLM costs</th><th>Train search USD</th></tr></thead>"
        f"<tbody>{optimizer_rows}</tbody></table></html>\n"
    )


def _save_summary(root: Path, summary: dict) -> None:
    _write_json(root / "summary.json", summary)
    (root / "report.html").write_text(_render_report(summary), encoding="utf-8")


def _run_pair(
    work, *, parallel: bool
) -> tuple[dict[str, RunResult], dict[str, Exception]]:
    results: dict[str, RunResult] = {}
    errors: dict[str, Exception] = {}
    splits = ("train", "test")
    if parallel:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {split: pool.submit(work, split) for split in splits}
            for split in splits:
                try:
                    results[split] = futures[split].result()
                except Exception as exc:
                    errors[split] = exc
    else:
        for split in splits:
            try:
                results[split] = work(split)
            except Exception as exc:
                errors[split] = exc
                break
    return results, errors


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
    optimizer: Optimizer | None,
    *,
    train: list[str],
    test: list[str],
    output_dir: Path,
    baseline_dir: Path | None = None,
    num_candidates: int = 1,
    parallel_phases: bool = False,
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
    if num_candidates < 1:
        raise ValueError("num_candidates must be positive")
    if optimizer is None and num_candidates != 1:
        raise ValueError("baseline-only runs cannot request candidates")
    if parallel_phases and not getattr(benchmark, "parallel_safe", False):
        raise ValueError("parallel phases require a benchmark with immutable toolsets")
    if parallel_phases and getattr(benchmark, "n_concurrent", 2) < 2:
        raise ValueError("parallel phases require at least two concurrency slots")
    original = tool_target.extract()
    if tool_target.effective_toolset() != original:
        raise ValueError("tool target must start at baseline")
    setup = {
        "toolset_sha256": _toolset_digest(original),
        "num_trials": getattr(benchmark, "num_trials", 1),
        "runs_per_task": getattr(benchmark, "runs_per_task", 1),
        "max_steps": getattr(benchmark, "max_steps", None),
        "user_model": getattr(agent, "user_model", None),
        "n_concurrent": getattr(benchmark, "n_concurrent", None),
        "parallel_phases": parallel_phases,
    }
    reused = (
        _load_reused_baseline(Path(baseline_dir), benchmark, agent, setup, train, test)
        if baseline_dir is not None
        else None
    )

    root = Path(output_dir).expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    summary: dict = {
        "benchmark": benchmark.name,
        "agent": agent.id,
        "optimizer": optimizer.id if optimizer is not None else None,
        "train_tasks": train,
        "test_tasks": test,
        "setup": setup,
        "phases": {},
    }
    if baseline_dir is not None:
        summary["baseline_source"] = str(Path(baseline_dir).expanduser().resolve())

    if reused is not None:
        baseline_runs, baseline_errors = reused, {}
    else:

        def baseline_work(split: str) -> RunResult:
            tasks = train if split == "train" else test
            if parallel_phases:
                return replace(
                    benchmark.evaluate_parallel(agent, tasks, original), split=split
                )
            return collect_baseline(benchmark, agent, tasks, tool_target, split=split)

        baseline_runs, baseline_errors = _run_pair(
            baseline_work, parallel=parallel_phases
        )
    baseline_artifacts = {}
    for split, run in baseline_runs.items():
        artifact = _phase(benchmark, run)
        baseline_artifacts[split] = artifact
        summary["phases"][f"baseline_{split}"] = _phase_metrics(artifact)
        if split == "train" or baseline_errors:
            _write_json(root / f"baseline_{split}.json", artifact)
    if baseline_errors:
        failed_split = next(iter(baseline_errors))
        error = baseline_errors[failed_split]
        summary["optimization"] = {
            "status": "baseline_failed",
            "failed_phase": f"baseline_{failed_split}",
            "error_type": type(error).__name__,
        }
        _save_summary(root, summary)
        raise error
    if optimizer is None:
        _write_json(root / "baseline_test.json", baseline_artifacts["test"])
        summary["optimization"] = {"status": "baseline_only"}
        _save_summary(root, summary)
        return summary

    train_eval = (
        make_train_evaluator(benchmark, agent, tool_target, train)
        if optimizer.wants_train_eval
        else None
    )
    records: list[dict] = []
    proposals: list[tuple[dict, Candidate, Path]] = []
    for index in range(num_candidates):
        candidate_root = (
            root if num_candidates == 1 else root / f"candidate_{index:02d}"
        )
        if num_candidates > 1:
            candidate_root.mkdir()
        record: dict = {"index": index, "phases": {}}
        records.append(record)
        try:
            with collect_optimizer_costs() as costs:
                candidate = propose(
                    optimizer,
                    tool_target,
                    baseline_runs["train"],
                    candidate_root / "optimize",
                    train_eval,
                )
        except Exception as exc:
            record["optimizer_cost"] = {
                "llm": _cost(costs.llm),
                "train_search": _cost(costs.search),
            }
            record["optimization"] = {
                "status": "optimizer_failed",
                "error_type": type(exc).__name__,
            }
            if num_candidates == 1:
                _write_json(root / "baseline_test.json", baseline_artifacts["test"])
                summary.update(
                    {key: value for key, value in record.items() if key != "phases"}
                )
                _save_summary(root, summary)
                raise
            continue
        record["optimizer_cost"] = {
            "llm": _cost(costs.llm),
            "train_search": _cost(costs.search),
        }
        validation = tool_target.validator().validate(candidate)
        changed = {
            name: text
            for name, text in candidate.files.items()
            if original.files.get(name) != text
        }
        if not validation.ok or not changed:
            record["optimization"] = {
                "status": "invalid_candidate" if not validation.ok else "no_edit",
                "validation": validation.log,
            }
            continue
        candidate = Candidate(changed)
        _save_candidate(candidate_root, candidate, original.allowlist)
        record["optimization"] = {
            "status": "candidate_applied",
            "changed_files": sorted(changed),
            "validation": validation.log,
        }
        proposals.append((record, candidate, candidate_root))

    # All proposals complete before the held-out transcript is materialized.
    _write_json(root / "baseline_test.json", baseline_artifacts["test"])
    for record, candidate, candidate_root in proposals:

        def candidate_work(split: str) -> RunResult:
            tasks = train if split == "train" else test
            if parallel_phases:
                effective = ToolSet(
                    {**original.files, **candidate.files},
                    original.allowlist,
                    original.language_rules,
                    original.context,
                )
                return replace(
                    benchmark.evaluate_parallel(agent, tasks, effective), split=split
                )
            return replace(
                evaluate_candidate(benchmark, agent, tool_target, candidate, tasks),
                split=split,
            )

        runs, errors = _run_pair(candidate_work, parallel=parallel_phases)
        for split, run in runs.items():
            phase_name = f"optimized_{split}"
            artifact = _phase(benchmark, run)
            record["phases"][phase_name] = _phase_metrics(artifact)
            _write_json(candidate_root / f"{phase_name}.json", artifact)
        if errors:
            failed_split = next(iter(errors))
            error = errors[failed_split]
            record["optimization"].update(
                status="evaluation_failed",
                failed_phase=f"optimized_{failed_split}",
                error_type=type(error).__name__,
            )
            if num_candidates == 1:
                summary.update(
                    {key: value for key, value in record.items() if key != "phases"}
                )
                summary["phases"].update(record["phases"])
                _save_summary(root, summary)
                raise error
            continue
        record["comparison"] = {
            split: _comparison(
                summary["phases"][f"baseline_{split}"],
                record["phases"][f"optimized_{split}"],
            )
            for split in ("train", "test")
        }

    if num_candidates == 1:
        record = records[0]
        summary.update({key: value for key, value in record.items() if key != "phases"})
        summary["phases"].update(record["phases"])
    else:
        summary["candidates"] = records
        completed = sum("comparison" in record for record in records)
        summary["optimization"] = {
            "status": "candidates_evaluated" if completed else "no_valid_candidate",
            "requested": num_candidates,
            "completed": completed,
        }
    _save_summary(root, summary)
    return summary
