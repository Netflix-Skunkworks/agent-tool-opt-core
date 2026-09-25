"""Task-paired recurring-cost analysis for saved tau2 Results files.

Report task-mean agent_cost + user_cost in USD per simulation, a paired
task-cluster bootstrap interval, and a two-sided paired sign-permutation p-value
for B - A. Negative deltas mean cheaper inference. One-time optimizer spend is
excluded.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from .paired import (
    add_analysis_arguments,
    comparisons_from_args,
    paired_estimate,
    permutation_description,
)


def per_task_costs(path: Path) -> tuple[dict[str, float], dict[str, set[int]]]:
    """Return task-mean recurring USD and exact trial coverage."""
    raw = json.loads(path.read_text())
    totals: dict[str, list[float]] = defaultdict(list)
    trials: dict[str, set[int]] = defaultdict(set)
    for simulation in raw["simulations"]:
        task = str(simulation["task_id"])
        trial = simulation.get("trial")
        if type(trial) is not int or trial < 0 or trial in trials[task]:
            raise ValueError(f"{path}: missing/invalid/duplicate trial for task {task}")
        amounts = [simulation.get("agent_cost"), simulation.get("user_cost")]
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in amounts
        ):
            raise ValueError(
                f"{path}: missing/invalid cost for task {task}, trial {trial}"
            )
        total = sum(amounts)
        if not math.isfinite(total):
            raise ValueError(
                f"{path}: nonfinite total cost for task {task}, trial {trial}"
            )
        totals[task].append(total)
        trials[task].add(trial)
    if not totals:
        raise ValueError(f"{path}: no simulations for cost comparison")
    return {task: fmean(values) for task, values in totals.items()}, dict(trials)


def compare(
    a_path: Path,
    b_path: Path,
    a_label: str,
    b_label: str,
    n_perm: int,
    n_boot: int,
    seed: int,
) -> None:
    """Print recurring-cost estimates and paired inference for one comparison."""
    a_cost, a_trials = per_task_costs(a_path)
    b_cost, b_trials = per_task_costs(b_path)
    if a_trials != b_trials:
        raise ValueError("Cost comparison requires identical task/trial coverage")
    estimate = paired_estimate(a_cost, b_cost, n_perm=n_perm, n_boot=n_boot, seed=seed)

    print(f"\nA = {a_label}\nB = {b_label}")
    print(
        f"Matched tasks: {len(a_cost)}; trials/task: "
        f"{sorted({len(task_trials) for task_trials in a_trials.values()})}"
    )
    print(f"Paired-permutation test: {permutation_description(len(a_cost), n_perm)}")
    print(
        "Recurring cost (agent + simulator USD/simulation; negative delta is cheaper)"
    )
    print(
        f"A={estimate.mean_a:.8f} B={estimate.mean_b:.8f} "
        f"delta={estimate.delta:+.8f} "
        f"95% CI=[{estimate.ci_low:+.8f}, {estimate.ci_high:+.8f}] "
        f"perm p={estimate.p_value:.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_analysis_arguments(parser)
    args = parser.parse_args()
    for comparison in comparisons_from_args(args, parser):
        compare(
            comparison.a_path,
            comparison.b_path,
            comparison.a_label,
            comparison.b_label,
            args.n_perm,
            args.n_boot,
            args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
