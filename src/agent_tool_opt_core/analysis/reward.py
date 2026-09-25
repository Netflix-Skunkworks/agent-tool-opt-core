"""Task-paired reward analysis for saved tau2 Results files.

For each requested k, report task-mean pass^k, a paired task-cluster bootstrap
interval, and a two-sided paired sign-permutation p-value for B - A. The pass^k
estimator is C(successes, k) / C(trials, k).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from .paired import (
    add_analysis_arguments,
    comparisons_from_args,
    paired_estimate,
    permutation_description,
)


def _load_simulations(path: Path) -> list[dict]:
    """Load task IDs and rewards directly, falling back to tau2 for old shapes."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    try:
        raw = json.loads(path.read_text())
        simulations = raw.get("simulations") if isinstance(raw, dict) else None
        if isinstance(simulations, list) and simulations:
            return [
                {
                    "task_id": simulation.get("task_id"),
                    "reward": float(
                        ((simulation.get("reward_info") or {}).get("reward")) or 0.0
                    ),
                }
                for simulation in simulations
            ]
    except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
        pass

    from tau2.data_model.simulation import Results  # noqa: PLC0415

    return [
        {
            "task_id": simulation.task_id,
            "reward": (
                simulation.reward_info.reward if simulation.reward_info else 0.0
            ),
        }
        for simulation in Results.load(path).simulations
    ]


def per_task_success_counts(path: Path) -> dict[str, tuple[int, int]]:
    """Return task ID -> (successes, trials), with reward >= 1 as success."""
    rewards: dict[str, list[float]] = defaultdict(list)
    for simulation in _load_simulations(path):
        rewards[str(simulation["task_id"])].append(float(simulation["reward"]))
    return {
        task: (sum(reward >= 1.0 for reward in values), len(values))
        for task, values in rewards.items()
    }


def pass_k_task(successes: int, trials: int, k: int) -> float:
    """Return C(successes, k) / C(trials, k) for one task."""
    if k > trials or successes < k:
        return 0.0
    return math.comb(successes, k) / math.comb(trials, k)


def compare(
    a_path: Path,
    b_path: Path,
    a_label: str,
    b_label: str,
    ks: list[int],
    n_perm: int,
    n_boot: int,
    seed: int,
) -> None:
    """Print reward estimates and paired inference for one comparison."""
    a_counts = per_task_success_counts(a_path)
    b_counts = per_task_success_counts(b_path)
    common = set(a_counts) & set(b_counts)
    print(f"\nA = {a_label}\nB = {b_label}")
    print(
        f"Tasks (in both): {len(common)}; trials/task: "
        f"A={sorted({trials for _, trials in a_counts.values()})} "
        f"B={sorted({trials for _, trials in b_counts.values()})}"
    )
    print(f"Paired-permutation test: {permutation_description(len(common), n_perm)}")
    print(
        f"\n{'k':>3}  {'pass^k(A)':>10}  {'pass^k(B)':>10}  {'Δ':>10}  "
        f"{'95% CI':>22}  {'perm p':>10}"
    )

    for k in ks:
        valid = sorted(
            task for task in common if a_counts[task][1] >= k and b_counts[task][1] >= k
        )
        if not valid:
            print(f"  k={k}: no tasks have >={k} trials in both runs")
            continue
        a_values = {task: pass_k_task(*a_counts[task], k) for task in valid}
        b_values = {task: pass_k_task(*b_counts[task], k) for task in valid}
        estimate = paired_estimate(
            a_values, b_values, n_perm=n_perm, n_boot=n_boot, seed=seed
        )
        print(
            f"{k:>3}  {estimate.mean_a:>10.4f}  {estimate.mean_b:>10.4f}  "
            f"{estimate.delta:>+10.4f}  "
            f"[{estimate.ci_low:>+.4f}, {estimate.ci_high:>+.4f}]  "
            f"{estimate.p_value:>10.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_analysis_arguments(parser)
    parser.add_argument("--ks", default="1,2,3", help="Comma-separated k values.")
    args = parser.parse_args()
    try:
        ks = [int(value) for value in args.ks.split(",")]
    except ValueError:
        parser.error("--ks must be a comma-separated list of integers")
    if not ks or any(k < 1 for k in ks):
        parser.error("--ks values must be positive")

    for comparison in comparisons_from_args(args, parser):
        compare(
            comparison.a_path,
            comparison.b_path,
            comparison.a_label,
            comparison.b_label,
            ks,
            args.n_perm,
            args.n_boot,
            args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
