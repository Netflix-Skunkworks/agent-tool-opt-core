"""Shared task-paired inference and run selection for saved tau2 results.

The two-sided permutation test pairs endpoint values by task and independently
flips each B - A difference. It enumerates every sign assignment for small task
sets and uses corrected Monte Carlo sampling otherwise. Confidence intervals
use a paired percentile bootstrap that resamples whole tasks.

The permutation design follows rycolab/paired-perm-test and Zmigrod, Vieira &
Cotterell, "Exact Paired-Permutation Testing for Structured Test Statistics",
NAACL 2022 (arXiv:2205.01416).
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Mapping

# Above this many tasks, 2^N enumeration is too expensive, so use Monte Carlo.
EXACT_MAX_TASKS = 22


@dataclass(frozen=True)
class PairedEstimate:
    """Mean endpoint estimates and paired inference for B - A."""

    mean_a: float
    mean_b: float
    ci_low: float
    ci_high: float
    p_value: float

    @property
    def delta(self) -> float:
        return self.mean_b - self.mean_a


@dataclass(frozen=True)
class Comparison:
    """Resolved Results files and display labels for one comparison."""

    a_path: Path
    b_path: Path
    a_label: str
    b_label: str


def _paired_deltas(
    a_per_task: Mapping[str, float], b_per_task: Mapping[str, float]
) -> list[float]:
    if not a_per_task or a_per_task.keys() != b_per_task.keys():
        raise ValueError("Paired inference requires identical nonempty task sets")
    return [b_per_task[task] - a_per_task[task] for task in sorted(a_per_task)]


def permutation_method(n_tasks: int) -> str:
    """Return the permutation strategy used for a task count."""
    return "exact" if n_tasks <= EXACT_MAX_TASKS else "monte-carlo"


def paired_permutation_p(
    a_per_task: Mapping[str, float],
    b_per_task: Mapping[str, float],
    n_perm: int = 100_000,
    seed: int = 0,
) -> float:
    """Two-sided paired sign-permutation p-value for the mean B - A effect."""
    if n_perm < 1:
        raise ValueError("n_perm must be positive")
    deltas = _paired_deltas(a_per_task, b_per_task)
    observed = abs(fmean(deltas))
    epsilon = 1e-12

    if len(deltas) <= EXACT_MAX_TASKS:
        extreme = 0
        for bits in range(1 << len(deltas)):
            permuted = sum(
                -delta if (bits >> index) & 1 else delta
                for index, delta in enumerate(deltas)
            ) / len(deltas)
            extreme += abs(permuted) >= observed - epsilon
        return extreme / (1 << len(deltas))

    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_perm):
        permuted = fmean(-delta if rng.random() < 0.5 else delta for delta in deltas)
        extreme += abs(permuted) >= observed - epsilon
    return (extreme + 1) / (n_perm + 1)


def paired_mean_ci(
    a_per_task: Mapping[str, float],
    b_per_task: Mapping[str, float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for mean B - A, resampling whole task pairs."""
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    deltas = _paired_deltas(a_per_task, b_per_task)
    rng = random.Random(seed)
    samples = sorted(
        fmean(deltas[rng.randrange(len(deltas))] for _ in deltas) for _ in range(n_boot)
    )
    low = int(alpha / 2 * n_boot)
    high = min(int((1 - alpha / 2) * n_boot), n_boot - 1)
    return samples[low], samples[high]


def paired_estimate(
    a_per_task: Mapping[str, float],
    b_per_task: Mapping[str, float],
    *,
    n_perm: int,
    n_boot: int,
    seed: int,
) -> PairedEstimate:
    """Compute condition means, a paired CI, and a paired permutation p-value."""
    low, high = paired_mean_ci(a_per_task, b_per_task, n_boot, seed)
    return PairedEstimate(
        mean_a=fmean(a_per_task.values()),
        mean_b=fmean(b_per_task.values()),
        ci_low=low,
        ci_high=high,
        p_value=paired_permutation_p(a_per_task, b_per_task, n_perm, seed),
    )


def permutation_description(n_tasks: int, n_perm: int) -> str:
    """Human-readable description of the selected permutation strategy."""
    method = permutation_method(n_tasks)
    detail = (
        f"exact 2^N for N<={EXACT_MAX_TASKS}"
        if method == "exact"
        else f"{n_perm} MC samples"
    )
    return f"{method} ({detail})"


def resolve_results_file(run_dir: Path, phase: str, split: str) -> Path:
    """Resolve run-local or Metaflow per-candidate Results filenames."""
    direct = run_dir / f"{phase}_{split}.json"
    candidates = [direct, run_dir / f"{phase}_{split}_c00.json"]
    candidates.extend(sorted(run_dir.glob(f"{phase}_{split}_c*.json")))
    return next((path for path in candidates if path.is_file()), direct)


def add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared run-selection and inference arguments to an analysis CLI."""
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=Path, help="Compare baseline and optimized.")
    group.add_argument("--a", type=Path, help="Run A directory (with --b).")
    group.add_argument("--ours", type=Path, help="Our run (with --baselines).")
    parser.add_argument("--b", type=Path, help="Run B directory (with --a).")
    parser.add_argument("--baselines", type=Path, nargs="+")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--phase", choices=("baseline", "optimized"), default="optimized"
    )
    parser.add_argument("--n-perm", type=int, default=100_000)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)


def comparisons_from_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> list[Comparison]:
    """Validate shared arguments and resolve all requested comparisons."""
    if args.n_perm < 1 or args.n_boot < 1:
        parser.error("--n-perm and --n-boot must be positive")
    if args.ours is not None:
        if not args.baselines:
            parser.error("--baselines is required with --ours")
        pairs = [(base, args.phase, args.ours, args.phase) for base in args.baselines]
    elif args.run is not None:
        pairs = [(args.run, "baseline", args.run, "optimized")]
    else:
        if args.b is None:
            parser.error("--b is required with --a")
        pairs = [(args.a, args.phase, args.b, args.phase)]

    return [
        Comparison(
            resolve_results_file(a_dir, a_phase, args.split),
            resolve_results_file(b_dir, b_phase, args.split),
            f"{a_dir.name}/{a_phase}_{args.split}",
            f"{b_dir.name}/{b_phase}_{args.split}",
        )
        for a_dir, a_phase, b_dir, b_phase in pairs
    ]
