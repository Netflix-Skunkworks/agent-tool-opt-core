"""Run the five-phase public TauBench Verified experiment.

Example: python -m agent_tool_opt_core.adapters.run_tau2 --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_tool_opt_core.adapters.run_phases import run_five_phases
from agent_tool_opt_core.adapters.tau2 import Tau2Agent, Tau2Benchmark, Tau2ToolTarget
from agent_tool_opt_core.adapters.terminal import load_splits
from agent_tool_opt_core.optimizers.catalog import build_optimizer
from agent_tool_opt_core.optimizers.pi_sandbox import (
    add_pi_sandbox_arguments,
    pi_sandbox_kwargs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain", choices=("airline", "retail", "telecom"), required=True
    )
    parser.add_argument(
        "--split", type=Path, required=True, help="Frozen train/test JSON manifest"
    )
    parser.add_argument("--agent-model", required=True)
    parser.add_argument("--user-model", required=True)
    parser.add_argument("--optimizer", choices=("pi", "llm"), default="pi")
    parser.add_argument("--optimizer-model")
    parser.add_argument("--pi-provider", help="Pi provider name, if needed")
    add_pi_sandbox_arguments(parser)
    parser.add_argument("--methods", default="reward_shaping,generalization")
    parser.add_argument(
        "--scope", choices=("descriptions", "full"), default="descriptions"
    )
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    args = parser.parse_args(argv)
    if not args.baseline_only and not args.optimizer_model:
        parser.error("--optimizer-model is required unless --baseline-only is set")
    try:
        sandbox_kwargs = pi_sandbox_kwargs(
            args.pi_sandbox,
            tuple(args.pi_sandbox_env),
            optimizer=args.optimizer,
            enabled=not args.baseline_only,
        )
    except ValueError as exc:
        parser.error(str(exc))

    train, test = load_splits(args.split)
    benchmark = Tau2Benchmark(
        args.domain,
        train_tasks=train,
        test_tasks=test,
        max_steps=args.max_steps,
        num_trials=args.num_trials,
    )
    agent = Tau2Agent(args.agent_model, user_model=args.user_model)
    target = Tau2ToolTarget(args.domain, descriptions_only=args.scope == "descriptions")
    optimizer = None
    if not args.baseline_only:
        optimizer_kwargs = {"model": args.optimizer_model, **sandbox_kwargs}
        if args.optimizer == "pi" and args.pi_provider:
            optimizer_kwargs["provider"] = args.pi_provider
        optimizer = build_optimizer(
            args.optimizer,
            methods=[name.strip() for name in args.methods.split(",") if name.strip()],
            **optimizer_kwargs,
        )
    result = run_five_phases(
        benchmark,
        agent,
        target,
        optimizer,
        train=train,
        test=test,
        output_dir=args.out,
        baseline_dir=args.baseline_dir,
        num_candidates=args.num_candidates,
    )
    print(f"Saved {result['optimization']['status']} run to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
