"""Run the five-phase public TerminalBench 2 or TBLite experiment.

Example: python -m agent_tool_opt_core.adapters.run_terminal --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_tool_opt_core.adapters.run_phases import run_five_phases
from agent_tool_opt_core.adapters.terminal import (
    OpenCodeAgent,
    OpenCodeToolTarget,
    OpenThoughtsTBLite,
    TerminalBench2,
)
from agent_tool_opt_core.optimizers.catalog import build_optimizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("tb2", "tblite"), required=True)
    parser.add_argument("--benchmark-checkout", type=Path, required=True)
    parser.add_argument("--opencode-checkout", type=Path, required=True)
    parser.add_argument(
        "--split", type=Path, required=True, help="Frozen train/test JSON manifest"
    )
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--bun-linux-binary", type=Path, required=True)
    parser.add_argument("--jobs-dir", type=Path, required=True)
    parser.add_argument("--agent-model", required=True)
    parser.add_argument("--optimizer", choices=("pi", "llm"), default="pi")
    parser.add_argument("--optimizer-model", required=True)
    parser.add_argument("--pi-provider", help="Pi provider name, if needed")
    parser.add_argument("--methods", default="reward_shaping,generalization")
    parser.add_argument(
        "--scope", choices=("descriptions", "full"), default="descriptions"
    )
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    args = parser.parse_args(argv)

    benchmark_class = TerminalBench2 if args.benchmark == "tb2" else OpenThoughtsTBLite
    benchmark = benchmark_class(
        benchmark_checkout=args.benchmark_checkout,
        opencode_checkout=args.opencode_checkout,
        split_manifest=args.split,
        source_bundle=args.source_bundle,
        bun_linux_binary=args.bun_linux_binary,
        jobs_dir=args.jobs_dir,
        num_trials=args.num_trials,
    )
    agent = OpenCodeAgent(args.agent_model)
    target = OpenCodeToolTarget(
        args.opencode_checkout, descriptions_only=args.scope == "descriptions"
    )
    optimizer_kwargs = {"model": args.optimizer_model}
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
        train=benchmark.tasks("train"),
        test=benchmark.tasks("test"),
        output_dir=args.out,
    )
    print(f"Saved {result['optimization']['status']} run to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
