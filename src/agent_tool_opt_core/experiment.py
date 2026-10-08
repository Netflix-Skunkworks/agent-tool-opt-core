"""The Experiment layer — a declarative sweep over the four roles.

Runs a ``{benchmark} x {subject_agent} x {optimizer}`` matrix, **shares
the baseline** per ``(benchmark, subject_agent)`` (computed once, fed to
every optimizer), evaluates each candidate across one or more
``eval_agents`` (cross-agent transfer), and produces a tidy
``ResultsTable`` + paired ``analyze``. The execution substrate (local /
local processes, public Metaflow, or remote sandboxes) lives inside each
here — the Experiment only knows the matrix, the sharing, and the analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from agent_tool_opt_core.api import (
    Agent,
    Benchmark,
    Metrics,
    Optimizer,
    RunResult,
    ToolTarget,
)
from agent_tool_opt_core.driver import (
    collect_baseline,
    evaluate_candidate,
    make_train_evaluator,
    propose,
)
from agent_tool_opt_core.orchestration import (
    ProposalMetadata,
    ProposalRetryPolicy,
    candidate_failure_metadata,
    record_proposal_metadata,
    run_proposal,
)

# Sentinel optimizer id for the no-optimization (baseline) rows.
BASELINE = "<baseline>"


@dataclass
class ExperimentSpec:
    benchmarks: list[Benchmark]
    subject_agents: list[Agent]  # whose transcripts the optimizers train on
    optimizers: list[Optimizer]  # each carries its own .id/config
    tool_target_for: Callable[[Benchmark], ToolTarget]
    eval_agents: Optional[list[Agent]] = None  # None -> eval on the subject agent
    train_trials: int = 5
    test_trials: int = 5
    proposal_retry: ProposalRetryPolicy = field(default_factory=ProposalRetryPolicy)


@dataclass(frozen=True)
class ResultRow:
    benchmark: str
    subject_agent: str
    optimizer: str  # BASELINE for the no-optimization row
    eval_agent: str
    split: str
    avg_reward: float | None
    pass_rate: float | None
    n: int
    status: str = "complete"
    proposal: dict | None = None


class ResultsTable:
    """Tidy results + the count of (expensive) baseline runs actually executed."""

    def __init__(self, rows: list[ResultRow], baseline_runs: int) -> None:
        self.rows = rows
        self.baseline_runs = baseline_runs

    def to_dicts(self) -> list[dict]:
        return [asdict(r) for r in self.rows]


def _row(
    b: Benchmark, sa: Agent, opt_id: str, ea: Agent, split: str, m: Metrics
) -> ResultRow:
    return ResultRow(
        benchmark=b.name,
        subject_agent=sa.id,
        optimizer=opt_id,
        eval_agent=ea.id,
        split=split,
        avg_reward=m.avg_reward,
        pass_rate=m.pass_rate,
        n=m.n,
    )


def _failed_row(
    b: Benchmark,
    sa: Agent,
    opt_id: str,
    ea: Agent,
    split: str,
    metadata: ProposalMetadata,
    n: int,
) -> ResultRow:
    candidate_failure = metadata.failure_kind == "candidate"
    return ResultRow(
        benchmark=b.name,
        subject_agent=sa.id,
        optimizer=opt_id,
        eval_agent=ea.id,
        split=split,
        avg_reward=0.0 if candidate_failure else None,
        pass_rate=0.0 if candidate_failure else None,
        n=n if candidate_failure else 0,
        status=metadata.status,
        proposal=metadata.to_dict(),
    )


class Experiment:
    def __init__(self) -> None:
        self.baseline_runs = 0

    def run(self, spec: ExperimentSpec, scratch: Path) -> ResultsTable:
        rows: list[ResultRow] = []
        # Shared baseline transcripts, one per (benchmark, subject_agent),
        # reused by every optimizer (the key variance-reduction property).
        shared: dict[tuple[str, str], RunResult] = {}

        for b in spec.benchmarks:
            tt = spec.tool_target_for(b)
            train, test = b.tasks("train"), b.tasks("test")
            eval_agents = spec.eval_agents or list(spec.subject_agents)

            for sa in spec.subject_agents:
                key = (b.name, sa.id)
                if key not in shared:
                    shared[key] = collect_baseline(b, sa, train, tt)
                    self.baseline_runs += 1
                baseline = shared[key]

                # Baseline test scores (per eval agent) — the paired anchor.
                for ea in eval_agents:
                    brun = b.evaluate(ea, test, tt.extract())
                    rows.append(_row(b, sa, BASELINE, ea, "test", b.score(brun)))

                for opt in spec.optimizers:
                    # Search optimizers (gepa) get a train-only evaluator over the
                    # shared train split; blind optimizers get None (cannot probe
                    # reward). Mirrors driver.optimize so sweeps don't silently
                    # degrade a search optimizer to a single proposal.
                    train_eval = (
                        make_train_evaluator(b, sa, tt, train)
                        if getattr(opt, "wants_train_eval", False)
                        else None
                    )
                    proposal_scratch = scratch / f"{b.name}__{sa.id}__{opt.id}"
                    proposal = run_proposal(
                        lambda attempt_scratch: propose(
                            opt,
                            tt,
                            baseline,
                            attempt_scratch,
                            train_eval,
                        ),
                        proposal_scratch,
                        spec.proposal_retry,
                    )
                    if proposal.candidate is None:
                        for ea in eval_agents:
                            rows.append(
                                _failed_row(
                                    b,
                                    sa,
                                    opt.id,
                                    ea,
                                    "test",
                                    proposal.metadata,
                                    len(test),
                                )
                            )
                        continue
                    cand = proposal.candidate
                    validation = tt.validator().validate(cand)
                    if not validation.ok:
                        metadata = candidate_failure_metadata(
                            f"candidate failed outer validation: {validation.log}",
                            attempts=proposal.metadata.attempt_count,
                            elapsed_seconds=proposal.metadata.elapsed_seconds,
                        )
                        record_proposal_metadata(proposal_scratch, metadata)
                        for ea in eval_agents:
                            rows.append(
                                _failed_row(
                                    b, sa, opt.id, ea, "test", metadata, len(test)
                                )
                            )
                        continue
                    for ea in eval_agents:
                        run = evaluate_candidate(b, ea, tt, cand, test)
                        rows.append(_row(b, sa, opt.id, ea, "test", b.score(run)))

        return ResultsTable(rows, self.baseline_runs)

    def analyze(self, table: ResultsTable) -> list[dict]:
        """Paired optimized-minus-baseline deltas per cell."""
        base = {
            (r.benchmark, r.subject_agent, r.eval_agent, r.split): r
            for r in table.rows
            if r.optimizer == BASELINE
        }
        out: list[dict] = []
        for r in table.rows:
            if r.optimizer == BASELINE:
                continue
            if r.avg_reward is None or r.pass_rate is None:
                continue
            b = base.get((r.benchmark, r.subject_agent, r.eval_agent, r.split))
            if b is None:
                continue
            out.append(
                {
                    "benchmark": r.benchmark,
                    "subject_agent": r.subject_agent,
                    "optimizer": r.optimizer,
                    "eval_agent": r.eval_agent,
                    "split": r.split,
                    "delta_reward": r.avg_reward - b.avg_reward,
                    "delta_pass_rate": r.pass_rate - b.pass_rate,
                }
            )
        return out
