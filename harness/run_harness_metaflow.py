"""Local, provider-neutral Metaflow harness for TauBench tool optimization.

This ports the phase graph of the Netflix harness to upstream Metaflow without
Titus, NCP, Metatron, or remote environment decorators. The synthetic mode is
for a no-key CI smoke; ``--benchmark tau2`` uses the real upstream adapter.

Examples:
    python harness/run_harness_metaflow.py show
    python harness/run_harness_metaflow.py run --benchmark synthetic --output-dir runs/mf-smoke
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from metaflow import FlowSpec, Parameter, step

from agent_tool_opt_core.adapters.run_phases import (
    _comparison,
    _cost,
    _load_reused_baseline,
    _phase,
    _phase_metrics,
    _save_candidate,
    _save_summary,
    _toolset_digest,
    _write_json,
)
from agent_tool_opt_core.adapters.tau2 import Tau2Agent, Tau2Benchmark, Tau2ToolTarget
from agent_tool_opt_core.adapters.terminal import load_splits
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
from agent_tool_opt_core.costs import collect_optimizer_costs
from agent_tool_opt_core.driver import (
    collect_baseline,
    evaluate_candidate,
    make_train_evaluator,
    propose,
)
from agent_tool_opt_core.optimizers.catalog import build_optimizer


class _SyntheticAgent(Agent):
    id = "synthetic-agent"


class _SyntheticBenchmark(Benchmark):
    name = "synthetic-benchmark"

    def tasks(self, split: str) -> list[str]:
        return [f"{split}-1"]

    def evaluate(self, agent: Agent, tasks: list[str], tools: ToolSet) -> RunResult:
        reward = float(tools.files["tool.txt"] == "edited")
        return RunResult(
            self.name,
            agent.id,
            tuple(
                TaskRun(task, reward, {"tool": tools.files["tool.txt"]}, cost_usd=0.0)
                for task in tasks
            ),
        )


class _SyntheticTarget(ToolTarget):
    kind = "txt"
    language_rules = "Edit tool.txt only"

    def __init__(self) -> None:
        self._original = ToolSet(
            {"tool.txt": "baseline"}, ("tool.txt",), self.language_rules
        )
        self._live = self._original

    def extract(self) -> ToolSet:
        return self._original

    def effective_toolset(self) -> ToolSet:
        return self._live

    def apply(self, candidate: Candidate) -> None:
        result = self.validator().validate(candidate)
        if not result.ok:
            raise ValueError(result.log)
        self._live = ToolSet(
            {**self._original.files, **candidate.files},
            self._original.allowlist,
            self.language_rules,
        )

    def restore(self) -> None:
        self._live = self._original


class _SyntheticOptimizer(Optimizer):
    id = "synthetic-optimizer"

    def propose(self, tools, run, validate, scratch, train_eval=None) -> Candidate:
        if run.split != "train":
            raise ValueError("optimizer received non-training evidence")
        return Candidate({"tool.txt": "edited"})


class ToolOptimizationHarness(FlowSpec):
    """baseline_train/test -> optimize -> optimized_train/test -> end."""

    benchmark = Parameter("benchmark", default="tau2", help="tau2|synthetic")
    domain = Parameter("domain", default="airline")
    split = Parameter("split", default="", help="Frozen train/test JSON manifest")
    agent_model = Parameter("agent-model", default="")
    user_model = Parameter("user-model", default="")
    optimizer = Parameter("optimizer", default="pi")
    optimizer_model = Parameter("optimizer-model", default="")
    pi_provider = Parameter("pi-provider", default="")
    methods = Parameter("methods", default="reward_shaping,generalization")
    scope = Parameter("scope", default="descriptions")
    num_trials = Parameter("num-trials", default=1, type=int)
    num_candidates = Parameter("num-candidates", default=1, type=int)
    max_steps = Parameter("max-steps", default=30, type=int)
    baseline_dir = Parameter("baseline-dir", default="")
    skip_optimize = Parameter("skip-optimize", default=False, is_flag=True)
    no_transcripts = Parameter("no-transcripts", default=False, is_flag=True)
    no_validation = Parameter("no-validation", default=False, is_flag=True)
    output_dir = Parameter("output-dir", required=True)

    def _components(self) -> tuple[Benchmark, Agent, ToolTarget]:
        if self.benchmark == "synthetic":
            return _SyntheticBenchmark(), _SyntheticAgent(), _SyntheticTarget()
        benchmark = Tau2Benchmark(
            self.domain,
            train_tasks=self.train_tasks,
            test_tasks=self.test_tasks,
            num_trials=self.num_trials,
            max_steps=self.max_steps,
        )
        agent = Tau2Agent(self.agent_model, user_model=self.user_model)
        target = Tau2ToolTarget(
            self.domain, descriptions_only=self.scope == "descriptions"
        )
        return benchmark, agent, target

    def _optimizer(self) -> Optimizer:
        if self.benchmark == "synthetic":
            return _SyntheticOptimizer()
        kwargs = {
            "model": self.optimizer_model,
            "use_transcripts": not self.no_transcripts,
            "require_validation": not self.no_validation,
        }
        if self.optimizer == "pi" and self.pi_provider:
            kwargs["provider"] = self.pi_provider
        return build_optimizer(
            self.optimizer,
            methods=[part.strip() for part in self.methods.split(",") if part.strip()],
            **kwargs,
        )

    def _baseline(self, split: str) -> RunResult:
        benchmark, agent, target = self._components()
        if self.baseline_dir:
            reused = _load_reused_baseline(
                Path(self.baseline_dir),
                benchmark,
                agent,
                self.setup,
                self.train_tasks,
                self.test_tasks,
            )
            return reused[split]
        tasks = self.train_tasks if split == "train" else self.test_tasks
        return collect_baseline(benchmark, agent, tasks, target, split=split)

    @step
    def start(self) -> None:
        if self.benchmark not in {"tau2", "synthetic"}:
            raise ValueError("benchmark must be tau2 or synthetic")
        if self.scope not in {"descriptions", "full"}:
            raise ValueError("scope must be descriptions or full")
        if self.num_candidates < 1 or self.num_trials < 1:
            raise ValueError("candidate and trial counts must be positive")
        if self.benchmark == "synthetic":
            self.train_tasks, self.test_tasks = ["train-1"], ["test-1"]
        else:
            if not self.split or not self.agent_model or not self.user_model:
                raise ValueError(
                    "TauBench needs --split, --agent-model and --user-model"
                )
            if not self.skip_optimize and not self.optimizer_model:
                raise ValueError("optimization needs --optimizer-model")
            self.train_tasks, self.test_tasks = load_splits(Path(self.split))
        benchmark, agent, target = self._components()
        tools = target.extract()
        self.setup = {
            "toolset_sha256": _toolset_digest(tools),
            "num_trials": getattr(benchmark, "num_trials", 1),
            "runs_per_task": getattr(benchmark, "runs_per_task", 1),
            "max_steps": getattr(benchmark, "max_steps", None),
            "user_model": getattr(agent, "user_model", None),
            "n_concurrent": getattr(benchmark, "n_concurrent", None),
            "parallel_phases": False,
        }
        path = Path(self.output_dir).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir(exist_ok=False)
        self.run_dir = str(path)
        self.next(self.baseline_train)

    @step
    def baseline_train(self) -> None:
        benchmark, _, _ = self._components()
        self.train_run = self._baseline("train")
        self.train_phase = _phase(benchmark, self.train_run)
        _write_json(Path(self.run_dir) / "baseline_train.json", self.train_phase)
        self.next(self.baseline_test)

    @step
    def baseline_test(self) -> None:
        benchmark, _, _ = self._components()
        self.test_run = self._baseline("test")
        self.test_phase = _phase(benchmark, self.test_run)
        # Delay the held-out file until all optimizer proposals return.
        self.next(self.optimize)

    @step
    def optimize(self) -> None:
        self.records: list[dict] = []
        self.candidates: list[Candidate | None] = []
        self.optimizer_id = None
        root = Path(self.run_dir)
        if not self.skip_optimize:
            benchmark, agent, target = self._components()
            optimizer = self._optimizer()
            self.optimizer_id = optimizer.id
            train_eval = (
                make_train_evaluator(benchmark, agent, target, self.train_tasks)
                if optimizer.wants_train_eval
                else None
            )
            original = target.extract()
            for index in range(self.num_candidates):
                candidate_root = (
                    root
                    if self.num_candidates == 1
                    else root / f"candidate_{index:02d}"
                )
                if self.num_candidates > 1:
                    candidate_root.mkdir()
                record: dict = {"index": index, "phases": {}}
                self.records.append(record)
                try:
                    with collect_optimizer_costs() as costs:
                        candidate = propose(
                            optimizer,
                            target,
                            self.train_run,
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
                    self.candidates.append(None)
                    continue
                record["optimizer_cost"] = {
                    "llm": _cost(costs.llm),
                    "train_search": _cost(costs.search),
                }
                validation = target.validator().validate(candidate)
                changed = {
                    name: text
                    for name, text in candidate.files.items()
                    if original.files.get(name) != text
                }
                if not validation.ok or not changed:
                    record["optimization"] = {
                        "status": "invalid_candidate"
                        if not validation.ok
                        else "no_edit",
                        "validation": validation.log,
                    }
                    self.candidates.append(None)
                    continue
                candidate = Candidate(changed)
                _save_candidate(candidate_root, candidate, original.allowlist)
                record["optimization"] = {
                    "status": "candidate_applied",
                    "changed_files": sorted(changed),
                    "validation": validation.log,
                }
                self.candidates.append(candidate)
        _write_json(root / "baseline_test.json", self.test_phase)
        self.next(self.optimized_train)

    def _evaluate_candidates(self, split: str) -> None:
        benchmark, agent, target = self._components()
        tasks = self.train_tasks if split == "train" else self.test_tasks
        for record, candidate in zip(self.records, self.candidates, strict=True):
            if (
                candidate is None
                or record["optimization"]["status"] != "candidate_applied"
            ):
                continue
            try:
                run = replace(
                    evaluate_candidate(benchmark, agent, target, candidate, tasks),
                    split=split,
                )
            except Exception as exc:
                record["optimization"].update(
                    status="evaluation_failed",
                    failed_phase=f"optimized_{split}",
                    error_type=type(exc).__name__,
                )
                continue
            phase_name = f"optimized_{split}"
            artifact = _phase(benchmark, run)
            record["phases"][phase_name] = _phase_metrics(artifact)
            candidate_root = (
                Path(self.run_dir)
                if self.num_candidates == 1
                else Path(self.run_dir) / f"candidate_{record['index']:02d}"
            )
            _write_json(candidate_root / f"{phase_name}.json", artifact)

    @step
    def optimized_train(self) -> None:
        self._evaluate_candidates("train")
        self.next(self.optimized_test)

    @step
    def optimized_test(self) -> None:
        self._evaluate_candidates("test")
        self.next(self.end)

    @step
    def end(self) -> None:
        benchmark, agent, _ = self._components()
        self.summary = {
            "benchmark": benchmark.name,
            "agent": agent.id,
            "optimizer": self.optimizer_id,
            "train_tasks": self.train_tasks,
            "test_tasks": self.test_tasks,
            "setup": self.setup,
            "phases": {
                "baseline_train": _phase_metrics(self.train_phase),
                "baseline_test": _phase_metrics(self.test_phase),
            },
        }
        if self.baseline_dir:
            self.summary["baseline_source"] = str(
                Path(self.baseline_dir).expanduser().resolve()
            )
        if self.skip_optimize:
            self.summary["optimization"] = {"status": "baseline_only"}
        elif self.num_candidates == 1:
            record = self.records[0]
            self.summary.update(
                {key: value for key, value in record.items() if key != "phases"}
            )
            self.summary["phases"].update(record["phases"])
            if all(
                f"optimized_{split}" in record["phases"] for split in ("train", "test")
            ):
                self.summary["comparison"] = {
                    split: _comparison(
                        self.summary["phases"][f"baseline_{split}"],
                        record["phases"][f"optimized_{split}"],
                    )
                    for split in ("train", "test")
                }
        else:
            for record in self.records:
                if all(
                    f"optimized_{split}" in record["phases"]
                    for split in ("train", "test")
                ):
                    record["comparison"] = {
                        split: _comparison(
                            self.summary["phases"][f"baseline_{split}"],
                            record["phases"][f"optimized_{split}"],
                        )
                        for split in ("train", "test")
                    }
            completed = sum("comparison" in record for record in self.records)
            self.summary["candidates"] = self.records
            self.summary["optimization"] = {
                "status": "candidates_evaluated" if completed else "no_valid_candidate",
                "requested": self.num_candidates,
                "completed": completed,
            }
        _save_summary(Path(self.run_dir), self.summary)
        print(f"Metaflow local optimization: {self.summary['optimization']['status']}")


if __name__ == "__main__":
    ToolOptimizationHarness()
