"""Local upstream-Metaflow compatibility smoke for the public optimizer core."""

from pathlib import Path

from metaflow import FlowSpec, Parameter, step

from agent_tool_opt_core.adapters.run_phases import run_five_phases
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


class SmokeAgent(Agent):
    id = "synthetic-agent"


class SmokeBenchmark(Benchmark):
    name = "synthetic-benchmark"

    def tasks(self, split):
        return ["train-1"] if split == "train" else ["test-1"]

    def evaluate(self, agent, tasks, tools):
        reward = float(tools.files["tool.txt"] == "edited")
        return RunResult(
            self.name,
            agent.id,
            tuple(
                TaskRun(task, reward, {"tool": tools.files["tool.txt"]}, cost_usd=0.0)
                for task in tasks
            ),
        )


class SmokeTarget(ToolTarget):
    kind = "txt"
    language_rules = "Edit tool.txt only"

    def __init__(self):
        self.original = ToolSet(
            {"tool.txt": "baseline"}, ("tool.txt",), self.language_rules
        )
        self.live = self.original

    def extract(self):
        return self.original

    def effective_toolset(self):
        return self.live

    def apply(self, candidate):
        self.live = ToolSet(
            {**self.original.files, **candidate.files},
            self.original.allowlist,
            self.language_rules,
        )

    def restore(self):
        self.live = self.original


class SmokeOptimizer(Optimizer):
    id = "synthetic-optimizer"

    def propose(self, tools, run, validate, scratch, train_eval=None):
        assert run.split == "train"
        return Candidate({"tool.txt": "edited"})


class PublicAdapterSmoke(FlowSpec):
    output_dir = Parameter("output-dir", required=True)

    @step
    def start(self):
        self.next(self.execute)

    @step
    def execute(self):
        self.summary = run_five_phases(
            SmokeBenchmark(),
            SmokeAgent(),
            SmokeTarget(),
            SmokeOptimizer(),
            train=["train-1"],
            test=["test-1"],
            output_dir=Path(self.output_dir),
        )
        self.next(self.end)

    @step
    def end(self):
        assert self.summary["optimization"]["status"] == "candidate_applied"
        assert self.summary["comparison"]["test"]["avg_reward_delta"] == 1.0
        print("public Metaflow smoke: passed")


if __name__ == "__main__":
    PublicAdapterSmoke()
