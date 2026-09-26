"""TerminalBench 2 and TBLite through Harbor and source-run OpenCode.

The tool target exposes only OpenCode's companion ``.txt`` descriptions. The
benchmark starts a fresh Harbor job for every evaluation; a custom Harbor agent
uploads a user-built OpenCode source bundle and overlays these descriptions in
the sandbox before the agent starts. Upstream repositories are never modified.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from agent_tool_opt_core.api import (
    Agent,
    Benchmark,
    Candidate,
    RunResult,
    TaskRun,
    ToolSet,
    ToolTarget,
    ValidationResult,
    Validator,
)

_RULES = (
    "Edit only the companion .txt tool descriptions. Preserve each tool's "
    "purpose, argument constraints, and safety guidance. The .ts implementations "
    "are read-only context; do not propose code changes."
)
_AGENT_IMPORT = "agent_tool_opt_core.adapters.harbor_opencode:SourceOpenCode"


@dataclass(frozen=True)
class OpenCodeDescriptionValidator(Validator):
    def validate(self, candidate: Candidate) -> ValidationResult:
        base = super().validate(candidate)
        if not base.ok:
            return base
        if any(not name.endswith(".txt") for name in candidate.files):
            return ValidationResult(False, "only .txt descriptions may change")
        if any(len(value) > 100_000 for value in candidate.files.values()):
            return ValidationResult(
                False, "tool description exceeds 100,000 characters"
            )
        return ValidationResult(True, "description edit")


class OpenCodeToolTarget(ToolTarget):
    kind = "txt"
    language_rules = _RULES

    def __init__(self, opencode_checkout: Path) -> None:
        root = Path(opencode_checkout).expanduser().resolve(strict=True)
        tool_dir = root / "packages" / "opencode" / "src" / "tool"
        if not tool_dir.is_dir() or not (root / "package.json").is_file():
            raise ValueError("not an OpenCode source checkout")
        self.tool_dir = tool_dir
        sources = sorted(tool_dir.rglob("*.ts"))
        imported: set[str] = set()
        for source in sources:
            text = source.read_text(encoding="utf-8")
            for rel in re.findall(r"""from\s+["'](\./[^"']+\.txt)["']""", text):
                target = (source.parent / rel).resolve()
                if target.is_relative_to(tool_dir) and target.is_file():
                    imported.add(target.relative_to(tool_dir).as_posix())
        self._baseline = {
            name: (tool_dir / name).read_text(encoding="utf-8")
            for name in sorted(imported)
        }
        if not self._baseline:
            raise ValueError("OpenCode checkout has no companion tool descriptions")
        self._context = {
            f"source/{source.relative_to(tool_dir).as_posix()}": source.read_text(
                encoding="utf-8"
            )
            for source in sources
        }
        self._live = dict(self._baseline)

    def extract(self) -> ToolSet:
        return ToolSet(
            dict(self._baseline), tuple(self._baseline), _RULES, dict(self._context)
        )

    def effective_toolset(self) -> ToolSet:
        return ToolSet(
            dict(self._live), tuple(self._baseline), _RULES, dict(self._context)
        )

    def validator(self) -> Validator:
        return OpenCodeDescriptionValidator(self.kind, tuple(self._baseline))

    def apply(self, candidate: Candidate) -> None:
        result = self.validator().validate(candidate)
        if not result.ok:
            raise ValueError(result.log)
        self._live.update(candidate.files)

    def restore(self) -> None:
        self._live = dict(self._baseline)


class OpenCodeAgent(Agent):
    def __init__(self, model: str) -> None:
        self.id = model
        self.model = model


def load_splits(path: Path) -> tuple[list[str], list[str]]:
    """Read a frozen, explicit split manifest; never derive it from outcomes."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("split manifest must be a JSON object")
    splits = []
    for name in ("train", "test"):
        values = data.get(name)
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) and value.strip() for value in values)
        ):
            raise ValueError(f"{name} must be a nonempty list of task names")
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {name} task")
        splits.append(values)
    if set(splits[0]) & set(splits[1]):
        raise ValueError("train and test tasks overlap")
    return splits[0], splits[1]


def _harbor_results(
    job_dir: Path, task_names: list[str], benchmark: str, agent: str
) -> RunResult:
    data = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    trials = data.get("trial_results")
    if not isinstance(trials, list) or len(trials) != len(task_names):
        raise RuntimeError("Harbor returned incomplete trial results")
    runs: dict[str, TaskRun] = {}
    for trial in trials:
        name = trial.get("task_name")
        if name not in task_names or name in runs:
            raise RuntimeError("Harbor returned an unexpected or duplicate task")
        if trial.get("exception_info"):
            raise RuntimeError(f"Harbor trial failed for {name}")
        verifier = trial.get("verifier_result") or {}
        rewards = verifier.get("rewards") or {}
        reward = rewards.get("reward")
        if not isinstance(reward, (int, float)) or isinstance(reward, bool):
            raise RuntimeError(f"Harbor trial has no numeric reward for {name}")
        trial_dir = (job_dir / trial["trial_name"]).resolve()
        if not trial_dir.is_relative_to(job_dir.resolve()):
            raise RuntimeError("Harbor trial path escapes its job directory")
        trajectory_path = trial_dir / "agent" / "trajectory.json"
        if not trajectory_path.is_file():
            raise RuntimeError(f"Harbor trial has no agent trajectory for {name}")
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        context = trial.get("agent_result") or {}
        cost = context.get("cost_usd")
        if cost is not None and (not isinstance(cost, (int, float)) or cost < 0):
            raise RuntimeError(f"Harbor trial has invalid cost for {name}")
        runs[name] = TaskRun(
            task_id=name,
            reward=float(reward),
            trajectory=trajectory,
            cost_usd=float(cost) if cost is not None else None,
            known_cost_usd=float(cost) if cost is not None else 0.0,
        )
    return RunResult(benchmark, agent, tuple(runs[name] for name in task_names))


class HarborBenchmark(Benchmark):
    """A local-Docker Harbor job on a frozen TerminalBench/TBLite task split."""

    def __init__(
        self,
        *,
        dataset: str,
        benchmark_checkout: Path,
        split_manifest: Path,
        source_bundle: Path,
        bun_linux_binary: Path,
        jobs_dir: Path,
        timeout_seconds: int = 7200,
    ) -> None:
        if dataset not in {"terminal-bench@2.0", "openthoughts-tblite"}:
            raise ValueError("unsupported Harbor dataset")
        self.name = dataset
        train, test = load_splits(split_manifest)
        self._splits = {"train": train, "test": test}
        self.benchmark_checkout = (
            Path(benchmark_checkout).expanduser().resolve(strict=True)
        )
        if not self.benchmark_checkout.is_dir():
            raise ValueError("benchmark_checkout must be a directory")
        for task_name in train + test:
            task_dir = (self.benchmark_checkout / task_name).resolve()
            if (
                not task_dir.is_relative_to(self.benchmark_checkout)
                or not (task_dir / "task.toml").is_file()
            ):
                raise ValueError(
                    f"task is not in the local benchmark checkout: {task_name}"
                )
        self.source_bundle = Path(source_bundle).expanduser().resolve(strict=True)
        self.bun_linux_binary = Path(bun_linux_binary).expanduser().resolve(strict=True)
        if not self.source_bundle.is_file() or not self.bun_linux_binary.is_file():
            raise ValueError("source bundle and Linux Bun binary must be files")
        self.jobs_dir = Path(jobs_dir).expanduser().resolve()
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds

    def tasks(self, split: str) -> list[str]:
        return list(self._splits[split])

    def evaluate(self, agent: Agent, tasks: list[str], tools: ToolSet) -> RunResult:
        if not isinstance(agent, OpenCodeAgent):
            raise TypeError("HarborBenchmark requires OpenCodeAgent")
        if not tasks or len(tasks) != len(set(tasks)):
            raise ValueError("provide distinct task IDs")
        if not set(tasks) <= set(self._splits["train"] + self._splits["test"]):
            raise ValueError("task outside configured splits")
        with tempfile.TemporaryDirectory(prefix="ato-opencode-overlay-") as staging:
            overlay = Path(staging)
            for name, content in tools.files.items():
                rel = PurePosixPath(name)
                if (
                    name not in tools.allowlist
                    or rel.is_absolute()
                    or ".." in rel.parts
                    or not name.endswith(".txt")
                ):
                    raise ValueError("unsafe tool overlay path")
                destination = overlay / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content, encoding="utf-8")
            job_name = f"ato-{uuid.uuid4().hex}"
            args = [
                "harbor",
                "run",
                "--path",
                str(self.benchmark_checkout),
                "--agent",
                _AGENT_IMPORT,
                "--model",
                agent.model,
                "--jobs-dir",
                str(self.jobs_dir),
                "--job-name",
                job_name,
                "--agent-kwarg",
                f"source_bundle={self.source_bundle}",
                "--agent-kwarg",
                f"bun_linux_binary={self.bun_linux_binary}",
                "--agent-kwarg",
                f"overlay_dir={overlay}",
                "--n-concurrent",
                "1",
            ]
            for name in tasks:
                args.extend(("--include-task-name", name))
            result = subprocess.run(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=self.timeout_seconds,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Harbor job failed (exit {result.returncode}); inspect the local job logs"
                )
            return _harbor_results(self.jobs_dir / job_name, tasks, self.name, agent.id)


class TerminalBench2(HarborBenchmark):
    def __init__(self, **kwargs) -> None:
        super().__init__(dataset="terminal-bench@2.0", **kwargs)


class OpenThoughtsTBLite(HarborBenchmark):
    def __init__(self, **kwargs) -> None:
        super().__init__(dataset="openthoughts-tblite", **kwargs)
