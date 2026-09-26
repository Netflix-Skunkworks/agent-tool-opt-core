"""TerminalBench 2 and TBLite through Harbor and source-run OpenCode.

The target exposes OpenCode's companion ``.txt`` descriptions by default and
the active ``.ts`` tool modules in opt-in full-code mode. The benchmark starts
a fresh Harbor job for every evaluation; a custom Harbor agent uploads a
user-built OpenCode source bundle and overlays candidate files in the sandbox
before the agent starts. Upstream repositories are never modified.
"""

from __future__ import annotations

import json
import math
import re
import shutil
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
from agent_tool_opt_core.costs import usd

_DESCRIPTION_RULES = (
    "Edit only the companion .txt tool descriptions. Preserve each tool's "
    "purpose, argument constraints, and safety guidance. The .ts implementations "
    "are read-only context; do not propose code changes."
)
_CODE_RULES = (
    "Edit the active OpenCode tool .ts modules and their companion .txt "
    "descriptions. Preserve Tool.define IDs, exported tool names, parameter "
    "contracts, and execute return shapes. Keep every module buildable."
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


@dataclass(frozen=True)
class OpenCodeCodeValidator(Validator):
    def validate(self, candidate: Candidate) -> ValidationResult:
        base = super().validate(candidate)
        if not base.ok:
            return base
        code = {
            name: value
            for name, value in candidate.files.items()
            if name.endswith(".ts")
        }
        if not code:
            return ValidationResult(True, "no TypeScript change")
        bun = shutil.which("bun")
        if bun is None:
            return ValidationResult(
                False, "Bun is required to validate TypeScript edits"
            )
        with tempfile.TemporaryDirectory(prefix="ato-bun-check-") as directory:
            for name, content in code.items():
                path = Path(directory) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                try:
                    result = subprocess.run(
                        [bun, "build", "--no-bundle", "--target=node", str(path)],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                        timeout=180,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    return ValidationResult(False, f"Bun validation timed out: {name}")
                if result.returncode != 0:
                    return ValidationResult(False, f"Bun could not parse {name}")
        return ValidationResult(True, "TypeScript parsed")


class OpenCodeToolTarget(ToolTarget):
    kind = "ts"
    language_rules = _DESCRIPTION_RULES

    def __init__(
        self, opencode_checkout: Path, *, descriptions_only: bool = True
    ) -> None:
        root = Path(opencode_checkout).expanduser().resolve(strict=True)
        tool_dir = root / "packages" / "opencode" / "src" / "tool"
        if not tool_dir.is_dir() or not (root / "package.json").is_file():
            raise ValueError("not an OpenCode source checkout")
        self.tool_dir = tool_dir
        self.descriptions_only = descriptions_only
        self.language_rules = _DESCRIPTION_RULES if descriptions_only else _CODE_RULES
        sources = sorted(tool_dir.rglob("*.ts"))
        imported: set[str] = set()
        for source in sources:
            text = source.read_text(encoding="utf-8")
            for rel in re.findall(r"""from\s+["'](\./[^"']+\.txt)["']""", text):
                target = (source.parent / rel).resolve()
                if target.is_relative_to(tool_dir) and target.is_file():
                    imported.add(target.relative_to(tool_dir).as_posix())
        descriptions = {
            name: (tool_dir / name).read_text(encoding="utf-8")
            for name in sorted(imported)
        }
        if not descriptions:
            raise ValueError("OpenCode checkout has no companion tool descriptions")
        active_code: dict[str, str] = {}
        registry = tool_dir / "registry.ts"
        if not descriptions_only:
            if not registry.is_file():
                raise ValueError("OpenCode checkout has no tool registry")
            for name in re.findall(
                r"""from\s+["']\./([a-zA-Z0-9_-]+)["']""",
                registry.read_text(encoding="utf-8"),
            ):
                source = tool_dir / f"{name}.ts"
                if source.is_file():
                    content = source.read_text(encoding="utf-8")
                    if "Tool.define" in content:
                        active_code[source.name] = content
            if not active_code:
                raise ValueError("OpenCode checkout has no active tool modules")
        self._baseline = {**descriptions, **active_code}
        self._context = {
            f"source/{source.relative_to(tool_dir).as_posix()}": source.read_text(
                encoding="utf-8"
            )
            for source in sources
            if source.name not in active_code
        }
        self._live = dict(self._baseline)

    def extract(self) -> ToolSet:
        return ToolSet(
            dict(self._baseline),
            tuple(self._baseline),
            self.language_rules,
            dict(self._context),
        )

    def effective_toolset(self) -> ToolSet:
        return ToolSet(
            dict(self._live),
            tuple(self._baseline),
            self.language_rules,
            dict(self._context),
        )

    def validator(self) -> Validator:
        if self.descriptions_only:
            return OpenCodeDescriptionValidator(self.kind, tuple(self._baseline))
        return OpenCodeCodeValidator(self.kind, tuple(self._baseline))

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
    job_dir: Path,
    task_names: list[str],
    benchmark: str,
    agent: str,
    num_trials: int = 1,
) -> RunResult:
    data = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    trials = data.get("trial_results")
    if not isinstance(trials, list) or len(trials) != len(task_names) * num_trials:
        raise RuntimeError("Harbor returned incomplete trial results")
    grouped: dict[str, list[TaskRun]] = {name: [] for name in task_names}
    seen_trials: set[str] = set()
    for trial in trials:
        name = trial.get("task_name")
        trial_name = trial.get("trial_name")
        if (
            name not in grouped
            or not isinstance(trial_name, str)
            or trial_name in seen_trials
        ):
            raise RuntimeError("Harbor returned an unexpected or duplicate trial")
        seen_trials.add(trial_name)
        if trial.get("exception_info"):
            raise RuntimeError(f"Harbor trial failed for {name}")
        verifier = trial.get("verifier_result") or {}
        rewards = verifier.get("rewards") or {}
        reward = rewards.get("reward")
        if (
            not isinstance(reward, (int, float))
            or isinstance(reward, bool)
            or not math.isfinite(reward)
        ):
            raise RuntimeError(f"Harbor trial has no numeric reward for {name}")
        trial_dir = (job_dir / trial_name).resolve()
        if not trial_dir.is_relative_to(job_dir.resolve()):
            raise RuntimeError("Harbor trial path escapes its job directory")
        trajectory_path = trial_dir / "agent" / "trajectory.json"
        if not trajectory_path.is_file():
            raise RuntimeError(f"Harbor trial has no agent trajectory for {name}")
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        context = trial.get("agent_result") or {}
        cost = context.get("cost_usd")
        if cost is not None and usd(cost) is None:
            raise RuntimeError(f"Harbor trial has invalid cost for {name}")
        grouped[name].append(
            TaskRun(
                task_id=name,
                reward=float(reward),
                trajectory=trajectory,
                cost_usd=float(cost) if cost is not None else None,
                known_cost_usd=float(cost) if cost is not None else 0.0,
            )
        )
    runs = []
    for name in task_names:
        attempts = grouped[name]
        if len(attempts) != num_trials:
            raise RuntimeError(f"Harbor returned incomplete attempts for {name}")
        runs.append(
            TaskRun(
                task_id=name,
                reward=sum(attempt.reward for attempt in attempts) / num_trials,
                trajectory=(
                    attempts[0].trajectory
                    if num_trials == 1
                    else [attempt.trajectory for attempt in attempts]
                ),
                cost_usd=(
                    sum(attempt.cost_usd for attempt in attempts)
                    if all(attempt.cost_usd is not None for attempt in attempts)
                    else None
                ),
                known_cost_usd=sum(attempt.known_cost_usd for attempt in attempts),
            )
        )
    return RunResult(benchmark, agent, tuple(runs))


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
        num_trials: int = 1,
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
        if num_trials < 1:
            raise ValueError("num_trials must be positive")
        self.num_trials = num_trials
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
                    or not name.endswith((".txt", ".ts"))
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
                "--n-attempts",
                str(self.num_trials),
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
            return _harbor_results(
                self.jobs_dir / job_name, tasks, self.name, agent.id, self.num_trials
            )


class TerminalBench2(HarborBenchmark):
    def __init__(self, **kwargs) -> None:
        super().__init__(dataset="terminal-bench@2.0", **kwargs)


class OpenThoughtsTBLite(HarborBenchmark):
    def __init__(self, **kwargs) -> None:
        super().__init__(dataset="openthoughts-tblite", **kwargs)
