"""Offline contract tests for the public benchmark adapters."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_tool_opt_core.adapters.tau2 import (
    Tau2Agent,
    Tau2Benchmark,
    Tau2DescriptionValidator,
    Tau2ToolTarget,
)
from agent_tool_opt_core.adapters.terminal import (
    OpenCodeAgent,
    OpenCodeToolTarget,
    OpenThoughtsTBLite,
    TerminalBench2,
    _harbor_results,
    load_splits,
)
from agent_tool_opt_core.api import Candidate
from agent_tool_opt_core.driver import collect_baseline, evaluate_candidate


_TOOLS = '''
def is_tool(fn):
    return fn

class AirlineTools:
    @is_tool
    def lookup(self, reservation_id: str):
        """Look up a reservation by ID."""
        return reservation_id
'''


def test_tau2_validator_rejects_code_and_signature_changes():
    validator = Tau2DescriptionValidator("py", ("tools.py",), _TOOLS)
    good = _TOOLS.replace(
        "Look up a reservation by ID.", "Use the exact reservation ID."
    )
    assert validator.validate(Candidate({"tools.py": good})).ok
    assert not validator.validate(
        Candidate({"tools.py": good.replace("return reservation_id", "return None")})
    ).ok
    assert not validator.validate(
        Candidate({"tools.py": good.replace("reservation_id: str", "id: str")})
    ).ok


def test_tau2_tool_target_swaps_and_restores_class(monkeypatch, tmp_path):
    source_file = tmp_path / "tools.py"
    source_file.write_text(_TOOLS)
    tools_module = types.ModuleType("tau2.domains.airline.tools")
    tools_module.__file__ = str(source_file)
    environment = types.ModuleType("tau2.domains.airline.environment")
    original = type("AirlineTools", (), {})
    environment.AirlineTools = original
    monkeypatch.setitem(sys.modules, tools_module.__name__, tools_module)
    monkeypatch.setitem(sys.modules, environment.__name__, environment)

    target = Tau2ToolTarget("airline")
    edited = _TOOLS.replace("Look up a reservation by ID.", "Use the exact ID.")
    target.apply(Candidate({"tools.py": edited}))
    assert environment.AirlineTools is not original
    assert environment.AirlineTools().lookup("ABC") == "ABC"
    assert "Use the exact ID" in target.effective_toolset().files["tools.py"]
    target.restore()
    assert environment.AirlineTools is original
    assert target.effective_toolset() == target.extract()


def test_tau2_runner_preserves_transcript_reward_and_cost(monkeypatch):
    run_module = types.ModuleType("tau2.run")
    calls = {}

    def run_tasks(**kwargs):
        calls.update(kwargs)
        sim = SimpleNamespace(
            task_id="t1",
            reward_info=SimpleNamespace(reward=0.75),
            messages=[
                SimpleNamespace(model_dump=lambda: {"role": "user", "content": "hi"})
            ],
            agent_cost=0.10,
            user_cost=0.05,
        )
        return SimpleNamespace(simulations=[sim])

    run_module.get_tasks = lambda domain, task_ids: [SimpleNamespace(id="t1")]
    run_module.run_tasks = run_tasks
    run_module.EvaluationType = SimpleNamespace(ALL="all")
    monkeypatch.setitem(sys.modules, "tau2.run", run_module)
    benchmark = Tau2Benchmark("airline", train_tasks=["t1"], test_tasks=["t2"])
    result = benchmark.evaluate(
        Tau2Agent("agent-model", user_model="user-model"), ["t1"], None
    )
    assert result.runs[0].reward == 0.75
    assert result.runs[0].trajectory == [{"role": "user", "content": "hi"}]
    assert result.runs[0].cost_usd == pytest.approx(0.15)
    assert calls["llm_agent"] == "agent-model"
    assert calls["llm_user"] == "user-model"


def _opencode_checkout(tmp_path: Path) -> Path:
    root = tmp_path / "opencode"
    tool_dir = root / "packages" / "opencode" / "src" / "tool"
    (tool_dir / "shell").mkdir(parents=True)
    (root / "package.json").write_text("{}")
    (tool_dir / "read.ts").write_text('import DESCRIPTION from "./read.txt"')
    (tool_dir / "read.txt").write_text("Read a file.")
    (tool_dir / "shell" / "prompt.ts").write_text(
        'import DESCRIPTION from "./shell.txt"'
    )
    (tool_dir / "shell" / "shell.txt").write_text("Run a shell command.")
    return root


def test_opencode_target_discovers_nested_imported_descriptions(tmp_path):
    target = OpenCodeToolTarget(_opencode_checkout(tmp_path))
    assert set(target.extract().files) == {"read.txt", "shell/shell.txt"}
    target.apply(Candidate({"shell/shell.txt": "Run one command."}))
    assert target.effective_toolset().files["shell/shell.txt"] == "Run one command."
    target.restore()
    assert target.effective_toolset().files["shell/shell.txt"] == "Run a shell command."
    with pytest.raises(ValueError):
        target.apply(Candidate({"read.ts": "bad"}))


def test_harbor_baseline_and_candidate_use_same_task_and_overlay(monkeypatch, tmp_path):
    source = _opencode_checkout(tmp_path)
    target = OpenCodeToolTarget(source)
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"train": ["task-1"], "test": ["task-2"]}))
    checkout = tmp_path / "terminal-bench-2"
    for name in ("task-1", "task-2"):
        (checkout / name).mkdir(parents=True)
        (checkout / name / "task.toml").write_text("")
    bundle = tmp_path / "source.tar"
    bundle.write_bytes(b"bundle")
    bun = tmp_path / "bun"
    bun.write_bytes(b"binary")
    snapshots = []

    def fake_run(args, **kwargs):
        assert kwargs["check"] is False
        task = args[args.index("--include-task-name") + 1]
        job_dir = (
            Path(args[args.index("--jobs-dir") + 1])
            / args[args.index("--job-name") + 1]
        )
        overlay = Path(
            next(arg.split("=", 1)[1] for arg in args if arg.startswith("overlay_dir="))
        )
        description = (overlay / "read.txt").read_text()
        snapshots.append((task, description, args[args.index("--model") + 1]))
        trial_dir = job_dir / "trial-1" / "agent"
        trial_dir.mkdir(parents=True)
        (trial_dir / "trajectory.json").write_text(json.dumps({"steps": [description]}))
        (job_dir / "result.json").write_text(
            json.dumps(
                {
                    "trial_results": [
                        {
                            "task_name": task,
                            "trial_name": "trial-1",
                            "exception_info": None,
                            "verifier_result": {"rewards": {"reward": 1.0}},
                            "agent_result": {"cost_usd": 0.2},
                        }
                    ],
                }
            )
        )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(
        "agent_tool_opt_core.adapters.terminal.subprocess.run", fake_run
    )
    benchmark = TerminalBench2(
        benchmark_checkout=checkout,
        split_manifest=manifest,
        source_bundle=bundle,
        bun_linux_binary=bun,
        jobs_dir=tmp_path / "jobs",
    )
    agent = OpenCodeAgent("provider/model")
    baseline = collect_baseline(benchmark, agent, ["task-1"], target)
    candidate = Candidate({"read.txt": "Read only the requested file."})
    optimized = evaluate_candidate(benchmark, agent, target, candidate, ["task-1"])
    assert baseline.runs[0].trajectory == {"steps": ["Read a file."]}
    assert optimized.runs[0].trajectory == {"steps": ["Read only the requested file."]}
    assert baseline.runs[0].cost_usd == optimized.runs[0].cost_usd == 0.2
    assert snapshots == [
        ("task-1", "Read a file.", "provider/model"),
        ("task-1", "Read only the requested file.", "provider/model"),
    ]
    assert target.effective_toolset() == target.extract()


def test_harbor_missing_reward_is_not_scored_as_zero(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "result.json").write_text(
        json.dumps(
            {
                "trial_results": [
                    {
                        "task_name": "task-1",
                        "trial_name": "trial-1",
                        "verifier_result": {"rewards": {}},
                    }
                ]
            }
        )
    )
    with pytest.raises(RuntimeError, match="no numeric reward"):
        _harbor_results(job, ["task-1"], "terminal-bench@2.0", "model")


def test_terminal_split_manifest_and_both_dataset_classes(tmp_path):
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"train": ["a"], "test": ["b"]}))
    assert load_splits(manifest) == (["a"], ["b"])
    bundle = tmp_path / "bundle.tar"
    bundle.write_bytes(b"x")
    bun = tmp_path / "bun"
    bun.write_bytes(b"x")
    checkout = tmp_path / "tasks"
    for name in ("a", "b"):
        (checkout / name).mkdir(parents=True)
        (checkout / name / "task.toml").write_text("")
    kwargs = dict(
        benchmark_checkout=checkout,
        split_manifest=manifest,
        source_bundle=bundle,
        bun_linux_binary=bun,
        jobs_dir=tmp_path / "jobs",
    )
    assert TerminalBench2(**kwargs).name == "terminal-bench@2.0"
    assert OpenThoughtsTBLite(**kwargs).name == "openthoughts-tblite"
    manifest.write_text(json.dumps({"train": ["a"], "test": ["a"]}))
    with pytest.raises(ValueError, match="overlap"):
        load_splits(manifest)
