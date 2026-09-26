"""Offline contract tests for the public benchmark adapters."""

from __future__ import annotations

import json
import io
import sys
import tarfile
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_tool_opt_core.adapters.tau2 import (
    Tau2Agent,
    Tau2Benchmark,
    Tau2CodeValidator,
    Tau2DescriptionValidator,
    Tau2ToolTarget,
    _schema_import_check,
)
from agent_tool_opt_core.adapters.tau2_descriptions import (
    parse_descriptions,
    render_descriptions,
    splice_docstrings,
)
from agent_tool_opt_core.adapters.terminal import (
    OpenCodeAgent,
    OpenCodeCodeValidator,
    OpenCodeToolTarget,
    OpenThoughtsTBLite,
    TerminalBench2,
    _harbor_results,
    load_splits,
    parse_splits,
)
from agent_tool_opt_core.adapters.opencode_bundle import check_bundle
from agent_tool_opt_core.api import Candidate, ValidationResult
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


def test_tau2_description_markdown_round_trip_and_heading_gate():
    original = render_descriptions(_TOOLS)
    revised = original.replace("Look up a reservation by ID.", "Look up café IDs.")
    docs = parse_descriptions(revised, {"lookup"})
    source = splice_docstrings(_TOOLS, docs)
    assert "Look up café IDs." in source
    assert (
        source.replace(repr("Look up café IDs."), '"""Look up a reservation by ID."""')
        == _TOOLS
    )
    assert "return reservation_id" in source
    with pytest.raises(ValueError, match="duplicate"):
        parse_descriptions(revised + "\n## lookup\n    duplicate\n", {"lookup"})
    with pytest.raises(ValueError, match="indented"):
        parse_descriptions("## lookup\nnot indented", {"lookup"})


def test_tau2_schema_check_uses_subprocess_without_api_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-for-candidate")

    def fake_run(args, **kwargs):
        compile(args[2], "schema-check", "exec")
        assert "OPENAI_API_KEY" not in kwargs["env"]
        assert (kwargs["cwd"] / "tools.py").read_text() == _TOOLS
        assert (kwargs["cwd"] / "baseline_tools.py").read_text() == _TOOLS
        return SimpleNamespace(returncode=0, stdout='ATOCHECK:["doc changed"]\n')

    monkeypatch.setattr("agent_tool_opt_core.adapters.tau2.subprocess.run", fake_run)
    result = _schema_import_check(_TOOLS, _TOOLS)
    assert result == ValidationResult(True, "doc changed")


def test_tau2_description_surface_and_validator(monkeypatch):
    checked = []
    monkeypatch.setattr(
        "agent_tool_opt_core.adapters.tau2._schema_import_check",
        lambda source, baseline=None: (
            checked.append((source, baseline))
            or ValidationResult(True, "schemas constructed")
        ),
    )
    validator = Tau2DescriptionValidator("py", ("descriptions.md",), _TOOLS)
    original = render_descriptions(_TOOLS)
    good = original.replace(
        "Look up a reservation by ID.", "Use the exact reservation ID."
    )
    assert validator.validate(Candidate({"descriptions.md": good})).ok
    assert checked[0][1] == _TOOLS
    assert "Use the exact reservation ID." in checked[0][0]
    assert "return reservation_id" in checked[0][0]
    assert not validator.validate(Candidate({"tools.py": _TOOLS})).ok
    assert not validator.validate(
        Candidate({"descriptions.md": good.replace("## lookup", "## wrong")})
    ).ok


def test_tau2_code_mode_preserves_tool_contract_but_allows_body_edit(monkeypatch):
    monkeypatch.setattr(
        "agent_tool_opt_core.adapters.tau2._schema_import_check",
        lambda source, baseline=None: ValidationResult(True, "schemas constructed"),
    )
    validator = Tau2CodeValidator("py", ("tools.py",), _TOOLS, "AirlineTools")
    body_edit = _TOOLS.replace("return reservation_id", "return reservation_id.upper()")
    assert validator.validate(Candidate({"tools.py": body_edit})).ok
    assert not validator.validate(
        Candidate({"tools.py": body_edit.replace("reservation_id: str", "id: str")})
    ).ok


def test_tau2_tool_target_swaps_and_restores_class(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "agent_tool_opt_core.adapters.tau2._schema_import_check",
        lambda source, baseline=None: ValidationResult(True, "schemas constructed"),
    )
    source_file = tmp_path / "tools.py"
    source_file.write_text(_TOOLS)
    tools_module = types.ModuleType("tau2.domains.airline.tools")
    tools_module.__file__ = str(source_file)
    environment = types.ModuleType("tau2.domains.airline.environment")
    original = type("AirlineTools", (), {})
    environment.AirlineTools = original
    monkeypatch.setitem(sys.modules, tools_module.__name__, tools_module)
    monkeypatch.setitem(sys.modules, environment.__name__, environment)
    data_dir = tmp_path / "data"
    policy = data_dir / "tau2" / "domains" / "airline" / "policy.md"
    policy.parent.mkdir(parents=True)
    policy.write_text("Airline policy")
    utils = types.ModuleType("tau2.utils.utils")
    utils.DATA_DIR = data_dir
    monkeypatch.setitem(sys.modules, utils.__name__, utils)

    target = Tau2ToolTarget("airline")
    assert set(target.extract().files) == {"descriptions.md"}
    assert target.extract().context["tools.py"] == _TOOLS
    assert target.extract().context["policy.md"] == "Airline policy"
    edited = render_descriptions(_TOOLS).replace(
        "Look up a reservation by ID.", "Use the exact ID."
    )
    target.apply(Candidate({"descriptions.md": edited}))
    assert environment.AirlineTools is not original
    assert environment.AirlineTools.lookup.__annotations__["reservation_id"] is str
    assert environment.AirlineTools().lookup("ABC") == "ABC"
    assert "Use the exact ID" in target.effective_toolset().files["descriptions.md"]
    target.restore()
    assert environment.AirlineTools is original
    assert target.effective_toolset() == target.extract()

    code_target = Tau2ToolTarget("airline", descriptions_only=False)
    code_target.apply(
        Candidate(
            {
                "tools.py": _TOOLS.replace(
                    "return reservation_id", "return reservation_id.upper()"
                )
            }
        )
    )
    assert environment.AirlineTools().lookup("abc") == "ABC"
    code_target.restore()
    assert environment.AirlineTools is original


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
    assert calls["num_trials"] == 1
    repeated = Tau2Benchmark(
        "airline", train_tasks=["t1"], test_tasks=["t2"], num_trials=2
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        repeated.evaluate(
            Tau2Agent("agent-model", user_model="user-model"), ["t1"], None
        )


def _opencode_checkout(tmp_path: Path) -> Path:
    root = tmp_path / "opencode"
    tool_dir = root / "packages" / "opencode" / "src" / "tool"
    (tool_dir / "shell").mkdir(parents=True)
    (root / "package.json").write_text("{}")
    (root / "bun.lock").write_text("lock")
    (root / "packages" / "opencode" / "package.json").write_text("{}")
    (root / "packages" / "opencode" / "src" / "index.ts").write_text("// entry")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "dependency.txt").write_text("installed")
    (tool_dir / "read.ts").write_text(
        'import DESCRIPTION from "./read.txt"\nexport const ReadTool = Tool.define("read", {})'
    )
    (tool_dir / "read.txt").write_text("Read a file.")
    (tool_dir / "registry.ts").write_text('import { ReadTool } from "./read"')
    (tool_dir / "shell" / "prompt.ts").write_text(
        'import DESCRIPTION from "./shell.txt"'
    )
    (tool_dir / "shell" / "shell.txt").write_text("Run a shell command.")
    return root


def _source_bundle(checkout: Path, path: Path) -> Path:
    with tarfile.open(path, "w") as archive:
        for source in sorted(checkout.rglob("*")):
            if source.is_file():
                archive.add(source, arcname=source.relative_to(checkout).as_posix())
    return path


def test_opencode_target_discovers_nested_imported_descriptions(tmp_path):
    target = OpenCodeToolTarget(_opencode_checkout(tmp_path))
    assert set(target.extract().files) == {"read.txt", "shell/shell.txt"}
    target.apply(Candidate({"shell/shell.txt": "Run one command."}))
    assert target.effective_toolset().files["shell/shell.txt"] == "Run one command."
    target.restore()
    assert target.effective_toolset().files["shell/shell.txt"] == "Run a shell command."
    with pytest.raises(ValueError):
        target.apply(Candidate({"read.ts": "bad"}))


def test_opencode_code_mode_exposes_active_ts_and_checks_bun(monkeypatch, tmp_path):
    target = OpenCodeToolTarget(_opencode_checkout(tmp_path), descriptions_only=False)
    assert "read.ts" in target.extract().allowlist
    assert "registry.ts" not in target.extract().allowlist
    source = target.extract().files["read.ts"]
    candidate = Candidate({"read.ts": source.replace('"read"', '"lookup"')})
    monkeypatch.setattr(
        "agent_tool_opt_core.adapters.terminal.shutil.which", lambda _: None
    )
    assert not target.validator().validate(candidate).ok

    calls = []
    monkeypatch.setattr(
        "agent_tool_opt_core.adapters.terminal.shutil.which", lambda _: "/bin/bun"
    )
    monkeypatch.setattr(
        "agent_tool_opt_core.adapters.terminal.subprocess.run",
        lambda args, **kwargs: calls.append(args) or SimpleNamespace(returncode=0),
    )
    assert isinstance(target.validator(), OpenCodeCodeValidator)
    target.apply(candidate)
    assert "lookup" in target.effective_toolset().files["read.ts"]
    assert calls[0][:2] == ["/bin/bun", "build"]
    target.restore()
    assert target.effective_toolset() == target.extract()


def test_harbor_baseline_and_candidate_use_same_task_and_overlay(monkeypatch, tmp_path):
    source = _opencode_checkout(tmp_path)
    target = OpenCodeToolTarget(source)
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"train": ["task-1"], "test": ["task-2"]}))
    checkout = tmp_path / "terminal-bench-2"
    for name in ("task-1", "task-2"):
        (checkout / name).mkdir(parents=True)
        (checkout / name / "task.toml").write_text("")
    bundle = _source_bundle(source, tmp_path / "source.tar")
    bun = tmp_path / "bun"
    bun.write_bytes(b"binary")
    snapshots = []
    concurrency = []

    def fake_run(args, **kwargs):
        assert kwargs["check"] is False
        assert args[:4] == [sys.executable, "-m", "harbor.cli.main", "run"]
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
        concurrency.append(int(args[args.index("--n-concurrent") + 1]))
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
        opencode_checkout=source,
        split_manifest=manifest,
        source_bundle=bundle,
        bun_linux_binary=bun,
        jobs_dir=tmp_path / "jobs",
        n_concurrent=5,
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
    assert concurrency == [5, 5]
    benchmark.evaluate_parallel(agent, ["task-1"], target.extract())
    assert concurrency[-1] == 2
    assert target.effective_toolset() == target.extract()


def test_harbor_failure_reports_type_without_stderr_secrets(monkeypatch, tmp_path):
    source = _opencode_checkout(tmp_path)
    bundle = _source_bundle(source, tmp_path / "source.tar")
    tasks = tmp_path / "tasks"
    for name in ("a", "b"):
        (tasks / name).mkdir(parents=True)
        (tasks / name / "task.toml").write_text("")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["a"], "test": ["b"]}))
    bun = tmp_path / "bun"
    bun.write_bytes(b"binary")
    benchmark = TerminalBench2(
        benchmark_checkout=tasks,
        opencode_checkout=source,
        split_manifest=split,
        source_bundle=bundle,
        bun_linux_binary=bun,
        jobs_dir=tmp_path / "jobs",
    )
    monkeypatch.setattr(
        "agent_tool_opt_core.adapters.terminal.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stderr="AuthenticationError: api_key=secret123"
        ),
    )
    with pytest.raises(RuntimeError) as error:
        benchmark.evaluate(
            OpenCodeAgent("provider/model"), ["a"], OpenCodeToolTarget(source).extract()
        )
    assert "AuthenticationError" in str(error.value)
    assert "secret123" not in str(error.value)


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


def test_harbor_trial_error_reports_only_safe_type(tmp_path):
    job = tmp_path / "job"
    trial = job / "trial-1"
    trial.mkdir(parents=True)
    (job / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": 1,
                "finished_at": "2026-09-25T20:00:00Z",
                "stats": {"n_completed_trials": 1, "n_errored_trials": 1},
            }
        )
    )
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-1",
                "trial_name": "trial-1",
                "exception_info": {
                    "exception_type": "AgentAuthenticationError",
                    "exception_message": "api_key=secret123",
                },
            }
        )
    )
    with pytest.raises(RuntimeError) as error:
        _harbor_results(job, ["task-1"], "terminal-bench@2.0", "model")
    assert "AgentAuthenticationError" in str(error.value)
    assert "secret123" not in str(error.value)


def test_harbor_restores_instruction_from_trial_result(tmp_path):
    job = tmp_path / "job"
    trial = job / "trial-1"
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "trajectory.json").write_text('{"steps": []}')
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "task-1",
                "trial_name": "trial-1",
                "instruction": "Fix the build",
                "verifier_result": {"rewards": {"reward": 0}},
            }
        )
    )
    (job / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": 1,
                "finished_at": "2026-09-25T20:00:00Z",
                "stats": {"n_completed_trials": 1, "n_errored_trials": 0},
            }
        )
    )
    run = _harbor_results(job, ["task-1"], "terminal-bench@2.0", "model")
    assert run.runs[0].trajectory == {
        "instruction": "Fix the build",
        "steps": [],
    }


def test_harbor_repeated_trials_aggregate_reward_transcripts_and_cost(tmp_path):
    job = tmp_path / "job"
    trials = []
    for index, reward in enumerate((0.0, 1.0), start=1):
        name = f"trial-{index}"
        trajectory = job / name / "agent"
        trajectory.mkdir(parents=True)
        (trajectory / "trajectory.json").write_text(json.dumps({"attempt": index}))
        trials.append(
            {
                "task_name": "task-1",
                "trial_name": name,
                "verifier_result": {"rewards": {"reward": reward}},
                "agent_result": {"cost_usd": 0.1 * index},
            }
        )
    (job / "result.json").write_text(json.dumps({"trial_results": trials}))
    run = _harbor_results(job, ["task-1"], "terminal-bench@2.0", "model", 2)
    assert run.runs[0].reward == 0.5
    assert run.runs[0].trajectory == [{"attempt": 1}, {"attempt": 2}]
    assert run.runs[0].cost_usd == pytest.approx(0.3)
    with pytest.raises(RuntimeError, match="incomplete"):
        _harbor_results(job, ["task-1"], "terminal-bench@2.0", "model", 3)


def test_terminal_split_manifest_and_both_dataset_classes(tmp_path):
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"train": ["a"], "test": ["b"]}))
    assert load_splits(manifest) == (["a"], ["b"])
    assert parse_splits({"train": ["a"], "test": ["b"]}) == (["a"], ["b"])
    opencode = _opencode_checkout(tmp_path)
    bundle = _source_bundle(opencode, tmp_path / "bundle.tar")
    bun = tmp_path / "bun"
    bun.write_bytes(b"x")
    checkout = tmp_path / "tasks"
    for name in ("a", "b"):
        (checkout / name).mkdir(parents=True)
        (checkout / name / "task.toml").write_text("")
    kwargs = dict(
        benchmark_checkout=checkout,
        opencode_checkout=opencode,
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


def test_opencode_bundle_must_match_local_tool_checkout(tmp_path):
    checkout = _opencode_checkout(tmp_path)
    bundle = _source_bundle(checkout, tmp_path / "source.tar")
    check_bundle(bundle, checkout)
    (checkout / "packages" / "opencode" / "src" / "tool" / "read.txt").write_text(
        "changed after bundling"
    )
    with pytest.raises(ValueError, match="differs"):
        check_bundle(bundle, checkout)
    tasks = tmp_path / "tasks"
    for name in ("a", "b"):
        (tasks / name).mkdir(parents=True)
        (tasks / name / "task.toml").write_text("")
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"train": ["a"], "test": ["b"]}))
    bun = tmp_path / "bun"
    bun.write_bytes(b"binary")
    with pytest.raises(ValueError, match="differs"):
        TerminalBench2(
            benchmark_checkout=tasks,
            opencode_checkout=checkout,
            split_manifest=manifest,
            source_bundle=bundle,
            bun_linux_binary=bun,
            jobs_dir=tmp_path / "jobs",
        )


def test_opencode_bundle_rejects_escaping_member(tmp_path):
    bundle = tmp_path / "malicious.tar"
    with tarfile.open(bundle, "w") as archive:
        payload = b"bad"
        info = tarfile.TarInfo("../escape")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe"):
        check_bundle(bundle)


def test_opencode_bundle_accepts_in_tree_workspace_links(tmp_path):
    checkout = _opencode_checkout(tmp_path)
    bundle = _source_bundle(checkout, tmp_path / "source.tar")
    with tarfile.open(bundle, "a") as archive:
        first = tarfile.TarInfo("node_modules/alias")
        first.type = tarfile.SYMTYPE
        first.linkname = "dependency.txt"
        archive.addfile(first)
        second = tarfile.TarInfo("node_modules/.bin/command")
        second.type = tarfile.SYMTYPE
        second.linkname = "../alias"
        archive.addfile(second)
    check_bundle(bundle, checkout)


def test_opencode_bundle_rejects_escaping_workspace_link(tmp_path):
    checkout = _opencode_checkout(tmp_path)
    bundle = _source_bundle(checkout, tmp_path / "source.tar")
    with tarfile.open(bundle, "a") as archive:
        link = tarfile.TarInfo("node_modules/escape")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        archive.addfile(link)
    with pytest.raises(ValueError, match="escaping"):
        check_bundle(bundle, checkout)
