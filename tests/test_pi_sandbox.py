"""Boundary checks for the optional, provider-neutral Pi sandbox."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent_tool_opt_core.optimizers import pi_sandbox
from agent_tool_opt_core.optimizers._common import OptimizerInfrastructureFailure
from agent_tool_opt_core.optimizers.pi import PiOptimizer
from agent_tool_opt_core.api import Candidate, RunResult, TaskRun, ToolSet, Validator


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    executables = {}
    for name in ("pi", "node", "bwrap"):
        target = root / name
        target.write_text("fixture")
        target.chmod(0o700)
        executables[name] = str(target)
    monkeypatch.setattr(pi_sandbox.sys, "platform", "linux")
    monkeypatch.setattr(pi_sandbox, "_RUNTIME_DIRS", (str(root),))
    monkeypatch.setattr(pi_sandbox, "_SYSTEM_FILES", ())
    monkeypatch.setattr(
        pi_sandbox.shutil,
        "which",
        lambda name, **kwargs: executables.get(Path(name).name),
    )
    return executables


def _workspace(tmp_path):
    scratch = tmp_path / "scratch"
    workspace = scratch / "workspace"
    workspace.mkdir(parents=True)
    (scratch / ".pi-session").mkdir()
    (workspace / "tool.txt").write_text("baseline")
    (workspace / "context.txt").write_text("read-only context")
    return workspace


def _mounts(argv, flag):
    return [tuple(argv[i + 1 : i + 3]) for i, arg in enumerate(argv) if arg == flag]


def test_only_editable_files_and_sessions_are_writable(runtime, tmp_path):
    workspace = _workspace(tmp_path)
    argv, env = pi_sandbox.bubblewrap_command(
        ["pi", "--session-dir", "/sessions"], workspace, ("tool.txt",), {}, ()
    )
    assert argv[0] == runtime["bwrap"]
    assert (str(workspace), "/workspace") in _mounts(argv, "--ro-bind")
    assert _mounts(argv, "--bind") == [
        (str(workspace.parent / ".pi-session"), "/sessions"),
        (str(workspace / "tool.txt"), "/workspace/tool.txt"),
    ]
    assert "--unshare-user" in argv and "--unshare-all" in argv
    assert "--new-session" in argv and "--die-with-parent" in argv
    assert "--share-net" in argv  # Model calls need the network; no egress claim.
    assert ("/", "/") not in _mounts(argv, "--ro-bind")
    assert env["HOME"] == "/home/pi"
    assert env["PI_TELEMETRY"] == "0"


def test_credentials_are_explicit_and_never_command_arguments(runtime, tmp_path):
    argv, env = pi_sandbox.bubblewrap_command(
        ["pi"],
        _workspace(tmp_path),
        ("tool.txt",),
        {
            "OPENAI_API_KEY": "TEST_ONLY_SELECTED_KEY",
            "ANTHROPIC_API_KEY": "TEST_ONLY_UNSELECTED_KEY",
            "AWS_SECRET_ACCESS_KEY": "TEST_ONLY_HOST_SECRET",
            "NODE_OPTIONS": "--require=/host/plugin.js",
            "HOME": "/host/home",
        },
        ("OPENAI_API_KEY",),
    )
    assert env["OPENAI_API_KEY"] == "TEST_ONLY_SELECTED_KEY"
    assert "TEST_ONLY_SELECTED_KEY" not in " ".join(argv)
    assert (
        not {"ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY", "NODE_OPTIONS"} & env.keys()
    )
    assert "/host/home" not in env.values()


@pytest.mark.parametrize(
    "name", ["NODE_OPTIONS", "LD_PRELOAD", "PATH", "AWS_SECRET_ACCESS_KEY"]
)
def test_runtime_injection_and_unrelated_credentials_are_rejected(name):
    with pytest.raises(ValueError, match="unsupported"):
        pi_sandbox.validate_sandbox_options("bubblewrap", (name,))


def test_missing_selected_credential_fails_before_launch(runtime, tmp_path):
    with pytest.raises(ValueError, match="unset"):
        pi_sandbox.bubblewrap_command(
            ["pi"], _workspace(tmp_path), ("tool.txt",), {}, ("OPENAI_API_KEY",)
        )


@pytest.mark.parametrize("path", ["../outside", "/outside"])
def test_editable_paths_cannot_escape_workspace(runtime, tmp_path, path):
    with pytest.raises(ValueError, match="unsafe"):
        pi_sandbox.bubblewrap_command(["pi"], _workspace(tmp_path), (path,), {}, ())


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_editable_links_are_rejected(runtime, tmp_path, kind):
    workspace = _workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.write_text("sentinel")
    target = workspace / "linked.txt"
    if kind == "symlink":
        target.symlink_to(outside)
    else:
        os.link(outside, target)
    with pytest.raises(ValueError, match="ordinary"):
        pi_sandbox.bubblewrap_command(["pi"], workspace, ("linked.txt",), {}, ())


def test_session_directory_cannot_be_a_symlink(runtime, tmp_path):
    workspace = _workspace(tmp_path)
    sessions = workspace.parent / ".pi-session"
    sessions.rmdir()
    sessions.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="sibling session"):
        pi_sandbox.bubblewrap_command(["pi"], workspace, ("tool.txt",), {}, ())


def test_home_installed_pi_is_not_implicitly_exposed(runtime, tmp_path):
    target = tmp_path / "private-home-pi"
    target.write_text("fixture")
    runtime["pi"] = str(target)
    with pytest.raises(RuntimeError, match="home-directory installations"):
        pi_sandbox.bubblewrap_runtime("pi", {})


def test_non_linux_and_missing_bwrap_fail_closed(monkeypatch):
    monkeypatch.setattr(pi_sandbox.sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="requires Linux"):
        PiOptimizer(sandbox="bubblewrap")
    monkeypatch.setattr(pi_sandbox.sys, "platform", "linux")
    monkeypatch.setattr(pi_sandbox.shutil, "which", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="bwrap executable"):
        PiOptimizer(sandbox="bubblewrap")


def test_custom_wrapper_cannot_be_combined_with_builtin_sandbox():
    with pytest.raises(ValueError, match="not both"):
        PiOptimizer(sandbox="bubblewrap", sandbox_cmd=["wrapper"])


@pytest.mark.parametrize("optimizer,enabled", [("llm", True), ("pi", False)])
def test_sandbox_is_not_silently_ignored(optimizer, enabled):
    with pytest.raises(ValueError, match="enabled Pi optimizer"):
        pi_sandbox.pi_sandbox_kwargs(
            "bubblewrap", (), optimizer=optimizer, enabled=enabled
        )


@pytest.mark.parametrize("returncode", [1, 137])
def test_sandbox_process_failure_is_not_an_accepted_noop(
    runtime, monkeypatch, tmp_path, returncode
):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            returncode,
            stdout=json.dumps({"type": "session", "id": "test"}),
            stderr="synthetic startup failure",
        )

    monkeypatch.setattr(subprocess, "run", run)
    tools = ToolSet({"tool.txt": "baseline"}, ("tool.txt",), "Edit tool.txt")
    optimizer = PiOptimizer(sandbox="bubblewrap", max_retries=1)
    with pytest.raises(
        OptimizerInfrastructureFailure, match="sandboxed process failed"
    ):
        optimizer.propose(
            tools,
            RunResult("b", "a", (TaskRun("task", 0.0),)),
            Validator("txt", tools.allowlist),
            tmp_path / "scratch",
        )
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[0] == runtime["bwrap"]
    assert "--no-extensions" in argv
    assert argv[argv.index("--session-dir") + 1] == "/sessions"
    assert kwargs["env"]["HOME"] == "/home/pi"


def test_sandboxed_pi_edit_returns_a_candidate(runtime, monkeypatch, tmp_path):
    def run(argv, **kwargs):
        (Path(kwargs["cwd"]) / "tool.txt").write_text("edited")
        return subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps({"type": "session", "id": "test"}), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", run)
    tools = ToolSet({"tool.txt": "baseline"}, ("tool.txt",), "Edit tool.txt")
    candidate = PiOptimizer(sandbox="bubblewrap", max_retries=1).propose(
        tools,
        RunResult("b", "a", (TaskRun("task", 0.0),)),
        Validator("txt", tools.allowlist),
        tmp_path / "scratch",
    )
    assert candidate == Candidate({"tool.txt": "edited"})


@pytest.mark.skipif(
    sys.platform != "linux"
    or shutil.which("bwrap") is None
    or shutil.which("node", path="/usr/local/bin:/usr/bin:/bin") is None,
    reason="requires Linux, Bubblewrap, and a system Node.js installation",
)
def test_real_bubblewrap_confinement(tmp_path, monkeypatch):
    """Exercise the kernel boundary using synthetic files and no model calls."""
    workspace = _workspace(tmp_path)
    outside = tmp_path / "outside-secret"
    outside.write_text("TEST_ONLY_HOST_FILE")
    monkeypatch.setenv("ATO_HOST_SECRET", "TEST_ONLY_HOST_ENV")
    script = """
import os
import sys
from pathlib import Path
assert not Path(sys.argv[1]).exists()
assert 'ATO_HOST_SECRET' not in os.environ
assert os.environ['HOME'] == '/home/pi'
assert Path('/workspace/context.txt').read_text() == 'read-only context'
for target in ['/workspace/context.txt', '/workspace/new-file']:
    try:
        Path(target).write_text('forbidden')
    except OSError:
        pass
    else:
        raise AssertionError('read-only workspace was writable')
Path('/workspace/tool.txt').write_text('edited')
Path('/sessions/session.json').write_text('{}')
assert not Path('/var/run/docker.sock').exists()
print('confined')
"""
    argv, env = pi_sandbox.bubblewrap_command(
        ["/usr/bin/python3", "-c", script, str(outside)],
        workspace,
        ("tool.txt",),
        os.environ,
        (),
    )
    result = subprocess.run(
        argv, env=env, capture_output=True, text=True, timeout=20, check=True
    )
    assert result.stdout.strip() == "confined"
    assert (workspace / "tool.txt").read_text() == "edited"
    assert (workspace.parent / ".pi-session" / "session.json").is_file()
    assert outside.read_text() == "TEST_ONLY_HOST_FILE"
