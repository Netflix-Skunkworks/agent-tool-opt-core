"""Tests for the Pi optimizer on api.Optimizer (subprocess + which mocked; no real pi)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent_tool_opt_core.api import (
    RunResult,
    TaskRun,
    ToolSet,
    ValidationResult,
    Validator,
)
from agent_tool_opt_core.optimizers._common import (
    OptimizerInfrastructureFailure,
    OptimizerLLMFailure,
    transcript_workspace_files,
)
from agent_tool_opt_core.optimizers.pi import (
    _MAX_ARGV_BYTES,
    _TRANSCRIPTS_PROMPT,
    PiOptimizer,
    _is_transient_upstream_failure,
    _session_never_started,
    parse_session_metrics,
)
from agent_tool_opt_core.costs import collect_optimizer_costs

# Linux MAX_ARG_STRLEN: the hard per-argv-element ceiling execve enforces.
_MAX_ARG_STRLEN = 32 * 4096


def _toolset():
    return ToolSet({"tools.py": "def t(): pass\n"}, ("tools.py",), "keep importable")


def _toolset_ctx():
    """A toolset with read-only context (ToolTarget-supplied + a transcript)."""
    return ToolSet(
        {"tools.py": "def t(): pass\n"},
        ("tools.py",),
        "keep importable",
        context={"policy.md": "be careful\n", "transcripts/failed/x.json": "{}\n"},
    )


def _run():
    return RunResult("b", "gpt", (TaskRun("t1", 0.0, trajectory="failed"),))


def _which(mocker, present=True):
    mocker.patch(
        "agent_tool_opt_core.optimizers.pi.shutil.which",
        return_value="/usr/local/bin/pi" if present else None,
    )


def _session(sid="s1"):
    return json.dumps({"type": "session", "id": sid})


def _billed(amount, identity="message-1"):
    return json.dumps(
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "id": identity,
                "usage": {"input": 1, "output": 1, "cost": {"total": amount}},
            },
        }
    )


@pytest.mark.parametrize("value", [None, -1, "broken", float("nan"), float("inf")])
def test_pi_missing_cost_is_not_zero(value):
    metrics = parse_session_metrics(
        _session() + "\n" + _billed(2) + "\n" + _billed(value, "m2")
    )
    assert metrics["cost_usd"] is None
    assert metrics["known_cost_usd"] == 2
    assert metrics["unknown_cost_count"] == 1


def test_pi_counts_replayed_messages_once_and_explicit_zero_as_known():
    events = "\n".join(
        [_session(), _billed(2), _session(), _billed(2), _billed(0, "m2")]
    )
    metrics = parse_session_metrics(events)
    assert metrics["cost_usd"] == 2
    assert metrics["assistant_message_count"] == 2


def test_pi_retry_cost_and_final_session_identity(mocker, tmp_path):
    _which(mocker)
    mocker.patch("agent_tool_opt_core.optimizers.pi.time.sleep")
    outputs = iter(
        [
            _session("old")
            + "\n"
            + _billed(1)
            + '\n{"type":"auto_retry_end","success":false,"error":"Overloaded 529"}',
            _session("new") + "\n" + _billed(2),
            _session("new") + "\n" + _billed(2) + "\n" + _billed(3, "m2"),
        ]
    )
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd)
        # Force a validation rethink after the transient session retry.
        (Path(kwargs["cwd"]) / "tools.py").write_text("def t(): return 1")
        return subprocess.CompletedProcess(cmd, 0, stdout=next(outputs), stderr="")

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=run)
    validator = mocker.Mock()
    validator.validate.side_effect = [
        ValidationResult(False, "retry"),
        ValidationResult(True),
    ]
    with collect_optimizer_costs() as costs:
        PiOptimizer(max_retries=2).propose(_toolset(), _run(), validator, tmp_path)
    assert _arg(commands[2], "--session") == "new"
    assert costs.llm.cost_usd == 6
    metrics = json.loads((tmp_path / "artifacts" / "metrics.json").read_text())
    assert metrics["cost_usd"] == 6
    assert (
        len(
            (tmp_path / "artifacts" / "session_attempts.jsonl").read_text().splitlines()
        )
        == 3
    )


def test_pi_timeout_keeps_known_spend_and_marks_unfinished_call_unknown(
    mocker, tmp_path
):
    _which(mocker)
    output = _session() + "\n" + _billed(2)
    mocker.patch(
        "agent_tool_opt_core.optimizers.pi.subprocess.run",
        side_effect=subprocess.TimeoutExpired("pi", 1, output=output),
    )
    with (
        collect_optimizer_costs() as costs,
        pytest.raises(OptimizerInfrastructureFailure),
    ):
        PiOptimizer(max_retries=1).propose(
            _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
        )
    assert costs.llm.cost_usd is None
    assert costs.llm.known_usd == 2


def test_pi_validation_exception_cannot_lose_billed_usage(mocker, tmp_path):
    _which(mocker)
    mocker.patch(
        "agent_tool_opt_core.optimizers.pi.subprocess.run",
        return_value=subprocess.CompletedProcess(
            [], 0, stdout=_session() + "\n" + _billed(2), stderr=""
        ),
    )
    validator = mocker.Mock()
    validator.validate.side_effect = RuntimeError("validator crashed")
    with (
        collect_optimizer_costs() as costs,
        pytest.raises(RuntimeError, match="validator crashed"),
    ):
        PiOptimizer(max_retries=1).propose(_toolset(), _run(), validator, tmp_path)
    assert costs.llm.cost_usd == 2


def test_unfinished_completion_in_earlier_session_remains_unknown():
    events = "\n".join(
        [
            _session("old"),
            '{"type":"message_start","message":{"role":"assistant"}}',
            _session("new"),
            _billed(2),
        ]
    )
    metrics = parse_session_metrics(events)
    assert metrics["cost_usd"] is None
    assert metrics["known_cost_usd"] == 2


def test_is_transient_upstream_failure():
    overloaded = '{"type":"auto_retry_end","success":false,"error":"Overloaded 529"}'
    assert _is_transient_upstream_failure(overloaded)
    assert not _is_transient_upstream_failure(_session())  # clean session
    # transient markers but a tool ran -> not a pure transient failure
    assert not _is_transient_upstream_failure(
        overloaded + '\n{"type":"tool_execution_end","toolName":"edit"}'
    )


def test_transient_upstream_exhaustion_is_infrastructure_failure(mocker, tmp_path):
    _which(mocker)
    overloaded = (
        _session()
        + '\n{"type":"auto_retry_end","success":false,"error":"Overloaded 529"}'
    )
    run = mocker.patch(
        "agent_tool_opt_core.optimizers.pi.subprocess.run",
        return_value=subprocess.CompletedProcess([], 1, stdout=overloaded, stderr=""),
    )
    mocker.patch("agent_tool_opt_core.optimizers.pi.time.sleep")

    with pytest.raises(OptimizerInfrastructureFailure, match="persisted"):
        PiOptimizer(max_retries=1).propose(
            _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
        )

    assert run.call_count == 6


def test_init_requires_binary(mocker):
    _which(mocker, present=False)
    with pytest.raises(RuntimeError, match="not found"):
        PiOptimizer()


def test_happy_path_returns_workspace_diff(mocker, tmp_path):
    _which(mocker)

    def side(cmd, **kw):
        (Path(kw["cwd"]) / "tools.py").write_text("def t():\n    return 1\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=_session(), stderr="")

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    cand = PiOptimizer(max_retries=2).propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
    )
    assert cand.files == {"tools.py": "def t():\n    return 1\n"}


def test_transcripts_are_full_workspace_files_not_initial_prompt(mocker, tmp_path):
    _which(mocker)
    seen = {}
    run = RunResult("b", "gpt", (TaskRun("unsafe/../id", 0.0, {"log": "x" * 20_000}),))

    def side(cmd, **kw):
        ws = Path(kw["cwd"])
        seen["prompt"] = _arg(cmd, "-p")
        seen["index"] = (ws / "BASELINE_TRANSCRIPTS_INDEX.md").read_text()
        seen["transcript"] = (ws / "baseline_transcripts" / "0000.json").read_text()
        (ws / "tools.py").write_text("def t():\n    return 1\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=_session(), stderr="")

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    PiOptimizer(max_retries=1).propose(
        _toolset(), run, Validator("py", ("tools.py",)), tmp_path
    )
    assert "x" * 100 not in seen["prompt"]
    assert "BASELINE_TRANSCRIPTS_INDEX.md" in seen["prompt"]
    assert "unsafe/../id" in seen["index"]
    assert "x" * 20_000 in seen["transcript"]


def test_executable_pi_smoke_reads_index_and_full_transcript(tmp_path):
    """Cross the real subprocess boundary without making a model call."""
    binary = tmp_path / "pi-filesystem-smoke"
    binary.write_text(
        """#!/usr/bin/env python3
import json
from pathlib import Path

index = Path("BASELINE_TRANSCRIPTS_INDEX.md").read_text()
transcript = Path("baseline_transcripts/0000.txt").read_text()
assert "task `t1`" in index
assert transcript == "FULL-TRANSCRIPT-SENTINEL"
Path("tools.py").write_text("def t():\\n    return 7\\n")
print(json.dumps({"type": "session", "id": "filesystem-smoke"}))
print(json.dumps({"type": "tool_execution_end", "toolName": "read"}))
"""
    )
    binary.chmod(0o755)
    run = RunResult("b", "gpt", (TaskRun("t1", 0.0, "FULL-TRANSCRIPT-SENTINEL"),))
    cand = PiOptimizer(binary=str(binary), provider=None, max_retries=1).propose(
        _toolset(), run, Validator("py", ("tools.py",)), tmp_path / "scratch"
    )
    assert cand.files == {"tools.py": "def t():\n    return 7\n"}
    metrics = json.loads(
        (tmp_path / "scratch" / "artifacts" / "metrics.json").read_text()
    )
    assert metrics["tool_calls"] == {"read": 1}


def test_change_gate_retries_on_offlimits_file(mocker, tmp_path):
    _which(mocker)
    calls = {"i": 0}

    def side(cmd, **kw):
        ws = Path(kw["cwd"])
        calls["i"] += 1
        if calls["i"] == 1:
            (ws / "stowaway.txt").write_text("x")  # new off-allowlist file -> gate fail
        else:
            (ws / "stowaway.txt").unlink(missing_ok=True)  # agent fixes it on retry
            (ws / "tools.py").write_text("def t():\n    return 2\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=_session(), stderr="")

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    cand = PiOptimizer(max_retries=3).propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
    )
    assert calls["i"] == 2  # retried once
    assert cand.files == {"tools.py": "def t():\n    return 2\n"}


def test_validate_failure_retries(mocker, tmp_path):
    _which(mocker)
    calls = {"i": 0}

    def side(cmd, **kw):
        ws = Path(kw["cwd"])
        calls["i"] += 1
        # attempt 1 leaves tools.py empty (language gate fails), then fixes it
        ws.joinpath("tools.py").write_text("   " if calls["i"] == 1 else "ok = 1\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=_session(), stderr="")

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    cand = PiOptimizer(max_retries=3).propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
    )
    assert calls["i"] == 2
    assert cand.files == {"tools.py": "ok = 1\n"}


def test_read_only_context_is_materialized_but_excluded(mocker, tmp_path):
    _which(mocker)
    seen = {}

    def side(cmd, **kw):
        ws = Path(kw["cwd"])
        seen["policy"] = (ws / "policy.md").read_text()  # agent can READ context
        seen["tr"] = (ws / "transcripts" / "failed" / "x.json").exists()
        (ws / "tools.py").write_text("def t():\n    return 1\n")  # edits allowlist only
        return subprocess.CompletedProcess(cmd, 0, stdout=_session(), stderr="")

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    cand = PiOptimizer(max_retries=1).propose(
        _toolset_ctx(), _run(), Validator("py", ("tools.py",)), tmp_path
    )
    assert seen["policy"] == "be careful\n" and seen["tr"] is True  # readable
    assert cand.files == {"tools.py": "def t():\n    return 1\n"}  # context excluded


def test_editing_read_only_context_fails_the_gate(mocker, tmp_path):
    _which(mocker)

    def side(cmd, **kw):
        (Path(kw["cwd"]) / "policy.md").write_text("HACKED\n")  # off-allowlist edit
        return subprocess.CompletedProcess(cmd, 0, stdout=_session(), stderr="")

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    # Rejected, and NOT returned as an empty candidate: that would be scored as
    # the baseline (a tie) rather than the loss it is.
    with pytest.raises(OptimizerLLMFailure, match="change_gate_ok=False"):
        PiOptimizer(max_retries=1).propose(
            _toolset_ctx(), _run(), Validator("py", ("tools.py",)), tmp_path
        )
    decision = (tmp_path / "artifacts" / "decision.txt").read_text()
    assert "no valid candidate after 1 attempts" in decision  # persisted first


def test_exhausted_validation_retries_raise(mocker, tmp_path):
    """Every attempt fails the language gate -> a loss, not an invalid candidate."""
    _which(mocker)
    calls = {"i": 0}

    def side(cmd, **kw):
        calls["i"] += 1
        # never importable, on every attempt
        (Path(kw["cwd"]) / "tools.py").write_text("def t(:\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=_session(), stderr="")

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    validator = mocker.Mock(spec=Validator)
    validator.validate.return_value = ValidationResult(False, "SyntaxError")
    with pytest.raises(OptimizerLLMFailure, match="validate_ok=False"):
        PiOptimizer(max_retries=3).propose(_toolset(), _run(), validator, tmp_path)
    assert calls["i"] == 3  # exhausted the retries before raising


def test_require_validation_false_still_gates_the_change_gate(mocker, tmp_path):
    """``require_validation=False`` ablates the LANGUAGE gate only."""
    _which(mocker)

    def side(cmd, **kw):
        (Path(kw["cwd"]) / "stowaway.txt").write_text("x")  # off-allowlist addition
        return subprocess.CompletedProcess(cmd, 0, stdout=_session(), stderr="")

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    with pytest.raises(OptimizerLLMFailure, match="change_gate_ok=False"):
        PiOptimizer(max_retries=1, require_validation=False).propose(
            _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
        )


def test_context_builder_output_is_read_only(mocker, tmp_path):
    _which(mocker)

    def side(cmd, **kw):
        ws = Path(kw["cwd"])
        # run-derived context (here a fake builder) is materialized + readable
        assert (ws / "TOOLS_RISK.md").read_text() == "risk\n"
        (ws / "tools.py").write_text("def t():\n    return 1\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=_session(), stderr="")

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    opt = PiOptimizer(
        max_retries=1, context_builder=lambda run: {"TOOLS_RISK.md": "risk\n"}
    )
    cand = opt.propose(_toolset(), _run(), Validator("py", ("tools.py",)), tmp_path)
    assert cand.files == {
        "tools.py": "def t():\n    return 1\n"
    }  # builder output excluded


def test_artifacts_and_metrics_persisted(mocker, tmp_path):
    _which(mocker)

    def side(cmd, **kw):
        (Path(kw["cwd"]) / "tools.py").write_text("def t():\n    return 1\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=_session(), stderr="")

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    PiOptimizer(max_retries=1).propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
    )
    art = tmp_path / "artifacts"
    assert json.loads((art / "metrics.json").read_text())["session_id"] == "s1"
    assert (art / "diff.patch").read_text().strip()  # non-empty diff
    assert "accepted" in (art / "decision.txt").read_text()


def test_parse_session_metrics():
    jsonl = "\n".join([
        json.dumps({"type": "session", "id": "s1"}),
        json.dumps({"type": "turn_start"}),
        json.dumps({"type": "tool_execution_end", "toolName": "edit"}),
        json.dumps({"type": "tool_execution_end", "toolName": "read", "isError": True}),
        json.dumps({"type": "message_end", "message": {
            "role": "assistant",
            "usage": {"input": 10, "output": 5, "totalTokens": 15, "cost": {"total": 0.01}},
            "stopReason": "end_turn", "model": "m",
        }}),
        "this is not json",
    ])  # fmt: skip
    m = parse_session_metrics(jsonl)
    assert m["total_tokens"] == 15 and m["input_tokens"] == 10
    assert m["tool_calls"] == {"edit": 1, "read": 1}
    assert m["tool_errors"] == {"read": 1}
    assert m["turn_count"] == 1 and m["session_id"] == "s1"
    assert m["cost_usd"] == 0.01 and m["parse_errors"] == 1


# ---- ablation knobs --------------------------------------------------------


def _capture(mocker, calls):
    def side(cmd, **kw):
        calls.append(cmd)
        (Path(kw["cwd"]) / "tools.py").write_text("def t():\n    return 1\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=_session(), stderr="")

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)


def _arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_use_transcripts_false_is_blind(mocker, tmp_path):
    _which(mocker)
    calls = []
    _capture(mocker, calls)
    seen = {"builder": 0}

    def builder(run):
        seen["builder"] += 1
        return {"X.md": "x"}

    PiOptimizer(max_retries=1, use_transcripts=False, context_builder=builder).propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
    )
    assert seen["builder"] == 0  # run-derived context skipped when blind
    assert "not been given any transcripts" in _arg(calls[0], "-p")  # cold prompt


def test_require_validation_false_accepts_invalid_candidate(mocker, tmp_path):
    _which(mocker)
    _capture(mocker, [])

    class _AlwaysFail(Validator):
        def validate(self, c):
            return ValidationResult(False, "nope")

    cand = PiOptimizer(max_retries=1, require_validation=False).propose(
        _toolset(), _run(), _AlwaysFail("py", ("tools.py",)), tmp_path
    )
    assert cand.files == {"tools.py": "def t():\n    return 1\n"}  # accepted anyway


# ---- transcript delivery (workspace reads, not argv) ------------------------


def _big_run(n_tasks=100):
    """A run whose summary blows past MAX_ARG_STRLEN (the regression scenario)."""
    return RunResult(
        "b",
        "gpt",
        tuple(
            TaskRun(f"t{i}", 0.0, trajectory="failure detail " * 300)
            for i in range(n_tasks)
        ),
    )


def test_transcript_prompt_is_a_short_workspace_pointer(mocker, tmp_path):
    _which(mocker)
    calls = []
    _capture(mocker, calls)
    run = _run()
    PiOptimizer(max_retries=1).propose(
        _toolset(), run, Validator("py", ("tools.py",)), tmp_path
    )
    assert _arg(calls[0], "-p") == _TRANSCRIPTS_PROMPT
    assert not any(a.startswith("@") for a in calls[0])
    expected = transcript_workspace_files(run)
    for rel, content in expected.items():
        assert (tmp_path / "workspace" / rel).read_text() == content


def test_transcripts_file_is_not_in_the_candidate(mocker, tmp_path):
    _which(mocker)
    _capture(mocker, [])
    cand = PiOptimizer(max_retries=1).propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
    )
    assert cand.files == {"tools.py": "def t():\n    return 1\n"}


def test_editing_the_transcripts_index_fails_the_gate(mocker, tmp_path):
    _which(mocker)

    def side(cmd, **kw):
        (Path(kw["cwd"]) / "BASELINE_TRANSCRIPTS_INDEX.md").write_text("HACKED\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=_session(), stderr="")

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    with pytest.raises(OptimizerLLMFailure, match="change_gate_ok=False"):
        PiOptimizer(max_retries=1).propose(
            _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
        )


def test_retry_does_not_resend_transcripts(mocker, tmp_path):
    _which(mocker)
    calls = []

    def side(cmd, **kw):
        ws = Path(kw["cwd"])
        calls.append(cmd)
        ws.joinpath("tools.py").write_text("   " if len(calls) == 1 else "ok = 1\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=_session(), stderr="")

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    PiOptimizer(max_retries=3).propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
    )
    assert len(calls) == 2
    # The resumed session already has the workspace instructions: nudge only.
    assert "did not pass validation" in _arg(calls[1], "-p")
    assert _TRANSCRIPTS_PROMPT not in calls[1]
    assert "--session" in calls[1]


def test_no_argv_element_exceeds_the_kernel_limit(mocker, tmp_path):
    """Regression test for OSError(E2BIG): a 100-task run used to put a ~400 KB
    transcript summary into a single argv element, which execve rejects."""
    _which(mocker)
    calls = []
    _capture(mocker, calls)
    run = _big_run()
    evidence = transcript_workspace_files(run)
    assert sum(len(v.encode()) for v in evidence.values()) > _MAX_ARG_STRLEN
    PiOptimizer(max_retries=1).propose(
        _toolset(), run, Validator("py", ("tools.py",)), tmp_path
    )
    assert max(len(a.encode()) for a in calls[0]) < _MAX_ARG_STRLEN


def test_oversized_argv_is_caught_before_exec(mocker, tmp_path):
    _which(mocker)
    ran = mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run")
    opt = PiOptimizer(max_retries=1, method_addendum="x" * (_MAX_ARGV_BYTES + 1))
    with pytest.raises(RuntimeError, match="--system-prompt"):
        opt.propose(_toolset(), _run(), Validator("py", ("tools.py",)), tmp_path)
    ran.assert_not_called()  # never even attempted


def test_launch_failure_persists_artifacts_and_raises(mocker, tmp_path):
    _which(mocker)
    mocker.patch(
        "agent_tool_opt_core.optimizers.pi.subprocess.run",
        side_effect=OSError(7, "Argument list too long"),
    )
    with pytest.raises(RuntimeError, match="failed to launch"):
        PiOptimizer(max_retries=5).propose(
            _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
        )
    # The old code let the OSError escape, leaving nothing to diagnose.
    art = tmp_path / "artifacts"
    assert "Argument list too long" in (art / "decision.txt").read_text()
    rec = [json.loads(x) for x in (art / "attempts.jsonl").read_text().splitlines()]
    assert len(rec) == 1 and "Argument list too long" in rec[0]["launch_error"]


def test_stdin_is_pinned_closed(mocker, tmp_path):
    _which(mocker)
    kwargs = {}

    def side(cmd, **kw):
        kwargs.update(kw)
        (Path(kw["cwd"]) / "tools.py").write_text("def t():\n    return 1\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=_session(), stderr="")

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    PiOptimizer(max_retries=1).propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
    )
    # pi merges piped stdin into the prompt whenever stdin is not a TTY.
    assert kwargs["stdin"] is subprocess.DEVNULL


def test_reserved_transcripts_path_conflict(mocker, tmp_path):
    _which(mocker)
    _capture(mocker, [])
    clashing = ToolSet(
        {"tools.py": "def t(): pass\n"},
        ("tools.py",),
        "keep importable",
        context={"BASELINE_TRANSCRIPTS_INDEX.md": "mine\n"},
    )
    with pytest.raises(ValueError, match="paths collide"):
        PiOptimizer(max_retries=1).propose(
            clashing, _run(), Validator("py", ("tools.py",)), tmp_path
        )


def test_reserved_transcripts_path_cannot_be_an_editable_tool(mocker, tmp_path):
    _which(mocker)
    ran = mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run")
    clashing = ToolSet(
        {
            "tools.py": "def t(): pass\n",
            "BASELINE_TRANSCRIPTS_INDEX.md": "editable collision\n",
        },
        ("tools.py", "BASELINE_TRANSCRIPTS_INDEX.md"),
        "keep importable",
    )
    with pytest.raises(ValueError, match="tool files"):
        PiOptimizer(max_retries=1).propose(
            clashing,
            _run(),
            Validator("py", ("tools.py", "BASELINE_TRANSCRIPTS_INDEX.md")),
            tmp_path,
        )
    ran.assert_not_called()


def test_session_dir_is_absolute_so_pi_cannot_write_into_the_workspace(
    mocker, tmp_path, monkeypatch
):
    """Regression test: with a RELATIVE scratch, pi (cwd=workspace) resolved
    --session-dir against the workspace and wrote its session file there, which
    the change gate then rejected as an added file on every attempt."""
    _which(mocker)
    calls = []
    _capture(mocker, calls)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run").mkdir()
    cand = PiOptimizer(max_retries=1).propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), Path("run/optimize")
    )
    session_dir = Path(_arg(calls[0], "--session-dir"))
    assert session_dir.is_absolute()
    ws = (tmp_path / "run" / "optimize" / "workspace").resolve()
    assert not session_dir.is_relative_to(ws)  # never inside the workspace
    assert cand.files == {"tools.py": "def t():\n    return 1\n"}  # gate passes


def _tool_event(name, text, *, is_error):
    return json.dumps({
        "type": "tool_execution_end",
        "toolName": name,
        "result": {"content": [{"type": "text", "text": text}]},
        "isError": is_error,
    })  # fmt: skip


# Verbatim from the run that exposed this: a scratch outside the sandbox's git
# root makes every read/edit fail while pi still exits 0.
_BWRAP_ERR = (
    "warning: cannot write to journal: not inside a git repository: use "
    "--journal global or a literal path bwrap: Can't chdir"
)


def test_sandbox_denial_is_detected_and_raises(mocker, tmp_path):
    _which(mocker)
    stdout = "\n".join([
        _session(),
        _tool_event("read", _BWRAP_ERR, is_error=True),
        _tool_event("edit", "Preflight failed before mutating files.", is_error=True),
    ])  # fmt: skip

    def side(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    # Without the guard this returns an empty Candidate reported as "accepted",
    # which downstream scores as a baseline tie instead of an environment bug.
    with pytest.raises(RuntimeError, match="sandbox could not reach the workspace"):
        PiOptimizer(max_retries=1).propose(
            _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
        )
    assert "git root" in (tmp_path / "artifacts" / "decision.txt").read_text()


def test_sandbox_marker_with_a_working_tool_call_is_not_a_denial(mocker, tmp_path):
    """A stray bwrap warning alongside successful tool use must NOT abort."""
    _which(mocker)
    stdout = "\n".join([
        _session(),
        _tool_event("read", _BWRAP_ERR, is_error=True),
        _tool_event("edit", "ok", is_error=False),
    ])  # fmt: skip

    def side(cmd, **kw):
        (Path(kw["cwd"]) / "tools.py").write_text("def t():\n    return 1\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    cand = PiOptimizer(max_retries=1).propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
    )
    assert cand.files == {"tools.py": "def t():\n    return 1\n"}


def test_sandbox_denial_on_stderr_only_is_still_detected(mocker, tmp_path):
    """A sandbox that cannot start at all writes to stderr, never into a JSON
    tool result — scanning stdout alone let that variant through as a null result."""
    _which(mocker)

    def side(cmd, **kw):
        return subprocess.CompletedProcess(
            cmd, 1, stdout=_session(), stderr="bwrap: Can't chdir: No such file"
        )

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    with pytest.raises(RuntimeError, match="sandbox could not reach the workspace"):
        PiOptimizer(max_retries=1).propose(
            _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
        )
    # stderr is persisted for diagnosis rather than discarded.
    assert "bwrap" in (tmp_path / "artifacts" / "stderr.txt").read_text()


def test_argv_preflight_covers_the_sandbox_prefix(mocker, tmp_path):
    """The kernel limit applies to the composed argv, so the preflight must see
    the sandbox prefix too — it previously checked only pi's own args."""
    _which(mocker)
    ran = mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run")
    opt = PiOptimizer(
        max_retries=1, sandbox_cmd=["sandbox", "--x", "y" * (_MAX_ARGV_BYTES + 1)]
    )
    with pytest.raises(RuntimeError, match="--x"):
        opt.propose(_toolset(), _run(), Validator("py", ("tools.py",)), tmp_path)
    ran.assert_not_called()


def test_blind_mode_writes_no_transcript_workspace(mocker, tmp_path):
    _which(mocker)
    calls = []
    _capture(mocker, calls)
    PiOptimizer(max_retries=1, use_transcripts=False).propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
    )
    assert not (tmp_path / "workspace" / "BASELINE_TRANSCRIPTS_INDEX.md").exists()
    assert not (tmp_path / "workspace" / "baseline_transcripts").exists()
    assert not any(a.startswith("@") for a in calls[0])


def test_sandbox_cmd_prefixes_the_pi_invocation(mocker, tmp_path):
    _which(mocker)
    calls = []

    def side(cmd, **kw):
        calls.append(cmd)
        (Path(kw["cwd"]) / "tools.py").write_text("def t():\n    return 1\n")
        return subprocess.CompletedProcess(cmd, 0, stdout=_session(), stderr="")

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    jail = ["bwrap", "--ro-bind", "/x", "/x", "--"]
    PiOptimizer(max_retries=1, sandbox_cmd=jail).propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
    )
    assert calls[0][: len(jail)] == jail  # pi is launched behind the jail prefix
    assert "pi" in calls[0][len(jail) :]


# --- pi never started -------------------------------------------------------
# `pi` on PATH is not `pi` runnable: a half-installed npm tree (orphaned
# @earendil-works staging dir -> ENOTEMPTY -> missing undici) leaves the wrapper
# executable, so execve succeeds and no OSError reaches launch_error, while node
# dies before pi emits anything. The workspace is then untouched, which the gate
# reads as "pi chose not to edit" -> empty Candidate "accepted" -> a baseline tie.
_NODE_ERR = "Error: Cannot find module 'undici'\n    at Module._resolveFilename"


def test_no_session_event_is_detected_and_raises(mocker, tmp_path):
    _which(mocker)

    def side(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=_NODE_ERR)

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    with pytest.raises(RuntimeError, match="emitted no session event"):
        PiOptimizer(max_retries=1).propose(
            _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
        )
    # The repair recipe and pi's own stderr both survive for diagnosis.
    decision = (tmp_path / "artifacts" / "decision.txt").read_text()
    assert "npm install --prefix" in decision
    assert "undici" in (tmp_path / "artifacts" / "stderr.txt").read_text()


def test_non_json_stdout_without_a_session_is_not_a_null_result(mocker, tmp_path):
    """Noise on stdout must not be mistaken for a session that ran."""
    _which(mocker)

    def side(cmd, **kw):
        return subprocess.CompletedProcess(
            cmd, 1, stdout="pi: unknown option --mode\n", stderr=""
        )

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    with pytest.raises(RuntimeError, match="emitted no session event"):
        PiOptimizer(max_retries=1).propose(
            _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
        )


def test_a_session_that_declines_to_edit_is_still_a_valid_empty_candidate(
    mocker, tmp_path
):
    """The discriminator: an empty diff is benign iff a session actually ran."""
    _which(mocker)

    def side(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout=_session(), stderr="")

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    cand = PiOptimizer(max_retries=1).propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
    )
    assert cand.files == {}


def test_session_never_started_predicate():
    assert _session_never_started("") is True
    assert _session_never_started("not json at all\n") is True
    # A malformed session line is not a session.
    assert _session_never_started('{"type":"turn_start"}\n') is True
    assert _session_never_started(_session()) is False
    assert _session_never_started("noise\n" + _session() + "\n") is False


def test_session_error_is_not_accepted_as_an_empty_candidate(mocker, tmp_path):
    _which(mocker)
    output = "\n".join(
        (
            _session(),
            json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "stopReason": "error",
                        "errorMessage": "authentication failed",
                    },
                }
            ),
        )
    )
    mocker.patch(
        "agent_tool_opt_core.optimizers.pi.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, stdout=output, stderr=""),
    )

    with pytest.raises(OptimizerInfrastructureFailure, match="session ended in error"):
        PiOptimizer(max_retries=1).propose(
            _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
        )

    assert parse_session_metrics(output)["stop_reason"] == "error"
    assert (
        "authentication failed"
        not in (tmp_path / "artifacts" / "decision.txt").read_text()
    )
    attempt = json.loads(
        (tmp_path / "artifacts" / "attempts.jsonl").read_text().splitlines()[0]
    )
    assert attempt["n_changed_files"] == 0


def test_session_error_before_tool_progress_retries_fresh_session(mocker, tmp_path):
    _which(mocker)
    calls = []
    error = "\n".join(
        [
            _session("failed"),
            json.dumps(
                {
                    "type": "message_end",
                    "message": {"role": "assistant", "stopReason": "error"},
                }
            ),
        ]
    )

    def side(cmd, **kw):
        calls.append(cmd)
        output = error if len(calls) == 1 else _session("fresh")
        return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    candidate = PiOptimizer(max_retries=2).propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
    )

    assert candidate.files == {}
    assert len(calls) == 2
    assert "--session" not in calls[1]
    assert _arg(calls[1], "-p") == _TRANSCRIPTS_PROMPT


def test_session_error_after_tool_progress_resumes_same_session(mocker, tmp_path):
    _which(mocker)
    calls = []

    def side(cmd, **kw):
        calls.append(cmd)
        workspace = Path(kw["cwd"])
        if len(calls) == 1:
            (workspace / "tools.py").write_text("def t():\n    return 2\n")
            output = "\n".join(
                (
                    _session("recoverable"),
                    json.dumps(
                        {
                            "type": "tool_execution_end",
                            "toolName": "edit",
                            "isError": False,
                        }
                    ),
                    json.dumps(
                        {
                            "type": "message_end",
                            "message": {
                                "role": "assistant",
                                "stopReason": "error",
                                "usage": {"totalTokens": 10},
                            },
                        }
                    ),
                )
            )
        else:
            output = _session("recoverable")
        return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")

    mocker.patch("agent_tool_opt_core.optimizers.pi.subprocess.run", side_effect=side)
    candidate = PiOptimizer(max_retries=2).propose(
        _toolset(), _run(), Validator("py", ("tools.py",)), tmp_path
    )

    assert candidate.files == {"tools.py": "def t():\n    return 2\n"}
    session_index = calls[1].index("--session")
    assert calls[1][session_index : session_index + 2] == [
        "--session",
        "recoverable",
    ]
    assert "session or model error" in _arg(calls[1], "-p")
    assert "not a validation failure" in _arg(calls[1], "-p")
    attempts = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "attempts.jsonl").read_text().splitlines()
    ]
    assert attempts[0]["n_changed_files"] == 1
