from __future__ import annotations

import json

import pytest

from agent_tool_opt_core.optimizers._common import (
    OptimizerInfrastructureFailure,
    OptimizerLLMFailure,
)
from agent_tool_opt_core.orchestration import (
    ProposalRetryPolicy,
    candidate_failure_metadata,
    record_proposal_metadata,
    run_proposal,
)


def _no_wait(monkeypatch):
    waits = []
    monkeypatch.setattr("agent_tool_opt_core.orchestration.time.sleep", waits.append)
    monkeypatch.setattr(
        "agent_tool_opt_core.orchestration.random.uniform", lambda low, high: 0
    )
    return waits


def test_infrastructure_recovery_retries_in_fresh_scratch(monkeypatch, tmp_path):
    waits = _no_wait(monkeypatch)
    scratches = []

    def proposal(scratch):
        scratches.append(scratch)
        if len(scratches) == 1:
            raise OptimizerInfrastructureFailure("overloaded", retry_after=4)
        return "candidate"

    outcome = run_proposal(
        proposal,
        tmp_path,
        ProposalRetryPolicy(max_attempts=2, base_delay=1, jitter=0),
    )

    assert outcome.candidate == "candidate"
    assert outcome.metadata.status == "complete"
    assert outcome.metadata.attempt_count == 2
    assert all(path.parent == tmp_path for path in scratches)
    assert scratches[0].name.startswith("attempt_01_")
    assert scratches[1].name.startswith("attempt_02_")
    assert scratches[0] != scratches[1]
    assert waits == [4]


@pytest.mark.parametrize("calls_per_proposal", [1, 3])
def test_persistent_infrastructure_failure_is_incomplete(
    monkeypatch, tmp_path, calls_per_proposal
):
    _no_wait(monkeypatch)
    calls = 0

    def proposal(scratch):
        nonlocal calls
        for _ in range(calls_per_proposal):
            calls += 1
        raise OptimizerInfrastructureFailure("gateway 503")

    outcome = run_proposal(
        proposal,
        tmp_path,
        ProposalRetryPolicy(max_attempts=2, base_delay=0, jitter=0),
    )

    assert outcome.candidate is None
    assert outcome.metadata.status == "incomplete"
    assert outcome.metadata.failure_kind == "infrastructure"
    assert outcome.metadata.attempt_count == 2
    assert calls == calls_per_proposal * 2


def test_candidate_failure_is_zero_retry(monkeypatch, tmp_path):
    waits = _no_wait(monkeypatch)
    calls = 0

    def proposal(scratch):
        nonlocal calls
        calls += 1
        raise OptimizerLLMFailure("invalid after validation retries")

    outcome = run_proposal(proposal, tmp_path)

    assert outcome.candidate is None
    assert outcome.metadata.status == "candidate_failed"
    assert outcome.metadata.failure_kind == "candidate"
    assert calls == 1
    assert waits == []


def test_unexpected_error_remains_fatal(tmp_path):
    error = RuntimeError("bug")

    def proposal(scratch):
        raise error

    with pytest.raises(RuntimeError, match="bug") as raised:
        run_proposal(proposal, tmp_path)

    assert raised.value is error
    assert raised.value.proposal_metadata.failure_kind == "fatal"
    assert json.loads((tmp_path / "proposal.json").read_text())["status"] == "fatal"


def test_outer_candidate_failure_preserves_attempt_timing(tmp_path):
    metadata = candidate_failure_metadata(
        "failed import",
        attempts=2,
        elapsed_seconds=3.5,
        exception_type="CandidateImportError",
    )

    record_proposal_metadata(tmp_path, metadata)

    persisted = json.loads((tmp_path / "proposal.json").read_text())
    assert persisted == metadata.to_dict()
