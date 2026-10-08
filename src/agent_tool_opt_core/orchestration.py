"""Shared candidate-proposal retry policy and structured outcomes."""

from __future__ import annotations

import json
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

from agent_tool_opt_core.optimizers._common import (
    OptimizerInfrastructureFailure,
    OptimizerLLMFailure,
)

_T = TypeVar("_T")


@dataclass(frozen=True)
class ProposalRetryPolicy:
    """Bounded full-proposal retries shared by every runner."""

    max_attempts: int = 2
    base_delay: float = 2.0
    max_delay: float = 60.0
    jitter: float = 0.25

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0 or self.max_delay < 0 or self.jitter < 0:
            raise ValueError("retry delays and jitter must not be negative")

    def delay(self, failed_attempt: int, retry_after: float | None = None) -> float:
        backoff = min(self.base_delay * (2 ** (failed_attempt - 1)), self.max_delay)
        wait = max(backoff, retry_after or 0.0)
        return wait + random.uniform(0.0, wait * self.jitter)


@dataclass(frozen=True)
class ProposalMetadata:
    """JSON-friendly proposal status recorded by runners and result tables."""

    status: str
    failure_kind: str | None
    attempt_count: int
    elapsed_seconds: float
    exception_type: str | None = None
    final_cause: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ProposalOutcome(Generic[_T]):
    candidate: _T | None
    metadata: ProposalMetadata


def _record(scratch: Path, outcome: ProposalOutcome[_T]) -> ProposalOutcome[_T]:
    record_proposal_metadata(scratch, outcome.metadata)
    return outcome


def record_proposal_metadata(scratch: Path, metadata: ProposalMetadata) -> None:
    """Persist proposal metadata at the stable root shared by all attempts."""
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "proposal.json").write_text(json.dumps(metadata.to_dict(), indent=2))


def _metadata(
    status: str,
    kind: str | None,
    attempts: int,
    started: float,
    exc: BaseException | None = None,
) -> ProposalMetadata:
    cause = (exc.__cause__ or exc) if exc is not None else None
    return ProposalMetadata(
        status=status,
        failure_kind=kind,
        attempt_count=attempts,
        elapsed_seconds=max(0.0, time.monotonic() - started),
        exception_type=type(cause).__name__ if cause is not None else None,
        final_cause=str(cause) if cause is not None else None,
    )


def candidate_failure_metadata(
    message: str,
    *,
    attempts: int = 1,
    elapsed_seconds: float = 0.0,
    exception_type: str = "ValidationFailure",
) -> ProposalMetadata:
    """Create structured metadata for a deterministic outer validation failure."""
    return ProposalMetadata(
        status="candidate_failed",
        failure_kind="candidate",
        attempt_count=attempts,
        elapsed_seconds=elapsed_seconds,
        exception_type=exception_type,
        final_cause=message,
    )


def run_proposal(
    propose: Callable[[Path], _T],
    scratch: Path,
    policy: ProposalRetryPolicy | None = None,
) -> ProposalOutcome[_T]:
    """Run a proposal with fresh attempt scratch and classify its final outcome.

    Deterministic candidate failures return immediately. Infrastructure failures
    retry under ``policy`` and return an incomplete outcome on exhaustion. Any
    other exception is tagged with fatal metadata and re-raised unchanged.
    """
    policy = policy or ProposalRetryPolicy()
    started = time.monotonic()
    scratch.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, policy.max_attempts + 1):
        attempt_scratch = Path(
            tempfile.mkdtemp(prefix=f"attempt_{attempt:02d}_", dir=scratch)
        )
        try:
            candidate = propose(attempt_scratch)
        except OptimizerLLMFailure as exc:
            return _record(
                scratch,
                ProposalOutcome(
                    None,
                    _metadata("candidate_failed", "candidate", attempt, started, exc),
                ),
            )
        except OptimizerInfrastructureFailure as exc:
            if attempt == policy.max_attempts:
                return _record(
                    scratch,
                    ProposalOutcome(
                        None,
                        _metadata(
                            "incomplete", "infrastructure", attempt, started, exc
                        ),
                    ),
                )
            time.sleep(policy.delay(attempt, exc.retry_after))
        except Exception as exc:
            metadata = _metadata("fatal", "fatal", attempt, started, exc)
            _record(scratch, ProposalOutcome(None, metadata))
            try:
                setattr(exc, "proposal_metadata", metadata)
            except (AttributeError, TypeError):
                pass
            raise
        else:
            return _record(
                scratch,
                ProposalOutcome(
                    candidate, _metadata("complete", None, attempt, started)
                ),
            )
    raise AssertionError("proposal retry loop terminated without an outcome")
