"""Public API for tool optimization — the five-concept model.

Roles are ABCs (we own every implementer): ``Benchmark``, ``Agent``,
``ToolTarget``, ``Optimizer``. Everything else is a value (dataclass). The
``Validator`` is plain data so it survives a remote distributed execution
boundary. Blind optimizers receive validation only; search optimizers may
also receive a ``TrainEvaluator`` restricted to training tasks. Neither
interface exposes held-out test evaluation.

Concepts:
  1. Benchmark   — tasks + running an agent on them + scoring
  2. Agent       — the executor under test (the subject; not the optimizer's model)
  3. ToolTarget  — where the optimizable tools live + extract/validate/apply/restore
  4. Optimizer   — propose an edit from tools + transcripts
  (+ Experiment  — the orchestration layer over the four; see experiment.py)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean

from agent_tool_opt_core.costs import total, usd

TaskId = str

# ---------------------------------------------------------------------------
# Value types (data, not interfaces)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSet:
    """The tools as the optimizer sees them: an editable allowlist plus
    optional read-only context the agent may consult but never edit."""

    files: dict[str, str]  # relpath -> content (the editable allowlist files)
    allowlist: tuple[str, ...]  # which relpaths may change
    language_rules: str  # how to edit THIS tool kind (feeds the optimizer prompt)
    # Read-only files (relpath -> content) the agent can READ for context but
    # not edit — e.g. the rest of a tool tree, domain policy. Materialized into
    # the workspace by the optimizer; edits to them are rejected by the change
    # gate. Supplied by the ToolTarget, which owns "what the agent sees".
    context: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    """A proposed edit. The single artifact every optimizer produces and
    every ToolTarget applies (replaces the optimize()/apply_to_tools() split)."""

    files: dict[str, str]  # relpath -> new content (subset of the allowlist)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    log: str = ""


@dataclass(frozen=True)
class TaskRun:
    task_id: TaskId
    reward: float
    # The benchmark's RAW per-task transcript, stored verbatim — whatever the
    # benchmark emits (its stdout / native trace / result object). Adapters do no
    # interpretation here; any processing (rendering, parsing) belongs to the
    # optimizer and must be domain-agnostic (see optimizers._common.render_trajectory).
    trajectory: object | None = None
    cost_usd: float | None = None
    known_cost_usd: float = 0.0


@dataclass(frozen=True)
class RunResult:
    benchmark: str
    agent: str
    runs: tuple[TaskRun, ...]
    # Provenance — "train" | "test" | None. The optimizer may only ever receive
    # a "train" run; driver.propose() rejects "test" (anti train/test leakage).
    split: str | None = None

    @property
    def transcripts(self) -> tuple[TaskRun, ...]:
        """Optimizer-facing view of a run (here: the per-task records)."""
        return self.runs

    def failing(self) -> tuple[TaskRun, ...]:
        return tuple(r for r in self.runs if r.reward < 1.0)

    @property
    def total_cost_usd(self) -> float | None:
        return total(r.cost_usd for r in self.runs) if self.runs else None

    @property
    def average_cost_usd(self) -> float | None:
        cost = self.total_cost_usd
        return cost / len(self.runs) if cost is not None else None

    @property
    def known_cost_usd(self) -> float:
        return (
            total(
                usd(r.cost_usd) if usd(r.cost_usd) is not None else r.known_cost_usd
                for r in self.runs
            )
            or 0.0
        )


@dataclass(frozen=True)
class Metrics:
    avg_reward: float
    pass_rate: float
    n: int

    def better_than(self, other: "Metrics") -> bool:
        return (self.avg_reward, self.pass_rate) > (other.avg_reward, other.pass_rate)


@dataclass(frozen=True)
class Validator:
    """Reconstructable validation handle handed to ``Optimizer.propose``.

    Deliberately plain data + a method (no captured ToolTarget / closure),
    so it is picklable and survives a remote worker boundary
    boundary. A real ToolTarget overrides ``validate`` (or supplies a
    richer Validator subclass) that rebuilds the language gate on the
    worker (``python -c "import tools"`` / ``bun build``) from ``kind`` +
    ``allowlist``.
    """

    kind: str  # e.g. "py" | "ts"
    allowlist: tuple[str, ...]

    def validate(self, c: Candidate) -> ValidationResult:
        for rel, txt in c.files.items():
            if rel not in self.allowlist:
                return ValidationResult(False, f"{rel} outside allowlist")
            if not txt.strip():
                return ValidationResult(False, f"{rel} is empty")
        return ValidationResult(True, "ok")


class TrainEvaluator(ABC):
    """A TRAIN-only reward oracle handed to search/select optimizers.

    Some optimizers must *score candidate tools* during their own loop — GEPA's
    Pareto search, k-fold cross-validation selection, minibatch
    propose-and-select. This is the capability that enables that whole family.

    Bound to the train task pool *by construction*: it cannot evaluate the test
    split (the anti-reward-hacking boundary in its "train jail" form — an
    optimizer may see the train signal, never test). Returned runs are stamped
    ``split="train"``. Handed to ``Optimizer.propose`` only when the optimizer
    declares ``wants_train_eval``; "blind" optimizers receive ``None`` and so
    cannot probe reward at all.
    """

    @property
    @abstractmethod
    def train_tasks(self) -> tuple[TaskId, ...]:
        """The full train task pool, so the optimizer can minibatch / fold it."""

    @abstractmethod
    def evaluate(
        self, candidate: Candidate, tasks: list[TaskId] | None = None
    ) -> RunResult:
        """Score ``candidate`` on ``tasks`` (default: all train). ``tasks`` must
        be a subset of ``train_tasks``; returns a ``split="train"`` RunResult."""


def score(run: RunResult) -> Metrics:
    """Default scoring; a Benchmark may override with pass^k etc."""
    rewards = [r.reward for r in run.runs]
    if not rewards:
        return Metrics(0.0, 0.0, 0)
    return Metrics(
        avg_reward=fmean(rewards),
        pass_rate=fmean(1.0 if r >= 1.0 else 0.0 for r in rewards),
        n=len(rewards),
    )


# ---------------------------------------------------------------------------
# The four roles
# ---------------------------------------------------------------------------


class Benchmark(ABC):
    """Concept 1 — tasks + running an agent on them + scoring."""

    name: str

    @abstractmethod
    def tasks(self, split: str) -> list[TaskId]: ...

    @abstractmethod
    def evaluate(
        self, agent: "Agent", tasks: list[TaskId], tools: ToolSet
    ) -> RunResult:
        """Run ``agent`` on ``tasks`` with the currently-effective ``tools``."""

    def score(self, run: RunResult) -> Metrics:
        return score(run)


class Agent(ABC):
    """Concept 2 — the executor under test (the subject). Often just config.

    This is the subject agent, NOT the optimizer's own model. The Pi
    optimizer's ``claude-opus`` is ``Optimizer`` config, not an ``Agent``.
    """

    id: str


class ToolTarget(ABC):
    """Concept 3 — where the optimizable tools live + how to swap them.

    Standalone and parameterized by location + injection strategy, so
    "tools owned by the Benchmark (tau2) vs the Agent (opencode)" is a
    construction detail, not a type hierarchy.
    """

    kind: str  # e.g. "py" | "ts"
    language_rules: str

    @abstractmethod
    def extract(self) -> ToolSet:
        """The baseline (unmodified) editable tools."""

    @abstractmethod
    def effective_toolset(self) -> ToolSet:
        """What the agent sees *right now* (reflects any applied candidate)."""

    @abstractmethod
    def apply(self, c: Candidate) -> None:
        """Inject ``c`` so the running agent uses it (in-process / overlay)."""

    @abstractmethod
    def restore(self) -> None:
        """Revert to baseline."""

    def validator(self) -> Validator:
        return Validator(self.kind, self.extract().allowlist)


class Optimizer(ABC):
    """Concept 4 — propose an edit from the tool source + train transcripts.

    Always gets a ``Validator`` (the cheap language gate) and nothing that
    reveals the *test* reward. Search/select optimizers additionally set
    ``wants_train_eval = True`` to receive a ``TrainEvaluator`` (train-only
    scoring); blind optimizers leave it False and get ``train_eval=None``, so
    they are structurally unable to probe reward.
    """

    id: str
    wants_train_eval: bool = False

    @abstractmethod
    def propose(
        self,
        tools: ToolSet,
        run: RunResult,
        validate: Validator,
        scratch: Path,
        train_eval: "TrainEvaluator | None" = None,
    ) -> Candidate: ...
