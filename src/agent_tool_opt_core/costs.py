"""Collect and aggregate optimizer costs for one proposal.

The active, context-local collector keeps optimizer LLM charges separate from
benchmark evaluations performed during search. Successful LLM responses are
priced from their token usage with LiteLLM; benchmark adapters add their own
recorded task costs. Failed calls that may have been billed are marked unknown.

Totals are exact only when every charge is known. Missing or invalid amounts
make the exact total unavailable while preserving the known USD subtotal;
explicit zero remains a valid measurement. Collection is optional, so the same
optimizer code runs unchanged outside an accounting scope.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from math import fsum, isfinite


def usd(value: object) -> float | None:
    """Accept only finite, nonnegative numeric amounts (including explicit zero)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        amount = float(value)
    except OverflowError:
        return None
    return amount if isfinite(amount) and amount >= 0 else None


def total(values) -> float | None:
    amounts = [usd(value) for value in values]
    return None if None in amounts else fsum(amounts)


@dataclass
class Cost:
    """Accumulate USD; any unknown charge makes ``cost_usd`` None.

    ``known_usd`` retains the known subtotal, including partial charges supplied
    with unknown totals. Explicit zero is valid; invalid amounts are unknown.
    """

    known_usd: float = 0.0
    unknown_count: int = 0
    count: int = 0
    sources: set[str] = field(default_factory=set)
    reasons: set[str] = field(default_factory=set)

    @property
    def cost_usd(self) -> float | None:
        return None if self.unknown_count else self.known_usd

    def add(self, value, *, known=0.0, source="", reason="cost_missing") -> None:
        amount = usd(value)
        self.count += 1
        self.known_usd = fsum(
            (self.known_usd, amount if amount is not None else usd(known) or 0.0)
        )
        if source:
            self.sources.add(source)
        if amount is None:
            self.unknown_count += 1
            self.reasons.add(reason)

    def add_run(self, run) -> None:
        for task in run.runs:
            self.add(
                task.cost_usd,
                known=task.known_cost_usd,
                source="tau2_results",
                reason="simulation_cost_missing",
            )


@dataclass
class OptimizerCosts:
    llm: Cost = field(default_factory=Cost)
    search: Cost = field(default_factory=Cost)


_active: ContextVar[OptimizerCosts | None] = ContextVar("optimizer_costs", default=None)


@contextmanager
def collect_optimizer_costs():
    costs = OptimizerCosts()
    token = _active.set(costs)
    try:
        yield costs
    finally:
        _active.reset(token)


def active_costs() -> OptimizerCosts | None:
    return _active.get()


def record_completion(response, model: str | None) -> None:
    """Estimate optimizer USD from response usage using LiteLLM's model rates.

    Source ``token_pricing``: LiteLLM's ``completion_cost()`` prices response
    input/output tokens and supported cache details for the requested model.
    Missing usage or pricing records an unknown charge; pricing failures
    never retry a completed call. These are estimates, not invoice amounts.
    """
    costs = active_costs()
    if costs is None:
        return
    normalized_cost = usd(getattr(response, "cost_usd", None))
    if normalized_cost is not None:
        source = getattr(response, "cost_source", None)
        costs.llm.add(
            normalized_cost,
            source=source if isinstance(source, str) else "provider_pricing",
        )
        return
    value = None
    reason = "completion_usage_missing"
    usage = getattr(response, "usage", None)
    if usage is not None:
        usage = usage.model_dump() if hasattr(usage, "model_dump") else usage
        if isinstance(usage, dict) and all(
            usd(usage.get(key)) is not None
            for key in ("prompt_tokens", "completion_tokens")
        ):
            reason = "model_price_missing"
            try:
                from litellm import completion_cost

                value = completion_cost(
                    completion_response={"model": model, "usage": usage}, model=model
                )
            except Exception:
                # Pricing is observability: it must not retry a billed completion
                # or change the optimizer's result when a model is unsupported.
                pass
    costs.llm.add(value, source="token_pricing", reason=reason)


def record_call_failure(exc: Exception) -> None:
    """A rejected request is free; a lost response may already have been billed."""
    costs = active_costs()
    if costs is not None and getattr(exc, "status_code", None) not in (
        400,
        401,
        403,
        404,
        422,
        429,
    ):
        costs.llm.add(None, reason="completion_response_lost")
