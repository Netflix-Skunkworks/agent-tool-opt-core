"""Costs are exact only when every charge is known, across all proposal attempts."""

from types import SimpleNamespace

import pytest

from agent_tool_opt_core.api import RunResult, TaskRun
from agent_tool_opt_core.costs import (
    Cost,
    active_costs,
    collect_optimizer_costs,
    record_completion,
    total,
)
from agent_tool_opt_core.optimizers._common import call_llm_with_retry


def response(**usage):
    return SimpleNamespace(usage=usage)


def test_run_cost_requires_complete_valid_measurements():
    run = RunResult(
        "b", "a", (TaskRun("1", 1, cost_usd=2), TaskRun("2", 0, cost_usd=4))
    )
    assert run.total_cost_usd == 6
    assert run.average_cost_usd == 3
    assert RunResult("b", "a", ()).total_cost_usd is None
    partial = RunResult("b", "a", (*run.runs, TaskRun("3", 0, known_cost_usd=1)))
    assert partial.total_cost_usd is None
    assert partial.average_cost_usd is None
    assert partial.known_cost_usd == 7


@pytest.mark.parametrize("unknown", [None, -1, True, "0", float("nan"), float("inf")])
def test_unknown_costs_preserve_lower_bound(unknown):
    cost = Cost()
    cost.add(2)
    cost.add(unknown, known=1)
    assert cost.cost_usd is None
    assert cost.known_usd == 3
    assert total([2, unknown]) is None


def test_scopes_are_isolated_and_reset_on_failure():
    with collect_optimizer_costs() as outer:
        outer.llm.add(1)
        with pytest.raises(RuntimeError), collect_optimizer_costs() as inner:
            inner.llm.add(2)
            raise RuntimeError("failed proposal")
        assert active_costs() is outer
        assert outer.llm.cost_usd == 1
    assert active_costs() is None


def test_actual_litellm_pricing_includes_cached_tokens():
    with collect_optimizer_costs() as costs:
        record_completion(
            response(
                prompt_tokens=1000,
                completion_tokens=200,
                prompt_tokens_details={"cached_tokens": 500},
            ),
            "gpt-4o-mini",
        )
    assert costs.llm.cost_usd == pytest.approx(0.0002325)
    assert costs.llm.sources == {"token_pricing"}


@pytest.mark.parametrize(
    "usage,model",
    [
        ({}, "gpt-4o-mini"),
        ({"prompt_tokens": 2, "completion_tokens": 1}, "unknown-test-model"),
    ],
)
def test_missing_usage_or_price_is_incomplete(usage, model):
    with collect_optimizer_costs() as costs:
        record_completion(response(**usage), model)
    assert costs.llm.cost_usd is None


def test_pricing_failure_does_not_retry_a_successful_call(monkeypatch):
    import litellm

    def broken_price(**kwargs):
        raise RuntimeError("pricing unavailable")

    monkeypatch.setattr(litellm, "completion_cost", broken_price)
    calls = []
    reply = response(prompt_tokens=3, completion_tokens=1)
    with collect_optimizer_costs() as costs:
        assert call_llm_with_retry(lambda: calls.append(1) or reply, model="m") is reply
    assert calls == [1]
    assert costs.llm.cost_usd is None


def test_rejected_retry_is_free_but_lost_response_is_unknown(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "completion_cost", lambda **kwargs: 2)
    reply = response(prompt_tokens=3, completion_tokens=1)
    for exception, expected in ((SimpleRateLimit(), 2), (TimeoutError(), None)):
        sequence = iter([exception, reply])

        def call():
            value = next(sequence)
            if isinstance(value, Exception):
                raise value
            return value

        with collect_optimizer_costs() as costs:
            assert call_llm_with_retry(call, model="m", base_delay=0) is reply
        assert costs.llm.cost_usd == expected
        assert costs.llm.known_usd == 2


@pytest.mark.parametrize(
    "name,expected", [("llm", 1), ("draft", 2), ("gepa", 1), ("toolobserver", 3)]
)
def test_catalog_optimizers_record_every_model_call(
    name, expected, monkeypatch, tmp_path
):
    import litellm
    from agent_tool_opt_core.api import ToolSet, Validator
    from agent_tool_opt_core.optimizers.catalog import build_optimizer
    from agent_tool_opt_core.testing import FakeLLMClient

    fake = FakeLLMClient(
        ["def t(): return 1"] * expected,
        usage={"prompt_tokens": 100, "completion_tokens": 20},
    )
    priced = []
    monkeypatch.setattr(
        litellm, "completion_cost", lambda **kw: priced.append(kw["model"]) or 2
    )
    extra = {"batch_size": 1} if name == "toolobserver" else {}
    optimizer = build_optimizer(name, model="gpt-4o-mini", llm=fake, **extra)
    run = RunResult(
        "b",
        "agent",
        (TaskRun("1", 0, "failed"), TaskRun("2", 0, "failed")),
        split="train",
    )
    with collect_optimizer_costs() as costs:
        optimizer.propose(
            ToolSet({"tools.py": "def t(): pass"}, ("tools.py",), ""),
            run,
            Validator("py", ("tools.py",)),
            tmp_path,
        )
    assert len(fake.calls) == expected
    assert priced == ["gpt-4o-mini"] * expected
    assert costs.llm.cost_usd == 2 * expected


class SimpleRateLimit(Exception):
    status_code = 429
