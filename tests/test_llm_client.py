from types import SimpleNamespace

import pytest

from agent_tool_opt_core.llm_client import LiteLLMClient


def test_litellm_client_returns_normalized_usage_cost_and_provenance(monkeypatch):
    calls = []
    response = SimpleNamespace(
        id="request-123",
        model="claude-sonnet",
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
        usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        _hidden_params={"custom_llm_provider": "anthropic"},
    )
    monkeypatch.setattr(
        "agent_tool_opt_core.llm_client.litellm.completion",
        lambda **kwargs: calls.append(kwargs) or response,
    )
    monkeypatch.setattr(
        "agent_tool_opt_core.llm_client.litellm.completion_cost",
        lambda **kwargs: 0.004,
    )

    completion = LiteLLMClient().complete(
        model="anthropic/claude-sonnet",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0,
    )

    assert completion.text == "answer"
    assert completion.model == "claude-sonnet"
    assert completion.provider == "anthropic"
    assert completion.usage == {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
    }
    assert completion.cost_usd == 0.004
    assert completion.cost_source == "litellm_pricing"
    assert completion.request_id == "request-123"
    assert calls == [
        {
            "model": "anthropic/claude-sonnet",
            "messages": [{"role": "user", "content": "hello"}],
            "num_retries": 0,
            "temperature": 0,
        }
    ]


def test_litellm_client_keeps_retry_policy_in_core(monkeypatch):
    monkeypatch.setattr(
        "agent_tool_opt_core.llm_client.litellm.completion",
        lambda **kwargs: pytest.fail("provider must not be called"),
    )
    with pytest.raises(ValueError, match="retry policy is owned"):
        LiteLLMClient().complete(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            num_retries=3,
        )


def test_litellm_client_does_not_fail_when_pricing_is_unavailable(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
        usage=None,
    )
    monkeypatch.setattr(
        "agent_tool_opt_core.llm_client.litellm.completion",
        lambda **kwargs: response,
    )
    monkeypatch.setattr(
        "agent_tool_opt_core.llm_client.litellm.completion_cost",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("no price")),
    )

    completion = LiteLLMClient().complete(
        model="openai/custom-model",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert completion.text == "answer"
    assert completion.cost_usd is None
    assert completion.provider == "openai"
