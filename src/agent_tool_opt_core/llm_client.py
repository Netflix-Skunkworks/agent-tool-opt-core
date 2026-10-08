"""Provider-neutral text completion clients for optimizers and judges."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import litellm

from agent_tool_opt_core.costs import usd


@dataclass(frozen=True)
class LLMCompletion:
    """Normalized completion result without credentials or raw provider headers."""

    text: str
    model: str
    provider: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    cost_source: str | None = None
    request_id: str | None = None


class LLMClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **params: Any,
    ) -> LLMCompletion: ...


class LiteLLMClient:
    """Thin LiteLLM transport with normalized, credential-free results."""

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **params: Any,
    ) -> LLMCompletion:
        if not isinstance(model, str) or not 1 <= len(model) <= 256:
            raise ValueError(
                "model must be a non-empty string of at most 256 characters"
            )
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        if "num_retries" in params:
            raise ValueError("retry policy is owned by agent-tool-opt-core")

        response = litellm.completion(
            model=model,
            messages=messages,
            num_retries=0,
            **params,
        )
        text = _response_text(response)
        usage = _usage_dict(getattr(response, "usage", None))
        response_model = getattr(response, "model", None)
        normalized_model = response_model if isinstance(response_model, str) else model
        provider = _provider_name(response, model)
        request_id = getattr(response, "id", None)
        if not isinstance(request_id, str) or len(request_id) > 512:
            request_id = None

        cost = None
        try:
            cost = usd(
                litellm.completion_cost(completion_response=response, model=model)
            )
        except Exception:
            # Cost is observability. Missing pricing must not alter the completion.
            pass
        return LLMCompletion(
            text=text,
            model=normalized_model,
            provider=provider,
            usage=usage,
            cost_usd=cost,
            cost_source="litellm_pricing" if cost is not None else None,
            request_id=request_id,
        )


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise ValueError("provider response contained no choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if content is None:
        return ""
    if not isinstance(content, str):
        raise ValueError("provider response content was not text")
    return content


def _usage_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, dict):
        return {}
    allowed = {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    }
    return {key: value[key] for key in allowed if key in value}


def _provider_name(response: Any, requested_model: str) -> str | None:
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        provider = hidden.get("custom_llm_provider")
        if isinstance(provider, str) and 1 <= len(provider) <= 128:
            return provider
    if "/" in requested_model:
        provider = requested_model.split("/", 1)[0]
        if provider:
            return provider
    return None


__all__ = ["LLMClient", "LLMCompletion", "LiteLLMClient"]
