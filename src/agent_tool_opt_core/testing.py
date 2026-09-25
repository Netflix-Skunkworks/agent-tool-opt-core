"""Small deterministic test doubles for downstream optimizer tests."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from agent_tool_opt_core.llm_client import LLMCompletion


class FakeLLMClient:
    """Return queued text completions and record normalized requests."""

    def __init__(
        self,
        replies: Iterable[str],
        *,
        usage: dict[str, Any] | None = None,
    ) -> None:
        self._replies = iter(replies)
        self.usage = usage or {"prompt_tokens": 0, "completion_tokens": 0}
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **params: Any,
    ) -> LLMCompletion:
        self.calls.append({"model": model, "messages": messages, **params})
        return LLMCompletion(
            text=next(self._replies),
            model=model,
            provider="fake",
            usage=dict(self.usage),
        )


__all__ = ["FakeLLMClient"]
