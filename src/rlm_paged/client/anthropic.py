"""Anthropic provider adapter.

Supports extended-thinking models: when `thinking_budget` > 0, requests
thinking blocks and surfaces them as `thinking_text` so the harness can
optionally route them through the chunk store.
"""

from __future__ import annotations

import os
from typing import Any

from rlm_paged.client._retry import with_retry
from rlm_paged.client.base import GenerationResult, LLMClient


class AnthropicClient(LLMClient):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        thinking_budget: int = 0,
    ) -> None:
        self.model = model
        self.thinking_budget = thinking_budget
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client: Any = None  # lazy

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    @property
    def supports_visible_thinking(self) -> bool:
        return self.thinking_budget > 0

    def _sdk(self) -> Any:
        if self._client is None:
            import anthropic  # type: ignore[import-not-found]

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        stop: list[str] | None = None,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> GenerationResult:
        import anthropic  # type: ignore[import-not-found]

        client = self._sdk()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if stop:
            kwargs["stop_sequences"] = stop
        if self.thinking_budget > 0:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
            # Extended thinking requires temperature=1 per current API contract.
            kwargs["temperature"] = 1.0

        def call() -> Any:
            return client.messages.create(**kwargs)

        retryable = (
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
        )
        resp = with_retry(call, retryable=retryable)

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        for block in resp.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(block.text)
            elif block_type == "thinking":
                thinking_parts.append(block.thinking)

        return GenerationResult(
            text="".join(text_parts),
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            finish_reason=str(resp.stop_reason or "stop"),
            thinking_text="".join(thinking_parts),
            thinking_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            raw={"id": resp.id, "model": resp.model},
        )
