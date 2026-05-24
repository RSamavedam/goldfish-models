"""OpenAI provider adapter.

Handles both standard chat models (gpt-4o, gpt-4.1) and reasoning models
(o1, o3, o4-mini). Reasoning models report `reasoning_tokens` in usage but
the reasoning content itself is hidden — `thinking_text` stays empty,
`thinking_tokens` is populated.
"""

from __future__ import annotations

import os
from typing import Any

from rlm_paged.client._retry import with_retry
from rlm_paged.client.base import GenerationResult, LLMClient


class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort  # "low" | "medium" | "high" | None
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client: Any = None

    @property
    def name(self) -> str:
        suffix = f":{self.reasoning_effort}" if self.reasoning_effort else ""
        return f"openai:{self.model}{suffix}"

    @property
    def supports_visible_thinking(self) -> bool:
        return False  # o-series reasoning is opaque

    def _sdk(self) -> Any:
        if self._client is None:
            import openai  # type: ignore[import-not-found]

            self._client = openai.OpenAI(api_key=self._api_key)
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
        import openai  # type: ignore[import-not-found]

        client = self._sdk()
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        is_reasoning = self._looks_like_reasoning_model()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        # Reasoning models use max_completion_tokens and reject temperature/stop.
        if is_reasoning:
            kwargs["max_completion_tokens"] = max_tokens
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
            if stop:
                kwargs["stop"] = stop

        def call() -> Any:
            return client.chat.completions.create(**kwargs)

        retryable = (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.InternalServerError,
        )
        resp = with_retry(call, retryable=retryable)
        choice = resp.choices[0]
        usage = resp.usage
        thinking_tokens = 0
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            thinking_tokens = getattr(details, "reasoning_tokens", 0) or 0

        return GenerationResult(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens - thinking_tokens,
            finish_reason=str(choice.finish_reason or "stop"),
            thinking_text="",
            thinking_tokens=thinking_tokens,
            raw={"id": resp.id, "model": resp.model},
        )

    def _looks_like_reasoning_model(self) -> bool:
        m = self.model.lower()
        return m.startswith("o1") or m.startswith("o3") or m.startswith("o4")
