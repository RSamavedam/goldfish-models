"""Google Gemini provider adapter.

Uses google-genai (the new SDK). Supports thinking-enabled Gemini 2.5
models — when thinking is on, `thoughts_token_count` is exposed in usage
metadata; the thought content itself is not surfaced through the API.
"""

from __future__ import annotations

import os
from typing import Any

from rlm_paged.client._retry import with_retry
from rlm_paged.client.base import GenerationResult, LLMClient


class GeminiClient(LLMClient):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        thinking_budget: int = 0,
    ) -> None:
        self.model = model
        self.thinking_budget = thinking_budget
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get(
            "GEMINI_API_KEY"
        )
        self._client: Any = None

    @property
    def name(self) -> str:
        return f"gemini:{self.model}"

    @property
    def supports_visible_thinking(self) -> bool:
        return False

    def _sdk(self) -> Any:
        if self._client is None:
            from google import genai  # type: ignore[import-not-found]

            self._client = genai.Client(api_key=self._api_key)
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
        from google.genai import types  # type: ignore[import-not-found]

        client = self._sdk()
        config_kwargs: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            config_kwargs["system_instruction"] = system
        if stop:
            config_kwargs["stop_sequences"] = stop
        if self.thinking_budget > 0:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
            )

        def call() -> Any:
            return client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )

        try:
            from google.genai import errors as gerrors  # type: ignore[import-not-found]
            retryable: tuple[type[BaseException], ...] = (
                getattr(gerrors, "ServerError", Exception),
                getattr(gerrors, "ClientError", Exception),
            )
        except ImportError:
            retryable = (Exception,)

        resp = with_retry(call, retryable=retryable)

        usage = resp.usage_metadata
        thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        return GenerationResult(
            text=(resp.text or "") if hasattr(resp, "text") else "",
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=output_tokens,
            finish_reason=str(
                (resp.candidates[0].finish_reason if resp.candidates else "stop")
            ),
            thinking_text="",
            thinking_tokens=thinking_tokens,
            raw={"model": self.model},
        )
