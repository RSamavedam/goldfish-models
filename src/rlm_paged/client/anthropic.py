"""Anthropic provider adapter. Stub — wire up in Phase 1 implementation."""

from __future__ import annotations

from rlm_paged.client.base import GenerationResult, LLMClient


class AnthropicClient(LLMClient):
    def __init__(self, model: str, *, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        stop: list[str] | None = None,
        temperature: float = 0.0,
    ) -> GenerationResult:
        raise NotImplementedError("AnthropicClient not yet implemented")
