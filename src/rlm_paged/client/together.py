"""Together AI hosted open-weight model adapter.

Together exposes an OpenAI-compatible chat-completions endpoint at
api.together.xyz with the broadest current catalog of open-weight models
(Llama 3.1/3.3, Qwen2.5, DeepSeek-R1/V3, Mixtral, etc.). We reuse the
`openai` SDK with `base_url=` override.

Key wrinkle: DeepSeek-R1 (and other "reasoning"-trained open models)
return their chain-of-thought as a visible `<think>...</think>` block at
the start of the response. We strip this into `thinking_text` so the
harness can optionally route reasoning through the chunk store — this is
the cleanest pre-RL test of structured-externalized vs unstructured-in-
window reasoning on a visible-thinking model.
"""

from __future__ import annotations

import os
import re
from typing import Any

from rlm_paged.client._retry import with_retry
from rlm_paged.client.base import GenerationResult, LLMClient

TOGETHER_BASE_URL = "https://api.together.xyz/v1"

# Models that emit <think>...</think> reasoning blocks in their output.
# Maintained by hand because Together doesn't advertise this per-model in
# the catalog API. Add new entries as the lineup grows.
_THINKS_OUT_LOUD = {
    "deepseek-ai/deepseek-r1",
    "deepseek-ai/deepseek-r1-distill-llama-70b",
    "deepseek-ai/deepseek-r1-distill-qwen-32b",
    "qwen/qwq-32b-preview",
}

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _split_thinking(text: str) -> tuple[str, str]:
    """Extract `<think>...</think>` reasoning from `text`.

    Returns (visible_text, thinking_text). The reasoning block is stripped
    out of `visible_text`. If no <think> block is found, thinking_text is "".
    """
    matches = _THINK_RE.findall(text)
    if not matches:
        return text, ""
    visible = _THINK_RE.sub("", text).strip()
    return visible, "\n".join(matches).strip()


class TogetherClient(LLMClient):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str = TOGETHER_BASE_URL,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        self._client: Any = None

    @property
    def name(self) -> str:
        return f"together:{self.model}"

    @property
    def supports_visible_thinking(self) -> bool:
        return self.model.lower() in _THINKS_OUT_LOUD

    def _sdk(self) -> Any:
        if self._client is None:
            import openai  # type: ignore[import-not-found]

            self._client = openai.OpenAI(
                api_key=self._api_key,
                base_url=self.base_url,
            )
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

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
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
        raw_text = choice.message.content or ""

        if self.supports_visible_thinking:
            visible, thinking = _split_thinking(raw_text)
        else:
            visible, thinking = raw_text, ""

        from rlm_paged.client.tokenizer import count as canonical_count

        thinking_tokens = canonical_count(thinking) if thinking else 0

        return GenerationResult(
            text=visible,
            input_tokens=usage.prompt_tokens,
            output_tokens=max(0, usage.completion_tokens - thinking_tokens),
            finish_reason=str(choice.finish_reason or "stop"),
            thinking_text=thinking,
            thinking_tokens=thinking_tokens,
            raw={"id": resp.id, "model": resp.model},
        )
