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

    def generate_with_warnings(
        self,
        prompt: str,
        *,
        max_tokens: int,
        system: str,
        warn_at_remaining: list[int] | None = None,
    ) -> GenerationResult:
        """Stream the response; inject ⟪WRAP⟫ warnings as output budget
        nears exhaustion.

        Mechanism:
          1. Stream the assistant response.
          2. Count tokens emitted (tiktoken cl100k_base).
          3. When `max_tokens - emitted` crosses each threshold in
             `warn_at_remaining` (e.g. [64, 32, 16]), abort the stream.
          4. Take the partial assistant text and restart the API call
             with: original system + original user + partial assistant
             (prefilled) + synthetic user message
             ⟪WRAP: N tokens left — wrap up quickly⟫
          5. The system prompt teaches the model to ignore the WRAP
             message content and continue mid-thought.
          6. Concatenate all assistant chunks into one final text.

        Reasoning-model caveat: gpt-5/o-series emit hidden reasoning
        tokens that don't stream visibly. We only watch the visible
        completion stream; reasoning is opaque.
        """
        import openai  # type: ignore[import-not-found]
        from rlm_paged.client.tokenizer import count as token_count

        if warn_at_remaining is None:
            warn_at_remaining = [64, 32, 16]
        # Sort descending so we trip the biggest threshold first.
        thresholds = sorted(set(warn_at_remaining), reverse=True)

        client = self._sdk()
        is_reasoning = self._looks_like_reasoning_model()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        emitted_chunks: list[str] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_thinking_tokens = 0
        finish_reason = "stop"

        warn_idx = 0  # which threshold to check next

        while True:
            # Determine the per-call budget. The first call gets
            # max_tokens. Each resume gets the remaining budget
            # minus a small safety margin so the API call itself
            # doesn't blow the cap.
            already = sum(token_count(c) for c in emitted_chunks)
            budget_left = max_tokens - already
            if budget_left <= 0:
                break

            call_max = budget_left
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": True,
            }
            if is_reasoning:
                kwargs["max_completion_tokens"] = call_max
                if self.reasoning_effort:
                    kwargs["reasoning_effort"] = self.reasoning_effort
                # Reasoning models also want usage stream-level info.
                kwargs["stream_options"] = {"include_usage": True}
            else:
                kwargs["max_tokens"] = call_max
                kwargs["temperature"] = 0.0
                kwargs["stream_options"] = {"include_usage": True}

            def call() -> Any:
                return client.chat.completions.create(**kwargs)

            retryable = (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.InternalServerError,
            )
            stream = with_retry(call, retryable=retryable)

            this_chunk = ""
            tripped_threshold: int | None = None
            last_usage = None

            for event in stream:
                # Usage may arrive on a final empty event.
                if getattr(event, "usage", None) is not None:
                    last_usage = event.usage
                choices = event.choices or []
                if not choices:
                    continue
                delta = choices[0].delta
                if delta is None:
                    continue
                piece = getattr(delta, "content", None) or ""
                if piece:
                    this_chunk += piece
                # Check if we've crossed the next remaining-budget threshold.
                if warn_idx < len(thresholds):
                    cumulative_out = sum(
                        token_count(c) for c in emitted_chunks
                    ) + token_count(this_chunk)
                    remaining = max_tokens - cumulative_out
                    if remaining <= thresholds[warn_idx]:
                        tripped_threshold = thresholds[warn_idx]
                        warn_idx += 1
                        try:
                            stream.close()
                        except Exception:
                            pass
                        break
                # Honor the natural end of stream.
                fr = getattr(choices[0], "finish_reason", None)
                if fr:
                    finish_reason = fr

            emitted_chunks.append(this_chunk)
            if last_usage is not None:
                total_input_tokens += getattr(last_usage, "prompt_tokens", 0) or 0
                total_output_tokens += getattr(last_usage, "completion_tokens", 0) or 0
                details = getattr(last_usage, "completion_tokens_details", None)
                if details is not None:
                    total_thinking_tokens += getattr(details, "reasoning_tokens", 0) or 0

            if tripped_threshold is None:
                # Stream ended naturally. Done.
                break

            # We tripped a threshold. Build the resume messages.
            # Prefill the partial assistant text, then add a synthetic
            # user message with the warning. The system prompt teaches
            # the model to continue mid-thought without acknowledging
            # the warning.
            assistant_so_far = "".join(emitted_chunks)
            warn_text = self._format_wrap_warning(tripped_threshold)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_so_far},
                {"role": "user", "content": warn_text},
            ]
            # Loop continues; next iteration calls the API again.

        return GenerationResult(
            text="".join(emitted_chunks),
            input_tokens=total_input_tokens,
            output_tokens=max(0, total_output_tokens - total_thinking_tokens),
            finish_reason=str(finish_reason or "stop"),
            thinking_text="",
            thinking_tokens=total_thinking_tokens,
            raw={"id": "wrap-stream", "model": self.model},
        )

    @staticmethod
    def _format_wrap_warning(remaining: int) -> str:
        if remaining <= 16:
            return f"⟪WRAP: {remaining} tokens left — STOP after this sentence⟫"
        if remaining <= 32:
            return f"⟪WRAP: {remaining} tokens left — finish this thought⟫"
        return f"⟪WRAP: {remaining} tokens left — wrap up quickly⟫"

    def _looks_like_reasoning_model(self) -> bool:
        """Detects models that require `max_completion_tokens` instead of
        `max_tokens` (and that reject explicit temperature/stop).

        Covers:
          - o1 / o3 / o4 reasoning families
          - gpt-5 family (uses the responses-style parameter set even on
            chat.completions)

        Confirmed empirically: a sweep run with `gpt-5` returned 400
        `Unsupported parameter: 'max_tokens'` until this matcher was
        widened to include the gpt-5 prefix.
        """
        m = self.model.lower()
        if m.startswith(("o1", "o3", "o4")):
            return True
        if m.startswith("gpt-5"):
            return True
        return False
