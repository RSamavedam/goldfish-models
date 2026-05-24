"""Anthropic provider adapter.

Three modes of operation:

1. Plain generation (no thinking, no tools) — standard chat completion.
2. Extended thinking — `thinking_budget > 0` requests `thinking` blocks
   that are returned alongside `text` blocks. The thinking content is
   visible to the harness as `thinking_text`. No tools.
3. Interleaved thinking with native tools — `interleaved_thinking=True`,
   `thinking_budget > 0`, and a tools list. Uses Anthropic's interleaved-
   thinking beta to let the model interleave thinking blocks with tool
   calls within a single turn. The client handles the multi-call dance
   internally because Anthropic requires signed thinking blocks to be
   round-tripped on follow-up messages.

The interleaved-thinking path is the only one that lets a *closed-source*
thinking model use our paging tools mid-reasoning. It's an approximation
of the L-enforcement we'd get with full RL training, not a substitute.
See docs/closed_thinking_asymmetry.md for honest constraints.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from rlm_paged.client._retry import with_retry
from rlm_paged.client.base import GenerationResult, LLMClient


INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"


class AnthropicClient(LLMClient):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        thinking_budget: int = 0,
        interleaved_thinking: bool = False,
        tools: list[dict[str, Any]] | None = None,
        max_interleaved_steps: int = 8,
    ) -> None:
        self.model = model
        self.thinking_budget = thinking_budget
        self.interleaved_thinking = interleaved_thinking
        self.tools = tools
        self.max_interleaved_steps = max_interleaved_steps
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client: Any = None  # lazy

    @property
    def name(self) -> str:
        suffix = "+interleaved" if self.interleaved_thinking else ""
        return f"anthropic:{self.model}{suffix}"

    @property
    def supports_visible_thinking(self) -> bool:
        return self.thinking_budget > 0

    @property
    def supports_interleaved_thinking(self) -> bool:
        return self.interleaved_thinking and self.thinking_budget > 0 and bool(self.tools)

    def _sdk(self) -> Any:
        if self._client is None:
            import anthropic  # type: ignore[import-not-found]

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    # ------------------------------------------------------------------ #
    # Plain / extended-thinking generation (no tools)                    #
    # ------------------------------------------------------------------ #

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
            kwargs["temperature"] = 1.0  # required by extended thinking

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

    # ------------------------------------------------------------------ #
    # Interleaved thinking + native tools                                #
    # ------------------------------------------------------------------ #

    def generate_with_dispatcher(
        self,
        prompt: str,
        *,
        max_tokens: int,
        system: str | None,
        dispatch: Callable[[str, dict[str, Any]], dict[str, Any]],
    ) -> GenerationResult:
        """Run a single 'turn' that may contain multiple thinking/tool_use cycles.

        `dispatch(tool_name, tool_input) -> tool_result_dict` is called for
        each tool_use block emitted by the model. The client passes the
        result back via a follow-up message and continues until the model
        emits a final text block with no pending tool_use.

        Returns one GenerationResult summarizing the trajectory: visible
        text is the concatenation of all text blocks, thinking_text is the
        concatenation of all thinking blocks.
        """
        if not self.supports_interleaved_thinking:
            raise RuntimeError(
                "generate_with_dispatcher requires interleaved_thinking=True, "
                "thinking_budget>0, and a tools list."
            )

        # The SDK is only required if a live API call would happen. Tests
        # pre-inject `self._client` to skip the real SDK; in that case we
        # never need the `anthropic` module at all.
        try:
            import anthropic  # type: ignore[import-not-found]

            retryable: tuple[type[BaseException], ...] = (
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            )
        except ImportError:
            if self._client is None:
                raise
            retryable = ()

        client = self._sdk()
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

        thinking_parts: list[str] = []
        text_parts: list[str] = []
        op_lines: list[str] = []  # rendered op-code lines for the harness loop
        total_input_tokens = 0
        total_output_tokens = 0
        finish_reason = "stop"

        for step in range(self.max_interleaved_steps):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 1.0,
                "messages": messages,
                "thinking": {"type": "enabled", "budget_tokens": self.thinking_budget},
                "tools": self.tools,
                "extra_headers": {"anthropic-beta": INTERLEAVED_THINKING_BETA},
            }
            if system:
                kwargs["system"] = system

            def call() -> Any:
                return client.messages.create(**kwargs)

            resp = with_retry(call, retryable=retryable)

            total_input_tokens += resp.usage.input_tokens
            total_output_tokens += resp.usage.output_tokens
            finish_reason = str(resp.stop_reason or "stop")

            # Build the assistant message echoing the model's content back
            # for the next call. Signed thinking blocks must be preserved
            # verbatim (signature field intact) for continuation.
            assistant_content: list[dict[str, Any]] = []
            tool_uses: list[tuple[str, str, dict[str, Any]]] = []  # (id, name, input)

            for block in resp.content:
                block_type = getattr(block, "type", None)
                if block_type == "thinking":
                    thinking_parts.append(block.thinking)
                    assistant_content.append(
                        {
                            "type": "thinking",
                            "thinking": block.thinking,
                            "signature": block.signature,
                        }
                    )
                elif block_type == "redacted_thinking":
                    # Preserve verbatim — we never see this content but the
                    # model needs it round-tripped.
                    assistant_content.append(
                        {"type": "redacted_thinking", "data": block.data}
                    )
                elif block_type == "text":
                    text_parts.append(block.text)
                    assistant_content.append({"type": "text", "text": block.text})
                elif block_type == "tool_use":
                    tool_uses.append((block.id, block.name, dict(block.input)))
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            if not tool_uses:
                # No more tool calls — we're done.
                break

            messages.append({"role": "assistant", "content": assistant_content})

            # Dispatch each tool call and collect the results into one user
            # message of tool_result blocks.
            tool_result_blocks: list[dict[str, Any]] = []
            for tool_id, tool_name, tool_input in tool_uses:
                # Record an op-code line so the harness loop's metrics still
                # see this op.
                op_lines.append(_format_op_line(tool_name, tool_input))
                try:
                    result = dispatch(tool_name, tool_input)
                    payload = json.dumps(result) if result is not None else "ok"
                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": payload,
                        }
                    )
                except Exception as exc:
                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": f"error: {type(exc).__name__}: {exc}",
                            "is_error": True,
                        }
                    )

            messages.append({"role": "user", "content": tool_result_blocks})
        else:
            finish_reason = "max_interleaved_steps"

        visible_text = "\n".join([*op_lines, *text_parts])

        return GenerationResult(
            text=visible_text,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            finish_reason=finish_reason,
            thinking_text="\n".join(thinking_parts),
            thinking_tokens=0,  # subsumed into output_tokens by Anthropic billing
            raw={"interleaved_steps": step + 1},
        )


def _format_op_line(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Render a native tool_use as a text-channel op-code line.

    Lets the existing op-code parser in the harness count and log the same
    way for interleaved-thinking runs as for visible-CoT runs.
    """
    if tool_name == "evict_head":
        return f"e {tool_input['n']}"
    if tool_name == "retrieve_chunk":
        return f"r {tool_input['chunk_id']} {tool_input['offset']} {tool_input['length']}"
    if tool_name == "query_refs":
        return f"q {tool_input['chunk_id']}"
    if tool_name == "annotate_chunk":
        return f"a {tool_input['chunk_id']} {tool_input['tag']}"
    if tool_name == "link_chunks":
        return f"l {tool_input['a']} {tool_input['b']}"
    return f"# unknown tool: {tool_name}"
