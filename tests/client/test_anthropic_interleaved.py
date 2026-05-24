"""Unit tests for the Anthropic interleaved-thinking dispatcher path.

We mock the Anthropic SDK at the `_sdk()` level so the test doesn't hit
the live API. The point of these tests isn't to verify Anthropic's API —
it's to verify our dispatcher wiring: tool calls round-trip, signed
thinking blocks get preserved in the message history, op-code lines get
generated for the harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from rlm_paged.client.anthropic import AnthropicClient
from rlm_paged.tools import ANTHROPIC_TOOLS


# -- Fake Anthropic SDK objects ---------------------------------------- #


@dataclass
class _FakeBlock:
    type: str
    # one of: text, thinking (+signature), tool_use (+id, name, input)
    text: str | None = None
    thinking: str | None = None
    signature: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None


@dataclass
class _FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class _FakeResponse:
    id: str
    model: str
    content: list[_FakeBlock]
    usage: _FakeUsage
    stop_reason: str


class _FakeMessages:
    def __init__(self, scripted: list[_FakeResponse]) -> None:
        self.scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        return self.scripted.pop(0)


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.messages = _FakeMessages(responses)


# -- Tests ------------------------------------------------------------- #


def _make_client_with_responses(responses: list[_FakeResponse]) -> AnthropicClient:
    client = AnthropicClient(
        "claude-opus-4-7",
        thinking_budget=1024,
        interleaved_thinking=True,
        tools=ANTHROPIC_TOOLS,
        api_key="test-key",
    )
    client._client = _FakeClient(responses)
    return client


def test_supports_interleaved_thinking_requires_all_three():
    plain = AnthropicClient("claude-opus-4-7", api_key="k")
    assert plain.supports_interleaved_thinking is False

    thinking_only = AnthropicClient("claude-opus-4-7", thinking_budget=1024, api_key="k")
    assert thinking_only.supports_interleaved_thinking is False

    no_tools = AnthropicClient(
        "claude-opus-4-7", thinking_budget=1024, interleaved_thinking=True, api_key="k"
    )
    assert no_tools.supports_interleaved_thinking is False

    full = _make_client_with_responses([])
    assert full.supports_interleaved_thinking is True


def test_generate_with_dispatcher_single_step_no_tools():
    """Model thinks, then emits final text — no tool calls."""
    resp = _FakeResponse(
        id="msg_1",
        model="claude-opus-4-7",
        content=[
            _FakeBlock(type="thinking", thinking="Let me work it out.", signature="sig1"),
            _FakeBlock(type="text", text="The answer is 42."),
        ],
        usage=_FakeUsage(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )
    client = _make_client_with_responses([resp])
    dispatched: list[tuple[str, dict]] = []

    def dispatch(name: str, inp: dict) -> dict:
        dispatched.append((name, inp))
        return {"ok": True}

    result = client.generate_with_dispatcher(
        "What is 6 * 7?",
        max_tokens=512,
        system=None,
        dispatch=dispatch,
    )

    assert dispatched == []
    assert "answer is 42" in result.text
    assert "work it out" in result.thinking_text
    assert result.input_tokens == 10
    assert result.output_tokens == 20


def test_generate_with_dispatcher_routes_tool_calls():
    """Model emits thinking + tool_use, then thinking + final text."""
    first = _FakeResponse(
        id="msg_1",
        model="claude-opus-4-7",
        content=[
            _FakeBlock(type="thinking", thinking="I need more room.", signature="s1"),
            _FakeBlock(
                type="tool_use",
                id="tool_use_1",
                name="evict_head",
                input={"n": 64},
            ),
        ],
        usage=_FakeUsage(input_tokens=15, output_tokens=25),
        stop_reason="tool_use",
    )
    second = _FakeResponse(
        id="msg_2",
        model="claude-opus-4-7",
        content=[
            _FakeBlock(type="thinking", thinking="Now I can continue.", signature="s2"),
            _FakeBlock(type="text", text="Final answer: 42"),
        ],
        usage=_FakeUsage(input_tokens=40, output_tokens=15),
        stop_reason="end_turn",
    )
    client = _make_client_with_responses([first, second])
    dispatched: list[tuple[str, dict]] = []

    def dispatch(name: str, inp: dict) -> dict:
        dispatched.append((name, inp))
        return {"ok": True, "freed": inp["n"]}

    result = client.generate_with_dispatcher(
        "Solve it.",
        max_tokens=512,
        system=None,
        dispatch=dispatch,
    )

    # The dispatcher was called once for the evict.
    assert dispatched == [("evict_head", {"n": 64})]
    # The visible text accumulates an op-code line for metrics + the final answer.
    assert "e 64" in result.text
    assert "Final answer: 42" in result.text
    # Both thinking blocks accumulated.
    assert "more room" in result.thinking_text
    assert "continue" in result.thinking_text
    # Token totals are summed across the two API calls.
    assert result.input_tokens == 55
    assert result.output_tokens == 40


def test_signed_thinking_blocks_are_round_tripped():
    """The follow-up message must echo back thinking blocks with signatures."""
    first = _FakeResponse(
        id="msg_1",
        model="claude-opus-4-7",
        content=[
            _FakeBlock(
                type="thinking", thinking="step 1", signature="signed-1"
            ),
            _FakeBlock(
                type="tool_use",
                id="tu_1",
                name="evict_head",
                input={"n": 32},
            ),
        ],
        usage=_FakeUsage(input_tokens=10, output_tokens=10),
        stop_reason="tool_use",
    )
    second = _FakeResponse(
        id="msg_2",
        model="claude-opus-4-7",
        content=[_FakeBlock(type="text", text="done")],
        usage=_FakeUsage(input_tokens=5, output_tokens=5),
        stop_reason="end_turn",
    )
    client = _make_client_with_responses([first, second])

    def dispatch(name: str, inp: dict) -> dict:
        return {"ok": True}

    client.generate_with_dispatcher(
        "go", max_tokens=512, system=None, dispatch=dispatch
    )

    # The second API call's messages should contain assistant content
    # echoing back the signed thinking block.
    second_call_messages = client._client.messages.calls[1]["messages"]
    assistant_msg = second_call_messages[1]
    assert assistant_msg["role"] == "assistant"
    thinking_block = next(
        b for b in assistant_msg["content"] if b["type"] == "thinking"
    )
    assert thinking_block["signature"] == "signed-1"
    assert thinking_block["thinking"] == "step 1"


def test_max_interleaved_steps_caps_loops():
    """If the model never stops calling tools, we bail at max_interleaved_steps."""
    looping_response = _FakeResponse(
        id="msg",
        model="claude-opus-4-7",
        content=[
            _FakeBlock(type="thinking", thinking="loop", signature="s"),
            _FakeBlock(
                type="tool_use",
                id="tu",
                name="evict_head",
                input={"n": 8},
            ),
        ],
        usage=_FakeUsage(input_tokens=5, output_tokens=5),
        stop_reason="tool_use",
    )
    # Provide enough scripted responses for the cap (default 8 steps).
    client = AnthropicClient(
        "claude-opus-4-7",
        thinking_budget=1024,
        interleaved_thinking=True,
        tools=ANTHROPIC_TOOLS,
        api_key="test-key",
        max_interleaved_steps=3,
    )
    client._client = _FakeClient([looping_response, looping_response, looping_response])

    def dispatch(name: str, inp: dict) -> dict:
        return {"ok": True}

    result = client.generate_with_dispatcher(
        "go", max_tokens=512, system=None, dispatch=dispatch
    )
    assert result.finish_reason == "max_interleaved_steps"
    assert result.raw["interleaved_steps"] == 3


def test_generate_with_dispatcher_rejects_clients_without_config():
    """Plain (non-interleaved) AnthropicClient should refuse the path."""
    plain = AnthropicClient("claude-opus-4-7", api_key="k")
    with pytest.raises(RuntimeError, match="interleaved_thinking"):
        plain.generate_with_dispatcher(
            "x", max_tokens=10, system=None, dispatch=lambda n, i: {}
        )
