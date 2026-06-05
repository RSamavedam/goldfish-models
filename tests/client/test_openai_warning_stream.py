"""Tests for the stop-and-resume warning mechanism.

The OpenAI client's generate_with_warnings() should:
  - Stream the response.
  - When emitted-token-count crosses a warn_at_remaining threshold,
    abort the stream and issue a fresh API call with the partial
    output prefilled as an assistant message + a synthetic user
    message containing the ⟪WRAP⟫ banner.
  - Concatenate all partials into one final text.

We mock the OpenAI client because hitting the API in CI is slow + costs $.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

from rlm_paged.client.openai import OpenAIClient


def _make_chunk(text: str, finish_reason: str | None = None, usage=None):
    delta = types.SimpleNamespace(content=text)
    choice = types.SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return types.SimpleNamespace(choices=[choice], usage=usage)


def _make_usage(prompt=10, completion=20, reasoning=0):
    details = types.SimpleNamespace(reasoning_tokens=reasoning)
    return types.SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        completion_tokens_details=details,
    )


class _FakeStream:
    def __init__(self, events):
        self._events = list(events)

    def __iter__(self):
        for e in self._events:
            yield e

    def close(self):
        pass


def test_warning_emitted_when_threshold_crossed():
    client = OpenAIClient("gpt-4o")
    long_text = "x " * 40
    first_events = [
        _make_chunk(long_text),
        _make_chunk(""),
        _make_chunk("", finish_reason=None, usage=_make_usage(completion=40)),
    ]
    second_events = [
        _make_chunk(" continued."),
        _make_chunk("", finish_reason="stop", usage=_make_usage(completion=2)),
    ]
    call_count = {"n": 0}
    call_args: list[dict] = []

    def fake_create(**kwargs):
        call_args.append(kwargs)
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _FakeStream(first_events)
        return _FakeStream(second_events)

    fake_sdk = MagicMock()
    fake_sdk.chat.completions.create.side_effect = fake_create

    with patch.object(client, "_sdk", return_value=fake_sdk):
        result = client.generate_with_warnings(
            "user prompt",
            max_tokens=100,
            system="sys",
            warn_at_remaining=[64],
        )

    assert "continued" in result.text
    assert call_count["n"] == 2
    second_messages = call_args[1]["messages"]
    roles = [m["role"] for m in second_messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert "⟪WRAP" in second_messages[-1]["content"]


def test_no_warning_if_response_short():
    client = OpenAIClient("gpt-4o")
    events = [
        _make_chunk("short answer"),
        _make_chunk("", finish_reason="stop", usage=_make_usage(completion=3)),
    ]
    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        return _FakeStream(events)

    fake_sdk = MagicMock()
    fake_sdk.chat.completions.create.side_effect = fake_create

    with patch.object(client, "_sdk", return_value=fake_sdk):
        result = client.generate_with_warnings(
            "user prompt",
            max_tokens=10000,
            system="sys",
            warn_at_remaining=[64, 32, 16],
        )

    assert result.text == "short answer"
    assert call_count["n"] == 1


def test_wrap_warning_format_escalates():
    assert "wrap up quickly" in OpenAIClient._format_wrap_warning(64)
    assert "finish this thought" in OpenAIClient._format_wrap_warning(32)
    assert "STOP after this sentence" in OpenAIClient._format_wrap_warning(16)
    assert "STOP" in OpenAIClient._format_wrap_warning(8)
