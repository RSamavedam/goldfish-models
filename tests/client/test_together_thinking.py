from __future__ import annotations

from rlm_paged.client import build_client
from rlm_paged.client.together import _split_thinking


def test_split_thinking_extracts_reasoning_block():
    raw = "<think>Let me work this out.\n1+1=2.</think>\n\nFinal answer: 2"
    visible, thinking = _split_thinking(raw)
    assert visible == "Final answer: 2"
    assert thinking == "Let me work this out.\n1+1=2."


def test_split_thinking_no_block_returns_empty_thinking():
    raw = "Just a regular response. Final answer: 5"
    visible, thinking = _split_thinking(raw)
    assert visible == raw
    assert thinking == ""


def test_split_thinking_concatenates_multiple_blocks():
    raw = (
        "<think>First branch.</think>"
        "Some prose.\n"
        "<think>Second branch.</think>"
        "Final answer: X"
    )
    visible, thinking = _split_thinking(raw)
    assert "First branch." in thinking
    assert "Second branch." in thinking
    assert "<think>" not in visible


def test_together_client_factory_and_name():
    c = build_client("together:deepseek-ai/DeepSeek-R1")
    assert c.name == "together:deepseek-ai/DeepSeek-R1"


def test_together_client_visible_thinking_only_for_known_reasoning_models():
    r1 = build_client("together:deepseek-ai/deepseek-r1")
    llama = build_client("together:meta-llama/Llama-3.3-70B-Instruct-Turbo")
    assert r1.supports_visible_thinking is True
    assert llama.supports_visible_thinking is False
