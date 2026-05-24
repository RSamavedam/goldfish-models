from __future__ import annotations

import pytest

from rlm_paged.store import ChunkStore
from rlm_paged.tools import (
    ANTHROPIC_TOOLS,
    ToolDispatcher,
    opcode_of,
    structured_to_parsed_op,
)
from rlm_paged.window import ActiveWindow, WindowConfig


def test_anthropic_tools_lineup_matches_opcodes():
    names = {tool["name"] for tool in ANTHROPIC_TOOLS}
    # All five core ops have structured mirrors (`s` similarity search is
    # deferred to Phase 1.5 — not in the structured set yet).
    assert names == {
        "evict_head",
        "retrieve_chunk",
        "query_refs",
        "annotate_chunk",
        "link_chunks",
    }


def test_each_tool_has_a_valid_input_schema():
    for tool in ANTHROPIC_TOOLS:
        assert "name" in tool
        assert "description" in tool
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema


def test_structured_to_parsed_op_evict():
    op = structured_to_parsed_op("evict_head", {"n": 128})
    assert op.code == "e" and op.args == ("128",)


def test_structured_to_parsed_op_retrieve():
    op = structured_to_parsed_op("retrieve_chunk", {"chunk_id": 7, "offset": 0, "length": 64})
    assert op.code == "r" and op.args == ("7", "0", "64")


def test_structured_to_parsed_op_link():
    op = structured_to_parsed_op("link_chunks", {"a": 3, "b": 5})
    assert op.code == "l" and op.args == ("3", "5")


def test_structured_to_parsed_op_unknown_raises():
    with pytest.raises(ValueError, match="unknown structured tool"):
        structured_to_parsed_op("not_a_real_tool", {})


def test_opcode_of_lookup():
    assert opcode_of("evict_head") == "e"
    assert opcode_of("retrieve_chunk") == "r"


def test_structured_op_dispatches_through_existing_dispatcher():
    # End-to-end: structured invocation should reach the same dispatcher
    # the text-channel ops use, with identical effects.
    window = ActiveWindow(WindowConfig(L=64, tail_max=32))
    store = ChunkStore(chunk_size=8)
    (cid,) = store.append(tokens=[10, 11, 12, 13], created_at_step=0, original_position=0)
    dispatcher = ToolDispatcher(window, store)

    op = structured_to_parsed_op(
        "retrieve_chunk", {"chunk_id": cid, "offset": 0, "length": 4}
    )
    result = dispatcher.dispatch(op)
    assert result.ok
    assert result.payload["tokens"] == [10, 11, 12, 13]
    assert window.tail == 4
