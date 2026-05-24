"""Structured tool schemas mirroring the text-channel op codes.

The terse op-code API (`e 128`, `r 42 0 64`) is what the model sees in the
visible text channel — it's token-frugal, designed for L=32-class budgets.
But native tool-calling APIs (Anthropic interleaved thinking, OpenAI
function calling, Gemini function calling) require structured JSON tool
definitions with names and explicit parameter schemas.

This module provides that mirror: same six ops, expressed as Anthropic-
style tool definitions. The same `ToolDispatcher` handles execution —
this just exposes a different surface to the model.

The tradeoff: structured tool calls cost more tokens than op codes
(roughly 20-40 tokens of JSON envelope per call vs. 5-8 for an op). At
L=32 this would consume the entire window per call, so the structured
path is meant for `L >= 256` regimes where the token budget can absorb
JSON overhead. At smaller L we still expect the model to use the op-code
text channel.
"""

from __future__ import annotations

from typing import Any

from rlm_paged.tools.api import ParsedOp


# Anthropic-style tool definitions. The format is also accepted by other
# providers with minor adaptations (OpenAI wraps in {"type": "function",
# "function": {...}}; Gemini uses FunctionDeclaration).
ANTHROPIC_TOOLS: list[dict[str, Any]] = [
    {
        "name": "evict_head",
        "description": (
            "Evict the oldest N tokens from the head of the active window's "
            "middle region. Use before retrieve_chunk when the window is "
            "near full. Returns the actual number of tokens freed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Number of tokens to evict",
                    "minimum": 1,
                },
            },
            "required": ["n"],
        },
    },
    {
        "name": "retrieve_chunk",
        "description": (
            "Retrieve a contiguous span of tokens from a previously stored "
            "chunk. The retrieved tokens are appended to the tail of the "
            "active window. Call evict_head first if the window cannot fit "
            "the requested span."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "integer", "minimum": 0},
                "offset": {
                    "type": "integer",
                    "description": "Offset within the chunk to start from",
                    "minimum": 0,
                },
                "length": {
                    "type": "integer",
                    "description": "Number of tokens to retrieve",
                    "minimum": 1,
                },
            },
            "required": ["chunk_id", "offset", "length"],
        },
    },
    {
        "name": "query_refs",
        "description": (
            "Return the outgoing and incoming references of a chunk. Use to "
            "navigate the chunk graph when you've forgotten where related "
            "content lives."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "integer", "minimum": 0},
            },
            "required": ["chunk_id"],
        },
    },
    {
        "name": "annotate_chunk",
        "description": "Attach a short tag (max 32 chars) to a chunk for later recall.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "integer", "minimum": 0},
                "tag": {"type": "string", "maxLength": 32},
            },
            "required": ["chunk_id", "tag"],
        },
    },
    {
        "name": "link_chunks",
        "description": "Record a directed reference edge from chunk a to chunk b.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "minimum": 0},
                "b": {"type": "integer", "minimum": 0},
            },
            "required": ["a", "b"],
        },
    },
]


# Map structured tool names back to the single-char op codes the dispatcher
# expects. This keeps the dispatcher source-of-truth and makes the
# structured path a thin façade.
_NAME_TO_OPCODE = {
    "evict_head": "e",
    "retrieve_chunk": "r",
    "query_refs": "q",
    "annotate_chunk": "a",
    "link_chunks": "l",
}


def structured_to_parsed_op(name: str, args: dict[str, Any]) -> ParsedOp:
    """Convert a structured tool invocation into a ParsedOp.

    The dispatcher's per-op handlers consume positional string arguments,
    so we serialize the structured args in the order each op expects.
    """
    if name == "evict_head":
        return ParsedOp(code="e", args=(str(args["n"]),))
    if name == "retrieve_chunk":
        return ParsedOp(
            code="r",
            args=(str(args["chunk_id"]), str(args["offset"]), str(args["length"])),
        )
    if name == "query_refs":
        return ParsedOp(code="q", args=(str(args["chunk_id"]),))
    if name == "annotate_chunk":
        return ParsedOp(code="a", args=(str(args["chunk_id"]), str(args["tag"])))
    if name == "link_chunks":
        return ParsedOp(code="l", args=(str(args["a"]), str(args["b"])))
    raise ValueError(f"unknown structured tool: {name!r}")


def opcode_of(name: str) -> str:
    return _NAME_TO_OPCODE[name]
