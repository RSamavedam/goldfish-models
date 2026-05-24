"""Execute parsed ops against the BlockStore and accumulate side effects.

The harness owns the execution loop. This module is a pure function on
(store, ops, current_turn) -> ExecutionResult: it writes new blocks
where required, *queues* retrievals (the harness performs them on the
next turn's assembly), and emits external-tool-call requests for the
harness to dispatch.

The executor does NOT enforce L. Budget enforcement is the harness's
job because it depends on rendering and tokenization. The executor's
output is a structured description of what the model asked for; the
harness can truncate, reject, or scale as needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rlm_paged.store.block_store import BlockStore
from rlm_paged.tools.ops import Op


@dataclass
class QueuedQuery:
    """A retrieval the model requested for the next turn's input."""

    type: str
    start: int | None = None
    end: int | None = None
    tag: str | None = None
    issued_by_op_idx: int = -1


@dataclass
class ExternalCall:
    """An external (non-memory) tool call the harness will execute."""

    tool: str
    args: str
    issued_by_op_idx: int = -1


@dataclass
class ExecutionResult:
    """Side effects produced by executing one turn's response."""

    notes_written: list[int] = field(default_factory=list)        # global indices
    continuing_instruction_index: int | None = None               # global index
    pending_queries: list[QueuedQuery] = field(default_factory=list)
    external_calls: list[ExternalCall] = field(default_factory=list)
    assistant_replies: list[int] = field(default_factory=list)    # global indices
    errors: list[str] = field(default_factory=list)
    has_mandatory_continue: bool = False


def execute(
    ops: list[Op],
    *,
    store: BlockStore,
    turn: int,
) -> ExecutionResult:
    result = ExecutionResult()
    seen_continue = False

    for idx, op in enumerate(ops):
        try:
            if op.name == "note":
                if not op.body.strip():
                    result.errors.append(f"op {idx}: empty note body")
                    continue
                block = store.append(
                    "note", op.body, created_at_turn=turn,
                    tags=_tag_list(op.args.get("tag")),
                )
                result.notes_written.append(block.global_index)

            elif op.name == "continue":
                if seen_continue:
                    result.errors.append(f"op {idx}: duplicate `continue` op")
                    continue
                if not op.body.strip():
                    result.errors.append(f"op {idx}: empty continue body")
                    continue
                block = store.append(
                    "continuing_instruction", op.body, created_at_turn=turn,
                )
                result.continuing_instruction_index = block.global_index
                seen_continue = True

            elif op.name == "query":
                type_name = op.args.get("type")
                if not isinstance(type_name, str):
                    result.errors.append(f"op {idx}: missing query type")
                    continue
                start = op.args.get("start")
                end = op.args.get("end")
                if start is not None and not isinstance(start, int):
                    result.errors.append(
                        f"op {idx}: query start not int: {start!r}"
                    )
                    continue
                if end is not None and not isinstance(end, int):
                    result.errors.append(f"op {idx}: query end not int: {end!r}")
                    continue
                result.pending_queries.append(
                    QueuedQuery(
                        type=type_name,
                        start=start,
                        end=end,
                        tag=op.args.get("tag"),
                        issued_by_op_idx=idx,
                    )
                )

            elif op.name == "pipe":
                # v1 implementation: minimal — record as an error so we
                # learn whether models actually try `pipe` in practice.
                # If they do, implement properly.
                result.errors.append(
                    f"op {idx}: `pipe` not implemented in v1; "
                    "use explicit query + note"
                )

            elif op.name == "call":
                tool = op.args.get("tool")
                if not isinstance(tool, str):
                    result.errors.append(f"op {idx}: missing call tool name")
                    continue
                result.external_calls.append(
                    ExternalCall(
                        tool=tool,
                        args=op.body,
                        issued_by_op_idx=idx,
                    )
                )

            elif op.name == "say":
                # Surface a message to the user. The harness can also pipe
                # it to a side channel (stdout, websocket, etc.) by reading
                # `result.assistant_replies` and looking up the block text.
                if not op.body.strip():
                    result.errors.append(f"op {idx}: empty say body")
                    continue
                block = store.append(
                    "assistant_reply",
                    op.body,
                    created_at_turn=turn,
                    tags=_tag_list(op.args.get("tag")),
                )
                result.assistant_replies.append(block.global_index)

            else:
                result.errors.append(f"op {idx}: unknown op `{op.name}`")
        except Exception as exc:
            result.errors.append(
                f"op {idx} ({op.name}): {type(exc).__name__}: {exc}"
            )

    result.has_mandatory_continue = seen_continue
    return result


def _tag_list(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return [str(t) for t in raw]
    return [str(raw)]
