from __future__ import annotations

from rlm_paged.store import BlockStore
from rlm_paged.tools.executor import execute
from rlm_paged.tools.ops import Op


def test_execute_writes_note():
    store = BlockStore()
    result = execute(
        [Op(name="note", args={"tag": "plan"}, body="A note.")],
        store=store, turn=0,
    )
    assert len(result.notes_written) == 1
    written = store.get(result.notes_written[0])
    assert written.text == "A note."
    assert "plan" in written.tags


def test_execute_writes_continue_exactly_once():
    store = BlockStore()
    result = execute(
        [Op(name="continue", body="next step")], store=store, turn=0,
    )
    assert result.has_mandatory_continue
    assert result.continuing_instruction_index is not None


def test_execute_duplicate_continue_errors():
    store = BlockStore()
    result = execute(
        [
            Op(name="continue", body="first"),
            Op(name="continue", body="second"),
        ],
        store=store, turn=0,
    )
    assert result.has_mandatory_continue
    assert any("duplicate `continue`" in e for e in result.errors)


def test_execute_missing_continue_flagged_false():
    store = BlockStore()
    result = execute([Op(name="note", body="just a note")], store=store, turn=0)
    assert result.has_mandatory_continue is False


def test_execute_queues_query_for_next_turn():
    store = BlockStore()
    result = execute(
        [Op(name="query", args={"type": "note", "start": 0, "end": -1})],
        store=store, turn=0,
    )
    assert len(result.pending_queries) == 1
    q = result.pending_queries[0]
    assert q.type == "note" and q.start == 0 and q.end == -1


def test_execute_unknown_op_records_error():
    store = BlockStore()
    result = execute([Op(name="banana")], store=store, turn=0)
    assert any("unknown op" in e for e in result.errors)


def test_execute_empty_body_note_errors():
    store = BlockStore()
    result = execute([Op(name="note", body="   ")], store=store, turn=0)
    assert any("empty note body" in e for e in result.errors)


def test_execute_records_external_call():
    store = BlockStore()
    result = execute(
        [Op(name="call", args={"tool": "echo"}, body="hello")],
        store=store, turn=0,
    )
    assert len(result.external_calls) == 1
    c = result.external_calls[0]
    assert c.tool == "echo"
    assert c.args == "hello"


def test_execute_pipe_currently_errors():
    """pipe is documented as v2; we want a clear error so we notice if
    models actually use it."""
    store = BlockStore()
    result = execute(
        [Op(name="pipe", args={"spec": "(query note 0 -1) -> note"})],
        store=store, turn=0,
    )
    assert any("`pipe` not implemented" in e for e in result.errors)
