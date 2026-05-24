from __future__ import annotations

from rlm_paged.store import BlockStore
from rlm_paged.tools.executor import execute
from rlm_paged.tools.ops import Op, parse_ops


def test_parse_say_op():
    _, ops = parse_ops(
        "say\n"
        "    Hello user, working on it.\n"
    )
    assert len(ops) == 1
    assert ops[0].name == "say"
    assert "Hello user" in ops[0].body


def test_execute_say_writes_assistant_reply():
    store = BlockStore()
    result = execute(
        [Op(name="say", body="Quick update for the user.")],
        store=store, turn=3,
    )
    assert len(result.assistant_replies) == 1
    block = store.get(result.assistant_replies[0])
    assert block.type == "assistant_reply"
    assert block.text == "Quick update for the user."
    assert block.created_at_turn == 3


def test_execute_say_empty_body_errors():
    store = BlockStore()
    result = execute([Op(name="say", body="   ")], store=store, turn=0)
    assert any("empty say body" in e for e in result.errors)
    assert result.assistant_replies == []


def test_say_with_tag():
    store = BlockStore()
    result = execute(
        [Op(name="say", args={"tag": "status"}, body="Working...")],
        store=store, turn=0,
    )
    block = store.get(result.assistant_replies[0])
    assert "status" in block.tags
