from __future__ import annotations

import pytest

from rlm_paged.store import BLOCK_TYPES, BlockStore


def test_block_types_canonical():
    assert BLOCK_TYPES == (
        "user_message",
        "assistant_reply",
        "observation",
        "note",
        "continuing_instruction",
    )


def test_append_assigns_monotonic_indices_per_type():
    s = BlockStore()
    b1 = s.append("note", "first", created_at_turn=0)
    b2 = s.append("note", "second", created_at_turn=0)
    b3 = s.append("user_message", "hi", created_at_turn=-1)
    assert (b1.type, b1.index, b1.global_index) == ("note", 0, 0)
    assert (b2.type, b2.index, b2.global_index) == ("note", 1, 1)
    assert (b3.type, b3.index, b3.global_index) == ("user_message", 0, 2)


def test_append_unknown_type_raises():
    s = BlockStore()
    with pytest.raises(ValueError, match="unknown block type"):
        s.append("garbage", "x", created_at_turn=0)


def test_query_basic_range():
    s = BlockStore()
    for i in range(5):
        s.append("note", f"n{i}", created_at_turn=0)
    out = s.query("note", start=1, end=3)
    assert [b.text for b in out] == ["n1", "n2", "n3"]


def test_query_open_ended():
    s = BlockStore()
    for i in range(3):
        s.append("note", f"n{i}", created_at_turn=0)
    assert [b.text for b in s.query("note")] == ["n0", "n1", "n2"]
    assert [b.text for b in s.query("note", start=1)] == ["n1", "n2"]
    assert [b.text for b in s.query("note", end=0)] == ["n0"]


def test_query_negative_indices():
    s = BlockStore()
    for i in range(5):
        s.append("note", f"n{i}", created_at_turn=0)
    assert [b.text for b in s.query("note", start=-2)] == ["n3", "n4"]
    assert [b.text for b in s.query("note", end=-1)] == ["n0", "n1", "n2", "n3", "n4"]


def test_query_tag_filter():
    s = BlockStore()
    s.append("note", "a", created_at_turn=0, tags=["plan"])
    s.append("note", "b", created_at_turn=0, tags=["scratch"])
    s.append("note", "c", created_at_turn=0, tags=["plan", "summary"])
    assert [b.text for b in s.query("note", tag="plan")] == ["a", "c"]


def test_query_empty_returns_empty():
    s = BlockStore()
    assert s.query("note") == []


def test_query_out_of_range_clamps():
    s = BlockStore()
    for i in range(3):
        s.append("note", f"n{i}", created_at_turn=0)
    # start > last legal index → empty
    assert s.query("note", start=99, end=200) == []


def test_query_updates_access_bookkeeping():
    s = BlockStore()
    s.append("note", "x", created_at_turn=0)
    blocks = s.query("note", at_turn=7)
    assert blocks[0].access_count == 1
    assert blocks[0].last_accessed_turn == 7


def test_link_is_bidirectional_and_idempotent():
    s = BlockStore()
    a = s.append("note", "a", created_at_turn=0)
    b = s.append("note", "b", created_at_turn=0)
    s.link(a.global_index, b.global_index)
    s.link(a.global_index, b.global_index)
    assert s.get(a.global_index).outgoing_refs == [b.global_index]
    assert s.get(b.global_index).incoming_refs == [a.global_index]


def test_tag_appends_and_dedups():
    s = BlockStore()
    b = s.append("note", "x", created_at_turn=0)
    s.tag(b.global_index, "plan")
    s.tag(b.global_index, "plan")
    s.tag(b.global_index, "summary")
    assert b.tags == ["plan", "summary"]


def test_stats_reports_count_and_last_index():
    s = BlockStore()
    s.append("note", "a", created_at_turn=0)
    s.append("note", "b", created_at_turn=0)
    s.append("observation", "o", created_at_turn=0)
    stats = s.stats()
    # Now includes unread fields; both notes are unread (never queried).
    assert stats["note"] == {
        "count": 2, "last_index": 1, "unread": 2, "earliest_unread": 0,
    }
    assert stats["observation"] == {
        "count": 1, "last_index": 0, "unread": 1, "earliest_unread": 0,
    }
    assert stats["user_message"] == {
        "count": 0, "last_index": -1, "unread": 0, "earliest_unread": -1,
    }
