from __future__ import annotations

from rlm_paged.store import BlockStore


def test_fresh_store_has_no_unread():
    s = BlockStore()
    assert s.unread_count("user_message") == 0
    assert s.earliest_unread_index("user_message") is None


def test_append_marks_block_unread():
    s = BlockStore()
    s.append("user_message", "hi", created_at_turn=-1)
    assert s.unread_count("user_message") == 1
    assert s.earliest_unread_index("user_message") == 0


def test_query_advances_cursor_through_returned_range():
    s = BlockStore()
    for i in range(5):
        s.append("user_message", f"m{i}", created_at_turn=-1)
    # Query the first two; cursor should advance past them.
    s.query("user_message", start=0, end=1)
    assert s.unread_count("user_message") == 3
    assert s.earliest_unread_index("user_message") == 2


def test_querying_oldest_only_does_not_skip_ahead():
    """Reading user_message:0 should leave user_message:1+ unread."""
    s = BlockStore()
    s.append("user_message", "a", created_at_turn=-1)
    s.append("user_message", "b", created_at_turn=-1)
    s.query("user_message", start=0, end=0)
    assert s.unread_count("user_message") == 1
    assert s.earliest_unread_index("user_message") == 1


def test_querying_newest_advances_cursor_past_skipped_blocks():
    """If the model reads only block 4 but skipped 0-3, the cursor still
    moves past 4. (Effectively: 'I've at least seen up to here.')"""
    s = BlockStore()
    for i in range(5):
        s.append("user_message", f"m{i}", created_at_turn=-1)
    s.query("user_message", start=4, end=4)
    assert s.unread_count("user_message") == 0
    assert s.earliest_unread_index("user_message") is None


def test_repeated_query_does_not_change_cursor():
    s = BlockStore()
    s.append("user_message", "x", created_at_turn=-1)
    s.query("user_message")
    s.query("user_message")
    assert s.unread_count("user_message") == 0


def test_appending_after_query_creates_new_unread():
    s = BlockStore()
    s.append("user_message", "first", created_at_turn=-1)
    s.query("user_message")
    assert s.unread_count("user_message") == 0
    s.append("user_message", "second", created_at_turn=0)
    assert s.unread_count("user_message") == 1
    assert s.earliest_unread_index("user_message") == 1


def test_unread_per_type_independent():
    s = BlockStore()
    s.append("user_message", "u", created_at_turn=0)
    s.append("note", "n", created_at_turn=0)
    s.query("user_message")
    assert s.unread_count("user_message") == 0
    assert s.unread_count("note") == 1
