from __future__ import annotations

from rlm_paged.store import ChunkStore


def test_append_splits_into_chunk_sized_pieces():
    store = ChunkStore(chunk_size=4)
    ids = store.append(
        tokens=[10, 11, 12, 13, 14, 15, 16, 17, 18],
        created_at_step=0,
        original_position=100,
    )
    assert ids == [0, 1, 2]
    assert store.get(0).tokens == [10, 11, 12, 13]
    assert store.get(1).tokens == [14, 15, 16, 17]
    assert store.get(2).tokens == [18]
    assert store.get(2).original_position == 108


def test_retrieve_updates_access_bookkeeping():
    store = ChunkStore(chunk_size=8)
    (cid,) = store.append(tokens=[1, 2, 3, 4, 5], created_at_step=0, original_position=0)
    tokens = store.retrieve(cid, offset=1, length=3, at_step=42)
    assert tokens == [2, 3, 4]
    chunk = store.get(cid)
    assert chunk.access_count == 1
    assert chunk.last_accessed_step == 42


def test_retrieve_out_of_range_raises():
    store = ChunkStore(chunk_size=8)
    (cid,) = store.append(tokens=[1, 2, 3], created_at_step=0, original_position=0)
    try:
        store.retrieve(cid, offset=2, length=5, at_step=0)
    except IndexError:
        pass
    else:
        raise AssertionError("expected IndexError")


def test_link_is_bidirectional_and_idempotent():
    store = ChunkStore(chunk_size=4)
    ids = store.append(tokens=[1, 2, 3, 4, 5, 6, 7, 8], created_at_step=0, original_position=0)
    a, b = ids[0], ids[1]
    store.link(a, b)
    store.link(a, b)  # idempotent
    out, inc = store.refs(a)
    assert out == [b]
    assert inc == []
    out_b, inc_b = store.refs(b)
    assert out_b == []
    assert inc_b == [a]


def test_annotate_unique_tags():
    store = ChunkStore(chunk_size=4)
    (cid,) = store.append(tokens=[1, 2], created_at_step=0, original_position=0)
    store.annotate(cid, "plan")
    store.annotate(cid, "plan")
    store.annotate(cid, "todo")
    assert store.get(cid).tags == ["plan", "todo"]
