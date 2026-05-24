from __future__ import annotations

from rlm_paged.store import ChunkStore
from rlm_paged.tools import ToolDispatcher, parse_op
from rlm_paged.window import ActiveWindow, WindowConfig


def make_dispatcher(L: int = 128, tail_max: int = 64):
    win = ActiveWindow(WindowConfig(L=L, tail_max=tail_max))
    store = ChunkStore(chunk_size=8)
    return ToolDispatcher(win, store), win, store


def test_parse_op_recognizes_known_opcodes():
    op = parse_op("r 7 0 8")
    assert op is not None
    assert op.code == "r" and op.args == ("7", "0", "8")


def test_parse_op_returns_none_for_free_text():
    assert parse_op("thinking about the next step") is None
    assert parse_op("") is None
    assert parse_op("xyz 1 2 3") is None


def test_evict_then_retrieve_roundtrip():
    disp, win, store = make_dispatcher(L=32, tail_max=16)
    # Pre-populate store with a chunk we can pull back.
    (cid,) = store.append(tokens=[100, 101, 102, 103], created_at_step=0, original_position=0)

    # Pretend the middle already has some old stuff.
    win.append_tail(12)
    win.freeze_tail_into_middle()
    assert win.middle == 12

    # Evict 4 from head.
    res = disp.dispatch(parse_op("e 4"))
    assert res.ok and res.payload == {"freed": 4}
    assert win.middle == 8

    # Retrieve a 4-token slice from the stored chunk.
    res = disp.dispatch(parse_op("r 0 0 4"))
    assert res.ok
    assert res.payload["tokens"] == [100, 101, 102, 103]
    assert win.tail == 4


def test_retrieve_when_window_full_returns_error():
    disp, win, store = make_dispatcher(L=8, tail_max=4)
    (cid,) = store.append(tokens=[1, 2, 3, 4], created_at_step=0, original_position=0)
    win.append_tail(4)
    win.freeze_tail_into_middle()
    win.append_tail(4)  # window now full
    res = disp.dispatch(parse_op("r 0 0 4"))
    assert not res.ok
    assert "window full" in (res.error or "")


def test_link_and_query_refs():
    disp, _, store = make_dispatcher()
    ids = store.append(tokens=[1, 2, 3, 4, 5, 6, 7, 8, 9], created_at_step=0, original_position=0)
    a, b = ids[0], ids[1]
    assert disp.dispatch(parse_op(f"l {a} {b}")).ok
    res = disp.dispatch(parse_op(f"q {a}"))
    assert res.ok and res.payload == {"out": [b], "in": []}


def test_annotate_attaches_tag():
    disp, _, store = make_dispatcher()
    (cid,) = store.append(tokens=[1, 2, 3], created_at_step=0, original_position=0)
    assert disp.dispatch(parse_op(f"a {cid} plan")).ok
    assert "plan" in store.get(cid).tags


def test_unknown_op_returns_error():
    disp, _, _ = make_dispatcher()
    fake = parse_op("e 1")
    fake.code = "z"  # type: ignore[misc]
    res = disp.dispatch(fake)
    assert not res.ok
    assert "unknown op" in (res.error or "")
