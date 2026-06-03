from __future__ import annotations

import pytest

from rlm_paged.shell import AgentFS


@pytest.fixture
def fs(tmp_path):
    f = AgentFS.make(tmp_path / "agent", instructions="solve this task")
    yield f
    f.cleanup()


def test_init_creates_expected_layout(fs):
    root = fs.root
    assert root.is_dir()
    assert fs.instructions_path.read_text() == "solve this task"
    assert (root / "stdin").is_dir()
    assert (root / "stdout").is_dir()
    assert (root / "user_output").is_dir()
    assert fs.history_path.exists()
    assert fs.history_path.read_text() == ""


def test_instructions_is_read_only(fs):
    import stat as st_mod

    mode = fs.instructions_path.stat().st_mode
    # Owner write bit should not be set.
    assert not (mode & st_mod.S_IWUSR)


def test_append_history_appends_in_order(fs):
    fs.append_history("first\n")
    fs.append_history("second\n")
    assert fs.history_path.read_text() == "first\nsecond\n"


def test_export_string_writes_to_user_output(fs):
    dest = fs.export("the answer is 42", is_string=True)
    assert dest.parent == fs.user_output_dir
    assert dest.read_text() == "the answer is 42"


def test_export_string_creates_unique_names(fs):
    a = fs.export("first", is_string=True)
    b = fs.export("second", is_string=True)
    assert a != b
    assert a.read_text() == "first"
    assert b.read_text() == "second"


def test_export_file_copies_into_user_output(fs):
    src = fs.root / "notes.txt"
    src.write_text("my notes")
    dest = fs.export("notes.txt")
    assert dest.parent == fs.user_output_dir
    assert dest.read_text() == "my notes"


def test_export_file_dedupes_on_collision(fs):
    (fs.root / "notes.txt").write_text("v1")
    fs.export("notes.txt")
    (fs.root / "notes.txt").write_text("v2")
    second = fs.export("notes.txt")
    assert second.name != "notes.txt"
    assert second.read_text() == "v2"


def test_export_rejects_absolute_path(fs):
    with pytest.raises(ValueError, match="absolute"):
        fs.export("/etc/passwd")


def test_export_rejects_parent_traversal(fs):
    with pytest.raises(ValueError, match="parent-traversal"):
        fs.export("../escape.txt")


def test_export_missing_file_raises(fs):
    with pytest.raises(FileNotFoundError):
        fs.export("nope.txt")


def test_list_user_outputs_orders_by_mtime(fs):
    import os
    import time

    paths = []
    for i, name in enumerate(["a.txt", "b.txt", "c.txt"]):
        p = fs.user_output_dir / name
        p.write_text(str(i))
        paths.append(p)
        t = time.time() + i
        os.utime(p, (t, t))
    ordered = fs.list_user_outputs()
    assert [p.name for p in ordered] == ["a.txt", "b.txt", "c.txt"]


def test_read_history_tail_returns_tail(fs):
    fs.append_history("a" * 100)
    fs.append_history("b" * 100)
    tail = fs.read_history_tail(max_chars=50)
    assert len(tail) == 50
    assert tail == "b" * 50


def test_read_history_tail_short_file_returns_all(fs):
    fs.append_history("short content")
    assert fs.read_history_tail(max_chars=1000) == "short content"


def test_cleanup_removes_directory(fs):
    root = fs.root
    assert root.exists()
    fs.cleanup()
    assert not root.exists()
