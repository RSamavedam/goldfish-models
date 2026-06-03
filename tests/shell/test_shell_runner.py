from __future__ import annotations

import pytest

from rlm_paged.shell import AgentFS, ShellRunner


@pytest.fixture
def runner(tmp_path):
    fs = AgentFS.make(tmp_path / "agent", instructions="test")
    yield ShellRunner(root=fs.root, timeout_s=3.0), fs
    fs.cleanup()


def test_simple_echo(runner):
    r, _ = runner
    out = r.run("echo hello")
    assert out.returncode == 0
    assert out.stdout.strip() == "hello"
    assert out.stderr == ""


def test_pipeline_echo_grep(runner):
    r, _ = runner
    out = r.run("echo 'hello world' | grep world")
    assert out.returncode == 0
    assert "world" in out.stdout


def test_write_and_read_file(runner):
    r, fs = runner
    a = r.run("echo data > notes.txt")
    assert a.returncode == 0
    b = r.run("cat notes.txt")
    assert b.stdout.strip() == "data"
    assert (fs.root / "notes.txt").read_text().strip() == "data"


def test_rejects_rm_with_absolute_path(runner):
    r, _ = runner
    out = r.run("rm -rf /")
    assert out.returncode != 0
    assert "absolute path" in out.stderr.lower() or "not allowed" in out.stderr.lower()


def test_rejects_unknown_executable(runner):
    r, _ = runner
    out = r.run("curl https://example.com")
    assert out.returncode != 0
    assert "not in allowlist" in out.stderr.lower()


def test_rejects_parent_traversal_in_arg(runner):
    r, _ = runner
    out = r.run("cat ../../etc/passwd")
    assert out.returncode != 0
    assert ".." in out.stderr


def test_rejects_absolute_path_outside_root(runner):
    r, _ = runner
    out = r.run("cat /etc/passwd")
    assert out.returncode != 0
    assert (
        "absolute" in out.stderr.lower()
        or "outside" in out.stderr.lower()
    )


def test_absolute_path_inside_root_allowed(runner):
    r, fs = runner
    (fs.root / "inside.txt").write_text("ok")
    out = r.run(f"cat {fs.root}/inside.txt")
    assert out.returncode == 0
    assert out.stdout.strip() == "ok"


def test_intercepts_done(runner):
    r, _ = runner
    out = r.run("done")
    assert out.intercepted_action == "done"
    assert out.returncode == 0


def test_intercepts_exit(runner):
    r, _ = runner
    out = r.run("exit")
    assert out.intercepted_action == "exit"


def test_intercepts_export(runner):
    r, _ = runner
    out = r.run("export answer.txt")
    assert out.intercepted_action == "export"


def test_timeout_returns_124(runner):
    r, _ = runner
    out = r.run("python3 -c 'import time; time.sleep(10)'")
    assert out.timed_out is True
    assert out.returncode == 124


def test_python3_inline_works(runner):
    r, _ = runner
    out = r.run("python3 -c 'print(sum(range(11)))'")
    assert out.returncode == 0
    assert out.stdout.strip() == "55"


def test_writes_stay_inside_root(runner):
    r, fs = runner
    out = r.run("echo bad > ../../escape.txt")
    assert out.returncode != 0  # parent traversal rejection


def test_max_output_truncation(tmp_path):
    fs = AgentFS.make(tmp_path / "agent", instructions="x")
    try:
        r = ShellRunner(root=fs.root, timeout_s=5.0, max_output_bytes=64)
        out = r.run("python3 -c 'print(\"a\" * 1000)'")
        assert len(out.stdout) <= 64
    finally:
        fs.cleanup()
