"""Regression tests for ShellRunner arg sanitization.

Smoke 4 trace showed:
  - `git diff > /tmp/answer.patch` → "absolute path outside agent root"
  - `cd testrepo` → "executable not in allowlist: cd"

Both fixed; pinning here.
"""

from __future__ import annotations

import pytest

from rlm_paged.shell import AgentFS, ShellRunner


@pytest.fixture
def runner(tmp_path):
    fs = AgentFS.make(tmp_path / "agent", instructions="test")
    yield ShellRunner(root=fs.root, timeout_s=5.0), fs
    fs.cleanup()


def test_tmp_path_accepted_in_args(runner):
    r, _ = runner
    result = r.run("echo hi > /tmp/test_sanitize.txt")
    assert result.returncode == 0
    # Don't read /tmp from the test; just confirm the sanitizer didn't
    # bounce the redirect.
    import os
    if os.path.exists("/tmp/test_sanitize.txt"):
        os.unlink("/tmp/test_sanitize.txt")


def test_etc_still_rejected(runner):
    r, _ = runner
    result = r.run("cat /etc/passwd")
    assert result.returncode != 0
    assert "absolute path" in result.stderr.lower() or "outside" in result.stderr.lower()


def test_home_still_rejected(runner):
    r, _ = runner
    result = r.run("cat /home/some/file")
    assert result.returncode != 0


def test_cd_in_allowlist(runner):
    """`cd foo` shouldn't error even though it's a shell builtin —
    most importantly so the model doesn't waste a turn on a security
    error every time it tries the most common shell idiom."""
    r, fs = runner
    (fs.root / "subdir").mkdir()
    result = r.run("cd subdir")
    assert result.returncode == 0


def test_cd_does_not_persist_between_commands(runner):
    """Document the limitation: cwd doesn't carry over between
    subprocess.run calls. The model must chain `cd && cmd`."""
    r, fs = runner
    (fs.root / "subdir").mkdir()
    (fs.root / "subdir" / "marker.txt").write_text("inside-sub")
    (fs.root / "marker.txt").write_text("outside-sub")
    # Two separate commands: the second sees root's marker, not subdir's.
    r.run("cd subdir")
    out2 = r.run("cat marker.txt")
    assert "outside-sub" in out2.stdout


def test_chained_cd_does_what_user_expects(runner):
    """Chained `cd && cmd` works because the chain is one subprocess."""
    r, fs = runner
    (fs.root / "subdir").mkdir()
    (fs.root / "subdir" / "marker.txt").write_text("inside-sub")
    (fs.root / "marker.txt").write_text("outside-sub")
    out = r.run("cd subdir && cat marker.txt")
    assert "inside-sub" in out.stdout
