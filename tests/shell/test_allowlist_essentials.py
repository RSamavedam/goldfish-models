"""Regression test: critical SWE-bench commands are in the allowlist.

Cloud smoke 3 confirmed git, pytest, and bash are essential for the
model to operate on a real repo. The first sweep failed because none
of them were allowed.
"""

from __future__ import annotations

from rlm_paged.shell.shell_runner import DEFAULT_ALLOWLIST


def test_git_in_default_allowlist():
    assert "git" in DEFAULT_ALLOWLIST


def test_pytest_in_default_allowlist():
    assert "pytest" in DEFAULT_ALLOWLIST


def test_bash_in_default_allowlist():
    """`bash -lc '...'` is a common idiom; needed for things like
    `bash -lc 'source venv/bin/activate && pytest'`."""
    assert "bash" in DEFAULT_ALLOWLIST


def test_chmod_in_default_allowlist():
    """Some test runners need chmod to make scripts executable."""
    assert "chmod" in DEFAULT_ALLOWLIST


def test_tox_and_make_in_default_allowlist():
    """Real Python projects often use tox or make targets for tests."""
    assert "tox" in DEFAULT_ALLOWLIST
    assert "make" in DEFAULT_ALLOWLIST
