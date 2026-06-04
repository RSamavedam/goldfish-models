"""Regression tests for heredoc-aware command splitting.

A smoke trace on SWE-bench showed the model writing:

    python3 - <<'PY'
    import sys, numpy as np
    print("hi")
    PY

The old extractor split each line into a separate command, so the body
lines (`import ...`, `print ...`) hit the allowlist rejection and the
heredoc was never delivered to python. This pins the new behavior:
heredoc body lines stay with their opener as a single command.
"""

from __future__ import annotations

from rlm_paged.shell.extractor import extract_shell_commands


def _wrap(body: str) -> str:
    return f"```bash\n{body}\n```\n"


def test_heredoc_unquoted_terminator_kept_as_one_command():
    body = (
        "cat > notes.txt <<EOF\n"
        "line one\n"
        "line two\n"
        "EOF\n"
    )
    cmds = extract_shell_commands(_wrap(body))
    assert len(cmds) == 1
    assert cmds[0].splitlines()[0].startswith("cat > notes.txt <<EOF")
    assert "line one" in cmds[0]
    assert "line two" in cmds[0]
    assert cmds[0].splitlines()[-1] == "EOF"


def test_heredoc_quoted_terminator_kept_as_one_command():
    body = (
        "python3 - <<'PY'\n"
        "import sys\n"
        "print(\"hi\")\n"
        "PY\n"
    )
    cmds = extract_shell_commands(_wrap(body))
    assert len(cmds) == 1
    assert "import sys" in cmds[0]
    assert "print(\"hi\")" in cmds[0]


def test_heredoc_dash_form_recognized():
    body = (
        "cat <<-EOF\n"
        "\tindented\n"
        "EOF\n"
    )
    cmds = extract_shell_commands(_wrap(body))
    assert len(cmds) == 1


def test_heredoc_followed_by_more_commands():
    body = (
        "cat > a.txt <<EOF\n"
        "x\n"
        "EOF\n"
        "echo done\n"
    )
    cmds = extract_shell_commands(_wrap(body))
    assert len(cmds) == 2
    assert cmds[0].splitlines()[-1] == "EOF"
    assert cmds[1] == "echo done"


def test_heredoc_missing_terminator_accumulates_to_block_end():
    body = (
        "cat > a.txt <<EOF\n"
        "x\n"
        "y\n"
    )
    cmds = extract_shell_commands(_wrap(body))
    assert len(cmds) == 1
    assert "x" in cmds[0]
    assert "y" in cmds[0]


def test_non_heredoc_line_with_arrow_left_left_not_mistaken():
    """A line like `echo foo<<bar` (no whitespace, not at end) shouldn't
    be misdetected. Conservative: regex requires the terminator to be at
    end-of-line."""
    body = "echo 'a<<b'\necho next\n"
    cmds = extract_shell_commands(_wrap(body))
    assert cmds == ["echo 'a<<b'", "echo next"]
