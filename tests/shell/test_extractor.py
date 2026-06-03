from __future__ import annotations

from rlm_paged.shell import extract_blocks, extract_shell_commands


def test_extract_blocks_basic():
    text = (
        "intro\n"
        "```bash\n"
        "echo hi\n"
        "```\n"
        "after\n"
    )
    blocks = extract_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].language == "bash"
    assert blocks[0].body.strip() == "echo hi"


def test_extract_blocks_preserves_order():
    text = (
        "first\n"
        "```bash\n"
        "echo one\n"
        "```\n"
        "middle\n"
        "```sh\n"
        "echo two\n"
        "```\n"
        "last\n"
    )
    blocks = extract_blocks(text)
    assert len(blocks) == 2
    assert "one" in blocks[0].body
    assert "two" in blocks[1].body


def test_extract_blocks_bare_fence_treated_as_shell():
    text = "```\necho hi\n```\n"
    blocks = extract_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].language == ""
    cmds = extract_shell_commands(text)
    assert cmds == ["echo hi"]


def test_extract_blocks_missing_close_treats_rest_as_body():
    text = "```bash\necho hi\nno close fence here"
    blocks = extract_blocks(text)
    assert len(blocks) == 1
    assert "no close fence" in blocks[0].body


def test_extract_shell_commands_strips_comments_and_blanks():
    text = (
        "```bash\n"
        "# this is a comment\n"
        "\n"
        "echo first\n"
        "\n"
        "# another comment\n"
        "echo second\n"
        "```\n"
    )
    assert extract_shell_commands(text) == ["echo first", "echo second"]


def test_extract_shell_commands_handles_line_continuation():
    text = (
        "```bash\n"
        "echo this is \\\n"
        "all one line\n"
        "echo second\n"
        "```\n"
    )
    cmds = extract_shell_commands(text)
    assert cmds[0] == "echo this is all one line"
    assert cmds[1] == "echo second"


def test_extract_shell_commands_skips_non_shell_blocks():
    text = (
        "```python\n"
        "print('this is python')\n"
        "```\n"
        "```bash\n"
        "echo hi\n"
        "```\n"
    )
    assert extract_shell_commands(text) == ["echo hi"]


def test_multiple_blocks_preserve_document_order():
    text = (
        "```bash\necho A\n```\n"
        "prose\n"
        "```bash\necho B\n```\n"
        "more prose\n"
        "```bash\necho C\n```\n"
    )
    assert extract_shell_commands(text) == ["echo A", "echo B", "echo C"]
