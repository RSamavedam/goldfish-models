from __future__ import annotations

from rlm_paged.tools.ops import (
    SCRATCH_CLOSE,
    SCRATCH_OPEN,
    extract_scratch,
    parse_ops,
)


def test_extract_scratch_basic():
    text = f"{SCRATCH_OPEN}thinking{SCRATCH_CLOSE}\n\nquery note 0 -1\n"
    scratch, rest = extract_scratch(text)
    assert scratch == "thinking"
    assert "query note" in rest


def test_extract_scratch_missing_open_returns_empty():
    scratch, rest = extract_scratch("no scratch here\n")
    assert scratch == ""
    assert rest == "no scratch here\n"


def test_extract_scratch_unclosed_swallows_everything():
    """Unclosed scratch tag => model burned its budget on thinking."""
    text = f"{SCRATCH_OPEN}they forgot to close it"
    scratch, rest = extract_scratch(text)
    assert "forgot to close" in scratch
    assert rest == ""


def test_parse_ops_single_line_quoted():
    scratch, ops = parse_ops('note "quick thought."')
    assert scratch == ""
    assert len(ops) == 1
    assert ops[0].name == "note"
    assert ops[0].body == "quick thought."


def test_parse_ops_multi_line_indented_body():
    text = (
        "note tag=plan\n"
        "    line one\n"
        "    line two\n"
        "\n"
        "continue\n"
        "    next turn\n"
    )
    _, ops = parse_ops(text)
    assert [o.name for o in ops] == ["note", "continue"]
    assert ops[0].args == {"tag": "plan"}
    assert ops[0].body == "line one\nline two"
    assert ops[1].body == "next turn"


def test_parse_ops_query_positional_args():
    _, ops = parse_ops("query note 0 -1 tag=plan\n")
    assert len(ops) == 1
    q = ops[0]
    assert q.name == "query"
    assert q.args["type"] == "note"
    assert q.args["start"] == 0
    assert q.args["end"] == -1
    assert q.args["tag"] == "plan"


def test_parse_ops_drops_unparseable_lines():
    _, ops = parse_ops(
        "garbage line 1\n"
        "note \"ok\"\n"
        "another garbage line\n"
    )
    assert [o.name for o in ops] == ["note"]


def test_parse_ops_call_with_body():
    text = (
        "call code_exec\n"
        "    for i in range(5):\n"
        "        print(i)\n"
    )
    _, ops = parse_ops(text)
    assert ops[0].name == "call"
    assert ops[0].args.get("tool") == "code_exec"
    assert "range(5)" in ops[0].body


def test_parse_ops_with_scratch_preserves_op_region():
    text = (
        f"{SCRATCH_OPEN}\n"
        "let me think about this\n"
        f"{SCRATCH_CLOSE}\n"
        "\n"
        "note tag=plan\n"
        "    a note\n"
        "continue\n"
        "    keep going\n"
    )
    scratch, ops = parse_ops(text)
    assert "let me think" in scratch
    assert [o.name for o in ops] == ["note", "continue"]
