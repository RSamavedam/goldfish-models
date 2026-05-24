"""Op surface for the stateless-turn architecture.

Five ops. The text-channel wire format is one op per line, name first,
arguments following. Multi-line argument values are written as a header
line followed by indented continuation lines (4 spaces), terminated by a
blank line or the next op.

    note tag=plan
        We're solving an AIME problem; the key trick is parity.
        Block index reserved as note:3.

    continue
        Try base case n=2 next turn. Query observations 7-8 for partial
        results.

    query observation 7 9
    query note 0 -1 tag=plan
    pipe (query note tag=plan) -> note tag=summary
    call code_exec
        for n in range(2, 5):
            print(n, n % 2)

The model can also use single-line shorthand for simple ops:

    query observation 7 9
    note "Quick thought."
    continue "Next: case n=3."

See DESIGN.md §2.7.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


OP_NAMES = ("note", "continue", "query", "pipe", "call", "scratch")


@dataclass
class Op:
    """One parsed op from the response region."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    body: str = ""             # multi-line value (notes, continues, code)
    raw: str = ""              # original text for debugging


# Public marker for the scratch region's opening/closing tags.
SCRATCH_OPEN = "<scratch>"
SCRATCH_CLOSE = "</scratch>"


# --------------------------------------------------------------------- #
# Parser                                                                #
# --------------------------------------------------------------------- #

_HEADER_RE = re.compile(
    r"^(?P<name>note|continue|query|pipe|call)(?:\s+(?P<rest>.*))?$"
)
_KV_RE = re.compile(r"(\w+)=(\S+)")


def extract_scratch(text: str) -> tuple[str, str]:
    """Return (scratch_text, remainder_without_scratch).

    Only the first `<scratch>...</scratch>` block at the start of the
    response is recognized. Anything outside that block is treated as op
    region.
    """
    stripped = text.lstrip()
    if not stripped.startswith(SCRATCH_OPEN):
        return "", text
    leading_ws_len = len(text) - len(stripped)
    after_open = stripped[len(SCRATCH_OPEN) :]
    close_at = after_open.find(SCRATCH_CLOSE)
    if close_at < 0:
        # Unclosed scratch — treat entire response as scratch (the model
        # blew its budget on thinking). Op region is empty.
        return after_open, ""
    scratch_text = after_open[:close_at]
    remainder = after_open[close_at + len(SCRATCH_CLOSE) :]
    return scratch_text, " " * leading_ws_len + remainder


def parse_ops(response: str) -> tuple[str, list[Op]]:
    """Parse the response region into (scratch_text, [Op, ...]).

    Robust to:
      - Single-line ops:    `note "quick thought."`
      - Multi-line ops:     `note tag=plan\\n    line one\\n    line two`
      - Mixed quote styles, missing args (caller validates).

    Unparseable lines are silently dropped — the harness counts them via
    `errors_dropped` in the runner.
    """
    scratch, body_text = extract_scratch(response)
    ops: list[Op] = []
    lines = body_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        m = _HEADER_RE.match(stripped)
        if not m:
            i += 1
            continue
        name = m.group("name")
        rest = (m.group("rest") or "").strip()

        # Single-line form: quoted body inline.
        if rest.startswith('"') or rest.startswith("'"):
            body, kv_rest = _take_quoted(rest)
            args = _parse_kv(kv_rest)
            ops.append(Op(name=name, args=args, body=body, raw=line))
            i += 1
            continue

        # Header may carry positional args + kwargs.
        args = _parse_positional_and_kv(name, rest)

        # Collect continuation lines (indented or blank-line terminated).
        body_lines: list[str] = []
        j = i + 1
        while j < len(lines):
            cont = lines[j]
            if not cont.strip():
                break
            if cont.startswith("    "):
                body_lines.append(cont[4:])
                j += 1
            elif cont.startswith("\t"):
                body_lines.append(cont[1:])
                j += 1
            else:
                break
        body = "\n".join(body_lines).rstrip()
        ops.append(Op(name=name, args=args, body=body, raw=line))
        i = j

    return scratch, ops


def _take_quoted(rest: str) -> tuple[str, str]:
    """Split a leading quoted string from `rest`. Returns (body, tail)."""
    quote = rest[0]
    end = rest.find(quote, 1)
    if end < 0:
        return rest[1:], ""
    return rest[1:end], rest[end + 1 :].strip()


def _parse_kv(rest: str) -> dict[str, Any]:
    return {m.group(1): m.group(2) for m in _KV_RE.finditer(rest)}


def _parse_positional_and_kv(name: str, rest: str) -> dict[str, Any]:
    """Per-op positional + kv arg parsing.

    Wire formats:
      query <type> <start> <end> [tag=T]
      pipe (query ...) -> <dest> [kv=...]
      call <tool_name>             (body holds tool args)
      note  [tag=T]                (body holds the note text)
      continue                     (body holds the message)
    """
    args: dict[str, Any] = {}
    if not rest:
        return args
    kv = _parse_kv(rest)
    args.update(kv)
    # Strip kv pairs and parse positional fragment.
    positional = _KV_RE.sub("", rest).strip()
    if name == "query" and positional:
        toks = positional.split()
        if toks:
            args["type"] = toks[0]
        if len(toks) >= 2:
            args["start"] = _maybe_int(toks[1])
        if len(toks) >= 3:
            args["end"] = _maybe_int(toks[2])
    elif name == "call" and positional:
        toks = positional.split(maxsplit=1)
        args["tool"] = toks[0]
    elif name == "pipe" and positional:
        # pipe (query note 0 -1) -> note tag=summary
        # We just keep the literal positional; the executor re-parses.
        args["spec"] = positional
    return args


def _maybe_int(s: str) -> Any:
    try:
        return int(s)
    except (TypeError, ValueError):
        return s
