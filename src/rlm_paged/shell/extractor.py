"""Extract fenced shell blocks from a model's response, in document order.

We recognize triple-backtick fences with optional `bash` / `sh` / `shell`
language tags. The model may write prose between blocks; only the
contents of fenced blocks are executed.

A single fenced block may contain MULTIPLE commands separated by
newlines. We treat each non-empty, non-comment line as a separate
command. Lines ending in `\\` are continuations.

Heredoc handling: when a line opens a heredoc (e.g. `cat > x <<EOF` or
`python3 - <<'PY'`), everything up to the matching terminator is taken
as ONE command. This was added after a smoke run where models reached
for `python3 - <<PY ... PY` to embed scripts and the harness split each
line into a separate command, every one of which failed allowlist.

Edge cases handled:
  - Missing close fence: the rest of the response is one block.
  - Lines starting with `#`: comments, skipped.
  - Heredoc terminator missing: we accumulate to end of block.
  - Heredoc with `<<-`: leading-tab stripping is the shell's problem;
    we just preserve the body verbatim.
  - Bare ``` with no language tag: treated as a shell block.
  - Nested triple-backticks inside a heredoc: still unsupported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Matches a heredoc-opening token at the END of a command:
#   <<EOF              → terminator EOF (interpreted)
#   <<-EOF             → terminator EOF (interpreted, leading tabs stripped)
#   << 'EOF'           → terminator EOF (literal)
#   <<-"EOF"           → terminator EOF (literal, leading tabs stripped)
# Captures the bare terminator word.
#
# Lookbehind requires whitespace or a `-` before `<<` so that text
# inside shell quotes (e.g. `echo 'a<<b'`) doesn't accidentally match.
# Real heredoc redirections always have whitespace before `<<`.
_HEREDOC_OPEN_RE = re.compile(
    r"(?<=\s)<<-?\s*[\"\']?(?P<term>[A-Za-z_][A-Za-z0-9_]*)[\"\']?\s*$"
)


_FENCE_OPEN = re.compile(
    r"^[ \t]*```(?P<lang>[a-zA-Z0-9_+\-]*)[ \t]*$",
    re.MULTILINE,
)
_FENCE_CLOSE = re.compile(r"^[ \t]*```[ \t]*$", re.MULTILINE)

# Languages we treat as shell blocks. Empty string (bare ```) also counts.
SHELL_LANGS = frozenset({"", "bash", "sh", "shell", "console"})


@dataclass
class FencedBlock:
    language: str           # the language tag (lowercased; "" for bare)
    body: str               # raw block contents (no fence lines)
    start_offset: int       # char offset of opening fence in original text
    end_offset: int         # char offset of closing fence (or len(text))


def extract_blocks(response: str) -> list[FencedBlock]:
    """Return all fenced blocks in document order."""
    blocks: list[FencedBlock] = []
    i = 0
    while i < len(response):
        m = _FENCE_OPEN.search(response, i)
        if not m:
            break
        lang = m.group("lang").lower()
        body_start = m.end() + 1  # past the trailing newline
        close = _FENCE_CLOSE.search(response, body_start)
        if close is None:
            body = response[body_start:]
            blocks.append(
                FencedBlock(
                    language=lang,
                    body=body,
                    start_offset=m.start(),
                    end_offset=len(response),
                )
            )
            break
        body = response[body_start : close.start()]
        blocks.append(
            FencedBlock(
                language=lang,
                body=body,
                start_offset=m.start(),
                end_offset=close.end(),
            )
        )
        i = close.end()
    return blocks


def extract_shell_commands(response: str) -> list[str]:
    """Return shell commands from all shell-flavored fenced blocks.

    Commands are split on newlines; comment lines (starting with `#`) and
    blank lines are dropped. Line-continuation (`\\` at end) joins the
    next line.
    """
    blocks = extract_blocks(response)
    commands: list[str] = []
    for blk in blocks:
        if blk.language not in SHELL_LANGS:
            continue
        for cmd in _split_commands(blk.body):
            if cmd:
                commands.append(cmd)
    return commands


def _split_commands(body: str) -> list[str]:
    """Split a block body into individual commands.

    A single line is one command unless one of these applies:
      - It ends with `\\` (continuation) → join with next line + space.
      - It begins with `#` (comment) → skipped.
      - It is blank → skipped, flushes any pending continuation buffer.
      - It opens a heredoc with `<<TERM` (or `<<-TERM`, `<<'TERM'`,
        `<<"TERM"`) → everything from this line up to and INCLUDING the
        terminator line is collected into ONE command, newlines
        preserved. This lets the shell see a real heredoc.
    """
    out: list[str] = []
    buffer: list[str] = []
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        raw_line = lines[i]
        stripped = raw_line.rstrip()

        if not stripped:
            if buffer:
                out.append(" ".join(buffer).strip())
                buffer = []
            i += 1
            continue

        if stripped.lstrip().startswith("#"):
            i += 1
            continue

        # Line continuation has precedence over heredoc detection.
        if stripped.endswith("\\"):
            buffer.append(stripped[:-1].rstrip())
            i += 1
            continue

        # Heredoc detection: scan the stripped line for an open marker.
        # The heredoc terminator detection runs against the line as it
        # would be after merging continuations, so flush the buffer
        # first.
        if buffer:
            merged_head = " ".join(buffer).strip() + " " + stripped
            buffer = []
        else:
            merged_head = stripped
        m = _HEREDOC_OPEN_RE.search(merged_head)
        if m:
            terminator = m.group("term")
            # Multi-line command: keep newlines intact so the shell
            # parses a real heredoc, not a single mashed-up line.
            heredoc_lines = [merged_head]
            i += 1
            while i < len(lines):
                next_line = lines[i]
                heredoc_lines.append(next_line)
                if next_line.strip() == terminator:
                    i += 1
                    break
                i += 1
            out.append("\n".join(heredoc_lines))
            continue

        out.append(merged_head)
        i += 1

    if buffer:
        out.append(" ".join(buffer).strip())
    return [c for c in out if c]
