"""Extract fenced shell blocks from a model's response, in document order.

We recognize triple-backtick fences with optional `bash` / `sh` / `shell`
language tags. The model may write prose between blocks; only the
contents of fenced blocks are executed.

A single fenced block may contain MULTIPLE commands separated by
newlines. We treat each non-empty, non-comment line as a separate
command. Lines ending in `\\` are continuations.

Edge cases handled:
  - Missing close fence: the rest of the response is one block.
  - Lines starting with `#`: comments, skipped.
  - Nested triple-backticks inside a heredoc: we do NOT try to be clever
    here — the model would need to use a different fence style or rely on
    the harness's "one fence per response" robustness. Documented in the
    system prompt.
  - Bare ``` with no language tag: treated as a shell block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


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

    - Each line is a command unless:
      - it ends with `\\` (continuation) → join with next line + a space
      - it begins with `#` (comment)
      - it is blank
    """
    out: list[str] = []
    buffer: list[str] = []
    for raw_line in body.splitlines():
        stripped = raw_line.rstrip()
        if not stripped:
            if buffer:
                out.append(" ".join(buffer).strip())
                buffer = []
            continue
        if stripped.lstrip().startswith("#"):
            continue
        if stripped.endswith("\\"):
            buffer.append(stripped[:-1].rstrip())
            continue
        buffer.append(stripped)
        out.append(" ".join(buffer).strip())
        buffer = []
    if buffer:
        out.append(" ".join(buffer).strip())
    return [c for c in out if c]
