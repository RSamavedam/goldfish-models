"""Typed block model for the stateless-turn architecture.

A `Block` is the unit of writeable state in goldfish-models. Five types:

- `user_message`            — the user's input (including the original
                              task prompt: first user_message IS the task).
                              Harness-written; ordered by timestamp.
- `assistant_reply`         — model's output to the user via the `say` op.
                              Symmetric to user_message; provides an audit
                              trail of model→user exchange.
- `observation`             — tool-call results, verbatim. Harness-written.
- `note`                    — model-authored compressed knowledge.
- `continuing_instruction`  — message from one turn to the next. One per turn.

All blocks are append-only. There is no delete, no update. A future
`supersede` op (not v1) provides logical invalidation without losing audit.

The `task` type from the v1 sketch has been removed — the original prompt
is the first `user_message`. This unifies the input channel.

See DESIGN.md §2.4 for the full spec.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

BLOCK_TYPES = (
    "user_message",
    "assistant_reply",
    "observation",
    "note",
    "continuing_instruction",
)


def _now() -> float:
    return time.time()


@dataclass
class Block:
    type: str                              # one of BLOCK_TYPES
    index: int                             # monotonic *within* type
    global_index: int                      # monotonic across all types (insertion order)
    text: str
    created_at_turn: int                   # -1 for harness-injected before turn 0
    timestamp: float = field(default_factory=_now)
    outgoing_refs: list[int] = field(default_factory=list)  # global indices
    incoming_refs: list[int] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    embedding: bytes | None = None
    access_count: int = 0
    last_accessed_turn: int = -1

    def __post_init__(self) -> None:
        if self.type not in BLOCK_TYPES:
            raise ValueError(f"unknown block type: {self.type!r}")

    @property
    def short_id(self) -> str:
        """Compact identifier for retrieved-content metadata: e.g. 'note:7'."""
        return f"{self.type}:{self.index}"
