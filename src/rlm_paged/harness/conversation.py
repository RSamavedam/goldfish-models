"""Conversation state for a multi-turn harness loop.

The model is called once per turn. Between turns the wrapper:
  1. Parses any ops out of the model's output and dispatches them.
  2. Accumulates non-op visible text into the tail.
  3. Applies the scheme's overflow policy if the window would exceed L.
  4. Renders the next prompt (pinned prefix + active window content).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rlm_paged.client.tokenizer import count


@dataclass
class Segment:
    """One contiguous text block in the conversation.

    `kind` is one of:
      - "task": the original user task prompt (always pinned)
      - "model": visible model output (CoT, intermediate reasoning)
      - "tool": tool result emitted by the dispatcher
      - "summary": replacement segment created by the summarized scheme
    """

    kind: str
    text: str
    tokens: int = 0
    chunk_ids: list[int] = field(default_factory=list)  # populated on eviction

    def __post_init__(self) -> None:
        if self.tokens == 0 and self.text:
            self.tokens = count(self.text)


@dataclass
class ConversationState:
    """All conversation history for a single task.

    The active window is a *view* over a suffix of `segments`. The first
    segment is always the task prompt and is pinned (counts against the
    pinned-prefix budget, not L).
    """

    segments: list[Segment] = field(default_factory=list)
    evicted: list[Segment] = field(default_factory=list)
    turns: int = 0

    def add_segment(self, segment: Segment) -> None:
        self.segments.append(segment)

    def active_segments(self) -> list[Segment]:
        """All segments except the task prompt (index 0)."""
        return self.segments[1:]

    def active_tokens(self) -> int:
        return sum(s.tokens for s in self.active_segments())

    def task_segment(self) -> Segment:
        return self.segments[0]
