"""Comparison schemes for the Phase 1 TTC sweep.

Each scheme decides what to do when the active window would exceed L. The
harness loop is scheme-agnostic; it calls `scheme.enforce_cap(state, L)`
after every turn and `scheme.render_window(state)` before every turn.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rlm_paged.harness.conversation import ConversationState, Segment

if TYPE_CHECKING:
    from rlm_paged.client.base import LLMClient
    from rlm_paged.store.store import ChunkStore


SCHEMES = ("native", "truncated", "paged", "summarized")


@dataclass
class SchemeContext:
    """Side state a scheme may need (e.g. a chunk store for paged)."""

    store: "ChunkStore | None" = None
    summarizer: "LLMClient | None" = None  # for summarized scheme


class Scheme(ABC):
    """Abstract overflow-policy interface."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def enforce_cap(self, state: ConversationState, L: int, ctx: SchemeContext) -> None:
        """Mutate `state` so the active window fits in L tokens."""

    def render_window(self, state: ConversationState) -> str:
        """Default: concatenate all active segments with blank-line separators."""
        return "\n\n".join(s.text for s in state.active_segments())


class NativeScheme(Scheme):
    """No cap — the entire conversation history is sent every turn."""

    @property
    def name(self) -> str:
        return "native"

    def enforce_cap(self, state: ConversationState, L: int, ctx: SchemeContext) -> None:
        return None


class TruncatedScheme(Scheme):
    """Hard cap, no paging. Oldest active segments are dropped on overflow.

    Accuracy ceiling for "what could the model do with L tokens of WM and no
    externalization." When paged beats this at the same L, externalization
    is the reason.
    """

    @property
    def name(self) -> str:
        return "truncated"

    def enforce_cap(self, state: ConversationState, L: int, ctx: SchemeContext) -> None:
        while state.active_tokens() > L and len(state.segments) > 1:
            dropped = state.segments.pop(1)
            state.evicted.append(dropped)


class PagedScheme(Scheme):
    """The goldfish-models scheme.

    On overflow, evict from the head of the middle (oldest non-task segments)
    into the chunk store. The model can also evict and retrieve explicitly
    via `e`/`r` ops handled by the dispatcher; this scheme only auto-evicts
    if the model itself didn't make room.
    """

    @property
    def name(self) -> str:
        return "paged"

    def enforce_cap(self, state: ConversationState, L: int, ctx: SchemeContext) -> None:
        if ctx.store is None:
            raise RuntimeError("PagedScheme requires a chunk store in SchemeContext")
        while state.active_tokens() > L and len(state.segments) > 1:
            dropped = state.segments.pop(1)
            from rlm_paged.client.tokenizer import encode

            token_ids = encode(dropped.text)
            new_ids = ctx.store.append(
                tokens=token_ids,
                created_at_step=state.turns,
                original_position=len(state.evicted),
                tags=[dropped.kind],
            )
            dropped.chunk_ids = new_ids
            state.evicted.append(dropped)


class SummarizedScheme(Scheme):
    """MemGPT-style: on overflow, replace the oldest active segments with a
    short summary segment produced by the summarizer client.

    Lossy by design. This is the baseline we want paged-CoT to beat.
    """

    SUMMARY_PROMPT = (
        "Summarize the following reasoning trace as concisely as possible while "
        "preserving information that may be needed to finish the problem. "
        "Aim for under {target} tokens.\n\nReasoning trace:\n{content}"
    )

    @property
    def name(self) -> str:
        return "summarized"

    def enforce_cap(self, state: ConversationState, L: int, ctx: SchemeContext) -> None:
        if ctx.summarizer is None:
            TruncatedScheme().enforce_cap(state, L, ctx)
            return
        if state.active_tokens() <= L:
            return
        active = state.active_segments()
        cut = max(1, len(active) // 2)
        to_summarize = active[:cut]
        keep = active[cut:]
        content = "\n\n".join(s.text for s in to_summarize)
        target = max(64, L // 4)
        prompt = self.SUMMARY_PROMPT.format(target=target, content=content)
        summary_resp = ctx.summarizer.generate(
            prompt, max_tokens=target * 2, temperature=0.0
        )
        summary_seg = Segment(kind="summary", text=summary_resp.text)
        state.segments = [state.task_segment(), summary_seg, *keep]
        TruncatedScheme().enforce_cap(state, L, ctx)


def build_scheme(name: str) -> Scheme:
    if name == "native":
        return NativeScheme()
    if name == "truncated":
        return TruncatedScheme()
    if name == "paged":
        return PagedScheme()
    if name == "summarized":
        return SummarizedScheme()
    raise ValueError(f"unknown scheme: {name!r}")
