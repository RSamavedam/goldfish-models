"""Typed, append-only block store for the stateless-turn architecture.

Distinct from the legacy `ChunkStore` (which stored token-ID chunks for
the old paged-active-window design). The block store works in *text* —
blocks have a `.text` field — because the stateless-turn architecture
operates on the model's visible output, not on internal KV state.

See DESIGN.md §2.4–2.5.
"""

from __future__ import annotations

from rlm_paged.store.block import BLOCK_TYPES, Block


class BlockStore:
    """In-process typed block store. CPU-DRAM tier only for v1."""

    def __init__(self) -> None:
        self._blocks: dict[int, Block] = {}              # global_index -> Block
        self._by_type: dict[str, list[Block]] = {t: [] for t in BLOCK_TYPES}
        self._next_global: int = 0

    # ----------------------------------------------------- writes

    def append(
        self,
        type: str,
        text: str,
        *,
        created_at_turn: int,
        tags: list[str] | None = None,
    ) -> Block:
        if type not in BLOCK_TYPES:
            raise ValueError(f"unknown block type: {type!r}")
        type_index = len(self._by_type[type])
        block = Block(
            type=type,
            index=type_index,
            global_index=self._next_global,
            text=text,
            created_at_turn=created_at_turn,
            tags=list(tags or []),
        )
        self._blocks[self._next_global] = block
        self._by_type[type].append(block)
        self._next_global += 1
        return block

    def link(self, src_global: int, dst_global: int) -> None:
        src = self._blocks[src_global]
        dst = self._blocks[dst_global]
        if dst_global not in src.outgoing_refs:
            src.outgoing_refs.append(dst_global)
        if src_global not in dst.incoming_refs:
            dst.incoming_refs.append(src_global)

    def tag(self, global_index: int, tag: str) -> None:
        b = self._blocks[global_index]
        if tag not in b.tags:
            b.tags.append(tag)

    # ----------------------------------------------------- reads

    def get(self, global_index: int) -> Block:
        return self._blocks[global_index]

    def by_type_index(self, type: str, index: int) -> Block:
        return self._by_type[type][index]

    def query(
        self,
        type: str,
        *,
        start: int | None = None,
        end: int | None = None,
        tag: str | None = None,
        at_turn: int = -1,
    ) -> list[Block]:
        """Return blocks of `type` whose per-type index is in [start, end].

        Bounds inclusive. Either may be omitted for open-ended. Negative
        indices count from the end (Python-list semantics). `tag` filters
        to only blocks carrying that tag. Empty results are not an error.
        """
        if type not in BLOCK_TYPES:
            raise ValueError(f"unknown block type: {type!r}")
        all_blocks = self._by_type[type]
        if not all_blocks:
            return []
        n = len(all_blocks)
        s = 0 if start is None else (start if start >= 0 else n + start)
        e = (n - 1) if end is None else (end if end >= 0 else n + end)
        # If the *requested* range falls entirely outside the valid window,
        # return empty rather than silently clamping to the last item — that
        # surfaces model errors more clearly than returning a nearby block.
        if s >= n or e < 0 or s > e:
            return []
        s = max(0, s)
        e = min(n - 1, e)
        results = all_blocks[s : e + 1]
        if tag is not None:
            results = [b for b in results if tag in b.tags]
        for b in results:
            b.access_count += 1
            b.last_accessed_turn = at_turn
        return list(results)

    # ----------------------------------------------------- stats (for system prompt)

    def stats(self) -> dict[str, dict[str, int]]:
        """Per-type counts + last index for the system-prompt prelude.

        Shape: {'note': {'count': 12, 'last_index': 11}, ...}
        """
        out: dict[str, dict[str, int]] = {}
        for t in BLOCK_TYPES:
            blocks = self._by_type[t]
            out[t] = {
                "count": len(blocks),
                "last_index": (blocks[-1].index if blocks else -1),
            }
        return out

    def __len__(self) -> int:
        return len(self._blocks)
