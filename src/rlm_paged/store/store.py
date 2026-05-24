from __future__ import annotations

from rlm_paged.store.chunk import Chunk


class ChunkStore:
    """In-process verbatim chunk store. CPU-DRAM tier only for now.

    Chunks are addressed by monotonically-increasing integer IDs.
    Retrieval returns the raw token IDs; positional re-binding is the
    window's responsibility (see DESIGN.md section 2.3).
    """

    def __init__(self, chunk_size: int = 256) -> None:
        self.chunk_size = chunk_size
        self._chunks: dict[int, Chunk] = {}
        self._next_id: int = 0

    def append(
        self,
        tokens: list[int],
        *,
        created_at_step: int,
        original_position: int,
        tags: list[str] | None = None,
    ) -> list[int]:
        """Split `tokens` into chunks of `self.chunk_size` and store them.

        Returns the list of new chunk IDs in order.
        """
        new_ids: list[int] = []
        for start in range(0, len(tokens), self.chunk_size):
            piece = tokens[start : start + self.chunk_size]
            chunk = Chunk(
                id=self._next_id,
                tokens=piece,
                created_at_step=created_at_step,
                original_position=original_position + start,
                tags=list(tags or []),
            )
            self._chunks[self._next_id] = chunk
            new_ids.append(self._next_id)
            self._next_id += 1
        return new_ids

    def get(self, chunk_id: int) -> Chunk:
        return self._chunks[chunk_id]

    def retrieve(
        self,
        chunk_id: int,
        offset: int,
        length: int,
        *,
        at_step: int,
    ) -> list[int]:
        """Return `length` tokens from chunk `chunk_id` starting at `offset`.

        Updates access bookkeeping. Does not span chunk boundaries — callers
        wanting a contiguous span across chunks issue multiple retrieves.
        """
        chunk = self._chunks[chunk_id]
        if offset < 0 or offset + length > len(chunk):
            raise IndexError(
                f"retrieve out of range: chunk {chunk_id} has {len(chunk)} tokens, "
                f"requested [{offset}, {offset + length})"
            )
        chunk.access_count += 1
        chunk.last_accessed_step = at_step
        return chunk.tokens[offset : offset + length]

    def link(self, src: int, dst: int) -> None:
        if dst not in self._chunks[src].outgoing_refs:
            self._chunks[src].outgoing_refs.append(dst)
        if src not in self._chunks[dst].incoming_refs:
            self._chunks[dst].incoming_refs.append(src)

    def annotate(self, chunk_id: int, tag: str) -> None:
        chunk = self._chunks[chunk_id]
        if tag not in chunk.tags:
            chunk.tags.append(tag)

    def refs(self, chunk_id: int) -> tuple[list[int], list[int]]:
        chunk = self._chunks[chunk_id]
        return list(chunk.outgoing_refs), list(chunk.incoming_refs)

    def __len__(self) -> int:
        return len(self._chunks)

    def total_tokens(self) -> int:
        return sum(len(c) for c in self._chunks.values())
