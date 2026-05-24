from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: int
    tokens: list[int]
    created_at_step: int
    original_position: int
    outgoing_refs: list[int] = field(default_factory=list)
    incoming_refs: list[int] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    embedding: bytes | None = None
    access_count: int = 0
    last_accessed_step: int = -1

    def __len__(self) -> int:
        return len(self.tokens)
