from __future__ import annotations

from dataclasses import dataclass


class WindowViolation(Exception):
    """Raised when a window op would violate len(middle) + len(tail) <= L."""


@dataclass(frozen=True)
class WindowConfig:
    """Hard-capped active-window config.

    The pinned prefix is *outside* L (see DESIGN.md section 2.1). L counts
    only middle + tail tokens.
    """

    L: int
    tail_max: int | None = None  # default L // 2 when None

    def resolved_tail_max(self) -> int:
        return self.tail_max if self.tail_max is not None else max(1, self.L // 2)


class ActiveWindow:
    """Tracks middle + tail token counts only. Prefix is implicit (not counted).

    Invariants enforced on every mutation:
      len(middle) + len(tail) <= L
      len(tail)               <= tail_max
    """

    def __init__(self, config: WindowConfig) -> None:
        self.config = config
        self._middle: int = 0
        self._tail: int = 0

    @property
    def middle(self) -> int:
        return self._middle

    @property
    def tail(self) -> int:
        return self._tail

    @property
    def used(self) -> int:
        return self._middle + self._tail

    @property
    def free(self) -> int:
        return self.config.L - self.used

    def append_tail(self, n: int) -> None:
        if n < 0:
            raise ValueError("append_tail requires n >= 0")
        if self._tail + n > self.config.resolved_tail_max():
            raise WindowViolation(
                f"tail overflow: {self._tail} + {n} > {self.config.resolved_tail_max()}"
            )
        if self.used + n > self.config.L:
            raise WindowViolation(f"window overflow: {self.used} + {n} > L={self.config.L}")
        self._tail += n

    def freeze_tail_into_middle(self) -> None:
        """Treat the current tail as committed history (becomes immutable middle)."""
        self._middle += self._tail
        self._tail = 0

    def evict_head(self, n: int) -> int:
        """Evict up to `n` tokens from the head of the middle. Returns tokens actually freed."""
        if n < 0:
            raise ValueError("evict_head requires n >= 0")
        freed = min(n, self._middle)
        self._middle -= freed
        return freed

    def can_fit(self, n: int) -> bool:
        return self.used + n <= self.config.L
