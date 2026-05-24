from __future__ import annotations


class CostCapExceeded(Exception):
    """Raised when a task exceeds its per-task tool-call token budget."""


class CostCap:
    """Per-task budget enforcer (see DESIGN.md section 3.5).

    Default ceiling is 100k tokens of tool-call traffic — input + output
    of every model call in the task's turn loop.
    """

    def __init__(self, max_tokens: int = 100_000) -> None:
        self.max_tokens = max_tokens
        self._spent = 0

    def charge(self, n: int) -> None:
        self._spent += n
        if self._spent > self.max_tokens:
            raise CostCapExceeded(f"spent {self._spent} > cap {self.max_tokens}")

    @property
    def spent(self) -> int:
        return self._spent

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self._spent)
