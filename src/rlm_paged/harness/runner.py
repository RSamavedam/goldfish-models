"""Run one (provider, scheme, L, task) cell of the Phase 1 sweep.

Stub — actual provider calls land in Phase 1 implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SweepCell:
    provider: str       # e.g. "anthropic:claude-opus-4-7"
    scheme: str         # "paged" | "summarized" | "rag" | "subagent" | "native"
    L: int              # active-window cap; ignored for "native"
    task_id: str        # bench-specific task identifier
    seed: int = 0


@dataclass
class RunResult:
    cell: SweepCell
    solved: bool
    score: float
    input_tokens: int
    output_tokens: int
    tool_calls: dict[str, int] = field(default_factory=dict)
    wall_seconds: float = 0.0
    active_window_mean_utilization: float = 0.0
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def run_cell(cell: SweepCell) -> RunResult:
    """Execute one sweep cell. Implementation lands in Phase 1."""
    raise NotImplementedError("run_cell pending Phase 1 implementation")
