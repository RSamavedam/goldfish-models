from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


# Benchmark families. Phase 1 is "ttc" (test-time-compute reasoning); the
# original long_doc / memory_dialogue / coding families from DESIGN.md
# remain reachable as a future direction.
FAMILY_TTC = "ttc"
FAMILY_LONG_DOC = "long_doc"
FAMILY_MEMORY_DIALOGUE = "memory_dialogue"
FAMILY_CODING = "coding"


@dataclass
class BenchTask:
    task_id: str
    family: str
    payload: dict        # bench-specific input (typically {"question": str, ...})
    expected: dict       # bench-specific gold (typically {"answer": str, ...})


class BenchSuite(ABC):
    @property
    @abstractmethod
    def family(self) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def tasks(self) -> list[BenchTask]: ...

    @abstractmethod
    def score(self, task: BenchTask, response: str) -> tuple[bool, float]:
        """Return (solved, fractional_score) for a model response."""

    def task_prompt(self, task: BenchTask) -> str:
        """Render the task's user prompt. Default: just the question text."""
        q = task.payload.get("question", "")
        return q
