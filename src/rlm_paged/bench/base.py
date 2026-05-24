from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BenchTask:
    task_id: str
    family: str          # "long_doc" | "memory_dialogue" | "coding"
    payload: dict        # bench-specific input
    expected: dict       # bench-specific gold


class BenchSuite(ABC):
    @property
    @abstractmethod
    def family(self) -> str: ...

    @abstractmethod
    def tasks(self) -> list[BenchTask]: ...

    @abstractmethod
    def score(self, task: BenchTask, response: str) -> tuple[bool, float]:
        """Return (solved, fractional_score) for a model response."""
