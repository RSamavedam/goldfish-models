from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RewardResult:
    reward: float
    valid: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class RewardFunction(ABC):
    @abstractmethod
    def __call__(self, solution: str) -> RewardResult:
        """Return reward, validity, and evaluator metadata for a candidate solution."""
