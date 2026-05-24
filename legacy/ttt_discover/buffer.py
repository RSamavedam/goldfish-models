from __future__ import annotations

import random
import statistics
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Solution:
    text: str
    reward: float
    parent_id: str | None
    step: int
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class BufferStats:
    count: int
    best_reward: float | None
    mean_reward: float | None
    reward_std: float
    unique_texts: int


class SolutionBuffer:
    def __init__(self, max_size: int | None = None, rng: random.Random | None = None) -> None:
        self.max_size = max_size
        self._solutions: list[Solution] = []
        self._visits: dict[str, int] = {}
        self._rng = rng or random.Random()

    def insert(self, solution: Solution) -> None:
        self._solutions.append(solution)
        self._visits.setdefault(solution.id, 0)
        if self.max_size is not None and len(self._solutions) > self.max_size:
            removed = self._solutions.pop(0)
            self._visits.pop(removed.id, None)

    def select(self, n: int, strategy: str = "puct") -> list[Solution]:
        if n <= 0 or not self._solutions:
            return []
        if strategy == "uniform":
            selected = self._rng.choices(self._solutions, k=n)
        elif strategy == "puct":
            from ttt_discover.search.puct import PUCTSelector

            selected = PUCTSelector(c_puct=1.5, buffer=self).select(n)
        else:
            raise ValueError(f"Unknown buffer selection strategy: {strategy}")
        for solution in selected:
            self.record_visit(solution.id)
        return selected

    def best(self, k: int = 1) -> list[Solution]:
        return sorted(self._solutions, key=lambda s: s.reward, reverse=True)[:k]

    def all(self) -> list[Solution]:
        return list(self._solutions)

    def stats(self) -> BufferStats:
        rewards = [solution.reward for solution in self._solutions]
        return BufferStats(
            count=len(self._solutions),
            best_reward=max(rewards) if rewards else None,
            mean_reward=statistics.fmean(rewards) if rewards else None,
            reward_std=statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            unique_texts=len({solution.text for solution in self._solutions}),
        )

    def visits(self, solution_id: str) -> int:
        return self._visits.get(solution_id, 0)

    def record_visit(self, solution_id: str) -> None:
        self._visits[solution_id] = self._visits.get(solution_id, 0) + 1
