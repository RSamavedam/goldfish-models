from __future__ import annotations

import math

from ttt_discover.buffer import Solution, SolutionBuffer


class PUCTSelector:
    def __init__(self, c_puct: float, buffer: SolutionBuffer) -> None:
        self.c_puct = c_puct
        self.buffer = buffer

    def select(self, n: int) -> list[Solution]:
        candidates = self.buffer.all()
        if n <= 0 or not candidates:
            return []

        total_visits = sum(self.buffer.visits(solution.id) for solution in candidates) + 1
        ranked = sorted(
            candidates,
            key=lambda solution: self._score(solution, total_visits),
            reverse=True,
        )
        selected: list[Solution] = []
        while len(selected) < n:
            selected.extend(ranked[: n - len(selected)])
        return selected

    def _score(self, solution: Solution, total_visits: int) -> float:
        visits = self.buffer.visits(solution.id)
        exploration = self.c_puct * math.sqrt(math.log(total_visits + 1) / (visits + 1))
        prior = float(solution.metadata.get("prior", 1.0))
        return solution.reward + prior * exploration
