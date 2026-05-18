from __future__ import annotations

from typing import Any

from ttt_discover.buffer import BufferStats


class AdaptiveBetaScheduler:
    def __init__(self, initial_beta: float, decay: str = "adaptive", config: dict[str, Any] | None = None) -> None:
        self.initial_beta = float(initial_beta)
        self.current_beta = float(initial_beta)
        self.decay = decay
        self.config = config or {}
        self.min_beta = float(self.config.get("min_beta", 0.01))
        self._steps = 0

    def step(self, buffer_stats: BufferStats) -> float:
        self._steps += 1
        if self.decay == "constant":
            return self.current_beta
        if self.decay == "linear_decay":
            decrement = float(self.config.get("linear_step", 0.01)) * self._steps
            self.current_beta = max(self.min_beta, self.initial_beta - decrement)
            return self.current_beta
        if self.decay == "adaptive":
            spread = buffer_stats.reward_std or 0.0
            target = self.initial_beta / (1.0 + spread + 0.05 * self._steps)
            self.current_beta = max(self.min_beta, target)
            return self.current_beta
        raise ValueError(f"Unknown beta schedule: {self.decay}")
