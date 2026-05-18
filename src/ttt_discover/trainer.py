from __future__ import annotations

import math
from typing import Any

from ttt_discover.policy import TrainingBatch

try:  # pragma: no cover - optional heavyweight dependency in skeleton tests
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class OnlineTrainer:
    def __init__(self, policy: Any, beta_scheduler: Any, config: dict[str, Any] | None = None) -> None:
        self.policy = policy
        self.beta_scheduler = beta_scheduler
        self.config = config or {}
        self._last_beta = float(getattr(beta_scheduler, "current_beta", self.config.get("beta", 1.0)))

    def compute_loss(self, batch: TrainingBatch):
        beta = max(self._last_beta, 1e-8)
        if torch is not None:
            rewards = torch.tensor([example.reward for example in batch.examples], dtype=torch.float32)
            logprobs = torch.tensor([example.logprob for example in batch.examples], dtype=torch.float32)
            utilities = torch.exp(rewards / beta)
            return -(utilities * logprobs).mean()

        losses = [-math.exp(example.reward / beta) * example.logprob for example in batch.examples]
        return sum(losses) / len(losses) if losses else 0.0

    def step(self, batch: TrainingBatch) -> dict[str, float]:
        if hasattr(self.beta_scheduler, "current_beta"):
            self._last_beta = float(self.beta_scheduler.current_beta)
        loss = self.compute_loss(batch)
        policy_metrics = self.policy.update(batch)
        loss_value = float(loss.detach().cpu().item()) if torch is not None and hasattr(loss, "detach") else float(loss)
        return {
            "loss": loss_value,
            "grad_norm": 0.0,
            "effective_beta": self._last_beta,
            **{key: float(value) for key, value in policy_metrics.items()},
        }
