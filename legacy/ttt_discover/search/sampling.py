from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 4096


def apply_temperature(logits: list[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return [logit / temperature for logit in logits]
