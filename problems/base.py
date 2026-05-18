from __future__ import annotations

from abc import ABC, abstractmethod

from ttt_discover.reward.base import RewardFunction


class Problem(ABC):
    name: str
    description: str

    @abstractmethod
    def prompt(self, seed_solution: str | None = None) -> str:
        """Format the LLM prompt, optionally asking it to improve a seed."""

    @abstractmethod
    def reward_fn(self) -> RewardFunction:
        """Return the reward function for this problem."""

    @abstractmethod
    def validate(self, solution: str) -> bool:
        """Run a cheap validity check before full reward computation."""
