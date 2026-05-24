from __future__ import annotations

from collections.abc import Callable

from rlm_paged.reward.base import RewardFunction

_REGISTRY: dict[str, RewardFunction] = {}


def register_reward(name: str, reward_fn: RewardFunction) -> RewardFunction:
    if name in _REGISTRY:
        raise ValueError(f"Reward already registered: {name}")
    _REGISTRY[name] = reward_fn
    return reward_fn


def get_reward(name: str) -> RewardFunction:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown reward function: {name}")
    return _REGISTRY[name]


def reward(name: str) -> Callable[[RewardFunction], RewardFunction]:
    def decorator(reward_fn: RewardFunction) -> RewardFunction:
        return register_reward(name, reward_fn)

    return decorator
