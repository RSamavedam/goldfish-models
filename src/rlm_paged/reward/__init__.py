"""Reward primitives reused from the legacy ttt_discover code.

The sandboxed Python executor + reward registry are unchanged from the
TTT-Discover skeleton; they're useful for the coding reward in Phase 2.
"""

from rlm_paged.reward.base import RewardFunction, RewardResult
from rlm_paged.reward.registry import get_reward, register_reward, reward
from rlm_paged.reward.sandbox import SandboxedReward

__all__ = [
    "RewardFunction",
    "RewardResult",
    "SandboxedReward",
    "get_reward",
    "register_reward",
    "reward",
]
