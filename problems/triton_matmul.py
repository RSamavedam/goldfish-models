from __future__ import annotations

from problems.base import Problem
from ttt_discover.reward.base import RewardFunction, RewardResult
from ttt_discover.reward.registry import register_reward


class TritonMatmulReward(RewardFunction):
    def __call__(self, solution: str) -> RewardResult:
        has_kernel = "triton" in solution.lower() and "matmul" in solution.lower()
        return RewardResult(0.1 if has_kernel else 0.0, has_kernel, {"stub": True})


class TritonMatmulProblem(Problem):
    name = "triton_matmul"
    description = "Optimize a Triton matrix multiplication kernel for correctness and latency."

    def prompt(self, seed_solution: str | None = None) -> str:
        base = "Write a Triton matmul kernel optimized for square FP16 matrices."
        if seed_solution:
            return f"{base}\nImprove this prior kernel:\n{seed_solution}"
        return base

    def reward_fn(self) -> RewardFunction:
        return TritonMatmulReward()

    def validate(self, solution: str) -> bool:
        return "triton" in solution.lower()


try:
    register_reward("triton_matmul", TritonMatmulReward())
except ValueError:
    pass
