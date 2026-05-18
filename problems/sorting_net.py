from __future__ import annotations

from problems.base import Problem
from ttt_discover.reward.base import RewardFunction, RewardResult
from ttt_discover.reward.registry import register_reward


class SortingNetReward(RewardFunction):
    def __call__(self, solution: str) -> RewardResult:
        namespace: dict[str, object] = {}
        try:
            exec(solution, {"__builtins__": {"sorted": sorted, "len": len, "range": range, "list": list}}, namespace)
            sort_values = namespace.get("sort_values")
            if not callable(sort_values):
                return RewardResult(0.0, False, {"error": "missing sort_values"})
            tests = [[], [1], [2, 1], [3, 1, 2], [5, -1, 5, 0], list(range(5, -1, -1))]
            passed = sum(1 for case in tests if list(sort_values(list(case))) == sorted(case))
        except Exception as exc:
            return RewardResult(0.0, False, {"error": repr(exc)})
        correctness = passed / len(tests)
        brevity_bonus = min(0.1, 20.0 / max(len(solution), 1))
        return RewardResult(correctness + (brevity_bonus if passed == len(tests) else 0.0), passed == len(tests), {"passed": passed, "total": len(tests)})


class SortingNetProblem(Problem):
    name = "sorting_net"
    description = "Synthesize a compact Python sorting routine for small integer lists."

    def prompt(self, seed_solution: str | None = None) -> str:
        base = "Write Python code defining sort_values(xs) that returns xs sorted ascending."
        if seed_solution:
            return f"{base}\nImprove this prior attempt:\n{seed_solution}"
        return base

    def reward_fn(self) -> RewardFunction:
        return SortingNetReward()

    def validate(self, solution: str) -> bool:
        return "def sort_values" in solution


try:
    register_reward("sorting_net", SortingNetReward())
except ValueError:
    pass
