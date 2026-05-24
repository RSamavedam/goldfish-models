from problems.sorting_net import SortingNetReward
from ttt_discover.reward.sandbox import SandboxedReward


def test_sandboxed_reward_executes_python():
    result = SandboxedReward(timeout_s=2)("print('ok')")

    assert result.valid
    assert result.reward == 1.0
    assert "ok" in result.metadata["stdout"]


def test_sorting_reward_scores_valid_solution():
    result = SortingNetReward()("def sort_values(xs):\n    return sorted(xs)\n")

    assert result.valid
    assert result.reward >= 1.0
