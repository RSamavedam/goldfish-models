from ttt_discover.buffer import Solution, SolutionBuffer
from ttt_discover.search.puct import PUCTSelector


def test_puct_prefers_high_reward_when_visits_equal():
    buffer = SolutionBuffer()
    low = Solution("low", 0.1, None, 0)
    high = Solution("high", 1.0, None, 0)
    buffer.insert(low)
    buffer.insert(high)

    selected = PUCTSelector(c_puct=1.0, buffer=buffer).select(1)

    assert selected == [high]


def test_puct_explores_less_visited_branch():
    buffer = SolutionBuffer()
    visited = Solution("visited", 1.0, None, 0)
    fresh = Solution("fresh", 0.9, None, 0)
    buffer.insert(visited)
    buffer.insert(fresh)
    for _ in range(100):
        buffer.record_visit(visited.id)

    selected = PUCTSelector(c_puct=2.0, buffer=buffer).select(1)

    assert selected == [fresh]
