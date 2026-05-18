from ttt_discover.buffer import Solution, SolutionBuffer


def test_buffer_insert_best_and_stats():
    buffer = SolutionBuffer()
    buffer.insert(Solution("a", 0.1, None, 0))
    buffer.insert(Solution("b", 0.9, None, 1))

    assert buffer.best()[0].text == "b"
    stats = buffer.stats()
    assert stats.count == 2
    assert stats.best_reward == 0.9
    assert stats.unique_texts == 2


def test_uniform_select_records_visits():
    buffer = SolutionBuffer()
    solution = Solution("a", 0.1, None, 0)
    buffer.insert(solution)

    assert buffer.select(3, strategy="uniform") == [solution, solution, solution]
    assert buffer.visits(solution.id) == 3
