"""Aggregation logic for analyze.py.

The script can't be imported as a normal module because it lives under
`scripts/`, but the aggregation functions are pure so we load them via
`importlib`. This isolates the test from CLI semantics.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ANALYZE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "analyze.py"
)


@pytest.fixture
def analyze():
    spec = importlib.util.spec_from_file_location("analyze_script", ANALYZE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(
    *,
    provider: str,
    benchmark: str,
    L: int,
    solved: bool,
    input_tokens: int = 100,
    output_tokens: int = 50,
    thinking_tokens: int = 0,
    turns: int = 2,
    op_counts: dict | None = None,
    failure_reason: str | None = None,
    score: float | None = None,
) -> dict:
    return {
        "cell": {"provider": provider, "benchmark": benchmark, "L": L},
        "solved": solved,
        "score": score if score is not None else (1.0 if solved else 0.0),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "turns": turns,
        "op_counts": op_counts or {},
        "failure_reason": failure_reason,
    }


def test_load_jsonl_handles_blank_lines_and_bad_json(analyze, tmp_path):
    p = tmp_path / "mixed.jsonl"
    p.write_text(
        "\n"
        + json.dumps(_row(provider="p1", benchmark="b1", L=32, solved=True))
        + "\n"
        + "not-json\n"
        + json.dumps(_row(provider="p1", benchmark="b1", L=32, solved=False))
        + "\n"
        + "\n"
    )
    rows = analyze.load_jsonl([p])
    assert len(rows) == 2


def test_load_jsonl_missing_file_warns_but_continues(analyze, tmp_path, capsys):
    rows = analyze.load_jsonl([tmp_path / "nope.jsonl"])
    assert rows == []
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_aggregate_solve_rate_basic(analyze):
    rows = [
        _row(provider="p1", benchmark="b1", L=32, solved=True),
        _row(provider="p1", benchmark="b1", L=32, solved=False),
        _row(provider="p1", benchmark="b1", L=32, solved=True),
        _row(provider="p1", benchmark="b1", L=128, solved=True),
    ]
    agg = analyze.aggregate(rows)
    cell32 = agg[("p1", "b1", 32)]
    assert cell32["n"] == 3
    assert cell32["solved"] == 2
    assert cell32["solve_rate"] == pytest.approx(2 / 3)
    cell128 = agg[("p1", "b1", 128)]
    assert cell128["n"] == 1
    assert cell128["solve_rate"] == 1.0


def test_aggregate_tokens_summed_across_channels(analyze):
    rows = [
        _row(
            provider="p1", benchmark="b1", L=32, solved=True,
            input_tokens=10, output_tokens=20, thinking_tokens=30,
        ),
        _row(
            provider="p1", benchmark="b1", L=32, solved=False,
            input_tokens=100, output_tokens=50, thinking_tokens=0,
        ),
    ]
    cell = analyze.aggregate(rows)[("p1", "b1", 32)]
    assert cell["tokens_mean"] == pytest.approx((60 + 150) / 2)
    assert cell["tokens_total"] == 210


def test_aggregate_op_per_turn_normalizes_correctly(analyze):
    rows = [
        _row(
            provider="p1", benchmark="b1", L=32, solved=True,
            turns=2, op_counts={"note": 1, "continue": 2, "query": 0},
        ),
        _row(
            provider="p1", benchmark="b1", L=32, solved=True,
            turns=4, op_counts={"note": 3, "continue": 4},
        ),
    ]
    cell = analyze.aggregate(rows)[("p1", "b1", 32)]
    # 6 total turns across 2 rows; total notes = 4; mean ops per turn = 4/6.
    assert cell["op_per_turn"]["note"] == pytest.approx(4 / 6)
    assert cell["op_per_turn"]["continue"] == pytest.approx(6 / 6)


def test_aggregate_failure_modes_use_solved_label_when_no_reason(analyze):
    rows = [
        _row(provider="p1", benchmark="b1", L=32, solved=True, failure_reason=None),
        _row(
            provider="p1", benchmark="b1", L=32, solved=False,
            failure_reason="missing_continue",
        ),
        _row(
            provider="p1", benchmark="b1", L=32, solved=False,
            failure_reason=None,
        ),
    ]
    cell = analyze.aggregate(rows)[("p1", "b1", 32)]
    modes = cell["failure_modes"]
    assert modes["solved"] == 1
    assert modes["missing_continue"] == 1
    assert modes["wrong_answer"] == 1


def test_print_solve_rate_table_handles_missing_cells(analyze, capsys):
    rows = [_row(provider="p1", benchmark="b1", L=32, solved=True)]
    agg = analyze.aggregate(rows)
    analyze.print_solve_rate_table(
        agg, benchmarks=["b1"], providers=["p1", "p2"], L_values=[32, 64]
    )
    out = capsys.readouterr().out
    # p1 at L=32 has data; everything else should be "—".
    assert "1.00 (1)" in out
    assert "—" in out
    assert "p2" in out


def test_aggregator_groups_distinct_providers_and_benchmarks(analyze):
    rows = [
        _row(provider="p1", benchmark="b1", L=32, solved=True),
        _row(provider="p2", benchmark="b1", L=32, solved=False),
        _row(provider="p1", benchmark="b2", L=32, solved=True),
        _row(provider="p1", benchmark="b1", L=64, solved=True),
    ]
    agg = analyze.aggregate(rows)
    assert set(agg.keys()) == {
        ("p1", "b1", 32),
        ("p2", "b1", 32),
        ("p1", "b2", 32),
        ("p1", "b1", 64),
    }
    for key in agg:
        assert agg[key]["n"] == 1
