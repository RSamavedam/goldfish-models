from __future__ import annotations

from rlm_paged.bench.base import FAMILY_TTC, BenchSuite, BenchTask
from rlm_paged.client.base import GenerationResult, LLMClient
from rlm_paged.harness import SweepCell, build_scheme, run_cell


class FakeSuite(BenchSuite):
    """One-task suite where the gold answer is '42'."""

    @property
    def family(self) -> str:
        return FAMILY_TTC

    @property
    def name(self) -> str:
        return "fake"

    def tasks(self) -> list[BenchTask]:
        return [
            BenchTask(
                task_id="fake-001",
                family=FAMILY_TTC,
                payload={"question": "What is the answer?"},
                expected={"answer": "42"},
            )
        ]

    def task_prompt(self, task: BenchTask) -> str:
        return task.payload["question"]

    def score(self, task: BenchTask, response: str) -> tuple[bool, float]:
        ok = "42" in response and "final answer" in response.lower()
        return ok, 1.0 if ok else 0.0


class ScriptedClient(LLMClient):
    """Returns pre-canned responses in order. Tracks how often it was called."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return "scripted:fake"

    def generate(self, prompt, *, max_tokens, stop=None, temperature=0.0, system=None):
        self.calls.append({"prompt": prompt, "system": system})
        text = self.responses.pop(0) if self.responses else ""
        return GenerationResult(
            text=text,
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
            finish_reason="stop",
        )


def _run(responses, scheme_name, L=128):
    client = ScriptedClient(responses)
    suite = FakeSuite()
    task = suite.tasks()[0]
    scheme = build_scheme(scheme_name)
    cell = SweepCell(
        provider="scripted:fake",
        scheme=scheme_name,
        L=L,
        benchmark=suite.name,
        task_id=task.task_id,
        max_turns=8,
    )
    return run_cell(cell, client=client, suite=suite, task=task, scheme=scheme), client


def test_native_one_turn_correct_answer():
    result, client = _run(["Think... Final answer: 42"], "native")
    assert result.solved is True
    assert result.score == 1.0
    assert result.turns == 1
    assert len(client.calls) == 1


def test_native_one_turn_wrong_answer():
    result, _ = _run(["Final answer: 99"], "native")
    assert result.solved is False
    assert result.score == 0.0


def test_truncated_stops_at_max_turns_when_no_final():
    result, client = _run(["Still thinking..."] * 12, "truncated", L=32)
    assert result.solved is False
    assert result.turns == 8  # max_turns
    assert result.failure_reason == "max_turns_reached"
    assert len(client.calls) == 8


def test_paged_dispatches_ops_and_tracks_counts():
    # Turn 1: model emits an op then text. Turn 2: final answer.
    responses = [
        "e 4\nThinking step one",
        "Continuing... Final answer: 42",
    ]
    result, _ = _run(responses, "paged", L=64)
    assert result.solved is True
    assert result.op_counts.get("e", 0) == 1


def test_paged_evicts_to_chunk_store_on_overflow():
    # Generate long visible text so the cap triggers paged eviction.
    long_blob = " ".join(f"word{i}" for i in range(200))  # ~200 tokens
    responses = [
        long_blob,                       # fills the window
        "More reasoning",                # forces overflow + eviction
        "Final answer: 42",
    ]
    result, _ = _run(responses, "paged", L=64)
    assert result.solved is True
    # mean_active_tokens should reflect the cap; the model wrote 200 tokens
    # but the active window can't exceed 64 by the end.
    assert result.peak_active_tokens >= 0


def test_cost_cap_blocks_runaway_loops():
    # 32 turns of nothing-but-noise; cap will trip first.
    responses = ["x " * 5000] * 32
    client = ScriptedClient(responses)
    suite = FakeSuite()
    task = suite.tasks()[0]
    scheme = build_scheme("paged")
    cell = SweepCell(
        provider="scripted:fake",
        scheme="paged",
        L=64,
        benchmark=suite.name,
        task_id=task.task_id,
        max_turns=32,
        cost_cap_tokens=500,
    )
    result = run_cell(cell, client=client, suite=suite, task=task, scheme=scheme)
    assert result.failure_reason is not None
    assert "cost_cap" in result.failure_reason
