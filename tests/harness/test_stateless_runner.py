from __future__ import annotations

from rlm_paged.bench.base import FAMILY_TTC, BenchSuite, BenchTask
from rlm_paged.client.base import GenerationResult, LLMClient
from rlm_paged.harness import StatelessCell, run_stateless_cell


class _FakeSuite(BenchSuite):
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
                payload={"question": "what is the answer?"},
                expected={"answer": "42"},
            )
        ]

    def task_prompt(self, task: BenchTask) -> str:
        return task.payload["question"]

    def score(self, task: BenchTask, response: str) -> tuple[bool, float]:
        ok = "42" in response and "final answer" in response.lower()
        return ok, 1.0 if ok else 0.0


class _ScriptedClient(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts_received: list[str] = []
        self.systems_received: list[str] = []

    @property
    def name(self) -> str:
        return "scripted:fake"

    def generate(self, prompt, *, max_tokens, stop=None, temperature=0.0, system=None):
        self.prompts_received.append(prompt)
        self.systems_received.append(system or "")
        text = self.responses.pop(0) if self.responses else ""
        return GenerationResult(
            text=text,
            input_tokens=max(1, len(prompt.split())),
            output_tokens=max(1, len(text.split())),
            finish_reason="stop",
        )


def _cell(L: int = 128, max_turns: int = 8) -> StatelessCell:
    return StatelessCell(
        provider="scripted:fake",
        L=L,
        benchmark="fake",
        task_id="fake-001",
        max_turns=max_turns,
    )


def test_one_turn_solve_through_continue():
    """Model emits a continue containing the final answer; harness scores."""
    client = _ScriptedClient(
        [
            "continue\n"
            "    Final answer: 42\n"
        ]
    )
    suite = _FakeSuite()
    result = run_stateless_cell(
        _cell(), client=client, suite=suite, task=suite.tasks()[0]
    )
    assert result.solved is True
    assert result.score == 1.0
    assert result.turns == 1


def test_missing_continue_retries_then_fails():
    """No `continue` op for three straight turns => failure_reason set."""
    client = _ScriptedClient(
        [
            "note \"trying\"\n",
            "note \"still trying\"\n",
            "note \"giving up\"\n",
        ]
    )
    suite = _FakeSuite()
    result = run_stateless_cell(
        _cell(max_turns=8), client=client, suite=suite, task=suite.tasks()[0]
    )
    assert result.solved is False
    assert result.failure_reason == "missing_continue"
    assert result.missing_continue_retries == 2


def test_system_prompt_includes_store_stats():
    client = _ScriptedClient(["continue\n    Final answer: 42\n"])
    suite = _FakeSuite()
    run_stateless_cell(_cell(), client=client, suite=suite, task=suite.tasks()[0])
    sys_prompt = client.systems_received[0]
    assert "task: 1 blocks" in sys_prompt
    assert "note: (none)" in sys_prompt


def test_multi_turn_uses_prior_continuing_instruction():
    """Second turn's user prompt should contain the first turn's continue."""
    client = _ScriptedClient(
        [
            "note \"start\"\ncontinue\n    work in progress\n",
            "continue\n    Final answer: 42\n",
        ]
    )
    suite = _FakeSuite()
    result = run_stateless_cell(
        _cell(), client=client, suite=suite, task=suite.tasks()[0]
    )
    assert result.solved is True
    assert result.turns == 2
    # Turn-2 prompt must echo turn-1's continuing_instruction.
    assert "work in progress" in client.prompts_received[1]


def test_query_results_land_in_next_turn_input():
    """First turn writes a note + queues a query; second turn sees the note."""
    client = _ScriptedClient(
        [
            "note tag=k\n"
            "    secret-token-XYZ\n"
            "query note 0 -1 tag=k\n"
            "continue\n"
            "    retrieving\n",
            "continue\n"
            "    Final answer: 42\n",
        ]
    )
    suite = _FakeSuite()
    result = run_stateless_cell(
        _cell(L=512), client=client, suite=suite, task=suite.tasks()[0]
    )
    assert result.solved is True
    # The second user prompt should contain the queried note.
    assert "secret-token-XYZ" in client.prompts_received[1]
    assert "[note:0]" in client.prompts_received[1]


def test_max_turns_terminates_with_failure_reason():
    client = _ScriptedClient(
        ["continue\n    keep going\n"] * 16
    )
    suite = _FakeSuite()
    result = run_stateless_cell(
        _cell(max_turns=3),
        client=client,
        suite=suite,
        task=suite.tasks()[0],
    )
    assert result.failure_reason == "max_turns_reached"
    assert result.turns == 3


def test_external_call_becomes_observation_next_turn():
    """A `call` op result lands as an observation block, queryable next turn."""
    client = _ScriptedClient(
        [
            "call echo\n"
            "    hello-world\n"
            "query observation 0 -1\n"
            "continue\n"
            "    using echo result\n",
            "continue\n"
            "    Final answer: 42\n",
        ]
    )
    suite = _FakeSuite()
    result = run_stateless_cell(
        _cell(L=512), client=client, suite=suite, task=suite.tasks()[0]
    )
    assert result.solved is True
    assert "echo: hello-world" in client.prompts_received[1]
    assert "[observation:0]" in client.prompts_received[1]
