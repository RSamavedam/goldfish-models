from __future__ import annotations

from rlm_paged.bench.base import FAMILY_TTC, BenchSuite, BenchTask
from rlm_paged.client.base import GenerationResult, LLMClient
from rlm_paged.harness import StatelessCell, run_stateless_cell


class _FakeSuite(BenchSuite):
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
                payload={"question": "tell me a joke"},
                expected={"answer": "any"},
            )
        ]

    def task_prompt(self, task: BenchTask) -> str:
        return task.payload["question"]

    def score(self, task: BenchTask, response: str) -> tuple[bool, float]:
        ok = "final answer" in response.lower()
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


def _cell(max_turns: int = 8) -> StatelessCell:
    return StatelessCell(
        provider="scripted:fake",
        L=512,
        benchmark="fake",
        task_id="fake-001",
        max_turns=max_turns,
    )


def test_initial_task_is_user_message_zero():
    """The benchmark task becomes user_message:0 and shows as unread."""
    client = _ScriptedClient(["continue\n    Final answer: ok\n"])
    suite = _FakeSuite()
    run_stateless_cell(_cell(), client=client, suite=suite, task=suite.tasks()[0])
    # The system prompt for turn 0 should flag 1 unread user_message.
    sys_0 = client.systems_received[0]
    assert "1 unread" in sys_0
    assert "user_message:0" in sys_0


def test_say_op_invokes_callback_with_text():
    captured: list[str] = []
    client = _ScriptedClient(
        [
            "say\n"
            "    Hi there, computing now.\n"
            "continue\n"
            "    Final answer: ok\n"
        ]
    )
    suite = _FakeSuite()
    result = run_stateless_cell(
        _cell(), client=client, suite=suite, task=suite.tasks()[0],
        on_assistant_reply=captured.append,
    )
    assert result.assistant_replies == 1
    assert captured == ["Hi there, computing now."]
    assert result.op_counts.get("say", 0) == 1


def test_user_input_callable_drops_messages_into_inbox():
    """A user message arriving between turns shows up as unread on turn N+1."""
    # Pre-script: turn 0 the model writes a continue but doesn't claim done.
    # Between turn 0 and 1 the user sends a message. Turn 1's system prompt
    # should reflect 1 new unread user_message.
    new_inputs = [
        [],                                        # before turn 0: nothing
        ["wait, also include the date"],           # before turn 1: new msg
        [],                                        # before turn 2
    ]
    pumps = iter(new_inputs)

    def user_input() -> list[str]:
        return next(pumps, [])

    client = _ScriptedClient(
        [
            "continue\n    working on it\n",       # turn 0: not done
            "query user_message 1 1\n"
            "continue\n    Final answer: noted\n",  # turn 1: queries the new msg, then done
        ]
    )
    suite = _FakeSuite()
    result = run_stateless_cell(
        _cell(), client=client, suite=suite, task=suite.tasks()[0],
        user_input=user_input,
    )

    # Two turns ran; the second's system prompt should have flagged the
    # mid-trajectory user message as unread before the model queried it.
    sys_1 = client.systems_received[1]
    assert "unread" in sys_1
    assert result.user_messages_received == 2  # original task + injected
    # Final answer should still register.
    assert result.solved is True


def test_done_blocked_when_inbox_has_unread():
    """If the model claims done but the user has a fresh unread message,
    the loop keeps going."""
    new_inputs = [
        [],                              # before turn 0
        ["new request"],                 # before turn 1: user interjects
        [],                              # before turn 2
    ]
    pumps = iter(new_inputs)

    def user_input() -> list[str]:
        return next(pumps, [])

    client = _ScriptedClient(
        [
            # Turn 0: model declares done.
            "continue\n    Final answer: 42\n",
            # Turn 1: user message has arrived; harness should NOT terminate
            # despite the prior "done" looking final. Turn 1's response
            # acknowledges and queries the new message.
            "query user_message -1 -1\n"
            "continue\n    handling the new request\n",
            # Turn 2: actually done.
            "continue\n    Final answer: complete\n",
        ]
    )
    suite = _FakeSuite()
    result = run_stateless_cell(
        _cell(), client=client, suite=suite, task=suite.tasks()[0],
        user_input=user_input,
    )
    # 3 turns ran, not 1. The "done" on turn 0 was blocked by the unread
    # user message that arrived between turns 0 and 1.
    assert result.turns == 3
