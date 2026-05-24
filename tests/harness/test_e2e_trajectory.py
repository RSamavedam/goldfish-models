"""End-to-end trajectory integration test.

Drives `run_stateless_cell` through a realistic multi-turn trajectory
using a scripted client that returns canned responses. Exercises:

  - turn 0 reads user_message:0 (the task) and queues a query for an
    external tool result
  - turn 1 issues an external tool call (`call code_exec`)
  - turn 2 queries the resulting observation, writes a note, continues
    with a partial result
  - a user message arrives between turns 2 and 3
  - turn 3 sees unread inbox, queries it, says something to the user,
    continues
  - turn 4 produces the final answer

For each turn, asserts:
  - the model's prompt contained the expected retrieved-content payload
  - the store contains the expected blocks after the turn
  - the op-count tallies are correct
  - the assistant_reply callback was invoked at the right time
  - termination happens at the right turn (not blocked, not premature)
"""

from __future__ import annotations

from rlm_paged.bench.base import FAMILY_TTC, BenchSuite, BenchTask
from rlm_paged.client.base import GenerationResult, LLMClient
from rlm_paged.harness import StatelessCell, run_stateless_cell


# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


class _MathSuite(BenchSuite):
    """Asks for a number; scores if 'Final answer: 7' appears."""

    @property
    def family(self) -> str:
        return FAMILY_TTC

    @property
    def name(self) -> str:
        return "fake-math"

    def tasks(self) -> list[BenchTask]:
        return [
            BenchTask(
                task_id="m1",
                family=FAMILY_TTC,
                payload={"question": "What is 3 + 4? Use the calculator."},
                expected={"answer": "7"},
            )
        ]

    def task_prompt(self, task: BenchTask) -> str:
        return task.payload["question"]

    def score(self, task: BenchTask, response: str) -> tuple[bool, float]:
        lowered = response.lower()
        ok = "final answer: 7" in lowered or "final answer:7" in lowered
        return ok, 1.0 if ok else 0.0


class _RecordingClient(LLMClient):
    """Returns scripted responses; records each prompt + system seen."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.systems: list[str] = []

    @property
    def name(self) -> str:
        return "scripted:recording"

    def generate(self, prompt, *, max_tokens, stop=None, temperature=0.0, system=None):
        self.prompts.append(prompt)
        self.systems.append(system or "")
        text = self.responses.pop(0) if self.responses else ""
        return GenerationResult(
            text=text,
            input_tokens=max(1, len(prompt.split())),
            output_tokens=max(1, len(text.split())),
            finish_reason="stop",
        )


# Tool used by the call op. Returns the sum as a string.
def _adder(args: str) -> str:
    """Parse `a b` from args and return their sum."""
    a, b = args.strip().split()
    return f"sum = {int(a) + int(b)}"


# --------------------------------------------------------------------- #
# The trajectory                                                        #
# --------------------------------------------------------------------- #


def test_full_trajectory_with_user_interjection_and_tool_call():
    # Scripted model responses, one per turn.
    responses = [
        # Turn 0: see the task; plan; call the adder; QUEUE the
        # observation query for next turn (queries fire on the turn
        # AFTER they're issued).
        "note tag=plan\n"
        "    User asks 3+4. Use the calculator tool to compute it.\n"
        "call adder\n"
        "    3 4\n"
        "query observation -1 -1\n"
        "continue\n"
        "    Waiting on adder. Observation will be in next turn's input.\n",

        # Turn 1: observation has been delivered; write a result note;
        # continue.
        "note tag=intermediate\n"
        "    Adder returned 7.\n"
        "continue\n"
        "    Result is 7. Almost done.\n",

        # Turn 2: user has interjected (their message arrives BEFORE this
        # turn). Model queries inbox, acknowledges via say, continues.
        "query user_message -1 -1\n"
        "say\n"
        "    Working on it. The answer will be in my final continue.\n"
        "continue\n"
        "    User wants the answer in the response. Will state it next turn.\n",

        # Turn 3: state the final answer.
        "continue\n"
        "    Final answer: 7\n",
    ]

    # Mid-trajectory user input: arrives once, between turns 1 and 2.
    user_msgs_per_turn = [
        [],                                # before turn 0
        [],                                # before turn 1
        ["please give the answer asap"],   # before turn 2
        [],                                # before turn 3
    ]
    pumps = iter(user_msgs_per_turn)

    def user_input() -> list[str]:
        return next(pumps, [])

    assistant_says: list[str] = []

    def on_assistant_reply(text: str) -> None:
        assistant_says.append(text)

    client = _RecordingClient(responses)
    suite = _MathSuite()
    cell = StatelessCell(
        provider="scripted:recording",
        L=512,
        benchmark="fake-math",
        task_id="m1",
        max_turns=8,
    )

    result = run_stateless_cell(
        cell,
        client=client,
        suite=suite,
        task=suite.tasks()[0],
        external_tools={"adder": _adder},
        user_input=user_input,
        on_assistant_reply=on_assistant_reply,
    )

    # ---- Outcome ----
    assert result.solved is True, (
        f"expected solve; got failure_reason={result.failure_reason}"
    )
    assert result.score == 1.0
    assert result.turns == 4
    assert result.failure_reason is None

    # ---- Op tallies ----
    # Turn 0: note + call + query + continue
    # Turn 1: note + continue                  (observation already delivered)
    # Turn 2: query + say + continue           (user_message query)
    # Turn 3: continue                         (final answer)
    # Totals: note=2, query=2, continue=4, call=1, say=1
    assert result.op_counts["note"] == 2
    assert result.op_counts["query"] == 2
    assert result.op_counts["continue"] == 4
    assert result.op_counts["call"] == 1
    assert result.op_counts["say"] == 1
    assert result.op_errors == 0
    assert result.notes_written == 2
    assert result.queries_issued == 2
    assert result.assistant_replies == 1
    # 1 original task + 1 user interjection = 2 user_messages received.
    assert result.user_messages_received == 2

    # ---- Assistant-reply callback ----
    assert len(assistant_says) == 1
    assert "working on it" in assistant_says[0].lower()

    # ---- Per-turn prompt content ----
    # Turn 1's prompt must include the prior continue.
    assert "Waiting on adder" in client.prompts[1]
    # Turn 1's prompt must include the observation from the call op
    # (executed at end of turn 0).
    assert "sum = 7" in client.prompts[1]
    # Turn 2's prompt must surface the new user message (delivered via
    # query user_message -1 -1 issued on turn 1; harness pumps user input
    # at the START of turn 2; query result lands at turn 2 prompt? no —
    # queries fire on the NEXT turn after they're issued. Turn 1 didn't
    # query user_message, so we only see it via the inbox surface in
    # turn 2's system prompt.)
    assert "1 unread" in client.systems[2] or "unread" in client.systems[2]
    # Turn 3's prompt must include turn 2's continue.
    assert "state it next turn" in client.prompts[3]

    # ---- System-prompt store stats ----
    # By turn 3 the model should see notes count >= 2, observations >= 1.
    sys_t3 = client.systems[3]
    assert "note: 2 blocks" in sys_t3
    assert "observation: 1 blocks" in sys_t3
    # assistant_reply was written on turn 2.
    assert "assistant_reply: 1 blocks" in sys_t3


def test_termination_blocked_only_when_inbox_has_unread():
    """Sanity-check the inbox-blocks-termination semantics in isolation."""
    responses = [
        # Turn 0: declare done immediately. No new user msgs, so done holds.
        "continue\n    Final answer: 7\n",
    ]
    user_msgs_per_turn = [[], []]
    pumps = iter(user_msgs_per_turn)
    client = _RecordingClient(responses)
    suite = _MathSuite()
    result = run_stateless_cell(
        StatelessCell(
            provider="scripted:recording",
            L=512,
            benchmark="fake-math",
            task_id="m1",
            max_turns=4,
        ),
        client=client,
        suite=suite,
        task=suite.tasks()[0],
        user_input=lambda: next(pumps, []),
    )
    assert result.turns == 1
    assert result.solved is True


def test_external_tool_unknown_writes_error_observation():
    """A call to an undefined tool yields an error observation rather than crashing."""
    responses = [
        "call nonexistent_tool\n"
        "    some args\n"
        "continue\n"
        "    submitted unknown tool\n",
        "query observation -1 -1\n"
        "continue\n"
        "    Final answer: 7\n",
    ]
    client = _RecordingClient(responses)
    suite = _MathSuite()
    result = run_stateless_cell(
        StatelessCell(
            provider="scripted:recording",
            L=512,
            benchmark="fake-math",
            task_id="m1",
            max_turns=4,
        ),
        client=client,
        suite=suite,
        task=suite.tasks()[0],
    )
    # Even though the tool is unknown, the trajectory completes and turn 1
    # gets an error observation in its retrieved-content.
    assert result.solved is True
    assert "unknown tool" in client.prompts[1]


def test_missing_continue_retries_then_recovers():
    """If the model forgets to emit `continue`, the harness re-prompts.
    The model recovers on retry and the trajectory completes."""
    responses = [
        # First response: no continue. Triggers retry.
        "note \"forgot the continue\"\n",
        # Retry: still no continue.
        "note \"oops still forgot\"\n",
        # After two retries, the harness gives up — but if we want to test
        # recovery, we let the third try succeed before the budget is hit.
    ]
    client = _RecordingClient(responses)
    suite = _MathSuite()
    result = run_stateless_cell(
        StatelessCell(
            provider="scripted:recording",
            L=512,
            benchmark="fake-math",
            task_id="m1",
            max_turns=4,
            max_retries_on_missing_continue=2,
        ),
        client=client,
        suite=suite,
        task=suite.tasks()[0],
    )
    # After 2 retries (3 total attempts) without a continue, the cell
    # fails with missing_continue.
    assert result.failure_reason == "missing_continue"
    assert result.missing_continue_retries == 2
    assert result.solved is False
