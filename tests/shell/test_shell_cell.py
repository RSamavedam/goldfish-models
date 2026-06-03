from __future__ import annotations

from rlm_paged.bench.base import FAMILY_TTC, BenchSuite, BenchTask
from rlm_paged.client.base import GenerationResult, LLMClient
from rlm_paged.shell import ShellCell, run_shell_cell


class _MathSuite(BenchSuite):
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
                payload={"question": "compute sum of 0..10"},
                expected={"answer": "55"},
            )
        ]

    def task_prompt(self, task: BenchTask) -> str:
        return task.payload["question"]

    def score(self, task: BenchTask, response: str) -> tuple[bool, float]:
        ok = "55" in response
        return ok, 1.0 if ok else 0.0


class _ScriptedClient(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.systems: list[str] = []

    @property
    def name(self) -> str:
        return "scripted:shell"

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


def _cell(L: int = 512, max_turns: int = 6) -> ShellCell:
    return ShellCell(
        provider="scripted:shell",
        L=L,
        benchmark="fake-math",
        task_id="m1",
        max_turns=max_turns,
    )


def test_one_turn_compute_export_done():
    """Model writes a python one-liner, exports the answer, and exits."""
    responses = [
        "I'll compute it.\n\n"
        "```bash\n"
        "python3 -c 'print(sum(range(11)))' > answer.txt\n"
        "export answer.txt\n"
        "done\n"
        "```\n",
    ]
    client = _ScriptedClient(responses)
    suite = _MathSuite()
    result = run_shell_cell(
        _cell(), client=client, suite=suite, task=suite.tasks()[0]
    )
    assert result.solved is True
    assert result.score == 1.0
    assert result.turns == 1
    assert result.user_output_files >= 1
    assert result.commands_executed == 1   # the python3 call
    assert result.commands_intercepted == 2  # export + done


def test_export_string_writes_literal_answer():
    responses = [
        '```bash\n'
        'export-string "55"\n'
        'done\n'
        '```\n',
    ]
    client = _ScriptedClient(responses)
    suite = _MathSuite()
    result = run_shell_cell(
        _cell(), client=client, suite=suite, task=suite.tasks()[0]
    )
    assert result.solved is True
    assert result.user_output_files == 1


def test_multi_turn_uses_prior_history_in_context():
    """First turn writes a fact to a notes file; second turn reads it."""
    responses = [
        # Turn 0: write 55 to a note, no done.
        "```bash\n"
        "echo 55 > notes.txt\n"
        "```\n",
        # Turn 1: read it, export, done.
        "```bash\n"
        "cat notes.txt > answer.txt\n"
        "export answer.txt\n"
        "done\n"
        "```\n",
    ]
    client = _ScriptedClient(responses)
    suite = _MathSuite()
    result = run_shell_cell(
        _cell(), client=client, suite=suite, task=suite.tasks()[0]
    )
    assert result.solved is True
    assert result.turns == 2


def test_max_turns_terminates_when_no_done():
    """Without `done`, the loop hits max_turns and scores whatever's in user_output/."""
    responses = ["```bash\necho still working\n```\n"] * 5
    client = _ScriptedClient(responses)
    suite = _MathSuite()
    result = run_shell_cell(
        _cell(max_turns=3), client=client, suite=suite, task=suite.tasks()[0]
    )
    assert result.failure_reason == "max_turns_reached"
    assert result.solved is False
    assert result.turns == 3


def test_failed_command_reflected_in_history_but_loop_continues():
    """If a command fails (e.g. cat of non-existent file), the run continues
    so subsequent commands in the same turn still execute."""
    responses = [
        "```bash\n"
        "cat does_not_exist.txt\n"
        "export-string \"55\"\n"
        "done\n"
        "```\n",
    ]
    client = _ScriptedClient(responses)
    suite = _MathSuite()
    result = run_shell_cell(
        _cell(), client=client, suite=suite, task=suite.tasks()[0]
    )
    assert result.solved is True
    assert result.commands_failed >= 1
    assert result.user_output_files == 1


def test_system_prompt_describes_shell_environment():
    responses = ["```bash\nexport-string \"55\"\ndone\n```\n"]
    client = _ScriptedClient(responses)
    suite = _MathSuite()
    run_shell_cell(_cell(), client=client, suite=suite, task=suite.tasks()[0])
    sp = client.systems[0]
    assert "instructions.txt" in sp
    assert "history.txt" in sp
    assert "export" in sp
    assert "done" in sp


def test_turn_zero_context_is_initial_hint():
    responses = ["```bash\nexport-string \"55\"\ndone\n```\n"]
    client = _ScriptedClient(responses)
    suite = _MathSuite()
    run_shell_cell(_cell(), client=client, suite=suite, task=suite.tasks()[0])
    assert "turn 0" in client.prompts[0].lower()
    assert "instructions.txt" in client.prompts[0]


def test_history_visible_to_model_via_cat():
    """Model can `cat history.txt` (or a tail) to see what it did before."""
    responses = [
        "```bash\n"
        "echo 'fact A' >> facts.txt\n"
        "```\n",
        # Now read history file
        "```bash\n"
        "wc -c history.txt\n"
        "tail -50 history.txt > recap.txt\n"
        "cat facts.txt >> answer.txt\n"
        "export-string \"55\"\n"
        "done\n"
        "```\n",
    ]
    client = _ScriptedClient(responses)
    suite = _MathSuite()
    result = run_shell_cell(
        _cell(), client=client, suite=suite, task=suite.tasks()[0]
    )
    assert result.solved is True
    assert result.turns == 2


def test_provider_error_does_not_crash_cell():
    """If client.generate raises, the cell records a provider_error and returns."""

    class _BrokenClient(LLMClient):
        @property
        def name(self) -> str:
            return "broken"

        def generate(self, prompt, *, max_tokens, stop=None, temperature=0.0, system=None):
            raise RuntimeError("API down")

    suite = _MathSuite()
    result = run_shell_cell(
        _cell(), client=_BrokenClient(), suite=suite, task=suite.tasks()[0]
    )
    assert result.failure_reason is not None
    assert "provider_error" in result.failure_reason
    assert result.solved is False
