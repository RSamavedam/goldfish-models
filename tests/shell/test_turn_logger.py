"""Tests for the per-turn transcript logger.

The harness exposes a `turn_logger: Callable[[dict], None]` parameter
on run_shell_cell. Each model call emits one row capturing the exact
prompt sent + response received + token telemetry. Downstream analysis
relies on these to understand model behavior."""

from __future__ import annotations

from rlm_paged.bench.base import FAMILY_CODING, BenchSuite, BenchTask
from rlm_paged.client.base import GenerationResult, LLMClient
from rlm_paged.shell import ShellCell, run_shell_cell


class _Scripted(LLMClient):
    def __init__(self, responses):
        self.responses = list(responses)
    @property
    def name(self): return "scripted:turnlog"
    def generate(self, prompt, *, max_tokens, stop=None, temperature=0.0, system=None):
        text = self.responses.pop(0) if self.responses else ""
        return GenerationResult(
            text=text, input_tokens=7, output_tokens=3,
            thinking_tokens=11, finish_reason="stop",
        )


class _Suite(BenchSuite):
    @property
    def family(self): return FAMILY_CODING
    @property
    def name(self): return "turnlog"
    def tasks(self): return [BenchTask("t", FAMILY_CODING, {}, {})]
    def task_prompt(self, t): return "do something"
    def score(self, t, r): return True, 1.0


def test_turn_logger_invoked_once_per_turn():
    rows = []
    client = _Scripted([
        "```bash\necho hi\n```",
        "```bash\nexport-string 'done'\ndone\n```",
    ])
    cell = ShellCell(
        provider="scripted:turnlog", L=2048, benchmark="turnlog",
        task_id="t", max_turns=5,
    )
    run_shell_cell(
        cell, client=client, suite=_Suite(), task=_Suite().tasks()[0],
        turn_logger=rows.append,
    )
    assert len(rows) == 2
    assert rows[0]["turn"] == 0
    assert rows[1]["turn"] == 1


def test_turn_logger_captures_required_fields():
    rows = []
    client = _Scripted(["```bash\nexport-string 'x'\ndone\n```"])
    cell = ShellCell(
        provider="scripted:turnlog", L=2048, benchmark="turnlog",
        task_id="t", max_turns=2,
    )
    run_shell_cell(
        cell, client=client, suite=_Suite(), task=_Suite().tasks()[0],
        turn_logger=rows.append,
    )
    row = rows[0]
    for key in (
        "cell_key", "provider", "benchmark", "task_id", "L", "turn",
        "system_prompt", "user_prompt", "response",
        "input_tokens", "output_tokens", "thinking_tokens", "finish_reason",
    ):
        assert key in row, f"missing key: {key}"
    # The actual prompt + response made it through verbatim.
    assert "export-string 'x'" in row["response"]
    assert row["L"] == 2048
    assert row["provider"] == "scripted:turnlog"
    assert row["input_tokens"] == 7
    assert row["thinking_tokens"] == 11


def test_turn_logger_callback_exception_does_not_crash_trajectory():
    """The harness must be robust to a buggy logger callback — never
    let log failures abort a real trajectory."""
    def boom(_row): raise RuntimeError("logger died")
    client = _Scripted(["```bash\nexport-string 'x'\ndone\n```"])
    cell = ShellCell(
        provider="scripted:turnlog", L=2048, benchmark="turnlog",
        task_id="t", max_turns=2,
    )
    result = run_shell_cell(
        cell, client=client, suite=_Suite(), task=_Suite().tasks()[0],
        turn_logger=boom,
    )
    assert result.solved is True
