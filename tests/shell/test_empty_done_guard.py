"""Regression test for bug 12.

o4-mini on smoke 5 said `done` with an empty 0-byte answer.patch in
user_output/, after its own sanity check (`wc -l ...`) clearly showed
the file was empty. Prompt rules failed to stop this. The harness now
refuses the `done` and gives the model another turn to fix it.
"""

from __future__ import annotations

from rlm_paged.bench.base import FAMILY_CODING, BenchSuite, BenchTask
from rlm_paged.client.base import GenerationResult, LLMClient
from rlm_paged.shell import ShellCell, run_shell_cell


class _Scripted(LLMClient):
    def __init__(self, responses):
        self.responses = list(responses)
    @property
    def name(self): return "scripted:done-guard"
    def generate(self, prompt, *, max_tokens, stop=None, temperature=0.0, system=None):
        text = self.responses.pop(0) if self.responses else ""
        return GenerationResult(
            text=text, input_tokens=1, output_tokens=1, finish_reason="stop"
        )


class _Suite(BenchSuite):
    @property
    def family(self): return FAMILY_CODING
    @property
    def name(self): return "done-guard"
    def tasks(self): return [BenchTask("t", FAMILY_CODING, {}, {})]
    def task_prompt(self, t): return "deliver a real artifact"
    def score(self, t, r):
        ok = "+real change" in r
        return ok, 1.0 if ok else 0.0


def test_done_with_empty_user_output_is_refused_and_retried():
    client = _Scripted([
        "```bash\ndone\n```\n",
        "```bash\n"
        "echo '+real change' > repo.patch\n"
        "export repo.patch\n"
        "done\n"
        "```\n",
    ])
    suite = _Suite()
    cell = ShellCell(
        provider="scripted:done-guard", L=4096, benchmark="done-guard",
        task_id="t", max_turns=6, command_timeout_s=5.0,
    )
    result = run_shell_cell(cell, client=client, suite=suite, task=suite.tasks()[0])
    assert result.solved is True, f"failure_reason={result.failure_reason}"
    assert result.empty_done_retries == 1
    assert result.exports_written == 1
    assert result.user_output_files == 1


def test_done_with_zero_byte_export_is_refused():
    client = _Scripted([
        "```bash\n"
        "touch empty.patch\n"
        "export empty.patch\n"
        "done\n"
        "```\n",
        "```bash\n"
        "echo '+real change' > real.patch\n"
        "export real.patch\n"
        "done\n"
        "```\n",
    ])
    suite = _Suite()
    cell = ShellCell(
        provider="scripted:done-guard", L=4096, benchmark="done-guard",
        task_id="t", max_turns=6, command_timeout_s=5.0,
    )
    result = run_shell_cell(cell, client=client, suite=suite, task=suite.tasks()[0])
    assert result.empty_done_retries == 1
    assert result.solved is True


def test_done_guard_eventually_gives_up():
    client = _Scripted([
        "```bash\ndone\n```\n",  # refused 1
        "```bash\ndone\n```\n",  # refused 2
        "```bash\ndone\n```\n",  # allowed (giving up)
    ])
    suite = _Suite()
    cell = ShellCell(
        provider="scripted:done-guard", L=4096, benchmark="done-guard",
        task_id="t", max_turns=8, command_timeout_s=5.0,
    )
    result = run_shell_cell(cell, client=client, suite=suite, task=suite.tasks()[0])
    assert result.empty_done_retries == 2
    assert result.solved is False
    assert result.exports_written == 0
