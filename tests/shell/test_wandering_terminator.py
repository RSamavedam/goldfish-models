"""Regression test for bug 13: terminate trajectories with N consecutive
empty turns.

Smoke 6 caught o4-mini stuck producing 7+ consecutive empty/prose-only
responses, burning ~$0.50 per turn for nothing. The runner now bails
after `max_consecutive_empty_turns` (default 4)."""

from __future__ import annotations

from rlm_paged.bench.base import FAMILY_CODING, BenchSuite, BenchTask
from rlm_paged.client.base import GenerationResult, LLMClient
from rlm_paged.shell import ShellCell, run_shell_cell


class _Scripted(LLMClient):
    def __init__(self, responses):
        self.responses = list(responses)
    @property
    def name(self): return "scripted:wander"
    def generate(self, prompt, *, max_tokens, stop=None, temperature=0.0, system=None):
        text = self.responses.pop(0) if self.responses else ""
        return GenerationResult(text=text, input_tokens=1, output_tokens=1, finish_reason="stop")


class _Suite(BenchSuite):
    @property
    def family(self): return FAMILY_CODING
    @property
    def name(self): return "wander"
    def tasks(self): return [BenchTask("t", FAMILY_CODING, {}, {})]
    def task_prompt(self, t): return "do something"
    def score(self, t, r): return False, 0.0


def test_consecutive_empty_turns_force_termination():
    """4 consecutive prose-only responses → harness bails."""
    client = _Scripted([
        "Let me think about this.",
        "Hmm.",
        "I'll plan more.",
        "Actually let me reconsider.",
        "```bash\necho late\n```",  # never executed
    ])
    cell = ShellCell(
        provider="scripted:wander", L=4096, benchmark="wander",
        task_id="t", max_turns=10, command_timeout_s=5.0,
    )
    result = run_shell_cell(cell, client=client, suite=_Suite(), task=_Suite().tasks()[0])
    assert result.failure_reason is not None
    assert "wandering" in result.failure_reason
    # Turns are 0-indexed and the break fires before the next increment,
    # so 4 empty turns (0,1,2,3) terminate at result.turns == 3.
    assert result.turns == 3


def test_command_resets_empty_counter():
    """A turn with at least one command resets the counter."""
    client = _Scripted([
        "thinking...",
        "still thinking",
        "```bash\necho actually-doing-something\n```",
        "back to thinking",
        "more thinking",
        "more more",
        "```bash\nexport-string 'fin'\ndone\n```",
    ])
    cell = ShellCell(
        provider="scripted:wander", L=4096, benchmark="wander",
        task_id="t", max_turns=10, command_timeout_s=5.0,
    )
    result = run_shell_cell(cell, client=client, suite=_Suite(), task=_Suite().tasks()[0])
    assert result.failure_reason is None or "wander" not in result.failure_reason
