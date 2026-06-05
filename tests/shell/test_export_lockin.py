"""Auto-terminate on first valid-diff export.

The harness ends the trajectory the moment a `export` lands a file
whose contents parse as a unified diff. Prevents the
"good patch -> iterate -> overwrite with garbage" failure mode.

Triggers ONLY for suites whose system_prompt_addendum() is non-None
(SWE-bench-shaped suites). Other suites may legitimately export
multiple non-diff artifacts.
"""

from __future__ import annotations

from rlm_paged.bench.base import FAMILY_CODING, BenchSuite, BenchTask
from rlm_paged.client.base import GenerationResult, LLMClient
from rlm_paged.shell import ShellCell, run_shell_cell


class _Scripted(LLMClient):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    @property
    def name(self): return "scripted:lockin"

    def generate(self, prompt, *, max_tokens, stop=None, temperature=0.0, system=None):
        self.calls += 1
        text = self.responses.pop(0) if self.responses else ""
        return GenerationResult(
            text=text, input_tokens=1, output_tokens=1,
            thinking_tokens=0, finish_reason="stop",
        )


_VALID_DIFF = """diff --git a/foo.py b/foo.py
index 0000001..0000002 100644
--- a/foo.py
+++ b/foo.py
@@ -1,1 +1,1 @@
-x = 1
+x = 2
"""


class _SweLikeSuite(BenchSuite):
    """Suite that returns a non-None addendum -> diff-lock applies."""
    @property
    def family(self): return FAMILY_CODING
    @property
    def name(self): return "swe_like"
    def tasks(self): return [BenchTask("t", FAMILY_CODING, {}, {})]
    def task_prompt(self, t): return "task"
    def score(self, t, r): return True, 1.0
    def system_prompt_addendum(self): return "SWE-LIKE"


class _NonSweSuite(BenchSuite):
    """Suite without addendum -> diff-lock does NOT apply."""
    @property
    def family(self): return FAMILY_CODING
    @property
    def name(self): return "non_swe"
    def tasks(self): return [BenchTask("t", FAMILY_CODING, {}, {})]
    def task_prompt(self, t): return "task"
    def score(self, t, r): return True, 1.0
    # no system_prompt_addendum override -> returns None


def test_valid_diff_export_locks_in_swe_like_suite():
    """First export that parses as a unified diff terminates the
    trajectory before subsequent turns can overwrite it."""
    client = _Scripted([
        # Turn 0: write the diff, export it. Should auto-terminate
        # before turn 1 even runs.
        f"```bash\ncat > /tmp/answer.patch <<'EOF'\n{_VALID_DIFF}EOF\nexport /tmp/answer.patch\n```",
        # Turn 1: model would corrupt the patch here. Should NEVER run.
        "```bash\necho 'this should never execute' > user_output/answer.patch\n```",
        "```bash\ndone\n```",
    ])
    cell = ShellCell(
        provider="scripted:lockin", L=2048, benchmark="swe_like",
        task_id="t", max_turns=8,
    )
    result = run_shell_cell(
        cell, client=client, suite=_SweLikeSuite(),
        task=_SweLikeSuite().tasks()[0],
    )
    # We should have made only ONE API call (turn 0) before terminating.
    assert client.calls == 1, f"expected 1 call, got {client.calls}"
    assert result.exports_written == 1
    # The locked-in file is still there, not overwritten.
    assert result.user_output_files == 1


def test_empty_export_does_not_lock_in():
    """Empty exports should not trigger auto-terminate — the model gets
    to keep trying."""
    client = _Scripted([
        # Turn 0: try to export an empty patch (heredoc edit didn't take).
        "```bash\ntouch /tmp/empty.patch\nexport /tmp/empty.patch\n```",
        # Turn 1: must still run.
        f"```bash\ncat > /tmp/real.patch <<'EOF'\n{_VALID_DIFF}EOF\nexport /tmp/real.patch\n```",
        "```bash\ndone\n```",
    ])
    cell = ShellCell(
        provider="scripted:lockin", L=2048, benchmark="swe_like",
        task_id="t", max_turns=8,
    )
    result = run_shell_cell(
        cell, client=client, suite=_SweLikeSuite(),
        task=_SweLikeSuite().tasks()[0],
    )
    # Turn 0's empty export should NOT have terminated.
    # Turn 1's valid diff should have.
    assert client.calls == 2, f"expected 2 calls, got {client.calls}"


def test_non_diff_export_does_not_lock_in():
    """Exporting a Python file (not a diff) should not trigger
    auto-terminate. We watched gpt-4o do this in paper5."""
    client = _Scripted([
        # Turn 0: export a python file (no --- / +++ / @@).
        "```bash\nprintf 'def foo():\\n    return 1\\n' > /tmp/code.py\nexport /tmp/code.py\n```",
        # Turn 1: now export a real diff.
        f"```bash\ncat > /tmp/real.patch <<'EOF'\n{_VALID_DIFF}EOF\nexport /tmp/real.patch\n```",
        "```bash\ndone\n```",
    ])
    cell = ShellCell(
        provider="scripted:lockin", L=2048, benchmark="swe_like",
        task_id="t", max_turns=8,
    )
    result = run_shell_cell(
        cell, client=client, suite=_SweLikeSuite(),
        task=_SweLikeSuite().tasks()[0],
    )
    assert client.calls == 2


def test_diff_lock_disabled_for_non_swe_suite():
    """For benchmarks without the SWE-bench addendum, diff exports do
    NOT auto-terminate — the suite may legitimately want multi-step
    or non-diff exports."""
    client = _Scripted([
        f"```bash\ncat > /tmp/answer.patch <<'EOF'\n{_VALID_DIFF}EOF\nexport /tmp/answer.patch\n```",
        # Turn 1: ALSO runs, even though turn 0 looked like a valid diff.
        "```bash\necho 'still running' >> notes.txt\n```",
        "```bash\ndone\n```",
    ])
    cell = ShellCell(
        provider="scripted:lockin", L=2048, benchmark="non_swe",
        task_id="t", max_turns=8,
    )
    result = run_shell_cell(
        cell, client=client, suite=_NonSweSuite(),
        task=_NonSweSuite().tasks()[0],
    )
    assert client.calls >= 2
