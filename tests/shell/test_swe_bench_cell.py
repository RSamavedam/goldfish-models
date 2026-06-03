"""Tests for run_swe_bench_cell + the before/after hooks on run_shell_cell.

We mock `bootstrap_repo_into_agent_root` to avoid hitting real git, and
use a fake SWE-bench-shaped suite that returns a known verdict.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rlm_paged.bench.base import FAMILY_CODING, BenchSuite, BenchTask
from rlm_paged.client.base import GenerationResult, LLMClient
from rlm_paged.shell import (
    ShellCell,
    run_shell_cell,
    run_swe_bench_cell,
)
from rlm_paged.shell.agent_fs import AgentFS


class _FakeSweBenchSuite(BenchSuite):
    """Scores by checking the response string for a sentinel."""

    def __init__(self, success_sentinel: str = "+++correct+++") -> None:
        self.success_sentinel = success_sentinel
        self.score_calls: list[tuple[str, str]] = []

    @property
    def family(self) -> str:
        return FAMILY_CODING

    @property
    def name(self) -> str:
        return "fake-swe"

    def tasks(self) -> list[BenchTask]:
        return [
            BenchTask(
                task_id="fake__repo-1",
                family=FAMILY_CODING,
                payload={
                    "instance_id": "fake__repo-1",
                    "repo": "fake/repo",
                    "base_commit": "deadbeef",
                    "problem_statement": "fix the bug",
                },
                expected={},
            )
        ]

    def task_prompt(self, task: BenchTask) -> str:
        return "fix the bug"

    def score(self, task: BenchTask, response: str) -> tuple[bool, float]:
        self.score_calls.append((task.task_id, response))
        ok = self.success_sentinel in response
        return ok, 1.0 if ok else 0.0


class _ScriptedClient(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    @property
    def name(self) -> str:
        return "scripted:swe"

    def generate(self, prompt, *, max_tokens, stop=None, temperature=0.0, system=None):
        text = self.responses.pop(0) if self.responses else ""
        return GenerationResult(
            text=text,
            input_tokens=max(1, len(prompt.split())),
            output_tokens=max(1, len(text.split())),
            finish_reason="stop",
        )


def _cell() -> ShellCell:
    return ShellCell(
        provider="scripted:swe",
        L=512,
        benchmark="fake-swe",
        task_id="fake__repo-1",
        max_turns=4,
    )


# ----- run_shell_cell hooks ----------------------------------------- #


def test_before_first_turn_runs_after_init_and_before_turn_zero(monkeypatch):
    """The hook can mutate the agent FS before the model's first call."""
    bootstrap_called: list[Path] = []

    def hook(fs: AgentFS) -> None:
        bootstrap_called.append(fs.root)
        (fs.root / "bootstrapped.txt").write_text("hello")

    responses = [
        "```bash\n"
        "cat bootstrapped.txt > answer.txt\n"
        "export answer.txt\n"
        "done\n"
        "```\n"
    ]
    suite = _FakeSweBenchSuite(success_sentinel="hello")
    result = run_shell_cell(
        _cell(),
        client=_ScriptedClient(responses),
        suite=suite,
        task=suite.tasks()[0],
        before_first_turn=hook,
    )
    assert len(bootstrap_called) == 1
    assert result.solved is True


def test_before_first_turn_failure_is_fatal_and_logged():
    """A bootstrap exception terminates the trajectory cleanly with a
    failure_reason."""
    def hook(_fs):
        raise RuntimeError("git network down")

    suite = _FakeSweBenchSuite()
    result = run_shell_cell(
        _cell(),
        client=_ScriptedClient([]),
        suite=suite,
        task=suite.tasks()[0],
        before_first_turn=hook,
    )
    assert result.solved is False
    assert result.failure_reason is not None
    assert "bootstrap_error" in result.failure_reason
    assert "git network down" in result.failure_reason


def test_after_last_turn_override_is_used_for_scoring():
    """The hook's returned string replaces the default user_output text."""
    def hook_after(fs, preliminary):
        return "+++correct+++"  # the success sentinel

    responses = ["```bash\ndone\n```\n"]
    suite = _FakeSweBenchSuite()
    result = run_shell_cell(
        _cell(),
        client=_ScriptedClient(responses),
        suite=suite,
        task=suite.tasks()[0],
        after_last_turn=hook_after,
    )
    assert result.solved is True
    # The score saw the override, not the (empty) user_output.
    assert suite.score_calls[0][1] == "+++correct+++"


def test_after_last_turn_returning_none_keeps_default():
    def hook_after(fs, preliminary):
        return None

    responses = ["```bash\nexport-string \"+++correct+++\"\ndone\n```\n"]
    suite = _FakeSweBenchSuite()
    result = run_shell_cell(
        _cell(),
        client=_ScriptedClient(responses),
        suite=suite,
        task=suite.tasks()[0],
        after_last_turn=hook_after,
    )
    assert result.solved is True


# ----- run_swe_bench_cell end-to-end -------------------------------- #


def test_run_swe_bench_cell_bootstraps_and_extracts(monkeypatch):
    """The wrapper calls the bootstrap and extracts the patch via the
    extract helper. We mock both so no git is required."""
    bootstraps: list[dict] = []

    def fake_bootstrap(root, *, repo, base_commit, cache_dir=None, **kwargs):
        bootstraps.append(
            {"root": root, "repo": repo, "base_commit": base_commit}
        )
        (root / "repo").mkdir(parents=True, exist_ok=True)

    def fake_extract(root, **kwargs):
        return "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n"

    monkeypatch.setattr(
        "rlm_paged.shell.swe_bench_cell.bootstrap_repo_into_agent_root",
        fake_bootstrap,
    )
    monkeypatch.setattr(
        "rlm_paged.shell.swe_bench_cell.extract_patch_from_agent_root",
        fake_extract,
    )

    class _DiffSuite(_FakeSweBenchSuite):
        def score(self, task, response):
            ok = "+new" in response  # crude "patch parsed" check
            return ok, 1.0 if ok else 0.0

    suite = _DiffSuite()
    responses = ["```bash\ndone\n```\n"]
    result = run_swe_bench_cell(
        _cell(),
        client=_ScriptedClient(responses),
        suite=suite,
        task=suite.tasks()[0],
    )
    assert len(bootstraps) == 1
    assert bootstraps[0]["repo"] == "fake/repo"
    assert bootstraps[0]["base_commit"] == "deadbeef"
    assert result.solved is True


def test_run_swe_bench_cell_rejects_task_missing_repo_or_commit():
    """A malformed task short-circuits with a clear failure_reason."""
    suite = _FakeSweBenchSuite()
    task = suite.tasks()[0]
    task.payload.pop("repo")
    result = run_swe_bench_cell(
        _cell(),
        client=_ScriptedClient([]),
        suite=suite,
        task=task,
    )
    assert result.solved is False
    assert "malformed_task" in (result.failure_reason or "")
