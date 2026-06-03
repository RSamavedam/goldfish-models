"""SWE-bench-specific wrapper around run_shell_cell.

Wires the `before_first_turn` and `after_last_turn` hooks so each
trajectory:

  - starts with the target repo checked out at base_commit under
    `<agent_root>/repo/`
  - is scored against an extracted unified-diff patch (NOT against the
    raw concatenation of user_output files, which would include
    arbitrary prose the model emitted alongside its patch).

Usage:

    suite = SweBenchVerifiedSuite(scorer_mode="subprocess")
    task = suite.tasks()[0]
    cell = ShellCell(
        provider="openai:gpt-5", L=2048, benchmark="swe_bench_verified",
        task_id=task.task_id, cost_cap_tokens=250_000,
        command_timeout_s=120.0,
    )
    result = run_swe_bench_cell(cell, client=client, suite=suite, task=task)

The `repo_cache_dir` kwarg points at a local cache of cloned repos
(shared across trajectories) to avoid re-cloning every task. Highly
recommended for the cloud sweep.
"""

from __future__ import annotations

from pathlib import Path

from rlm_paged.bench.base import BenchSuite, BenchTask
from rlm_paged.bench.swe_bench import (
    bootstrap_repo_into_agent_root,
    extract_patch_from_agent_root,
)
from rlm_paged.client.base import LLMClient
from rlm_paged.shell.agent_fs import AgentFS
from rlm_paged.shell.shell_runner_cell import (
    ShellCell,
    ShellCellResult,
    run_shell_cell,
)


def run_swe_bench_cell(
    cell: ShellCell,
    *,
    client: LLMClient,
    suite: BenchSuite,
    task: BenchTask,
    repo_cache_dir: Path | None = None,
    keep_agent_dir: bool = False,
) -> ShellCellResult:
    """Run one SWE-bench trajectory under the shell harness.

    The task is expected to be from `SweBenchVerifiedSuite` (its payload
    must carry `repo` and `base_commit`).
    """
    repo = task.payload.get("repo")
    base_commit = task.payload.get("base_commit")
    if not repo or not base_commit:
        return ShellCellResult(
            cell=cell,
            solved=False,
            score=0.0,
            final_answer_text="",
            failure_reason="malformed_task: missing repo or base_commit",
        )

    def _bootstrap(fs: AgentFS) -> None:
        bootstrap_repo_into_agent_root(
            fs.root,
            repo=repo,
            base_commit=base_commit,
            cache_dir=repo_cache_dir,
        )

    def _extract_patch(fs: AgentFS, _preliminary: ShellCellResult) -> str:
        return extract_patch_from_agent_root(fs.root)

    return run_shell_cell(
        cell,
        client=client,
        suite=suite,
        task=task,
        keep_agent_dir=keep_agent_dir,
        before_first_turn=_bootstrap,
        after_last_turn=_extract_patch,
    )
