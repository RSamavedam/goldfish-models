"""Multi-turn driver for the shell-harness architecture.

One `run_shell_cell` invocation = one trajectory. Per turn:

  1. Read the last `k` tokens of history.txt → context.
  2. Prompt = system_prompt (outside k) + context.
  3. Call the model once. Cap output at `k` tokens (post-hoc truncation).
  4. Extract fenced shell blocks in document order; split into commands.
  5. For each command:
       - intercept `export <file>` → copy file to user_output/
       - intercept `export-string "..."` → write literal to user_output/
       - intercept `done` / `exit` → set termination flag
       - else: ShellRunner.run() with cwd at agent root
     Append `$ <cmd>` + stdout + stderr to history.txt as we go.
  6. Append the model's full response to history.txt FIRST, then the
     command transcripts.
  7. If termination flag set OR max_turns hit: stop, score user_output/.

Returns a `ShellCellResult` with metrics structurally compatible with
`StatelessResult` (so analyze.py works unchanged).
"""

from __future__ import annotations

import shlex
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rlm_paged.bench.base import BenchSuite, BenchTask
from rlm_paged.client.base import LLMClient
from rlm_paged.client.tokenizer import count as token_count
from rlm_paged.client.tokenizer import decode as token_decode
from rlm_paged.client.tokenizer import encode as token_encode
from rlm_paged.harness.cost_cap import CostCap, CostCapExceeded
from rlm_paged.shell.agent_fs import AgentFS
from rlm_paged.shell.extractor import extract_shell_commands
from rlm_paged.shell.shell_runner import (
    CommandResult,
    ShellRunner,
)
from rlm_paged.shell.system_prompt import render_system_prompt


@dataclass
class ShellCell:
    """One cell of a shell-harness sweep."""

    provider: str
    L: int                            # = k: per-turn context cap; 0 = unlimited
    benchmark: str
    task_id: str
    seed: int = 0
    max_turns: int = 16
    max_tokens_per_turn: int | None = None  # default: same as L
    cost_cap_tokens: int = 100_000
    command_timeout_s: float = 10.0


@dataclass
class ShellCellResult:
    cell: ShellCell
    solved: bool
    score: float
    final_answer_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    turns: int = 0
    commands_executed: int = 0
    commands_intercepted: int = 0
    commands_failed: int = 0
    exports_written: int = 0
    history_chars_end: int = 0
    user_output_files: int = 0
    wall_seconds: float = 0.0
    finish_reason: str = "stop"
    failure_reason: str | None = None
    op_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _truncate_to_k_tokens(text: str, k: int) -> tuple[str, bool]:
    """Truncate `text` to at most `k` tokens (cl100k_base). Returns (text, truncated)."""
    if token_count(text) <= k:
        return text, False
    ids = token_encode(text)[-k:]
    return token_decode(ids), True


def _truncate_response_to_k(text: str, k: int) -> tuple[str, bool]:
    """Truncate from the FRONT (keep head)."""
    if token_count(text) <= k:
        return text, False
    ids = token_encode(text)[:k]
    return token_decode(ids), True


_EXPORT_STRING_RE = ("export-string", "export_string")


def _parse_export_string(command: str) -> str | None:
    """If `command` is `export-string "..."` return the literal payload."""
    head = command.strip().split(maxsplit=1)
    if not head:
        return None
    if head[0] not in _EXPORT_STRING_RE:
        return None
    rest = head[1] if len(head) > 1 else ""
    try:
        parts = shlex.split(rest, posix=True)
    except ValueError:
        return rest
    return " ".join(parts) if parts else ""


def _parse_export(command: str) -> str | None:
    """If `command` is `export <path>` return the path. Returns None on string-export."""
    head = command.strip().split(maxsplit=1)
    if not head or head[0] != "export":
        return None
    rest = head[1].strip() if len(head) > 1 else ""
    try:
        parts = shlex.split(rest, posix=True)
    except ValueError:
        return rest
    return parts[0] if parts else ""


# --------------------------------------------------------------------- #
# Cell runner                                                           #
# --------------------------------------------------------------------- #


def run_shell_cell(
    cell: ShellCell,
    *,
    client: LLMClient,
    suite: BenchSuite,
    task: BenchTask,
    agent_root: Path | None = None,
    keep_agent_dir: bool = False,
) -> ShellCellResult:
    """Run one trajectory under the shell-harness architecture."""
    started = time.perf_counter()

    k = cell.L if cell.L > 0 else 10**9
    max_out = cell.max_tokens_per_turn or (cell.L if cell.L > 0 else 4096)

    # Build the agent's walled-off filesystem.
    if agent_root is None:
        tmp = tempfile.mkdtemp(prefix="goldfish-agent-")
        root = Path(tmp)
    else:
        root = Path(agent_root)

    # The instructions.txt file gets the task prompt + (optionally) brief
    # benchmark context. We deliberately keep it short — the model has to
    # `cat` it explicitly to see it, and re-cat costs tokens.
    instructions = suite.task_prompt(task)
    fs = AgentFS.make(root, instructions=instructions)
    runner = ShellRunner(root=root, timeout_s=cell.command_timeout_s)
    cap = CostCap(max_tokens=cell.cost_cap_tokens)

    system_prompt = render_system_prompt(k=k, timeout_s=cell.command_timeout_s)

    input_tokens = 0
    output_tokens = 0
    thinking_tokens = 0
    commands_executed = 0
    commands_intercepted = 0
    commands_failed = 0
    exports_written = 0
    terminated_by_done = False
    finish_reason = "stop"
    failure_reason: str | None = None
    op_counts: dict[str, int] = {
        "command": 0,
        "export": 0,
        "export_string": 0,
        "done": 0,
    }

    turn = 0
    try:
        while turn < cell.max_turns:
            # Build the context as the last-k-tokens of history.
            context = fs.read_history_tail(max_chars=k * 8)  # generous byte slice
            context, _ = _truncate_to_k_tokens(context, k)

            if turn == 0:
                # Hint the model where to start.
                context = (
                    "(this is turn 0; history.txt is empty. "
                    "`cat instructions.txt` to see the task.)"
                )

            prompt = context
            fs.write_stdin(turn, prompt)

            try:
                cap.charge(token_count(prompt))
                cap.charge(token_count(system_prompt))
            except CostCapExceeded as exc:
                failure_reason = f"cost_cap: {exc}"
                break

            try:
                gen = client.generate(
                    prompt,
                    max_tokens=max_out,
                    system=system_prompt,
                    temperature=0.0,
                )
            except Exception as exc:
                failure_reason = f"provider_error: {type(exc).__name__}: {exc}"
                break

            input_tokens += gen.input_tokens
            output_tokens += gen.output_tokens
            thinking_tokens += gen.thinking_tokens
            try:
                cap.charge(gen.output_tokens + gen.thinking_tokens)
            except CostCapExceeded as exc:
                failure_reason = f"cost_cap: {exc}"
                break

            response, _ = _truncate_response_to_k(gen.text, k)
            fs.write_stdout(turn, response)

            # Append the model's response to history before running cmds.
            fs.append_history(
                f"\n--- turn {turn} model response ---\n{response}\n"
            )

            commands = extract_shell_commands(response)

            for cmd in commands:
                cap_violation = False
                # Try the intercepts before any allowlist check.
                exp_string = _parse_export_string(cmd)
                if exp_string is not None:
                    dest = fs.export(exp_string, is_string=True)
                    fs.append_history(
                        f"\n$ {cmd}\n[export-string] wrote {dest.name} "
                        f"({len(exp_string)} chars)\n"
                    )
                    op_counts["export_string"] += 1
                    commands_intercepted += 1
                    exports_written += 1
                    continue

                exp_path = _parse_export(cmd)
                if exp_path is not None:
                    try:
                        dest = fs.export(exp_path, is_string=False)
                        fs.append_history(
                            f"\n$ {cmd}\n[export] copied to {dest.name}\n"
                        )
                        exports_written += 1
                    except (FileNotFoundError, ValueError) as exc:
                        fs.append_history(
                            f"\n$ {cmd}\n[export error] {exc}\n"
                        )
                        commands_failed += 1
                    op_counts["export"] += 1
                    commands_intercepted += 1
                    continue

                # Normal shell command.
                result = runner.run(cmd)
                if result.intercepted_action in ("done", "exit"):
                    terminated_by_done = True
                    op_counts["done"] += 1
                    commands_intercepted += 1
                    fs.append_history(f"\n$ {cmd}\n[done]\n")
                    break
                op_counts["command"] += 1
                commands_executed += 1
                if result.returncode != 0:
                    commands_failed += 1
                _append_command_to_history(fs, result)

            if terminated_by_done:
                turn += 1
                break

            finish_reason = gen.finish_reason
            turn += 1
        else:
            failure_reason = "max_turns_reached"
    except Exception as exc:
        failure_reason = f"harness_error: {type(exc).__name__}: {exc}"

    # Score against user_output/.
    user_files = fs.list_user_outputs()
    scoring_text = _read_user_outputs_for_scoring(user_files)
    solved, score = suite.score(task, scoring_text)
    final_answer = scoring_text or ""

    result_obj = ShellCellResult(
        cell=cell,
        solved=solved,
        score=score,
        final_answer_text=final_answer,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        turns=turn,
        commands_executed=commands_executed,
        commands_intercepted=commands_intercepted,
        commands_failed=commands_failed,
        exports_written=exports_written,
        history_chars_end=fs.history_path.stat().st_size if fs.history_path.exists() else 0,
        user_output_files=len(user_files),
        wall_seconds=time.perf_counter() - started,
        finish_reason="done" if terminated_by_done else finish_reason,
        failure_reason=failure_reason,
        op_counts=op_counts,
        metadata={
            "scheme": "shell",
            "client": client.name,
            "benchmark": suite.name,
            "L": cell.L,
            "cost_cap_spent": cap.spent,
            "agent_root": str(fs.root),
        },
    )

    if not keep_agent_dir:
        fs.cleanup()
    return result_obj


def _append_command_to_history(fs: AgentFS, result: CommandResult) -> None:
    chunks = [f"\n$ {result.command}\n"]
    if result.stdout:
        chunks.append(result.stdout)
        if not result.stdout.endswith("\n"):
            chunks.append("\n")
    if result.stderr:
        chunks.append(f"[stderr]\n{result.stderr}")
        if not result.stderr.endswith("\n"):
            chunks.append("\n")
    if result.returncode != 0 and not result.stderr:
        chunks.append(f"[exit {result.returncode}]\n")
    fs.append_history("".join(chunks))


def _read_user_outputs_for_scoring(files: list[Path]) -> str:
    """Concatenate all exported files (newest last) for the grader."""
    parts: list[str] = []
    for p in files:
        try:
            parts.append(f"=== {p.name} ===\n{p.read_text(encoding='utf-8', errors='replace')}")
        except Exception as exc:
            parts.append(f"=== {p.name} === [unreadable: {exc}]")
    return "\n\n".join(parts)
