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
from typing import Any, Callable

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


# Sentinel: a ShellCell with L=L_NATIVE is the **no-truncation control**,
# NOT a zero-context run. Picked 0 originally because Python YAML scalars
# default to int and 0 was convenient; the name is misleading but the
# convention is wired through analyze.py, every JSONL we've stored, and
# every figure script. Treat 0 here as "infinity" — the bench's native
# context limit. Everywhere it's RENDERED to a human (figures, tables,
# the paper), it appears as "native" or "L=∞", never "L=0".
L_NATIVE: int = 0


def render_L(L: int) -> str:
    """Human-facing label for a context cap. Use this in every figure,
    table, log line, and journal entry. NEVER print bare `L=0`."""
    return "L=∞" if L == L_NATIVE else f"L={L}"


@dataclass
class ShellCell:
    """One cell of a shell-harness sweep."""

    provider: str
    # Per-turn context cap, in tokens. L = L_NATIVE (0) is the
    # no-truncation control (native bench context). L > 0 is the
    # goldfish regime under study. See render_L() for human-facing
    # output — never display the bare integer 0.
    L: int
    benchmark: str
    task_id: str
    seed: int = 0
    max_turns: int = 16
    max_tokens_per_turn: int | None = None  # default: same as L
    cost_cap_tokens: int = 100_000
    command_timeout_s: float = 10.0
    # "baseline" or "scratchpad" — the latter enables the paged-memory
    # protocol that makes the model maintain notes.md every turn.
    prompt_variant: str = "baseline"


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
    empty_done_retries: int = 0
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


def _inject_position_markers(text: str, every: int = 16) -> str:
    """Insert «@N» markers every `every` tokens of `text`.

    These markers tell the model where in its budget it is. They are
    inserted AFTER truncation so they don't count toward the L cap.
    They ARE counted toward billing (every char goes to the API), but
    that's a fixed overhead the user-supplied prompt tolerates.

    The marker uses guillemets («») rather than ASCII so it's
    visually distinct from any code/output in the agent's history.
    """
    if every <= 0:
        return text
    ids = token_encode(text)
    if len(ids) <= every:
        # Still annotate with one marker at end so the model knows we
        # could have annotated if there were enough tokens.
        return f"{text}«@{len(ids)}»"
    chunks: list[str] = []
    for i in range(0, len(ids), every):
        slice_ids = ids[i : i + every]
        chunks.append(token_decode(slice_ids))
        # Place a marker labeled with the cumulative token count.
        chunks.append(f"«@{min(i + every, len(ids))}»")
    return "".join(chunks)


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
    before_first_turn: Callable[[Any], None] | None = None,
    after_last_turn: Callable[[Any, "ShellCellResult"], str | None] | None = None,
    turn_logger: Callable[[dict], None] | None = None,
) -> ShellCellResult:
    """Run one trajectory under the shell-harness architecture.

    `before_first_turn(fs)` runs after the AgentFS is initialized but
    before turn 0. Use it to prepopulate the agent's filesystem (e.g.
    clone a target repo for SWE-bench-style tasks). The callback receives
    the AgentFS and may mutate it freely.

    `after_last_turn(fs, result)` runs after the trajectory exits but
    before scoring. If it returns a string, that string is used as the
    `response` argument to `suite.score()` instead of the user_output
    concatenation. Use it to inject a custom artifact (e.g. an extracted
    unified-diff patch) into the scorer's input. Returns None to keep
    default behavior.
    """
    started = time.perf_counter()

    k = cell.L if cell.L > 0 else 10**9
    # The per-turn output budget is INDEPENDENT of L. L is the
    # context-window cap (goldfish constraint on what survives into
    # the next turn). The model still needs room to produce a useful
    # response THIS turn. With reasoning models, thinking-tokens
    # compete with output-tokens for the same budget — capping at L
    # tokens at small L causes 100% LENGTH-CAP turns with zero output.
    # Default to 4096 across all L; override via max_tokens_per_turn
    # in the config if needed.
    max_out = cell.max_tokens_per_turn or 4096

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
    if before_first_turn is not None:
        try:
            before_first_turn(fs)
        except Exception as exc:
            # Bootstrap failure is fatal — return early with a clear reason.
            if not keep_agent_dir:
                fs.cleanup()
            return ShellCellResult(
                cell=cell,
                solved=False,
                score=0.0,
                final_answer_text="",
                turns=0,
                failure_reason=f"bootstrap_error: {type(exc).__name__}: {exc}",
                wall_seconds=time.perf_counter() - started,
                op_counts={
                    "command": 0, "export": 0,
                    "export_string": 0, "done": 0,
                },
                metadata={
                    "scheme": "shell",
                    "client": client.name,
                    "benchmark": suite.name,
                    "L": cell.L,
                },
            )
    runner = ShellRunner(root=root, timeout_s=cell.command_timeout_s)
    cap = CostCap(max_tokens=cell.cost_cap_tokens)

    system_prompt = render_system_prompt(
        k=k,
        timeout_s=cell.command_timeout_s,
        max_out=max_out,
        variant=cell.prompt_variant,
    )

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
        "empty_done_retried": 0,
    }
    # Bug 12 guard: if the model says `done` while user_output/ is empty
    # (no files) OR every file in user_output/ is 0 bytes, refuse to
    # terminate and append a nag to history. After this many retries
    # we give up and let `done` go through anyway (so a genuinely
    # impossible task can still terminate).
    max_empty_done_retries = 2
    empty_done_retries_used = 0

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

            # Tinystate variant: inject position markers every 16 tokens
            # AFTER truncation. Markers are visual hints («@N») that show
            # the model how much of its L-token budget it has used. They
            # do NOT count toward L (truncation already happened).
            if cell.prompt_variant == "tinystate":
                context = _inject_position_markers(context, every=16)

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

            # Per-turn transcript log: gives downstream analysis access to
            # exactly what the model saw and wrote. The harness driver
            # provides a callback that appends each row to runs/turns.jsonl
            # (or wherever it wants). On-error in the callback we never
            # let the trajectory crash — log failures are best-effort.
            if turn_logger is not None:
                try:
                    turn_logger({
                        "cell_key": (
                            f"{cell.provider}|shell|{cell.L}|{cell.benchmark}|"
                            f"{cell.task_id}|{cell.seed}"
                        ),
                        "provider": cell.provider,
                        "benchmark": cell.benchmark,
                        "task_id": cell.task_id,
                        "L": cell.L,
                        "turn": turn,
                        "system_prompt": system_prompt,
                        "user_prompt": prompt,
                        "response": response,
                        "thinking_text": gen.thinking_text or "",
                        "input_tokens": gen.input_tokens,
                        "output_tokens": gen.output_tokens,
                        "thinking_tokens": gen.thinking_tokens,
                        "finish_reason": gen.finish_reason,
                    })
                except Exception:
                    pass

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
                    # Bug 12: refuse to terminate if nothing has been
                    # delivered. The model has a strong RLHF bias to
                    # finish-and-deliver-something even when its output
                    # is empty; prompt rules alone don't reliably stop
                    # this. Refuse + nag + retry up to N times.
                    files = fs.list_user_outputs()
                    nonempty = [p for p in files if p.stat().st_size > 0]
                    if not nonempty and empty_done_retries_used < max_empty_done_retries:
                        empty_done_retries_used += 1
                        op_counts["empty_done_retried"] += 1
                        commands_intercepted += 1
                        if not files:
                            reason = "user_output/ is empty (no files exported)"
                        else:
                            reason = (
                                "all files in user_output/ are 0 bytes: "
                                + ", ".join(p.name for p in files)
                            )
                        fs.append_history(
                            f"\n$ {cmd}\n"
                            f"[done REFUSED] {reason}. "
                            f"Re-do the work and EXPORT a real artifact "
                            f"before issuing `done` again. "
                            f"(retry {empty_done_retries_used}/{max_empty_done_retries})\n"
                        )
                        # Do NOT break — continue the loop, model gets
                        # another turn. We still consider this `done`
                        # as an emitted op for tallying.
                        op_counts["done"] += 1
                        continue
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

    # Score against user_output/ — or, if the caller hooked it, against
    # whatever string `after_last_turn` returns.
    user_files = fs.list_user_outputs()
    default_scoring_text = _read_user_outputs_for_scoring(user_files)
    scoring_text = default_scoring_text
    if after_last_turn is not None:
        try:
            preliminary = ShellCellResult(
                cell=cell,
                solved=False,
                score=0.0,
                final_answer_text=default_scoring_text,
                turns=turn,
            )
            override = after_last_turn(fs, preliminary)
            if isinstance(override, str):
                scoring_text = override
        except Exception as exc:
            failure_reason = (
                failure_reason
                or f"after_last_turn_error: {type(exc).__name__}: {exc}"
            )
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
        empty_done_retries=empty_done_retries_used,
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
