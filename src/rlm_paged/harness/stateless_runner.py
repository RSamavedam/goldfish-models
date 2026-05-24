"""Sweep-cell driver for the stateless-turn architecture.

One `run_stateless_cell` invocation = one (provider, L, benchmark, task)
trajectory. The trajectory is a sequence of turns, each consisting of:

  1. assemble_input: system prompt (with inbox + store stats) +
     retrieved-content region (prior continuing_instruction + queued
     query results).
  2. client.generate: one API call.
  3. process_response: parse ops, execute against the store.
  4. dispatch external tool calls -> observation blocks.
  5. surface assistant_reply blocks via on_assistant_reply callback.
  6. pump pending user_input -> user_message blocks.
  7. continue or terminate.

This runner is provider-agnostic. Every provider runs the same loop.

The trajectory's bootstrap state is the original benchmark question,
written as user_message:0 before turn 0 starts.

Termination conditions, in priority order:
  - cost cap exceeded
  - provider error
  - max_turns reached
  - model emitted continue containing "done" / boxed answer / final-answer
    line AND there are no unread user_messages
  - missing-continue retries exhausted
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from rlm_paged.bench.answer_extraction import (
    extract_boxed,
    extract_final_answer_line,
)
from rlm_paged.bench.base import BenchSuite, BenchTask
from rlm_paged.client.base import LLMClient
from rlm_paged.client.tokenizer import count
from rlm_paged.harness.cost_cap import CostCap, CostCapExceeded
from rlm_paged.harness.turn import (
    assemble_input,
    process_response,
)
from rlm_paged.store.block_store import BlockStore
from rlm_paged.tools.executor import ExternalCall, QueuedQuery


@dataclass
class StatelessCell:
    """One cell of the Phase 1 sweep under the stateless-turn architecture."""

    provider: str
    L: int                       # 0 => "native" (effectively unlimited)
    benchmark: str
    task_id: str
    seed: int = 0
    max_turns: int = 16
    max_tokens_per_turn: int = 1024
    cost_cap_tokens: int = 100_000
    max_retries_on_missing_continue: int = 2


@dataclass
class StatelessResult:
    cell: StatelessCell
    solved: bool
    score: float
    final_answer_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    turns: int = 0
    op_counts: dict[str, int] = field(default_factory=dict)
    op_errors: int = 0
    notes_written: int = 0
    queries_issued: int = 0
    assistant_replies: int = 0
    user_messages_received: int = 0
    retrieval_truncations: int = 0
    response_truncations: int = 0
    continue_truncations: int = 0
    missing_continue_retries: int = 0
    wall_seconds: float = 0.0
    finish_reason: str = "stop"
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #

_DONE_SENTINELS = ("continue done", "done.", "answer is")


def _looks_like_done(continuing_text: str) -> bool:
    lowered = continuing_text.strip().lower()
    if not lowered:
        return False
    return (
        lowered.startswith("done")
        or "final answer:" in lowered
        or "\\boxed{" in lowered
        or any(s in lowered for s in _DONE_SENTINELS)
    )


def _last_continuing_instruction_text(store: BlockStore) -> str | None:
    blocks = store._by_type["continuing_instruction"]  # avoid marking as read
    return blocks[-1].text if blocks else None


def _extract_answer(text: str) -> str | None:
    return extract_boxed(text) or extract_final_answer_line(text)


# --------------------------------------------------------------------- #
# External-tool dispatch                                                #
# --------------------------------------------------------------------- #

def _builtin_echo(args: str) -> str:
    """Trivial tool used in tests; just echoes its args."""
    return f"echo: {args}"


DEFAULT_EXTERNAL_TOOLS: dict[str, Any] = {"echo": _builtin_echo}


def _dispatch_external_calls(
    calls: list[ExternalCall],
    *,
    store: BlockStore,
    registry: dict[str, Any],
    turn: int,
) -> int:
    """Execute external calls, writing each result as an observation block."""
    written = 0
    for call in calls:
        tool = registry.get(call.tool)
        if tool is None:
            store.append(
                "observation",
                f"[tool={call.tool} error] unknown tool",
                created_at_turn=turn,
            )
            continue
        try:
            output = tool(call.args)
        except Exception as exc:
            output = f"[tool={call.tool} error] {type(exc).__name__}: {exc}"
        store.append(
            "observation",
            f"[tool={call.tool}]\n{output}",
            created_at_turn=turn,
        )
        written += 1
    return written


def _pump_user_inputs(
    user_input: Callable[[], list[str]] | None,
    *,
    store: BlockStore,
    turn: int,
) -> int:
    """Poll the user_input callable and append any returned strings as user_message blocks."""
    if user_input is None:
        return 0
    try:
        new_messages = user_input()
    except Exception:
        return 0
    if not new_messages:
        return 0
    for msg in new_messages:
        if msg is None:
            continue
        text = str(msg).strip()
        if not text:
            continue
        store.append("user_message", text, created_at_turn=turn)
    return len(new_messages)


def _emit_assistant_replies(
    reply_global_indices: list[int],
    *,
    store: BlockStore,
    callback: Callable[[str], None] | None,
) -> None:
    if not reply_global_indices or callback is None:
        return
    for gid in reply_global_indices:
        try:
            callback(store.get(gid).text)
        except Exception:
            pass  # never let a user-callback crash the loop


# --------------------------------------------------------------------- #
# The cell runner                                                       #
# --------------------------------------------------------------------- #

def run_stateless_cell(
    cell: StatelessCell,
    *,
    client: LLMClient,
    suite: BenchSuite,
    task: BenchTask,
    external_tools: dict[str, Any] | None = None,
    user_input: Callable[[], list[str]] | None = None,
    on_assistant_reply: Callable[[str], None] | None = None,
) -> StatelessResult:
    """Run one trajectory.

    `user_input` is an optional callable polled between turns; it returns
    a list of new user messages (possibly empty). Each becomes a
    user_message block in the store, which the model can query via the
    inbox surface.

    `on_assistant_reply` is invoked once per assistant_reply block the
    model writes via `say`. Lets callers surface model output to a user
    interface, websocket, etc.
    """
    started = time.perf_counter()

    L_effective = cell.L if cell.L > 0 else 10**9  # "native" path
    store = BlockStore()
    # The original task is the first user_message. (Unifies inbox + task.)
    store.append("user_message", suite.task_prompt(task), created_at_turn=-1)

    cap = CostCap(max_tokens=cell.cost_cap_tokens)
    registry = dict(DEFAULT_EXTERNAL_TOOLS)
    if external_tools:
        registry.update(external_tools)

    pending_queries: list[QueuedQuery] = []
    last_ci_text: str | None = None

    input_tokens = 0
    output_tokens = 0
    thinking_tokens = 0
    op_counts: Counter[str] = Counter()
    op_errors = 0
    notes_written_total = 0
    queries_issued_total = 0
    assistant_replies_total = 0
    user_messages_received_total = 1  # the initial task counts
    retrieval_truncations = 0
    response_truncations = 0
    continue_truncations = 0
    missing_continue_retries = 0
    finish_reason = "stop"
    failure_reason: str | None = None
    final_answer_text = ""

    turn = 0
    extra_system_notes = ""
    try:
        while turn < cell.max_turns:
            # Pump any user input that arrived since the last turn.
            new_user_msgs = _pump_user_inputs(user_input, store=store, turn=turn)
            user_messages_received_total += new_user_msgs

            turn_input = assemble_input(
                store=store,
                pending_queries=pending_queries,
                L=L_effective,
                turn=turn,
                task_prompt=suite.task_prompt(task),
                last_continuing_instruction=last_ci_text,
                extra_system_notes=extra_system_notes,
            )
            retrieval_truncations += turn_input.truncations

            try:
                cap.charge(count(turn_input.user_prompt))
                cap.charge(count(turn_input.system_prompt))
            except CostCapExceeded as exc:
                failure_reason = f"cost_cap: {exc}"
                break

            try:
                gen = client.generate(
                    turn_input.user_prompt,
                    max_tokens=cell.max_tokens_per_turn,
                    system=turn_input.system_prompt,
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

            turn_output = process_response(
                gen.text, store=store, L=L_effective, turn=turn
            )
            response_truncations += int(turn_output.response_truncated)
            continue_truncations += int(turn_output.continuing_instruction_truncated)

            ex = turn_output.execution
            op_counts["note"] += len(ex.notes_written)
            op_counts["continue"] += int(ex.has_mandatory_continue)
            op_counts["query"] += len(ex.pending_queries)
            op_counts["call"] += len(ex.external_calls)
            op_counts["say"] += len(ex.assistant_replies)
            op_errors += len(ex.errors)
            notes_written_total += len(ex.notes_written)
            queries_issued_total += len(ex.pending_queries)
            assistant_replies_total += len(ex.assistant_replies)
            finish_reason = gen.finish_reason

            # Surface any user-facing replies.
            _emit_assistant_replies(
                ex.assistant_replies, store=store, callback=on_assistant_reply
            )

            # Mandatory continue: retry up to N times if missing.
            if not ex.has_mandatory_continue:
                if missing_continue_retries < cell.max_retries_on_missing_continue:
                    missing_continue_retries += 1
                    extra_system_notes = (
                        "PROTOCOL ERROR last turn: you must emit exactly one "
                        "`continue` op with a non-empty body. Try again."
                    )
                    continue
                failure_reason = "missing_continue"
                break
            extra_system_notes = ""  # reset after a good turn

            # Did the model declare done? Only honor that termination if
            # the user inbox is empty — otherwise the user just said
            # something and we need to let the model address it.
            last_ci_text = _last_continuing_instruction_text(store)
            looks_done = bool(last_ci_text) and (
                _looks_like_done(last_ci_text)
                or _extract_answer(last_ci_text) is not None
            )
            if looks_done and store.unread_count("user_message") == 0:
                final_answer_text = last_ci_text or ""
                turn += 1
                break

            # Dispatch any external tool calls (their results become
            # observations the model can query next turn).
            if ex.external_calls:
                _dispatch_external_calls(
                    ex.external_calls,
                    store=store,
                    registry=registry,
                    turn=turn,
                )

            pending_queries = list(ex.pending_queries)
            turn += 1
        else:
            failure_reason = "max_turns_reached"
            last_ci_text = _last_continuing_instruction_text(store) or ""
            final_answer_text = last_ci_text
    except Exception as exc:
        failure_reason = f"harness_error: {type(exc).__name__}: {exc}"

    # If we didn't get a clean done, fall back to using whatever
    # continuing_instruction we have as the answer text.
    if not final_answer_text:
        ci_text = _last_continuing_instruction_text(store)
        if ci_text:
            final_answer_text = ci_text

    # Score against notes + assistant_replies + final answer. Some
    # benchmarks may have answers expressed in any of these.
    note_texts = [b.text for b in store._by_type["note"]]
    reply_texts = [b.text for b in store._by_type["assistant_reply"]]
    scoring_text = "\n".join(note_texts + reply_texts + [final_answer_text])
    solved, score = suite.score(task, scoring_text)

    return StatelessResult(
        cell=cell,
        solved=solved,
        score=score,
        final_answer_text=final_answer_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        turns=turn,
        op_counts=dict(op_counts),
        op_errors=op_errors,
        notes_written=notes_written_total,
        queries_issued=queries_issued_total,
        assistant_replies=assistant_replies_total,
        user_messages_received=user_messages_received_total,
        retrieval_truncations=retrieval_truncations,
        response_truncations=response_truncations,
        continue_truncations=continue_truncations,
        missing_continue_retries=missing_continue_retries,
        wall_seconds=time.perf_counter() - started,
        finish_reason=finish_reason,
        failure_reason=failure_reason,
        metadata={
            "scheme": "stateless_turn",
            "client": client.name,
            "benchmark": suite.name,
            "L": cell.L,
            "cost_cap_spent": cap.spent,
            "store_blocks": len(store),
        },
    )
