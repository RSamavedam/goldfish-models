"""Run one (provider, scheme, L, benchmark, task) cell of the Phase 1 sweep.

The harness loop is multi-turn:

  Turn 0: prompt = pinned_prefix + task.payload
  Turn N: prompt = pinned_prefix + render_window(state)

Each turn the model emits a generation. We parse it line-by-line:
  - Op-code lines (e.g. "r 3 0 64") are dispatched against the chunk store.
    Their outputs become tool Segments in the conversation.
  - Other lines accumulate into a model Segment.

The loop terminates when:
  - The model emits a recognized final-answer marker (\\boxed{}, "Final answer:")
  - The cost cap is hit
  - max_turns is reached
  - The provider returns finish_reason="length" with no more progress

After the loop, the bench suite scores the concatenation of all visible model
output. Metrics are accumulated as we go.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from rlm_paged.bench.answer_extraction import extract_boxed, extract_final_answer_line
from rlm_paged.bench.base import BenchSuite, BenchTask
from rlm_paged.client.base import LLMClient
from rlm_paged.client.tokenizer import count
from rlm_paged.harness.conversation import ConversationState, Segment
from rlm_paged.harness.cost_cap import CostCap, CostCapExceeded
from rlm_paged.harness.schemes import Scheme, SchemeContext
from rlm_paged.store.store import ChunkStore
from rlm_paged.tools.api import ToolDispatcher, parse_op
from rlm_paged.tools.schema import PREFIX_SCHEMA
from rlm_paged.window.state import ActiveWindow, WindowConfig


@dataclass
class SweepCell:
    provider: str
    scheme: str
    L: int                       # 0 means no cap (native)
    benchmark: str
    task_id: str
    seed: int = 0
    max_turns: int = 16
    max_tokens_per_turn: int = 1024
    cost_cap_tokens: int = 100_000


@dataclass
class RunResult:
    cell: SweepCell
    solved: bool
    score: float
    final_response: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    turns: int = 0
    op_counts: dict[str, int] = field(default_factory=dict)
    op_errors: int = 0
    wall_seconds: float = 0.0
    mean_active_tokens: float = 0.0
    peak_active_tokens: int = 0
    finish_reason: str = "stop"
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


FINAL_ANSWER_MARKERS = ("final answer", "\\boxed{")


def _has_final_answer(text: str) -> bool:
    lower = text.lower()
    return any(m in lower for m in FINAL_ANSWER_MARKERS)


def _build_system_prompt(scheme_name: str, L: int) -> str:
    if scheme_name == "paged":
        return (
            "You are solving a reasoning problem with a hard working-memory cap of "
            f"L={L} tokens on your active context. To externalize intermediate work, "
            "you may issue ops (one per line, alone on the line):\n"
            f"{PREFIX_SCHEMA}\n"
            "Tool results return inline. Issue evicts before retrieves when the "
            "window is near full. End your solution with: \"Final answer: ...\" "
            "or place the answer in \\boxed{...}."
        )
    if scheme_name == "truncated":
        return (
            f"You are solving a reasoning problem under a hard working-memory cap "
            f"of L={L} tokens. Older content will be silently dropped when the cap "
            "is exceeded. End with \"Final answer: ...\" or \\boxed{...}."
        )
    if scheme_name == "summarized":
        return (
            f"You are solving a reasoning problem under a hard working-memory cap "
            f"of L={L} tokens. Older content will be replaced by a short summary "
            "when the cap is exceeded. End with \"Final answer: ...\" or \\boxed{...}."
        )
    # native
    return (
        "You are solving a reasoning problem. Think step-by-step. "
        "End with \"Final answer: ...\" or \\boxed{...}."
    )


def _dispatch_ops(
    text: str,
    dispatcher: ToolDispatcher,
    op_counts: Counter,
) -> tuple[str, list[Segment], int]:
    """Walk lines, dispatching any op-code lines. Returns (visible_text, tool_segments, errors)."""
    visible_lines: list[str] = []
    tool_segments: list[Segment] = []
    errors = 0
    for line in text.splitlines():
        op = parse_op(line)
        if op is None:
            visible_lines.append(line)
            continue
        op_counts[op.code] += 1
        result = dispatcher.dispatch(op)
        if not result.ok:
            errors += 1
            tool_segments.append(
                Segment(kind="tool", text=f"[op {op.code} error: {result.error}]")
            )
            continue
        if op.code == "r":
            from rlm_paged.client.tokenizer import decode

            tokens = result.payload["tokens"]  # type: ignore[index]
            decoded = decode(tokens)
            tool_segments.append(Segment(kind="tool", text=f"[retrieved] {decoded}"))
        elif op.code == "q":
            tool_segments.append(
                Segment(kind="tool", text=f"[refs] {result.payload}")
            )
        elif op.code == "e":
            tool_segments.append(
                Segment(kind="tool", text=f"[evicted {result.payload['freed']} tokens]")  # type: ignore[index]
            )
        else:
            tool_segments.append(Segment(kind="tool", text=f"[{op.code} ok]"))
    return "\n".join(visible_lines), tool_segments, errors


def run_cell(
    cell: SweepCell,
    *,
    client: LLMClient,
    suite: BenchSuite,
    task: BenchTask,
    scheme: Scheme,
    summarizer: LLMClient | None = None,
) -> RunResult:
    """Execute one (provider, scheme, L, benchmark, task) sweep cell."""
    started = time.perf_counter()

    L_effective = cell.L if cell.L > 0 else 10**9  # 1B-token cap == native
    window = ActiveWindow(WindowConfig(L=L_effective, tail_max=max(L_effective // 2, 1)))
    store = ChunkStore(chunk_size=256)
    dispatcher = ToolDispatcher(window, store)
    ctx = SchemeContext(store=store, summarizer=summarizer)
    cap = CostCap(max_tokens=cell.cost_cap_tokens)

    system_prompt = _build_system_prompt(scheme.name, L_effective)
    task_prompt = suite.task_prompt(task)
    state = ConversationState(segments=[Segment(kind="task", text=task_prompt)])

    op_counts: Counter[str] = Counter()
    op_errors = 0
    input_tokens = 0
    output_tokens = 0
    thinking_tokens = 0
    active_token_samples: list[int] = []
    final_response_parts: list[str] = []
    finish_reason = "stop"
    failure_reason: str | None = None

    try:
        for turn in range(cell.max_turns):
            state.turns = turn
            window_text = scheme.render_window(state)
            prompt = task_prompt + ("\n\n" + window_text if window_text else "")

            try:
                cap.charge(count(prompt))
            except CostCapExceeded as exc:
                failure_reason = f"cost_cap: {exc}"
                break

            try:
                gen = client.generate(
                    prompt,
                    max_tokens=cell.max_tokens_per_turn,
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

            dispatcher.step = turn
            visible_text, tool_segments, errs = _dispatch_ops(
                gen.text, dispatcher, op_counts
            )
            op_errors += errs

            if visible_text.strip():
                model_seg = Segment(kind="model", text=visible_text)
                state.add_segment(model_seg)
                final_response_parts.append(visible_text)
            for ts in tool_segments:
                state.add_segment(ts)

            scheme.enforce_cap(state, L_effective, ctx)
            active_token_samples.append(state.active_tokens())

            finish_reason = gen.finish_reason
            if _has_final_answer(visible_text):
                break
            if gen.finish_reason == "length" and not visible_text.strip():
                # Model is stuck — finished a turn with no progress.
                failure_reason = "length_no_progress"
                break
        else:
            failure_reason = "max_turns_reached"
    except Exception as exc:
        failure_reason = f"harness_error: {type(exc).__name__}: {exc}"

    final_response = "\n".join(final_response_parts)
    solved, score = suite.score(task, final_response)
    mean_active = (
        sum(active_token_samples) / len(active_token_samples)
        if active_token_samples
        else 0.0
    )
    peak_active = max(active_token_samples) if active_token_samples else 0

    return RunResult(
        cell=cell,
        solved=solved,
        score=score,
        final_response=final_response,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        turns=state.turns + 1,
        op_counts=dict(op_counts),
        op_errors=op_errors,
        wall_seconds=time.perf_counter() - started,
        mean_active_tokens=mean_active,
        peak_active_tokens=peak_active,
        finish_reason=finish_reason,
        failure_reason=failure_reason,
        metadata={
            "scheme": scheme.name,
            "client": client.name,
            "benchmark": suite.name,
            "L": cell.L,
            "cost_cap_spent": cap.spent,
        },
    )
