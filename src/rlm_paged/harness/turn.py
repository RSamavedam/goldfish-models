"""Per-turn process for the stateless-turn architecture.

One turn = one API call. The harness:

  1. ASSEMBLE the input prompt: system prompt (with current store stats)
     + retrieved-content region (prior turn's continuing_instruction +
     queries the prior turn queued, executed against the store now).
     Hard cap: L/2 tokens for the retrieved-content region.
  2. CALL the model once.
  3. PARSE the response: extract <scratch> (discarded), parse ops.
  4. EXECUTE ops against the store: writes notes, sets next turn's
     continuing_instruction, queues queries for next turn, records
     external tool calls.
  5. VALIDATE the mandatory `continue` op. If missing, return a retry
     signal (the caller decides whether to retry or fail).

The harness loop calls `assemble_input` -> client -> `process_response`,
threading the QueuedQuery list between turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rlm_paged.client.tokenizer import count, decode, encode
from rlm_paged.store.block_store import BlockStore
from rlm_paged.tools.executor import (
    ExecutionResult,
    QueuedQuery,
    execute,
)
from rlm_paged.tools.ops import parse_ops
from rlm_paged.tools.schema import render_system_prompt


# Sentinels we inject into the input when something didn't fit.
TRUNCATED_MARKER = "[TRUNCATED]"
OVERBUDGET_MARKER = "[RETRIEVAL_OVER_BUDGET]"


@dataclass
class TurnInput:
    """The prompt and system message handed to the model this turn."""

    system_prompt: str
    user_prompt: str
    retrieved_tokens: int       # token count of the retrieved-content region
    truncations: int            # how many retrievals were dropped or shortened
    delivered_queries: int      # how many queued queries actually landed


@dataclass
class TurnOutput:
    """What came back from the model + what the harness did with it."""

    scratch_text: str
    ops_parsed: int
    execution: ExecutionResult
    response_tokens: int
    response_truncated: bool
    continuing_instruction_truncated: bool


def assemble_input(
    *,
    store: BlockStore,
    pending_queries: list[QueuedQuery],
    L: int,
    turn: int,
    task_prompt: str,
    last_continuing_instruction: str | None,
    extra_system_notes: str = "",
) -> TurnInput:
    """Render the prompt for turn `turn`.

    `task_prompt` is the original problem statement, which we put at the
    top of the user prompt and never count against L/2 because it's the
    invariant problem statement (every turn needs to know what it's
    solving). The retrieved-content region is purely
    continuing_instruction + queued retrievals.
    """
    half = max(1, L // 2)
    system = render_system_prompt(L=L, store_stats=store.stats())
    if extra_system_notes:
        system = system + "\n\n" + extra_system_notes

    # Build the retrieved-content region under L/2.
    retrieved_pieces: list[str] = []
    used = 0
    truncations = 0
    delivered = 0

    # 1) Guaranteed: prior turn's continuing_instruction, truncated if needed.
    if last_continuing_instruction is not None:
        ci_tokens = count(last_continuing_instruction)
        if ci_tokens <= half:
            retrieved_pieces.append(
                f"[continuing_instruction from turn {turn - 1}]\n"
                f"{last_continuing_instruction}"
            )
            used += ci_tokens
        else:
            # Truncate to fit.
            tok_ids = encode(last_continuing_instruction)[: half - 1]
            truncated_text = decode(tok_ids)
            retrieved_pieces.append(
                f"[continuing_instruction from turn {turn - 1}] "
                f"{TRUNCATED_MARKER}\n{truncated_text}"
            )
            used += half
            truncations += 1

    # 2) Queued queries from the prior turn. Execute each; stop adding once
    #    we'd exceed L/2; record overflow markers for the rest.
    for qi, q in enumerate(pending_queries):
        try:
            blocks = store.query(
                q.type,
                start=q.start,
                end=q.end,
                tag=q.tag,
                at_turn=turn,
            )
        except Exception as exc:
            piece = (
                f"[query {qi}: error] {type(exc).__name__}: {exc}"
            )
            piece_tokens = count(piece)
            if used + piece_tokens > half:
                truncations += 1
                continue
            retrieved_pieces.append(piece)
            used += piece_tokens
            continue

        if not blocks:
            piece = f"[query {qi}: no results for {q.type}]"
            piece_tokens = count(piece)
            if used + piece_tokens > half:
                truncations += 1
                continue
            retrieved_pieces.append(piece)
            used += piece_tokens
            continue

        for b in blocks:
            piece = f"[{b.short_id}] {b.text}"
            piece_tokens = count(piece)
            if used + piece_tokens > half:
                truncations += 1
                retrieved_pieces.append(
                    f"{OVERBUDGET_MARKER} query {qi} stopped at "
                    f"{b.short_id} (remaining results dropped)"
                )
                # Soft-stop: no more retrievals; don't double-count for
                # subsequent queries.
                used = half  # poison so further appends bail
                break
            retrieved_pieces.append(piece)
            used += piece_tokens
        delivered += 1
        if used >= half:
            # Mark every later query as dropped without trying.
            for later_idx in range(qi + 1, len(pending_queries)):
                truncations += 1
            break

    retrieved_region = "\n\n".join(retrieved_pieces) if retrieved_pieces else ""

    # User prompt = task + retrieved region.
    user = task_prompt
    if retrieved_region:
        user = f"{task_prompt}\n\n--- RETRIEVED ---\n{retrieved_region}"

    return TurnInput(
        system_prompt=system,
        user_prompt=user,
        retrieved_tokens=used,
        truncations=truncations,
        delivered_queries=delivered,
    )


def process_response(
    response_text: str,
    *,
    store: BlockStore,
    L: int,
    turn: int,
) -> TurnOutput:
    """Parse + execute one turn's response against the store.

    Enforces the L/2 response cap by truncating before parsing if the
    response is over budget. The model can still emit a (truncated) `continue`
    because we leave the head of the response intact.
    """
    half = max(1, L // 2)
    response_tokens = count(response_text)
    truncated = False
    if response_tokens > half:
        truncated = True
        ids = encode(response_text)[:half]
        response_text = decode(ids)
        response_tokens = half

    scratch, ops = parse_ops(response_text)
    execution = execute(ops, store=store, turn=turn)

    # Note: the body of any single op is necessarily <= response budget
    # (the response was already truncated to half tokens above, and any
    # op's body is a substring of the response). So continue-truncation
    # at this layer would be unreachable. We enforce the L/2 cap when the
    # continue is injected into the *next* turn's retrieved region in
    # assemble_input — that's where overshoot actually matters for the
    # model's experience.
    ci_truncated = False

    return TurnOutput(
        scratch_text=scratch,
        ops_parsed=len(ops),
        execution=execution,
        response_tokens=response_tokens,
        response_truncated=truncated,
        continuing_instruction_truncated=ci_truncated,
    )
