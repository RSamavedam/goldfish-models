from __future__ import annotations

from rlm_paged.store import BlockStore
from rlm_paged.harness.turn import (
    OVERBUDGET_MARKER,
    TRUNCATED_MARKER,
    assemble_input,
    process_response,
)
from rlm_paged.tools.executor import QueuedQuery


def test_assemble_input_with_no_prior_state():
    store = BlockStore()
    store.append("user_message", "compute the answer", created_at_turn=-1)
    out = assemble_input(
        store=store,
        pending_queries=[],
        L=64,
        turn=0,
        task_prompt="compute the answer",
        last_continuing_instruction=None,
    )
    assert "compute the answer" in out.user_prompt
    assert "RETRIEVED" not in out.user_prompt
    assert out.retrieved_tokens == 0
    assert out.truncations == 0


def test_assemble_input_injects_prior_continuing_instruction():
    store = BlockStore()
    out = assemble_input(
        store=store,
        pending_queries=[],
        L=256,
        turn=1,
        task_prompt="solve it",
        last_continuing_instruction="finish step 2",
    )
    assert "finish step 2" in out.user_prompt
    assert "continuing_instruction from turn 0" in out.user_prompt


def test_assemble_input_truncates_oversize_continuing_instruction():
    store = BlockStore()
    huge = "x " * 500  # comfortably over L/2 at L=32
    out = assemble_input(
        store=store,
        pending_queries=[],
        L=32,
        turn=1,
        task_prompt="t",
        last_continuing_instruction=huge,
    )
    assert TRUNCATED_MARKER in out.user_prompt
    assert out.truncations == 1


def test_assemble_input_executes_queries_and_marks_overflow():
    store = BlockStore()
    # 6 notes with substantial text; with small L only a few fit.
    for i in range(6):
        store.append("note", "lorem ipsum " * 20, created_at_turn=0)
    queries = [QueuedQuery(type="note", start=0, end=5)]
    out = assemble_input(
        store=store,
        pending_queries=queries,
        L=128,  # L/2 == 64 tokens for retrieved region
        turn=1,
        task_prompt="t",
        last_continuing_instruction=None,
    )
    # Should have overflowed and emitted the marker.
    assert OVERBUDGET_MARKER in out.user_prompt
    assert out.truncations >= 1


def test_assemble_input_handles_empty_query_result():
    store = BlockStore()
    queries = [QueuedQuery(type="note", start=0, end=5)]
    out = assemble_input(
        store=store,
        pending_queries=queries,
        L=128,
        turn=1,
        task_prompt="t",
        last_continuing_instruction=None,
    )
    assert "no results for note" in out.user_prompt


def test_process_response_writes_note_and_continue():
    store = BlockStore()
    response = (
        "note tag=plan\n"
        "    Approach: enumerate cases.\n"
        "continue\n"
        "    Compute n=3 next.\n"
    )
    out = process_response(response, store=store, L=128, turn=0)
    assert out.ops_parsed == 2
    assert out.execution.has_mandatory_continue
    assert len(out.execution.notes_written) == 1
    assert out.continuing_instruction_truncated is False


def test_oversize_continue_is_truncated_when_replayed_next_turn():
    """The body of a single op is always <= response budget (already
    truncated). The L/2 cap kicks in when the continue is *replayed* into
    the next turn's retrieved-content region — that's where truncation
    becomes visible to the model."""
    store = BlockStore()
    # First store a continue manually (simulating a turn that wrote one).
    huge_body = "go " * 1000  # >> L/2 at L=32
    store.append("continuing_instruction", huge_body, created_at_turn=0)

    out = assemble_input(
        store=store,
        pending_queries=[],
        L=32,
        turn=1,
        task_prompt="t",
        last_continuing_instruction=huge_body,
    )
    assert TRUNCATED_MARKER in out.user_prompt
    assert out.truncations >= 1


def test_process_response_truncates_oversize_response_itself():
    store = BlockStore()
    response = "note \"x\"\n" * 500  # way over L/2
    out = process_response(response, store=store, L=32, turn=0)
    assert out.response_truncated is True
