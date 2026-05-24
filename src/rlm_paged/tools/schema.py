"""System-prompt prelude for the stateless-turn architecture.

This is what the model sees at the top of every turn, outside L. It
explains the op surface, the retrieval/response budget split, and what
the harness will do on the model's behalf between turns.

The old token-frugal single-char schema (`e`/`r`/`q`/`a`/`l`/`s`) is
kept below as `LEGACY_PREFIX_SCHEMA` for backward compatibility with
the deprecated paths.
"""

from __future__ import annotations

OP_REFERENCE: dict[str, str] = {
    "note": "Append a note block. Body holds the text. Optional `tag=...`.",
    "continue": "Set this turn's continuing_instruction. MANDATORY; exactly one per turn.",
    "query": "Queue a retrieval for the next turn: `query <type> <start> <end> [tag=T]`.",
    "pipe": "Execute a query and feed its result into another op in one step.",
    "call": "External tool call. Body holds tool arguments. Result -> observation block.",
    "say": "Optional: send a message to the user. Body holds the text. Writes assistant_reply.",
}


SYSTEM_PROMPT_TEMPLATE = """You are a stateless reasoning agent with a goldfish-sized working memory.

WORKING MEMORY MODEL
====================
Each turn you receive a tightly budgeted input (at most {half_L} tokens) and
produce a budgeted response (at most {half_L} tokens). You do NOT remember
this turn after it ends — anything you need later must be explicitly
written to the block store.

The block store is append-only and typed:
    user_message[*]           messages from the user. The FIRST user_message
                              is the original task. New ones can arrive at
                              any time between turns; check the INBOX below.
    assistant_reply[*]        your messages to the user (you write via `say`).
    observation[*]            tool-call results (harness-written, verbatim).
    note[*]                   your knowledge (you write; cannot edit/delete).
    continuing_instruction[*] one message per turn (you write; one per turn).

CURRENT STORE STATS
===================
{store_stats}

INBOX
=====
{inbox_status}

OP SURFACE
==========
Issue ops one per line in your response. You may open an optional
`<scratch>...</scratch>` block at the very top (max {scratch_budget}
tokens) for in-turn thinking; it is discarded after the turn.

  note [tag=T]              Append a note (body holds the text).
  continue                  MANDATORY. Body holds your message to the next turn.
                            Exactly one per turn. Will be truncated if oversize.
  query <type> <start> <end> [tag=T]
                            Queue a retrieval for the next turn. Indices
                            are inclusive; -1 means "last". Querying a
                            user_message block marks it as read.
  pipe (query ...) -> <dest>
                            Execute query and route results to `dest`
                            (note / continue / say / call).
  call <tool_name>          External tool (body holds args). Result becomes
                            a new observation block.
  say [tag=T]               Optional. Send a message to the user. Body holds
                            the text. Writes an assistant_reply block.

EXAMPLE
=======
    <scratch>
    The prior turn's continue said to compute case n=3.
    User has 1 unread message.
    </scratch>

    query user_message -1 -1
    note tag=plan
        Approach: enumerate small cases, look for a pattern, prove by
        induction.
    say
        Working on it — will share once I have a clean answer.
    continue
        Computed n=3 yields 7. Next: try n=4, then conjecture closed form.

RULES OF THUMB
==============
- Read the inbox FIRST when there are unread user_messages; the user's
  most recent intent may supersede prior work.
- Tight queries beat wide ones. The retrieved-content budget is hard.
- Notes should be compressed knowledge, not verbatim transcripts.
- Refer to blocks by `<type>:<index>` in your continuing_instruction so
  you can re-query specifically next turn.
- If you run out of budget, prioritize the continue over notes.
- Use `say` sparingly: only when the user needs to hear something. Most
  reasoning belongs in notes / continue, not in user-facing messages.
"""


def _render_stats(store_stats: dict) -> str:
    lines = []
    for type_name, info in store_stats.items():
        count = info["count"]
        last = info["last_index"]
        if count == 0:
            lines.append(f"    {type_name}: (none)")
        else:
            lines.append(
                f"    {type_name}: {count} blocks, last index {last}"
            )
    return "\n".join(lines)


def _render_inbox(store_stats: dict) -> str:
    info = store_stats.get("user_message", {"count": 0, "unread": 0, "earliest_unread": -1})
    unread = info.get("unread", 0)
    earliest = info.get("earliest_unread", -1)
    total = info.get("count", 0)
    if unread == 0:
        if total == 0:
            return "    (no user messages yet)"
        return f"    all caught up — {total} user_message blocks, none unread."
    return (
        f"    {unread} unread (earliest: user_message:{earliest}); "
        f"{total} total user_message blocks."
    )


def render_system_prompt(
    L: int,
    store_stats: dict,
    *,
    scratch_budget: int | None = None,
) -> str:
    """Render the system prompt with current store stats and budget figures."""
    half = max(1, L // 2)
    scratch = scratch_budget if scratch_budget is not None else max(1, L // 8)
    return SYSTEM_PROMPT_TEMPLATE.format(
        half_L=half,
        scratch_budget=scratch,
        store_stats=_render_stats(store_stats),
        inbox_status=_render_inbox(store_stats),
    )


# --------------------------------------------------------------------- #
# Legacy schema (paged-tokens architecture; deprecated but preserved)   #
# --------------------------------------------------------------------- #

LEGACY_OP_REFERENCE: dict[str, str] = {
    "e": "evict N tokens from head of middle",
    "r": "retrieve LEN tokens from chunk CID at offset OFS",
    "q": "query refs of chunk CID",
    "a": "annotate chunk CID with TAG",
    "l": "link chunk A -> chunk B",
    "s": "top-K similarity search for QUERY (embeddings)",
}

LEGACY_PREFIX_SCHEMA: str = (
    "OPS: e N | r CID OFS LEN | q CID | a CID TAG | l A B | s Q K\n"
    "Window cap L. Issue ops one per line. Use r to bring back evicted tokens.\n"
    "Evict before retrieving when full."
)

# Keep `PREFIX_SCHEMA` as an alias for `LEGACY_PREFIX_SCHEMA` so existing
# callers (the deprecated runner path, the interleaved-thinking client)
# don't break. New code uses `render_system_prompt`.
PREFIX_SCHEMA = LEGACY_PREFIX_SCHEMA
