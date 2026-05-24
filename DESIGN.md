# goldfish-models: Design Document

> Language models with goldfish-sized working memory.
>
> Each turn is a stateless process. The model receives a tightly budgeted
> input region assembled from prior-turn retrievals, emits structured
> memory ops (queries, append-only notes, a mandatory `continue` message
> to the next turn, optional external tool calls), and is then discarded.
> The next turn starts fresh with whatever the previous turn explicitly
> wrote to the block store.
>
> The bet: at the human-working-memory regime (`L` from 2k down to 32
> tokens), reasoning is forced to externalize into the block store. The
> model learns (via RL) to manage memory hierarchies the way a programmer
> manages files — typed blocks, explicit reads, append-only writes.

This file is the canonical spec. The rest of the repo implements it.

---

## 1. Problem framing

### 1.1 The regime nobody has built

Three lines of work exist in the long-context / memory-agent literature:

- **MemAgent (Tsinghua, 2025)**: trained model + fixed-length **lossy**
  memory panel, one-pass streaming, no revisitation.
- **RLM (Zhang & Khattab, MIT, 2025)**: prompt-as-Python-variable with a
  REPL, recursive sub-calls, but the **root window is unbounded** in
  principle. Apple's SRLM paper (2026) showed it degrades when the input
  fits in window.
- **MemGPT / Letta**: lossy summarization on eviction, soft cap, no
  programmatic access to verbatim history.

The unexplored design point:

> **Hard-capped per-turn input + output, lossless append-only block store,
> stateless turns with explicit message-passing between them, model trained
> via RL to manage memory hierarchies under that strict cap.**

This combines RLM's interface idea (model writes ops over its context)
with MemAgent's training story (RL teaches the access policy) and adds
the strict small-`L` regime, *plus* a structural shift: the model never
"holds" working state across turns — it has to deliberately externalize
or lose it.

### 1.2 Why "human working memory" framing matters

Human WM is famously small — Miller's 7±2 chunks, Cowan's revised ~4
chunks. That's ~20–40 tokens of "actively held" content, behind a vast
long-term store accessed by structured retrieval. We target that regime
directly: the sweep bottoms out at `L=32`, which is roughly Miller's 7
chunks at ~5 tokens per chunk.

Three consequences:

1. **The sweep is bottom-up**, not "how small can we go before it
   breaks". We start at `L ∈ {32, 64, 128}` and work *up* to 2k. The
   failure curve at small `L` under prompting alone is the motivation
   for RL training.
2. **The system prompt lives outside `L`.** Long-term knowledge (tool
   schemas, memory-hierarchy tips, current store stats) is "always on" —
   it doesn't count against working memory. `L` is purely the per-turn
   I/O budget.
3. **The Phase 3 probes target known cognitive-psych curves** (serial
   position, chunking, rehearsal, recency) so emergent structure can be
   compared to human data, not just narrated post-hoc.

### 1.3 Why stateless turns

The earlier draft of this design built a single mutable active window
with a four-region structure (pinned prefix / immutable middle / mutable
tail) and tried to enforce `L` by paging tokens between calls. That
design has three problems that this restructured one fixes:

1. **It only works on visible-CoT models.** Closed APIs (OpenAI o-series,
   Gemini) hide reasoning server-side; Anthropic with extended thinking
   exposes it but only between thinking blocks. There was no single
   architecture that worked across all providers.
2. **L was a soft cap inside a turn.** A thinking block could overshoot;
   reasoning was uninterruptible. The cap was only really enforced
   between turns.
3. **State was implicit.** What got carried forward depended on what
   tokens happened to still be in the window. Hard to reason about, hard
   to debug.

The new design makes the turn the atomic unit, makes state-carrying
explicit (via `continuing_instruction` and the block store), and works
identically on every provider because the model isn't asked to do
anything provider-specific — it just reads structured input and writes
structured output.

---

## 2. System architecture

### 2.1 The per-turn process model

Each turn is a single API call to the model. Conceptually:

```
turn(prev_continuing_instruction, retrieved_blocks, store) -> {
    notes_to_append: list[str],
    continuing_instruction: str,             # mandatory
    queries_for_next_turn: list[Query],
    external_tool_calls: list[ToolCall],
}
```

The model has no persistent state across turns. Every turn is a fresh
process. The block store is the only shared mutable state.

The prompt the model sees at turn `t`:

```
┌────────────────────────────────────────────────────────────┐
│  SYSTEM PROMPT  (outside L; same every turn modulo stats)  │
│  - Op schemas and how to use them                          │
│  - Memory-hierarchy tips ("write tight notes, query by     │
│    type", "use notes to compress observations", etc.)      │
│  - Current store stats: |task|, |observation|, |note|,     │
│    |continuing_instruction| block counts + last index of   │
│    each.                                                   │
├────────────────────────────────────────────────────────────┤
│  RETRIEVED-CONTENT REGION  (size ≤ L/2; "stdin")           │
│                                                            │
│  1. Mandatory: prior turn's `continuing_instruction`       │
│     (truncated to L/2 if oversize, with a marker).         │
│  2. Each query that the prior turn requested, executed by  │
│     the harness against the store. Each result carries     │
│     minimal metadata: (block_type, block_index, char       │
│     range within block). Multiple results from one query   │
│     are concatenated with separators.                      │
│  3. If the queue would push past L/2, the harness rejects  │
│     later retrievals and includes a brief error block      │
│     in their place ("RETRIEVAL_OVER_BUDGET: query N").     │
├────────────────────────────────────────────────────────────┤
│  RESPONSE REGION  (size ≤ L/2; "stdout")                   │
│                                                            │
│  Model output, structured. Sections:                       │
│  - Optional `<scratch>...</scratch>` (≤ L/8 tokens) for    │
│    in-turn thinking. Discarded after the turn. Not         │
│    counted against next turn's L; this turn's response     │
│    budget still includes it.                               │
│  - Ops (one per line or one structured tool call each):    │
│    - `note <text>`        — append a note block            │
│    - `continue <text>`    — set the next turn's            │
│      continuing_instruction (exactly one per turn)         │
│    - `query <type> <indices>` — request a block range to   │
│      land in the next turn's retrieved-content region      │
│    - `pipe <query> <dest>` — pipe the result of a stored   │
│      query into `note` / `continue` / an external tool     │
│      call (collapses query+write/call into one op)         │
│    - `call <tool_name> <args>` — external tool call        │
│      (code-exec, web search, etc.); result becomes an      │
│      observation block in the store and may be queried     │
│      by the next turn                                      │
│    - `say <text>` — optional. Sends a message to the user; │
│      writes an assistant_reply block and (if the harness   │
│      has a callback wired) surfaces the text to a UI side  │
│      channel. Most turns don't use this — reasoning lives  │
│      in notes/continue, not in user-facing messages.       │
└────────────────────────────────────────────────────────────┘

           writes ↑    ↓ reads
┌────────────────────────────────────────────────────────────┐
│  BLOCK STORE  (off-GPU; CPU DRAM + NVMe spill)             │
│                                                            │
│  Append-only, typed:                                       │
│    - user_message[0..]      : the user's input. The FIRST  │
│                                user_message IS the original│
│                                task prompt. New messages   │
│                                can arrive between turns.   │
│    - assistant_reply[0..]   : the model's `say` outputs.   │
│                                Audit trail of model→user.  │
│    - observation[0..]       : harness-injected tool        │
│                                results, verbatim.          │
│    - note[0..]              : model-authored notes.        │
│    - continuing_instruction : one per turn, indexed by     │
│                                turn number.                │
│                                                            │
│  Each block carries: id (per-type monotonic), global       │
│  index, created_at_turn, real-wall-clock timestamp, refs   │
│  (outgoing+incoming), tags (model-assigned), optional      │
│  embedding.                                                │
│                                                            │
│  The store tracks an unread cursor per type (most useful   │
│  for user_message). When the model queries a range that    │
│  includes previously-unseen blocks, the cursor advances    │
│  past them. The system prompt's INBOX section reports the  │
│  current unread count + earliest unread index so the model │
│  knows to read new user input before continuing.           │
└────────────────────────────────────────────────────────────┘
```

### 2.2 The two-half budget

`L` is split:

- **L/2 retrieved-content cap** on what the harness can inject into the
  next turn's input (excluding system prompt).
- **L/2 response cap** on what the model can emit, including any
  `<scratch>` and all op text.

This is symmetric and conceptually simple. It also means a model with a
2k context window can in principle handle `L=2048` (giving the model 1k
in and 1k out) without ever overflowing the API's actual context limit.

### 2.3 Optional scratch region

The model can open a `<scratch>...</scratch>` block at the start of its
response, up to L/8 tokens. The scratch is:

- **Discarded after the turn.** Not stored anywhere. The model can't
  reference it next turn.
- **Counted toward the L/2 response budget.** Spending tokens on scratch
  means fewer tokens for ops.
- **The only place free-form thinking lives** in this architecture.

Why this exists: forbidding scratch entirely forces reasoning models
(o-series, R1) to skip their natural thinking step, which is a known
quality hit. A small scratch budget gives them a register-file-sized
space to think coherently for the current turn without substituting for
the block store. Anything they want to *remember* still has to go through
`note` or `continue`.

L/8 is the default; configurable per sweep cell. At `L=32` that's 4
tokens of scratch — essentially none, which is the point.

### 2.4 Block types and store semantics

Four block types, monotonically indexed within each type:

| Type                       | Author  | Mutable | Purpose                                  |
|----------------------------|---------|---------|------------------------------------------|
| `task[i]`                  | harness | no      | The original problem prompt(s).          |
| `observation[i]`           | harness | no      | Tool-call results, dataset content, etc. |
| `note[i]`                  | model   | append  | Model-authored compressed knowledge.     |
| `continuing_instruction[t]`| model   | append  | The message to turn `t+1`. One per turn. |

All blocks are append-only. There is no `delete`, no `update`, no
`evict` — the store grows monotonically. Logical invalidation of stale
notes is left to a future `supersede(old_index, new_index)` op (not v1).

### 2.5 Queries

A query selects a contiguous range from a single block type:

```
query(type, [start_index, end_index], optional: tag_filter)
```

Returns the matching blocks' contents, each prefixed with minimal
metadata (`[type:index]`). Multiple queries from one turn are unioned
into the next turn's retrieved-content region in the order they were
issued.

The L/2 cap is enforced retrieval-by-retrieval: as the harness assembles
the next turn's input, it stops adding new retrieval payloads once L/2
is reached and emits a `RETRIEVAL_OVER_BUDGET` marker for the dropped
ones. The model sees explicitly which of its queries were honored and
which weren't.

The guaranteed first slot is the prior turn's `continuing_instruction`,
truncated (with a marker) if it overshoots L/2 on its own.

### 2.6 The mandatory `continue` op

Every turn must emit exactly one `continue <text>` op. This is:

- **The only state carried forward to the next turn.** Everything else
  is implicit (must be re-queried).
- **Truncated to L/2 if oversize**, but with a `[TRUNCATED]` marker
  visible to the next turn so the model knows to retrieve more state.
- **A retry trigger if missing.** A response without a `continue` op is
  rejected; the harness re-prompts with an error message in the
  retrieved-content region of the same turn.

The motivation is identical to OS pipe semantics: a process must produce
*something* on stdout before it terminates, or the pipeline breaks. The
"something" can be `continue done` (the convention for "this trajectory
is complete, score me").

### 2.7 The op surface

| Op       | Args                                  | Semantics                                                              |
|----------|---------------------------------------|------------------------------------------------------------------------|
| `note`   | `<text>`                              | Append a new note block.                                               |
| `continue` | `<text>`                            | Set this turn's continuing_instruction. Mandatory; exactly one per turn. |
| `query`  | `<type> <start> <end>` `[tag=T]`       | Queue a retrieval for the next turn's input region.                   |
| `pipe`   | `<query_spec> -> <dest>`              | Execute the query and feed results into `dest` (note/continue/call).   |
| `call`   | `<tool_name> <args...>`               | External tool call. Result becomes a new observation block.            |

The op surface is text-channel by default (one op per line, terse) for
provider compatibility. A structured-tools mirror (Anthropic JSON
tool definitions) exists for providers with native tool calling — same
ops, structured envelope. See `src/rlm_paged/tools/structured.py`.

The old single-character op codes (`e`/`r`/`q`/`a`/`l`/`s`) from the
previous architecture are removed. `e` (evict) was never needed in this
model — eviction is structural, not user-controlled. The rest are
subsumed by the new op set.

### 2.8 Why this works across providers (closing the asymmetry)

The original four-region design depended on the wrapper seeing every
reasoning token to enforce `L`. Closed APIs (OpenAI o-series, Gemini)
hid the reasoning, breaking the architecture. Anthropic with extended
thinking was a halfway house that required signed-block round-tripping.

The new design has no such dependency. The model's input and output are
both observable text (or text + structured tool calls). What the model
does internally — whether it thinks for 10 tokens or 10k server-side
reasoning tokens — *doesn't change what the wrapper does or what it
enforces*. The provider's internal thinking is billed and counted; the
wrapper doesn't try to interrupt or page it. Per-turn `L` is enforced on
input (which the harness controls) and output (which the harness reads
and truncates if needed).

Reasoning models still get to think — they just do it server-side per
turn, with a small per-turn `<scratch>` available if helpful. Whatever
they decide is worth remembering they have to externalize via `note` or
`continue` like everyone else.

**This makes the design provider-agnostic.** The same harness loop runs
identically on GPT-5, Claude (with or without thinking), Gemini, R1,
Llama, Qwen — no special cases.

The closed-thinking-asymmetry doc
(`docs/closed_thinking_asymmetry.md`) describes the *old* design's
limitations and is preserved for context, but the architecture pivot
makes most of its content obsolete. We'll update that doc to reflect
the resolution.

### 2.9 OS / systems parallels

- **Stateless coroutines with explicit state-passing.** Each turn is one
  yield point; the value passed to the next yield is `continuing_
  instruction`; the heap is the block store.
- **Cooperative process pipeline with shared filesystem.** Each turn is
  a process; stdin = retrieved-content region; stdout =
  continuing_instruction; the chunk store is a mounted append-only FS.
- **Stateless RPC handler + database.** Each turn is one request handler
  invocation; the block store is the database; the continuing_instruction
  is the payload between coordinator and worker.
- **Event-sourced architecture.** All writes are append-only events;
  state is reconstructable from the log; future `supersede` op gives us
  logical update without losing audit.

We avoid framing it as "OS paging" — that metaphor breaks because pages
in a VM are involuntary and the resident set is hardware-managed.
Goldfish queries are *voluntary* and *predictive*: the model has to
guess what it'll need next and request it before the turn starts.

### 2.10 Storage tiers

- **Pinned CPU DRAM** for the working set (~last 10M tokens).
- **NVMe** spill for cold blocks past the working set.
- **GPU HBM** — never. Blocks enter HBM only via re-prefill during
  Phase 4 kernel work.

---

## 3. The sweep (Phase 1)

### 3.1 L values

`L ∈ {32, 64, 128, 256, 512, 1024, 2048}`.

Native-context baseline (no cap) is a separate line on every plot, not
a sweep point.

### 3.2 Benchmarks (test-time compute family)

| Benchmark           | Why                                                                |
|---------------------|--------------------------------------------------------------------|
| GPQA Diamond        | Graduate-science MC; classic TTC benchmark.                        |
| MATH-500            | Free-response math; clean answer extraction.                       |
| AIME 2024/25        | Integer answers, hard reasoning, small set.                        |
| HLE (text-only)     | Hardest open benchmark; LLM-as-judge scoring (placeholder strict). |
| ARC-AGI             | Visual reasoning, grids; deferred until raw JSON loader.            |

### 3.3 Schemes

Schemes now differ in *how the response region is structured* and
*how the harness handles overflow*:

- **Stateless-turn (paged)**: the new default. Mandatory `continue`,
  L/2 each half, scratch ≤ L/8.
- **Truncated**: prior turn's full visible response gets sent back as
  input; oldest content drops when exceeding L. No `continue`,
  no notes, no queries — measures "what if the model just had `L`
  tokens of history every turn."
- **Summarized**: MemGPT-style — when L is exceeded, a summarizer call
  replaces the dropped content with a short paraphrase.
- **Native**: no L cap. Full visible history sent every turn (or the
  full multi-turn assistant/user/tool_use thread for native-tool
  providers). Baseline.

The stateless-turn scheme is the system under test. Truncated and
summarized are the lossy baselines we want to beat. Native is the
upper-bound reference.

### 3.4 Providers

Same lineup as before, but with the architectural pivot every provider
runs the same harness loop:

- `openai:gpt-5`, `openai:o3`, etc.
- `anthropic:claude-opus-4-7` (extended thinking can be on or off — the
  per-turn budget is unchanged; thinking lives server-side).
- `gemini:gemini-2.5-pro`
- `together:meta-llama/Llama-3.3-70B-Instruct-Turbo`
- `together:meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo`
- `together:Qwen/Qwen2.5-72B-Instruct-Turbo`
- `together:Qwen/Qwen2.5-Coder-32B-Instruct`
- `together:deepseek-ai/DeepSeek-V3`
- `together:deepseek-ai/DeepSeek-R1`

The `+interleaved` Claude variant from the previous design is kept as a
**Regime-B alternate baseline**, not the primary path. Its purpose now
is the ablation "does interleaved thinking with paging tools beat the
new stateless-turn architecture?" — we expect no, since the new design
strictly generalizes.

### 3.5 Metrics

Per (task, provider, scheme, `L`):

- **Solve rate** (task-specific success metric).
- **Tokens consumed** (input + output + server-side thinking + external
  tool tokens).
- **Op counts** (note / continue / query / pipe / call).
- **Wall-clock seconds per task** (with cost cap).
- **Retrieval-budget overflow rate** (how often L/2 input is hit).
- **Continue-truncation rate** (how often the model's `continue`
  overshoots and gets truncated).
- **Notes written / queries issued per turn** (memory hygiene).
- **Store growth rate** (blocks per turn).

### 3.6 Cost cap

Hard ceiling of **100k tokens** of total billed tokens per task. Beyond
that, mark `failed-by-budget`. Revisit if many cells trip the cap.

---

## 4. Phase 2: RL finetune

(Unchanged from previous draft; the architectural pivot doesn't change
the training shape, just what the model is being trained to do.)

### 4.1 Hardware

p5.48xlarge: 8×H100 80GB, 640GB HBM aggregate.

### 4.2 Base model

**Qwen2.5-Coder-14B** as the primary target.

### 4.3 LoRA configuration

Rank 64, alpha 128. Adapters on q/k/v/o + MLP gate/up/down.

### 4.4 RL algorithm

Baseline **GRPO** with credit-assignment adaptation for multi-turn
trajectories. The new architecture actually *simplifies* this somewhat:
because each turn is stateless, the trajectory factors into independent
per-turn decisions modulo the chain through `continuing_instruction` and
queries. We can use turn-level advantage with a discount factor more
cleanly than the previous multi-conv DAPO required.

### 4.5 Reward

```
R = task_success
  - λ · max(0, total_billed_tokens - target_tokens)
  - μ · retrieval_overflow_rate
  - ν · continue_truncation_rate
  - ρ · store_bloat_penalty       # discourages dumping everything into notes
```

The shaping terms now align with the architecture: penalize sloppy
memory hygiene (over-querying, oversize continues, dumping verbatim
observations into notes without compression).

### 4.6 Environment

Long-context coding env. Repository QA where the full repo doesn't fit
in `L`. SWE-Gym + BigCodeBench-Hard + custom repo-QA.

### 4.7 Stack

verl + vLLM + FSDP. The multi-turn rollout structure is naturally
expressible because turns are independent API calls — no signed-block
round-tripping to fight.

---

## 5. Phase 3: Re-eval + emergent structure

(Largely unchanged.)

Re-run the Phase 1 sweep on the RL'd model. Cognitive-psych probes:
chunking, rehearsal, serial-position curve, articulatory-loop analog,
topical clustering, reference-graph structure. Capacity probe (clamp
`L` below training value at eval).

New probes specific to the architecture:

- **Note-vs-continue ratio** — does the model learn to use notes for
  durable knowledge and continues for short-term passing?
- **Query specificity** — distribution of query ranges; does the model
  learn tight queries instead of wide-net ones?
- **Pipe usage** — does the model use `pipe` to collapse query+write
  into single ops, saving response tokens?

---

## 6. Phase 4: Triton kernels

(Unchanged in goals; the architectural pivot is about the model-facing
surface, not the kernel-facing surface.)

Custom Triton attention kernel for fixed-`L` + small batched contexts.
Persistent decoder loop via CUDA Graphs. Throughput vs vLLM
PagedAttention baseline.

At `L=128` on Qwen2.5-14B: KV per sequence ≈ 16 MB; ~37,500 concurrent
sequences fit in 640 GB HBM minus weights. Realistic ceiling: a few
thousand before kernel overhead dominates. Target: **5–10k tokens/sec
aggregate on 1×H100 at batch ≥ 1000.**

---

## 7. Module layout (updated)

```
src/rlm_paged/
  store/
    block.py             # NEW: typed Block dataclass
    store.py             # block store with typed query()
    chunk.py             # DEPRECATED: kept for legacy/interleaved path
  tools/
    api.py               # Op parser + dispatcher (new op surface)
    schema.py            # System-prompt prelude
    structured.py        # Native-tool mirror of new ops
  client/
    base.py              # LLMClient interface (unchanged)
    openai.py / anthropic.py / gemini.py / together.py
  harness/
    turn.py              # NEW: per-turn process; assembles input,
                         #      parses output, executes ops
    runner.py            # Sweep cell driver
    schemes.py           # stateless_turn / truncated / summarized / native
    cost_cap.py
  bench/
    *                    # unchanged
  reward/
  utils/

scripts/
  sweep_phase1.py
  analyze.py
  rl_train.py            # Phase 2

configs/
  default.yaml
  sweep/phase1.yaml
```

### 7.1 Deprecated but preserved

- `window/` — the four-region active window is gone. We may delete this
  directory entirely after the rewrite stabilizes; keep for one or two
  commits as reference.
- `client/anthropic.py::generate_with_dispatcher` — the interleaved-
  thinking path. Kept as the Regime-B baseline scheme. Marked clearly
  in docstrings.
- `tools/api.py`'s old single-char op codes (`e`/`r`/`q`/`a`/`l`/`s`)
  — replaced by the new ops. The new parser still accepts the old
  syntax for one release cycle so legacy tests keep passing while
  they're migrated.

### 7.2 Reused as-is

- `reward/sandbox.py`, `reward/registry.py`, `reward/base.py`
- `utils/logging.py`, `utils/config.py`
- `bench/*` (no changes needed; benchmarks are scheme-agnostic)
- `client/*` adapters except for the deprecated dispatcher path

---

## 8. Open questions

1. **Scratch budget calibration.** Default `L/8`. At L=32 that's 4
   tokens (effectively none). At L=2048 it's 256 tokens (genuinely
   useful for a turn's worth of thinking). Whether this is the right
   curve is empirical.

2. **`pipe` semantics for very large query results.** If a `pipe` would
   produce a result that exceeds L/2 (for note dest) or the destination
   tool's input limit (for call dest), do we (a) truncate silently with
   a marker, (b) reject and surface error, (c) chunk into multiple
   notes? Default: (b) with a clear error so the model learns to
   constrain queries.

3. **Tool-call observation handling.** External tool results go into an
   `observation` block. If the result is large (a search returning 50KB
   of HTML), do we (a) store verbatim and trust the model to query
   narrowly, (b) summarize on the way in, (c) chunk and let the model
   query by sub-block? Default: (a) — verbatim is the contract; the
   model has to learn to query narrowly. May need to revisit.

4. **Mandatory-continue retry policy.** If the model doesn't emit a
   `continue`, we re-prompt with an error. How many retries before
   declaring the cell `failed-by-protocol`? Default: 2 retries, then
   fail.

5. **Embedding store for semantic queries.** The current query op is
   structural (by type/index/tag). A future variant could allow
   semantic similarity search via embeddings — deferred until we see
   whether the structural ops are sufficient.

---

## 9. Status

| Item                                              | State          |
|---------------------------------------------------|----------------|
| Design doc (this file, rewritten)                 | ✅              |
| Repo pivot in place                               | ✅              |
| Stateless-turn architecture in code               | in progress    |
| Phase 1 sweep                                     | not started    |
| Phase 2 RL                                        | not started    |
| Phase 3 probes                                    | not started    |
| Phase 4 kernels                                   | not started    |
