# goldfish-models: Design Document

> Language models with goldfish-sized working memory.
>
> A wrapper that enforces a hard active-context cap `L` on top of any LLM,
> with verbatim, append-only paging of evicted content to off-GPU storage
> and a token-frugal tool API for the model to manage its own working memory.
>
> The bet: at the human-working-memory regime (`L` from 2k down to 32 tokens),
> the small active window is not a limitation to work around — it is the
> point. Reasoning gets externalized into a queryable chunk store, and the
> model learns (via RL) to manage that store the way a CPU manages cache.

This file is the canonical spec. The rest of the repo implements it.

---

## 1. Problem framing

### 1.1 The regime nobody has built

Three lines of work exist in the long-context / memory-agent literature:

- **MemAgent (Tsinghua, 2025)**: trained model + fixed-length **lossy** memory
  panel, one-pass streaming, no revisitation.
- **RLM (Zhang & Khattab, MIT, 2025)**: prompt-as-Python-variable with a REPL,
  recursive sub-calls, but the **root window is unbounded** in principle.
  Apple's SRLM paper (2026) showed it degrades when the input fits in window.
- **MemGPT / Letta**: lossy summarization on eviction, soft cap, no
  programmatic access to verbatim history.

The unexplored design point:

> **Hard-capped active window `L`, with lossless verbatim paging to off-GPU
> storage, append-only re-prefill on retrieval, and the model trained via RL
> to manage the store under a strict cap.**

This combines RLM's interface idea (model writes ops over its context) with
MemAgent's training story (RL teaches the access policy) and adds the strict
small-`L` regime neither has tested.

### 1.2 Why "human working memory" framing matters

Human WM is famously small — Miller's 7±2 chunks, Cowan's revised ~4 chunks.
That's ~20–40 tokens of "actively held" content, behind a vast long-term
store accessed by structured retrieval. We're targeting that regime directly:
the sweep bottoms out at `L=32`, which is roughly Miller's 7 chunks at ~5
tokens per chunk.

This framing has three consequences:

1. **The sweep is bottom-up**, not "how small can we go before it breaks".
   We start at `L ∈ {32, 64, 128}` and work *up* to 2k. The failure curve at
   small `L` under prompting alone is the motivation for RL training.
2. **The pinned prefix lives outside `L`.** Long-term knowledge (system
   prompt, tool schemas, op codes) is "always on" — it doesn't count against
   working memory. `L` is purely the scratch budget.
3. **The Phase 3 probes target known cognitive-psych curves** (serial
   position, chunking, rehearsal, recency) so emergent structure can be
   compared to human data, not just narrated post-hoc.

---

## 2. System architecture

### 2.1 The four regions

At any decode step the model sees:

```
┌────────────────────────────────────────────────────────────┐
│  PINNED PREFIX  (outside L; always on; never invalidated)  │
│  - System prompt                                           │
│  - Tool schemas (terse op-code form)                       │
│  - Persistent task state                                   │
├────────────────────────────────────────────────────────────┤
│  ACTIVE WINDOW  (size ≤ L; the scratch budget)             │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  MIDDLE  (immutable; KV cache stable)                │  │
│  │  - Retrieved chunks pending consumption              │  │
│  │  - Recent tool-call results                          │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  TAIL SCRATCH  (only mutable region)                 │  │
│  │  - Current reasoning / generation                    │  │
│  │  - This is where eviction & append happen            │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘

           ↑ tool calls ↓
┌────────────────────────────────────────────────────────────┐
│  CHUNK STORE  (off-GPU; CPU DRAM + NVMe spill)             │
│  Verbatim token IDs, chunk metadata, reference graph       │
└────────────────────────────────────────────────────────────┘
```

The pinned prefix is implemented as a constant prefix on every request; the
wrapper guarantees its KV cache is never invalidated. It does not count
against `L`.

### 2.2 The eviction discipline

Three rules. They are the entire reason re-prefill on retrieval is cheap:

1. **Prefix is pinned.** Never touched.
2. **Middle is immutable.** Once a chunk lands in the middle, it stays
   until evicted from the head of the active window. No insert, no replace,
   no reorder.
3. **All mutation happens at the tail.** Eviction removes from the *head* of
   the active window (oldest content first, after the prefix). Retrieval
   appends to the *tail*. Both ops keep the prefix's KV valid and only
   re-prefill `O(k_r)` new tokens.

When the active window hits `L` and the model wants to retrieve `k_r` more
tokens, it must first evict at least `k_r` tokens from the head. The wrapper
enforces this: a retrieve that would overflow `L` either fails or
auto-evicts oldest content.

### 2.3 What gets stored on eviction

**Token IDs**, not KV vectors. Reasons:

- Token IDs: 4 B/token. 1M tokens = 4 MB. Free.
- KV vectors: ~131 KB/token for Llama-3-8B. 1M tokens = 131 GB. NVMe only.
- KV vectors carry RoPE rotations tied to original position. Bringing them
  back at a new position requires either pre-RoPE storage + re-rotation, or
  "reactivate at original position" (which breaks the append-only discipline).
- Re-prefill of `k_r` tokens at the tail is `~10 ms` on H100 for `k_r=500`.
  The "savings" from storing KV is `~6 ms`, not worth the 30,000× storage
  multiplier or the positional headache.

Verdict: store token IDs. Re-prefill on retrieve.

### 2.4 The chunk store

A chunk is a contiguous run of token IDs plus metadata. Default chunk size
is **256 tokens** (fixed, kernel-friendly). The model can retrieve any
contiguous span of one or more chunks.

```python
class Chunk:
    id: int                    # monotonically increasing
    tokens: list[int]          # length ≤ chunk_size
    created_at_step: int       # global decode step at creation
    original_position: int     # position in active window when evicted
    outgoing_refs: list[int]   # chunk IDs this one references
    incoming_refs: list[int]   # chunk IDs that reference this one
    tags: list[str]            # model-assigned topic / role tags
    embedding: bytes | None    # optional dense vector for similarity ops
    access_count: int          # times retrieved since creation
    last_accessed_step: int    # most recent retrieval step
```

References are populated two ways:

- **Implicit**: when the model retrieves chunk B while chunk A is still in
  the active window, an edge `A → B` is recorded by the wrapper.
- **Explicit**: the model calls `link(a, b)` or `annotate(chunk_id, tag)`.

Storage tiers:

- **Pinned CPU DRAM** (`cudaMallocHost`) for the working set (~ last 10M
  tokens of trajectory).
- **NVMe** spill for cold chunks past the working set.
- **GPU HBM** — never. Chunks only enter HBM via the re-prefill path.

### 2.5 The tool API

Token-frugal by design. At `L=32` a JSON tool call would be wider than the
window itself. Op codes are single characters; arguments are positional
integers; returns are packed.

| Op   | Args              | Semantics                                                   | Approx. tokens |
|------|-------------------|-------------------------------------------------------------|----------------|
| `e`  | `n`               | Evict `n` tokens from head of active window                 | ~5             |
| `r`  | `cid, ofs, len`   | Retrieve `len` tokens from chunk `cid` at offset `ofs`      | ~8             |
| `q`  | `cid`             | Return outgoing+incoming refs of chunk `cid`                | ~5             |
| `a`  | `cid, tag`        | Annotate chunk `cid` with `tag` (≤8 tokens)                 | ~6             |
| `l`  | `a, b`            | Link chunks `a → b`                                         | ~5             |
| `s`  | `query, k`        | Top-k similarity search (only if embeddings enabled)        | varies         |

Returns are *also* terse: `r` returns the raw tokens (the point); `q`
returns a comma-separated integer list; `a`/`l` return a single status byte.

The wrapper parses these calls out of the model's generation stream and
executes them inline. The tool-call traffic counts against `L` while it sits
in the active window, but is evictable once it has been consumed.

### 2.6 The active-window invariant

The wrapper enforces, at every decode step:

```
len(prefix)               <= prefix_max          (fixed, outside L)
len(middle) + len(tail)   <= L
len(tail)                 <= tail_max            (default L/2)
```

A retrieval that would violate the invariant either fails (returns an error
the model can see) or auto-evicts from the head of the middle. The choice is
configurable per training run; Phase 1 default is **fail loudly** so we can
measure how often the model bumps the cap.

---

## 3. The sweep (Phase 1)

### 3.1 L values

`L ∈ {32, 64, 128, 256, 512, 1024, 2048}`.

Native-context baseline (no cap) is a separate line on every plot, not a
sweep point.

### 3.2 Three benchmark families

| Family            | Benchmarks                                          | Why                                             |
|-------------------|-----------------------------------------------------|-------------------------------------------------|
| Long-doc QA       | RULER (NIAH variants), ∞Bench, LongBench v2          | Cleanest NIAH-style stress test for retrieval   |
| Memory/dialogue   | LoCoMo, MemoryAgentBench (CR-MH track)              | Mem0 / MemAgent published baselines             |
| Coding            | SWE-bench Verified (50–100 task subset), repo-QA    | Aligns with Phase 2 RL training target          |

Sweeping across all three addresses the generalization gap Claude flagged in
the transcript — existing memory-agent papers train *and* test on one family.

### 3.3 Providers and baselines

Per L value, per task family, run:

- **Paged**: our wrapper enforcing cap `L` with chunk-store tools.
- **Summarized**: MemGPT-style recursive-summary eviction at cap `L`.
- **RAG**: top-k embedding retrieval (no eviction discipline).
- **Sub-agent isolation**: orchestrator with cap `L`, sub-agents at native.
- **Native baseline**: no cap (single line per benchmark, not per `L`).

Providers: OpenAI, Anthropic, Gemini. Same harness, one `LLMClient`
interface.

### 3.4 Metrics

Per (task, provider, scheme, `L`):

- **Solve rate** (task-specific success metric).
- **Tokens consumed** (input + output + tool-call traffic).
- **Tool calls per task** (count by op).
- **Wall-clock seconds per task** (with cost cap; mark `failed-by-budget`).
- **Active-window utilization** (mean fraction of `L` used).

### 3.5 Cost cap

Hard ceiling of **100k tokens** of tool-call traffic per task. Beyond that,
mark `failed-by-budget`. This prevents L=32 runs that thrash forever from
draining API credits.

---

## 4. Phase 2: RL finetune

### 4.1 Hardware

p5.48xlarge: 8×H100 80GB, 640GB HBM aggregate.

### 4.2 Base model

**Qwen2.5-Coder-14B** as the primary target.

- 14B fits in HBM under LoRA with vLLM rollouts on 4 GPUs and FSDP training
  on the other 4.
- Coder variant is essential because Phase 1 includes coding in the sweep
  and the model needs to be strong at structured manipulation.
- 32B is reachable with aggressive sharding but RL step-time becomes painful.
- 7B leaves performance on the table given we have 8×H100.

### 4.3 LoRA configuration

- Rank 64, alpha 128.
- Adapters on q/k/v/o projections + MLP gate/up/down.
- bf16 master weights, bf16 LoRA, fp32 optimizer state.
- ~30 GB per GPU including KV cache.

### 4.4 RL algorithm

Baseline **GRPO**. Real contribution is the credit-assignment adaptation:

- An eviction at step 5 only pays off when a retrieve at step 47 succeeds.
  Vanilla GRPO assigns advantage at trajectory level — too sparse.
- Two candidate adaptations to ablate:
  - **Multi-conv DAPO** (after MemAgent): each (window-state, decision) is a
    sub-trajectory, single terminal reward propagated to all.
  - **Step-wise GRPO** (after AgeMem): dense step-level shaping reward for
    memory ops, terminal reward for task.

### 4.5 Reward

```
R = task_success
  - λ · max(0, tool_calls - target_tool_calls)
  - μ · mean(active_window_size) / L
  - ν · failed_tool_calls
```

The shaping terms are critical. Without them the model thrashes the store
or fills the active window with garbage. Coefficients tuned per task family.

### 4.6 Environment

Long-context coding env. Repository QA where the full repo doesn't fit in
`L`. Candidates:

- SWE-Gym (training-friendly subset of SWE-bench)
- BigCodeBench-Hard
- Custom: a held-out set of mid-size repos with synthetic Q→answer pairs

### 4.7 Stack

- **verl** (NVIDIA) for the RL backbone. Closer to what MemAgent built on.
- **vLLM** for rollouts (Phase 4 will eventually swap to our custom kernels).
- **FSDP** for training shards.
- Multi-context rollout structure has to be added on top of verl — that's
  part of the contribution.

---

## 5. Phase 3: Re-eval + emergent structure

### 5.1 Headline experiment

Re-run the Phase 1 sweep on the RL'd model. Expected result: the post-RL
model holds accuracy at small `L` where the prompted base model collapses.

### 5.2 Cognitive-psych probes

Designed up front so claims are measurable:

- **Chunking** — does the model pack multi-step reasoning into single chunks
  before evicting? Measure: information density of chunks (entropy / token).
- **Rehearsal** — does it cycle the same chunks back in to keep them
  "active"? Measure: chunk re-retrieval rate vs. random baseline.
- **Serial-position curve** — `P(retrieve | chunk age)`. Compare to human
  serial-position data (Murdock 1962): primacy + recency humps.
- **Articulatory loop analog** — does a tight register of last-K tokens act
  like phonological WM? Probe by clamping tail size and measuring task drop.
- **Topical clustering** — embed retrieved chunks; cluster by topic across
  trajectory. Compare to random baseline.
- **Reference-graph structure** — degree distribution of the implicit
  reference graph. Hubs would suggest topical anchors.

### 5.3 Capacity probe

At eval time, artificially clamp `L` below the training value. The RL'd
model should degrade gracefully; the base model should fall off a cliff.
This is the "did it learn to externalize, or did it memorize a window-size"
test.

---

## 6. Phase 4: Triton kernels

### 6.1 The structural opportunities

Mainstream serving stacks don't exploit three properties our system has:

1. **Pinned prefix** — its KV is permanently valid. A custom kernel can
   skip generic prefix invalidation logic.
2. **Fixed `L`** across all sequences in a batch — no ragged-length overhead,
   no block table.
3. **Append-only tail** — re-prefill is small and predictable; persistent
   kernels can keep state across many steps.

### 6.2 Concrete targets

- Custom Triton attention kernel for fixed-`L` + pinned-prefix.
- Persistent decoder loop via CUDA Graphs.
- Async page-fault path: overlap CPU→GPU transfer of retrieved chunks with
  the current decode step.
- Throughput benchmark vs. vLLM PagedAttention on the same workload.

### 6.3 The throughput math at human-WM regime

At `L=128` on Qwen2.5-14B:
- KV per sequence ≈ 16 MB.
- 640 GB HBM minus ~28 GB model = ~600 GB available.
- ~37,500 concurrent sequences theoretically. Realistically: a few thousand
  before kernel overhead dominates.

Target: **5–10k tokens/sec aggregate on 1×H100 at batch ≥ 1000.**

---

## 7. Module layout

```
src/rlm_paged/
  store/
    chunk.py            # Chunk dataclass + metadata
    store.py            # In-process chunk store (CPU DRAM tier)
    nvme.py             # Cold tier (deferred)
    refs.py             # Reference graph operations
  window/
    state.py            # ActiveWindowState (prefix/middle/tail tracking)
    invariants.py       # Enforce len + tail constraints
  tools/
    api.py              # Op codes, parser, dispatcher
    schema.py           # Terse format for prefix injection
  client/
    base.py             # LLMClient interface
    openai.py           # OpenAI adapter
    anthropic.py        # Anthropic adapter
    gemini.py           # Gemini adapter
    local_vllm.py       # vLLM adapter (Phase 2+)
  harness/
    runner.py           # Run a single (model, scheme, L, task) cell
    schemes.py          # paged / summarized / rag / subagent / native
    cost_cap.py         # Per-task budget enforcement
  bench/
    base.py             # Task interface
    ruler.py            # RULER NIAH variants
    longbench.py
    locomo.py
    swe_bench.py
    repo_qa.py
  reward/                       # REUSED FROM ttt_discover
  utils/
    config.py                   # REUSED FROM ttt_discover
    logging.py                  # REUSED FROM ttt_discover

scripts/
  sweep_phase1.py       # Drive the L sweep
  analyze.py            # Plot curves
  rl_train.py           # Phase 2 entry point (deferred)

configs/
  default.yaml
  sweep/                # Per-benchmark configs
  rl/                   # Phase 2 configs
```

### 7.1 What we keep from `ttt_discover`

- `reward/sandbox.py` — verbatim, will be used as the SWE-bench / coding
  reward executor.
- `reward/registry.py` — verbatim, registry pattern for task rewards.
- `utils/logging.py` — verbatim, JSONL logger.
- `utils/config.py` — verbatim, OmegaConf loader.
- `checkpointing/manager.py` — for Phase 2.

### 7.2 What we delete or archive

- `engine.py`, `buffer.py`, `policy.py`, `trainer.py` — TTT-Discover specific,
  not reusable. Archived under `legacy/` for one commit, then removed.
- `search/puct.py`, `search/sampling.py` — same.
- `scheduler/beta.py` — same.
- `problems/sorting_net.py`, `problems/triton_matmul.py` — irrelevant.

---

## 8. Open questions

1. **Tool API token budget at L=32.** Even with single-char op codes, a
   single retrieve + return might saturate the window. Do we need a "no-op
   mode" where the model emits only op codes and the tool result is the
   *only* thing that lands in the tail?
2. **Implicit reference edges**: every co-occurrence in the window creates
   an edge, which blows up the graph quickly. Sparsify (only chunks the
   model explicitly references in its tail tokens)?
3. **Embedding store for `s` op**: do we maintain dense embeddings from the
   start, or defer until Phase 2 when the model might learn to use them?
   Phase 1 default: off.
4. **Cost cap at 100k tokens** may be too tight at L=32 where every retrieve
   is small. Revisit after first dry run.

These get resolved by the first round of Phase 1 dry runs.

---

## 9. Status

| Item | State |
|---|---|
| Design doc (this file) | ✅ |
| Repo pivot in place | in progress |
| Phase 1 harness | not started |
| Phase 1 sweep | not started |
| Phase 2 RL | not started |
| Phase 3 probes | not started |
| Phase 4 kernels | not started |
