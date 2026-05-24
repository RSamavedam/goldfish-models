# Closed-model thinking: what L-enforcement is actually possible

The goldfish-models thesis is that a hard `L` cap on a model's *active
reasoning context* + a paging tool API forces structured externalization.
That story works cleanly when the wrapper can see and bound every token
of the model's reasoning. For frontier closed APIs, what we can see and
what we can control varies — and the gap is real enough that Phase 1
results have to be reported in two regimes.

This doc lays out, per provider, exactly what we can and can't do.

## Regime A: full L-enforcement on reasoning

Available when the model emits reasoning as **visible text tokens** that
the wrapper can read, count, and interject between.

Providers in this regime:

- Any model that produces CoT as plain output text. The wrapper enforces
  L on the whole visible stream.
- **DeepSeek-R1** (via Together) — emits `<think>...</think>` blocks that
  we strip into `thinking_text` and route as `kind="thinking"` Segments.
  The PagedScheme will evict overflowing thinking content into the chunk
  store like any other segment.
- Any open-weight model we run ourselves (vLLM, HF) where every output
  token is visible.

This is the regime the experiment is designed around. The paged scheme
beats truncated when the model uses the chunk store to recover content
that would otherwise have been dropped.

## Regime B: soft cap via budget + interleaved tool calls

**Anthropic Claude (Opus / Sonnet 4-4.7 family) with extended thinking
+ interleaved-thinking beta + native paging tools.**

What we *can* do:

- Set `thinking.budget_tokens = L` as a *soft* cap. The model usually
  stops thinking around the budget but may overshoot — Anthropic docs
  explicitly warn about this.
- Expose our five paging ops as native tools (`evict_head`,
  `retrieve_chunk`, `query_refs`, `annotate_chunk`, `link_chunks`).
- With `interleaved-thinking-2025-05-14` beta, the model can interleave
  `thinking` blocks with `tool_use` blocks within a single logical turn.
  Between thinking blocks it can call our tools to externalize prior
  thinking into the chunk store.
- Read every thinking block's text (so we see the reasoning, score it,
  and route it through scheme logic).

What we *cannot* do:

- Hard-cap thinking at exactly L tokens. The budget is advisory.
- Interrupt mid-thinking-block. A thinking block runs to its budget
  before the model can emit a tool call.
- Modify, summarize, or strip thinking blocks that get round-tripped
  between API calls. Signed thinking blocks must be preserved verbatim
  on follow-up messages or continuation breaks.

So this regime approximates Regime A. The paging happens *between*
thinking blocks, not *within* them. This is close enough to test the
core claim ("structured externalization helps") but the L cap is fuzzy.

## Regime C: opaque reasoning, no L-enforcement

**OpenAI o-series (o1, o3, o4-mini).** Reasoning happens server-side
before any tokens reach the client. We can:

- Set `reasoning_effort: low | medium | high` as a coarse budget hint.
- See `completion_tokens_details.reasoning_tokens` (the count only).
- Continue a reasoning chain across calls via the Responses API's
  encrypted `reasoning.encrypted_content` items — but the content is
  opaque; we can't read it, summarize it, or page it.

We cannot:

- Read any token of the reasoning.
- Stop reasoning before it completes.
- Inject tool results mid-reasoning.

For these models, "applying goldfish-models" means restricting the
*visible response* with our paging tools — but the reasoning happens in
the model's full native context regardless. So results from this regime
aren't a clean test of the L story. They're a baseline showing how the
provider behaves under our scheme without actually enforcing L on its
reasoning.

**Gemini 2.5 with thinking** is the same shape: `thinking_budget` is a
soft cap, thought summaries can be exposed (paraphrases, not raw tokens),
and no tool interjection mid-thinking. Treated as Regime C for Phase 1.

## What this means for the paper

Phase 1 results split into two columns:

| Regime | Providers | What "L=128 paged" means |
|---|---|---|
| A | DeepSeek-R1, Together open-weight non-thinking models, vLLM-local models | Hard cap on visible reasoning. Real L-enforcement. |
| B | Anthropic Claude w/ extended thinking + interleaved tools | Soft cap per thinking block + paging between blocks. Approximate. |
| C | OpenAI o-series, Gemini 2.5 thinking | No reasoning-level enforcement. Visible-output cap only. |

The headline plots (solve rate vs L) should be drawn separately for each
regime. The cleanest cross-regime comparison is at the small-L end of
Regime A vs the smallest-budget configuration of Regime B — that's where
the experiments inform each other.

The DeepSeek-R1 result is the single most important data point in Phase
1: it's the only place where a *reasoning-trained* model's thinking goes
through Regime A enforcement without any RL training of our own. If
paged-CoT outperforms truncated-CoT for R1 at small L, that's the
strongest pre-RL evidence for the thesis. If not, Phase 2 has to
discover the behavior the existing model can't learn from prompting
alone.

## How to read the data

The JSONL output schema doesn't distinguish regimes explicitly. Use
these heuristics:

- `metadata.client` ends with `+interleaved` → Regime B
- `metadata.client` is `openai:o*` or `gemini:*` with `thinking_tokens > 0`
  → Regime C
- everything else → Regime A

When plotting, group by regime and don't overlay regimes on the same
axes without footnoting the difference.
