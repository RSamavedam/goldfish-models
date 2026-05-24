# goldfish-models

> Language models with goldfish-sized working memory.

Hard-capped active context window (`L` from 2k down to 32 tokens), verbatim
off-GPU paging of evicted content, and a token-frugal tool API the model uses
to manage its own working memory. The bet: at the human-working-memory
regime, the tiny active window is not a limitation — it is the point.
Reasoning gets externalized into a queryable chunk store, and the model
learns (via RL) to manage that store the way a CPU manages cache.

See [DESIGN.md](DESIGN.md) for the full spec, motivation, and four-phase plan.

## What's here

- `src/rlm_paged/store/` — verbatim chunk store with metadata + reference graph
- `src/rlm_paged/window/` — active-window state with hard-cap invariants
- `src/rlm_paged/tools/` — single-char op codes (`e`, `r`, `q`, `a`, `l`, `s`)
- `src/rlm_paged/client/` — provider adapters (OpenAI / Anthropic / Gemini) — stubs
- `src/rlm_paged/harness/` — sweep runner + cost cap
- `src/rlm_paged/bench/` — benchmark loaders — stubs
- `src/rlm_paged/reward/` — sandboxed reward primitives (reused from legacy)
- `tests/` — unit tests for store, window, tools, cost cap
- `legacy/` — the previous TTT-Discover skeleton, archived in place

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Phases

1. **Phase 1** — API harness + L-sweep across long-doc, dialogue, and coding benchmarks
2. **Phase 2** — RL finetune Qwen2.5-Coder-14B on p5 (8×H100) under the paged regime
3. **Phase 3** — Re-eval + cognitive-psych probes for emergent memory structure
4. **Phase 4** — Triton kernels exploiting the pinned-prefix + extreme-batch regime
