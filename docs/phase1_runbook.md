# Phase 1 Runbook

How to actually run the test-time-compute (TTC) sweep across providers,
schemes, L values, and benchmarks.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,api]'
```

API keys (any subset of providers you want to sweep):

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...        # or GEMINI_API_KEY
export TOGETHER_API_KEY=...      # open-weight models (Llama, Qwen, DeepSeek)
```

## Provider lineup (default config)

Frontier closed-source APIs:

- `openai:gpt-5`
- `anthropic:claude-opus-4-7`
- `gemini:gemini-2.5-pro`

Open-weight, biggest of each family, hosted on Together:

- `together:meta-llama/Llama-3.3-70B-Instruct-Turbo`
- `together:meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo`
- `together:Qwen/Qwen2.5-72B-Instruct-Turbo`
- `together:Qwen/Qwen2.5-Coder-32B-Instruct`
- `together:deepseek-ai/DeepSeek-V3`
- `together:deepseek-ai/DeepSeek-R1`  ← visible thinking, routed to chunk store

DeepSeek-R1 is the only model in the lineup whose reasoning chain is
visible to the harness (it emits `<think>...</think>` blocks). When run
under the paged scheme, R1's thinking is added as `kind="thinking"`
segments that the chunk store can page out and the model can retrieve via
`r` ops on later turns. For every other model, thinking (if any) is
token-billed but opaque.

## Smoke test (no API calls)

```bash
PYTHONPATH=src python -m pytest tests/ -q
PYTHONPATH=src python scripts/sweep_phase1.py --dry-run
```

The dry run prints every (provider, scheme, L, benchmark) cell the full
sweep would visit.

## Small live run

Cheapest viable test: one provider, a few tasks. Together's Llama 3.3-70B
is typically the cheapest competent model in the lineup (~$0.88/MTok), so
it's the right place to start.

```bash
PYTHONPATH=src python scripts/sweep_phase1.py \
  --config configs/sweep/phase1.yaml \
  --output runs/smoke.jsonl \
  --only-provider together:meta-llama/Llama-3.3-70B-Instruct-Turbo \
  --limit-tasks 3
```

Expansion: 4 benchmarks × (3 schemes × 7 L values + 1 native) = ~88 cells
× 3 tasks = ~264 runs.

If you want to start with DeepSeek-R1 (the visible-thinking test):

```bash
PYTHONPATH=src python scripts/sweep_phase1.py \
  --config configs/sweep/phase1.yaml \
  --output runs/smoke_r1.jsonl \
  --only-provider together:deepseek-ai/DeepSeek-R1 \
  --limit-tasks 3
```

R1 is more expensive than Llama (~$3/MTok input, $7/MTok output as of
late 2025) but it's the only model where the paged-CoT story is testable
without RL training.

## Full sweep

```bash
PYTHONPATH=src python scripts/sweep_phase1.py \
  --config configs/sweep/phase1.yaml \
  --output runs/phase1.jsonl
```

Resumable — already-completed cell keys are read from the output JSONL on
startup and skipped. Safe to ctrl-c and re-run.

## What's in the output

One JSONL row per (provider, scheme, L, benchmark, task, seed) cell.
Schema is `RunResult` from `rlm_paged.harness.runner`:

- `cell` — the SweepCell that was run (provider/scheme/L/benchmark/task_id)
- `solved` — bool, did the bench suite mark this correct
- `score` — float in [0, 1]
- `input_tokens` / `output_tokens` / `thinking_tokens`
- `turns` — number of multi-turn iterations
- `op_counts` — dict mapping op code → count (only for paged scheme)
- `op_errors` — count of failed op dispatches
- `wall_seconds`
- `mean_active_tokens` / `peak_active_tokens` — active-window utilization
- `finish_reason` — last turn's reason ("stop" | "length" | ...)
- `failure_reason` — populated iff the loop exited without a final answer
- `metadata.cost_cap_spent` — total tokens charged against the cap

## Cost estimation

Worst case per cell with the default 100k cost cap:

- 9 providers × ~88 cells × ~57 avg tasks × 100k tokens = ~4.5B tokens
  worst-case
- Real usage is much lower because most cells terminate at `final_answer`
  well before the cap

Rough per-provider total at the default config, assuming most cells use
~10-20k tokens not 100k:

| Provider                                | Est. cost (full sweep) |
|-----------------------------------------|------------------------|
| openai:gpt-5                            | $30-80                 |
| anthropic:claude-opus-4-7               | $40-100                |
| gemini:gemini-2.5-pro                   | $20-60                 |
| together: Llama-3.3-70B                 | $5-15                  |
| together: Llama-3.1-405B                | $30-80                 |
| together: Qwen2.5-72B                   | $5-15                  |
| together: Qwen2.5-Coder-32B             | $3-10                  |
| together: DeepSeek-V3                   | $5-15                  |
| together: DeepSeek-R1                   | $40-120                |

Total estimate for a complete sweep: **$200-500**. Always run small first
and check the `metadata.cost_cap_spent` distribution before scaling.

## Analysis

`scripts/analyze.py` is currently a stub. The intended plots:

1. **L-sweep curve**: solve rate vs L, one line per (provider, scheme),
   one figure per benchmark.
2. **Token-efficiency curve**: solve rate vs total tokens spent, same
   layout.
3. **Op-frequency heatmap**: for paged-scheme runs, which op codes get
   used and how often, by L value.
4. **Failure-mode breakdown**: stacked bar of `failure_reason` by L value
   and scheme.
5. **Thinking-routed comparison** (R1 only): solve rate under paged with
   thinking-as-chunks vs. paged with thinking-discarded vs. native. This
   is the cleanest pre-RL test of the structured-externalization claim.
