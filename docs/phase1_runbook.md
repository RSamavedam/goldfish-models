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
```

## Smoke test (no API calls)

```bash
PYTHONPATH=src python -m pytest tests/ -q
PYTHONPATH=src python scripts/sweep_phase1.py --dry-run
```

The dry run prints every (provider, scheme, L, benchmark) cell the full
sweep would visit. With the default config that's ~264 cells, which
expand task-by-task into ~25k total runs.

## Small live run (one provider, one benchmark, a handful of tasks)

```bash
PYTHONPATH=src python scripts/sweep_phase1.py \
  --config configs/sweep/phase1.yaml \
  --output runs/smoke.jsonl \
  --only-provider anthropic:claude-opus-4-7 \
  --limit-tasks 3
```

This still expands over all schemes × all L values × all benchmarks for
that one provider — about 88 cells × 3 tasks = ~264 runs. Cost-bound at
100k tokens per task by default.

## Full sweep

```bash
PYTHONPATH=src python scripts/sweep_phase1.py \
  --config configs/sweep/phase1.yaml \
  --output runs/phase1.jsonl
```

The script is resumable — already-completed cell keys are read from the
output JSONL on startup and skipped. Safe to ctrl-c and re-run.

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
- 100k tokens × 3 providers × ~88 cells × ~50 tasks = ~1.3B tokens
- At GPT-5 list pricing (~$10/MTok input, $30/MTok output) that's roughly
  a few hundred USD if every cell actually spends its cap.
- Real usage will be much lower because most cells terminate via
  `final_answer` long before 100k tokens. Expect ~$50-100 total for a
  full sweep across all three providers.

Run small first, then scale.

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
