# Observation logs

Live record of how models actually behave under the hard-capped
context regime. The harness writes raw per-turn transcripts; humans
(and the assistant on demand) turn those into structured notes.

## Layout

- **`journal.md`** — append-only, chronological. One dated entry per
  observation session. What I (Claude) was looking at, what jumped
  out, hypotheses, predictions. Never edited after the fact.
- **`synthesis.md`** — current-state digest. Rewritten on demand.
  Hypotheses I currently believe, with pointers into journal entries
  that support or refute them. Stale entries get explicitly retired,
  not deleted.
- **Raw transcripts** — NOT committed. Live under `runs/turns_*.jsonl`
  and grow large fast. Each row is one model call:

      {
        cell_key, provider, benchmark, task_id, L, turn,
        system_prompt, user_prompt, response, thinking_text,
        input_tokens, output_tokens, thinking_tokens, finish_reason
      }

## How to produce raw transcripts

Pass `--turn-log PATH` to `scripts/sweep_shell.py`:

    python scripts/sweep_shell.py \
        --config configs/sweep/swe_bench.yaml \
        --output runs/smokeN.jsonl \
        --turn-log runs/turns_smokeN.jsonl \
        --use-swe-bench-cell yes

The harness flushes after every turn so live tailing works (`tail -f`,
`jq`, etc.).

## How to update the docs

Ask the assistant: *"refresh observations from runs/turns_smokeN.jsonl"*.

The assistant runs `scripts/analyze_turns.py` against the JSONL,
adds a dated entry to `journal.md` (with concrete examples + counts),
and rewrites `synthesis.md` to reflect the current best understanding.
No automation — patterns surface when a human asks, never silently.
