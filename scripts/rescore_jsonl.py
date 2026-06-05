"""Rescore patches in an existing JSONL by re-running the swebench harness.

Walks each row in the input, takes its final_answer_text (the patch),
and calls SweBenchVerifiedSuite.score() again. Writes a new JSONL with
solved/score updated; leaves all other fields untouched.

Use this after a bug-16-style scorer failure (e.g. parallel container
name collisions) to recover real solve counts without re-running the
agent trajectories.

Usage:
    python scripts/rescore_jsonl.py runs/paper8/baseline.jsonl \\
        --output runs/paper8/baseline.rescored.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rlm_paged.bench.swe_bench import SweBenchVerifiedSuite


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--diag-dir", type=Path, default=None,
                   help="Optional dir to drop scorer stdout/stderr per instance.")
    p.add_argument("--scorer-timeout-s", type=float, default=1800.0)
    p.add_argument("--only-non-empty", action="store_true",
                   help="Skip rows with empty final_answer_text.")
    args = p.parse_args()

    rows = []
    with args.input.open() as h:
        for line in h:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"loaded {len(rows)} rows from {args.input}", file=sys.stderr)

    # Build one suite per instance lazily — we need its task payload to
    # know the FAIL_TO_PASS / PASS_TO_PASS sets the scorer uses.
    suite = SweBenchVerifiedSuite(
        scorer_mode="subprocess",
        scorer_timeout_s=args.scorer_timeout_s,
        scorer_diag_dir=args.diag_dir,
    )
    tasks_by_id = {t.task_id: t for t in suite.tasks()}

    new_solves = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for i, r in enumerate(rows):
            patch = r.get("final_answer_text", "") or ""
            tid = r.get("cell", {}).get("task_id")
            old_solved = r.get("solved", False)

            if args.only_non_empty and not patch.strip():
                out.write(json.dumps(r, default=str) + "\n")
                continue

            if not patch.strip():
                # nothing to rescore
                out.write(json.dumps(r, default=str) + "\n")
                continue

            task = tasks_by_id.get(tid)
            if task is None:
                print(f"  ! unknown task {tid}, skipping", file=sys.stderr)
                out.write(json.dumps(r, default=str) + "\n")
                continue

            t0 = time.monotonic()
            try:
                solved, score = suite.score(task, patch)
            except Exception as exc:
                print(f"  ! score error on {tid}: {exc}", file=sys.stderr)
                out.write(json.dumps(r, default=str) + "\n")
                continue
            elapsed = time.monotonic() - t0
            print(
                f"  [{i+1}/{len(rows)}] L={r['cell']['L']:>5} {tid:<30s} "
                f"was_solved={old_solved} now_solved={solved} ({elapsed:.0f}s)",
                file=sys.stderr,
            )
            if solved and not old_solved:
                new_solves += 1
            r["solved"] = bool(solved)
            r["score"] = float(score)
            r["rescored"] = True
            out.write(json.dumps(r, default=str) + "\n")

    print(
        f"\nrescore complete: {new_solves} new solves recovered "
        f"(was {sum(1 for r in rows if r.get('solved'))} → now "
        f"{sum(1 for r in rows if r.get('solved')) + new_solves})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
