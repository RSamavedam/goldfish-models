"""Surface model-behavior patterns from per-turn transcripts.

Reads a JSONL produced by `sweep_shell.py --turn-log` (one row per
model call) and prints a structured digest of:

  - per-cell turn count, token spend, finish-reason distribution
  - per-provider token / turn distribution
  - empty-done refusals (turns where the response tried to terminate
    without writing anything to user_output)
  - cwd-thrash / re-orientation indicators (`ls`, `cat`, `git status`
    rates per cell)
  - response length percentiles per provider
  - top-K "wandering" cells (most empty turns)

Designed to be re-run on the same JSONL as it grows; output is
deterministic given the same input. No silent truncation — if we cap
output (top-K, percentiles), we say so.

Usage:
    python scripts/analyze_turns.py runs/turns_smokeN.jsonl
    python scripts/analyze_turns.py runs/turns_smokeN.jsonl --cell '*Astropy-12907*'
    python scripts/analyze_turns.py runs/turns_smokeN.jsonl --turn-detail 'gpt-5*Astropy*'
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

# Heuristics for classifying what a turn is "doing". Cheap regex over
# the response text — wrong sometimes, but consistently wrong, which
# is what we need for relative comparisons across cells/providers.
_RE_EXPLORE = re.compile(r"\b(ls|find|tree|pwd|git status|git log)\b")
_RE_READ = re.compile(r"\b(cat|head|tail|less|nl|grep|sed -n)\b")
_RE_EDIT = re.compile(r"\b(patch|sed -i|>\s*[\w./-]+\.\w+|cat\s*<<)")
_RE_VERIFY = re.compile(r"\b(pytest|python -m pytest|tox|make test|python\s+-c)\b")
_RE_EXPORT = re.compile(r"\bexport(-string)?\b")
_RE_DONE = re.compile(r"^\s*done\s*$", re.M)


def _iter_rows(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _classify(resp: str) -> set[str]:
    tags: set[str] = set()
    if _RE_EXPLORE.search(resp):
        tags.add("explore")
    if _RE_READ.search(resp):
        tags.add("read")
    if _RE_EDIT.search(resp):
        tags.add("edit")
    if _RE_VERIFY.search(resp):
        tags.add("verify")
    if _RE_EXPORT.search(resp):
        tags.add("export")
    if _RE_DONE.search(resp):
        tags.add("done")
    if not tags:
        tags.add("other")
    return tags


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _digest(rows: list[dict], cell_filter: str | None) -> None:
    if cell_filter:
        rows = [r for r in rows if fnmatch.fnmatch(r["cell_key"], cell_filter)]
    if not rows:
        print("no rows match filter", file=sys.stderr)
        return

    by_cell: dict[str, list[dict]] = defaultdict(list)
    by_provider: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[r["cell_key"]].append(r)
        by_provider[r["provider"]].append(r)

    print(f"=== analyzed {len(rows)} turns across {len(by_cell)} cells")
    print()

    # ---------- per-provider response-length distribution
    print("--- per-provider output tokens (median / p90 / max) ---")
    for prov, prov_rows in sorted(by_provider.items()):
        outs = [r.get("output_tokens", 0) or 0 for r in prov_rows]
        print(
            f"  {prov:30s}  n={len(prov_rows):4d}  "
            f"median={statistics.median(outs):6.0f}  "
            f"p90={_percentile(outs, 0.9):6.0f}  "
            f"max={max(outs):6d}"
        )
    print()

    # ---------- per-provider turn-classification rates
    print("--- per-provider turn-tag rates (% of turns with each tag) ---")
    for prov, prov_rows in sorted(by_provider.items()):
        tag_counts: Counter[str] = Counter()
        for r in prov_rows:
            for t in _classify(r.get("response") or ""):
                tag_counts[t] += 1
        n = len(prov_rows)
        bits = "  ".join(
            f"{tag}={100 * tag_counts[tag] / n:4.1f}%"
            for tag in ("explore", "read", "edit", "verify", "export", "done", "other")
        )
        print(f"  {prov:30s}  {bits}")
    print()

    # ---------- cells with most "other" / empty-looking turns
    print("--- top-10 cells by 'other' turn count (wandering indicator) ---")
    wandering = []
    for key, crows in by_cell.items():
        other_n = sum(1 for r in crows if "other" in _classify(r.get("response") or ""))
        if other_n:
            wandering.append((other_n, len(crows), key))
    wandering.sort(reverse=True)
    for other_n, total, key in wandering[:10]:
        print(f"  {other_n:3d}/{total:3d}  {key}")
    if len(wandering) > 10:
        print(f"  ({len(wandering) - 10} more cells with wandering not shown)")
    print()

    # ---------- empty-done attempts (done tag + no preceding export in cell)
    print("--- cells with 'done' but no 'export' in the same trajectory ---")
    suspect = []
    for key, crows in by_cell.items():
        tags = set()
        for r in crows:
            tags |= _classify(r.get("response") or "")
        if "done" in tags and "export" not in tags:
            suspect.append((len(crows), key))
    suspect.sort(reverse=True)
    for n, key in suspect[:10]:
        print(f"  turns={n:3d}  {key}")
    if not suspect:
        print("  (none — every 'done' was preceded by at least one export attempt)")
    print()

    # ---------- finish-reason distribution
    print("--- finish_reason distribution ---")
    fr = Counter(r.get("finish_reason") or "?" for r in rows)
    for k, v in fr.most_common():
        print(f"  {k:20s}  {v:5d}  ({100 * v / len(rows):4.1f}%)")
    print()


def _turn_detail(rows: list[dict], cell_filter: str) -> None:
    """Dump every turn for a specific cell, prompts and all."""
    matching = [r for r in rows if fnmatch.fnmatch(r["cell_key"], cell_filter)]
    if not matching:
        print(f"no cells match: {cell_filter}", file=sys.stderr)
        return
    keys = sorted({r["cell_key"] for r in matching})
    for key in keys:
        crows = sorted(
            (r for r in matching if r["cell_key"] == key), key=lambda r: r["turn"]
        )
        print("=" * 72)
        print(f"CELL: {key}")
        print("=" * 72)
        for r in crows:
            print(f"\n--- turn {r['turn']} "
                  f"(in={r.get('input_tokens')} out={r.get('output_tokens')} "
                  f"think={r.get('thinking_tokens')} finish={r.get('finish_reason')}) ---")
            print("[USER PROMPT]")
            print((r.get("user_prompt") or "")[-2000:])
            if r.get("thinking_text"):
                print("\n[THINKING]")
                print(r["thinking_text"][:1500])
            print("\n[RESPONSE]")
            print((r.get("response") or "")[:3000])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--cell", default=None,
                        help="glob filter on cell_key for the digest")
    parser.add_argument("--turn-detail", default=None,
                        help="glob filter — dump every turn of matching cells")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"file not found: {args.path}", file=sys.stderr)
        return 1

    rows = list(_iter_rows(args.path))
    if not rows:
        print("no rows in file", file=sys.stderr)
        return 1

    if args.turn_detail:
        _turn_detail(rows, args.turn_detail)
    else:
        _digest(rows, args.cell)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
