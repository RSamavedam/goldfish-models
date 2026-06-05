"""Quick status snapshot of the paper sweep runs.

Reads runs/paper2/{baseline,scratchpad}.jsonl and prints a compact
table of cells completed + solved breakdown by L. Run locally after
`aws s3 sync s3://...goldfish.../paper2 runs/paper2 --quiet` to get
the freshest data.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def render_L(L: int) -> str:
    return "∞" if L == 0 else str(L)


def sort_L(L: int) -> float:
    return float("inf") if L == 0 else float(L)


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open() if l.strip()]


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/paper2")
    b = load(root / "baseline.jsonl")
    s = load(root / "scratchpad.jsonl")
    print(f"sweep root: {root}")
    print(f"  baseline:   {len(b):>2}/25 cells, solved {sum(1 for r in b if r['solved'])}")
    print(f"  scratchpad: {len(s):>2}/25 cells, solved {sum(1 for r in s if r['solved'])}")
    print()

    # Per-L tally
    by_L: dict[int, dict[str, list[dict]]] = defaultdict(lambda: {"B": [], "S": []})
    for r in b:
        by_L[r["cell"]["L"]]["B"].append(r)
    for r in s:
        by_L[r["cell"]["L"]]["S"].append(r)

    print(f"{'L':>4}  baseline       scratchpad")
    print("-" * 40)
    for L in sorted(by_L, key=sort_L):
        bL = by_L[L]["B"]
        sL = by_L[L]["S"]
        b_str = f"{sum(1 for r in bL if r['solved'])}/{len(bL)} solved" if bL else "—"
        s_str = f"{sum(1 for r in sL if r['solved'])}/{len(sL)} solved" if sL else "—"
        print(f"{render_L(L):>4}  {b_str:<14} {s_str}")
    print()

    # List every solved cell explicitly
    solves = [(r, "baseline") for r in b if r["solved"]] + [(r, "scratchpad") for r in s if r["solved"]]
    if solves:
        print("Solved cells:")
        for r, var in solves:
            c = r["cell"]
            print(f"  {var:<12} L={render_L(c['L']):>3}  {c['task_id']:<28}  turns={r['turns']:>2}")
    else:
        print("(no solves yet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
