"""Analyze sweep JSONL output and emit summary tables + optional plots.

Reads one or more JSONL files produced by `sweep_stateless.py` (or the
legacy `sweep_phase1.py`), aggregates by (provider, benchmark, L), and
prints:

  - solve-rate table per benchmark (rows = providers, cols = L values)
  - mean tokens-per-task per cell
  - op-usage table (mean ops per turn, by op type)
  - failure-mode breakdown per L

With `--plots <dir>`, also writes PNGs:
  - solve_rate_vs_L_<benchmark>.png  — one line per provider
  - op_distribution.png              — heatmap of mean ops/turn
  - failure_modes.png                — stacked bar by L
  - token_efficiency_<benchmark>.png — scatter of solve-rate vs tokens

Plotting requires matplotlib. If it isn't installed, plots are skipped
with a warning; the tables still print.

Robust to partial data: cells with no rows render as "—" in tables.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------- #
# Loading                                                               #
# --------------------------------------------------------------------- #


def load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            print(f"warning: {path} not found, skipping", file=sys.stderr)
            continue
        with path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(
                        f"warning: {path}:{lineno} bad JSON ({exc}); skipping",
                        file=sys.stderr,
                    )
    return rows


def row_provider(row: dict) -> str:
    cell = row.get("cell") or {}
    return cell.get("provider", "<unknown>")


def row_benchmark(row: dict) -> str:
    cell = row.get("cell") or {}
    return cell.get("benchmark", "<unknown>")


def row_L(row: dict) -> int:
    cell = row.get("cell") or {}
    return int(cell.get("L", 0))


# --------------------------------------------------------------------- #
# Aggregation                                                           #
# --------------------------------------------------------------------- #


def aggregate(
    rows: list[dict],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Group rows by (provider, benchmark, L). Returns aggregate dicts."""
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for r in rows:
        key = (row_provider(r), row_benchmark(r), row_L(r))
        groups[key].append(r)

    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    for key, group in groups.items():
        n = len(group)
        solved = sum(1 for r in group if r.get("solved"))
        score_mean = statistics.fmean(float(r.get("score", 0)) for r in group)
        toks = [
            (r.get("input_tokens", 0) or 0)
            + (r.get("output_tokens", 0) or 0)
            + (r.get("thinking_tokens", 0) or 0)
            for r in group
        ]
        turns = [int(r.get("turns", 0) or 0) for r in group]
        op_totals: Counter[str] = Counter()
        for r in group:
            for op_name, cnt in (r.get("op_counts") or {}).items():
                op_totals[op_name] += int(cnt)
        failure_modes = Counter(
            (r.get("failure_reason") or ("solved" if r.get("solved") else "wrong_answer"))
            for r in group
        )
        total_turns = sum(turns) or 1
        out[key] = {
            "n": n,
            "solved": solved,
            "solve_rate": solved / n if n else 0.0,
            "score_mean": score_mean,
            "tokens_mean": statistics.fmean(toks) if toks else 0.0,
            "tokens_total": sum(toks),
            "turns_mean": statistics.fmean(turns) if turns else 0.0,
            "op_counts_total": dict(op_totals),
            "op_per_turn": {k: v / total_turns for k, v in op_totals.items()},
            "failure_modes": dict(failure_modes),
        }
    return out


# --------------------------------------------------------------------- #
# Tables                                                                #
# --------------------------------------------------------------------- #


def print_solve_rate_table(
    agg: dict, *, benchmarks: list[str], providers: list[str], L_values: list[int]
) -> None:
    for bench in benchmarks:
        print(f"\n## Solve rate — {bench}")
        header = ["provider"] + [f"L={L}" if L > 0 else "native" for L in L_values] + ["n"]
        rows: list[list[str]] = []
        for prov in providers:
            line = [prov]
            n_total = 0
            for L in L_values:
                cell = agg.get((prov, bench, L))
                if cell is None:
                    line.append("—")
                else:
                    n_total += cell["n"]
                    line.append(f"{cell['solve_rate']:.2f} ({cell['n']})")
            line.append(str(n_total))
            rows.append(line)
        _print_md_table(header, rows)


def print_token_table(
    agg: dict, *, benchmarks: list[str], providers: list[str], L_values: list[int]
) -> None:
    for bench in benchmarks:
        print(f"\n## Mean tokens/task — {bench}")
        header = ["provider"] + [f"L={L}" if L > 0 else "native" for L in L_values]
        rows: list[list[str]] = []
        for prov in providers:
            line = [prov]
            for L in L_values:
                cell = agg.get((prov, bench, L))
                line.append("—" if cell is None else f"{cell['tokens_mean']:.0f}")
            rows.append(line)
        _print_md_table(header, rows)


def print_op_usage_table(agg: dict) -> None:
    print("\n## Op usage (mean ops per turn)")
    op_universe = sorted({k for v in agg.values() for k in v["op_per_turn"].keys()})
    if not op_universe:
        print("  (no op data)")
        return
    header = ["provider", "benchmark", "L"] + op_universe + ["turns_mean"]
    rows: list[list[str]] = []
    for (prov, bench, L), cell in sorted(agg.items()):
        line = [prov, bench, f"L={L}" if L > 0 else "native"]
        for op in op_universe:
            v = cell["op_per_turn"].get(op, 0.0)
            line.append(f"{v:.2f}" if v else "—")
        line.append(f"{cell['turns_mean']:.1f}")
        rows.append(line)
    _print_md_table(header, rows)


def print_failure_modes(agg: dict, *, L_values: list[int]) -> None:
    print("\n## Failure-mode breakdown (summed across providers/benchmarks)")
    mode_universe = sorted(
        {m for v in agg.values() for m in v["failure_modes"].keys()}
    )
    if not mode_universe:
        print("  (no data)")
        return
    header = ["L"] + mode_universe
    rows: list[list[str]] = []
    for L in L_values:
        counts: Counter[str] = Counter()
        for (_p, _b, L_), cell in agg.items():
            if L_ != L:
                continue
            for mode, n in cell["failure_modes"].items():
                counts[mode] += n
        line = [f"L={L}" if L > 0 else "native"]
        for mode in mode_universe:
            line.append(str(counts.get(mode, 0)) if counts.get(mode, 0) else "—")
        rows.append(line)
    _print_md_table(header, rows)


def _print_md_table(header: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(str(c)) for c in [h] + [row[i] for row in rows])
        for i, h in enumerate(header)
    ]
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    line = "|" + "|".join(f" {h:<{widths[i]}} " for i, h in enumerate(header)) + "|"
    print(line)
    print(sep)
    for row in rows:
        line = "|" + "|".join(f" {row[i]:<{widths[i]}} " for i in range(len(header))) + "|"
        print(line)


# --------------------------------------------------------------------- #
# Plots                                                                 #
# --------------------------------------------------------------------- #


def _try_import_matplotlib():
    try:
        import matplotlib  # type: ignore[import-not-found]
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]

        return plt
    except ImportError:
        return None


def plot_solve_rate_vs_L(agg, *, benchmarks, providers, L_values, out_dir: Path):
    plt = _try_import_matplotlib()
    if plt is None:
        print("warning: matplotlib not installed; skipping plots", file=sys.stderr)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    capped_L = [L for L in L_values if L > 0]
    has_native = any(L == 0 for L in L_values)

    for bench in benchmarks:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for prov in providers:
            xs, ys = [], []
            for L in capped_L:
                cell = agg.get((prov, bench, L))
                if cell is None or cell["n"] == 0:
                    continue
                xs.append(L)
                ys.append(cell["solve_rate"])
            if xs:
                ax.plot(xs, ys, marker="o", label=prov)
            if has_native:
                nat = agg.get((prov, bench, 0))
                if nat is not None:
                    ax.axhline(nat["solve_rate"], linestyle=":", alpha=0.4)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("L (canonical tokens)")
        ax.set_ylabel("solve rate")
        ax.set_title(f"goldfish-models · solve rate vs L · {bench}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        out = out_dir / f"solve_rate_vs_L_{bench}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"  wrote {out}", file=sys.stderr)


def plot_op_distribution(agg, *, providers, L_values, out_dir: Path):
    plt = _try_import_matplotlib()
    if plt is None:
        return
    op_universe = sorted({k for v in agg.values() for k in v["op_per_turn"].keys()})
    if not op_universe:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    capped_L = [L for L in L_values if L > 0]
    row_labels: list[str] = []
    matrix: list[list[float]] = []
    for prov in providers:
        for L in capped_L:
            cells = [c for (p, _b, L_), c in agg.items() if p == prov and L_ == L]
            if not cells:
                continue
            mean_ops = {
                op: statistics.fmean(
                    [c["op_per_turn"].get(op, 0.0) for c in cells]
                )
                for op in op_universe
            }
            row_labels.append(f"{prov}|L={L}")
            matrix.append([mean_ops[op] for op in op_universe])
    if not matrix:
        return
    fig, ax = plt.subplots(
        figsize=(1.0 + 0.7 * len(op_universe), 0.4 * len(row_labels) + 1.5)
    )
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(op_universe)))
    ax.set_xticklabels(op_universe)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title("mean ops per turn")
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    out = out_dir / "op_distribution.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}", file=sys.stderr)


def plot_failure_modes(agg, *, L_values, out_dir: Path):
    plt = _try_import_matplotlib()
    if plt is None:
        return
    mode_universe = sorted({m for v in agg.values() for m in v["failure_modes"].keys()})
    if not mode_universe:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    capped_L = [L for L in L_values if L > 0]
    if not capped_L:
        return
    counts_by_L: list[Counter[str]] = []
    for L in capped_L:
        c: Counter[str] = Counter()
        for (_p, _b, L_), cell in agg.items():
            if L_ != L:
                continue
            for mode, n in cell["failure_modes"].items():
                c[mode] += n
        counts_by_L.append(c)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottoms = [0.0] * len(capped_L)
    for mode in mode_universe:
        ys = [counts_by_L[i].get(mode, 0) for i in range(len(capped_L))]
        ax.bar([str(L) for L in capped_L], ys, bottom=bottoms, label=mode)
        bottoms = [bottoms[i] + ys[i] for i in range(len(capped_L))]
    ax.set_xlabel("L")
    ax.set_ylabel("count")
    ax.set_title("trajectory outcomes by L")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = out_dir / "failure_modes.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}", file=sys.stderr)


def plot_token_efficiency(agg, *, benchmarks, providers, out_dir: Path):
    plt = _try_import_matplotlib()
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for bench in benchmarks:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for prov in providers:
            xs, ys = [], []
            for (p, b, _L), cell in agg.items():
                if p != prov or b != bench:
                    continue
                if cell["n"] == 0 or cell["tokens_mean"] <= 0:
                    continue
                xs.append(cell["tokens_mean"])
                ys.append(cell["solve_rate"])
            if xs:
                ax.scatter(xs, ys, label=prov, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("mean tokens per task")
        ax.set_ylabel("solve rate")
        ax.set_title(f"solve rate vs token spend · {bench}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        out = out_dir / f"token_efficiency_{bench}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"  wrote {out}", file=sys.stderr)


# --------------------------------------------------------------------- #
# CLI                                                                   #
# --------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs", nargs="+", help="One or more JSONL files (e.g. runs/phase1.jsonl)."
    )
    parser.add_argument(
        "--plots",
        default=None,
        help="If set, write PNGs to this directory.",
    )
    parser.add_argument(
        "--L-order",
        default="0,32,64,128,256,512,1024,2048",
        help="Comma-separated L values to display. 0 = native.",
    )
    args = parser.parse_args()

    rows = load_jsonl([Path(p) for p in args.inputs])
    if not rows:
        print("no rows loaded; nothing to do", file=sys.stderr)
        return 1
    agg = aggregate(rows)

    providers = sorted({p for (p, _b, _L) in agg.keys()})
    benchmarks = sorted({b for (_p, b, _L) in agg.keys()})
    L_values = [int(x) for x in args.L_order.split(",") if x.strip()]

    print(f"# goldfish-models · sweep summary ({len(rows)} rows)")
    print(f"providers ({len(providers)}): {', '.join(providers)}")
    print(f"benchmarks ({len(benchmarks)}): {', '.join(benchmarks)}")
    print(f"L values: {L_values}")

    print_solve_rate_table(
        agg, benchmarks=benchmarks, providers=providers, L_values=L_values
    )
    print_token_table(
        agg, benchmarks=benchmarks, providers=providers, L_values=L_values
    )
    print_op_usage_table(agg)
    print_failure_modes(agg, L_values=L_values)

    if args.plots:
        out_dir = Path(args.plots)
        plot_solve_rate_vs_L(
            agg,
            benchmarks=benchmarks,
            providers=providers,
            L_values=L_values,
            out_dir=out_dir,
        )
        plot_op_distribution(
            agg, providers=providers, L_values=L_values, out_dir=out_dir
        )
        plot_failure_modes(agg, L_values=L_values, out_dir=out_dir)
        plot_token_efficiency(
            agg, benchmarks=benchmarks, providers=providers, out_dir=out_dir
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
