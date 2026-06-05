"""Generate paper figures from cell-level + per-turn JSONLs.

Inputs:
  --baseline   runs/.../baseline.jsonl   (one row per cell)
  --scratchpad runs/.../scratchpad.jsonl
  --turns-baseline   runs/.../turns_baseline.jsonl   (optional, per-turn)
  --turns-scratchpad runs/.../turns_scratchpad.jsonl
  --out paper/figures/

Outputs:
  fig1_solve_rate.png   solve rate vs L, two lines (baseline, scratchpad)
  fig2_tokens.png       median tokens-per-cell vs L, two lines
  fig3_failure_modes.png  stacked bar of failure_reason breakdown vs L × variant
  fig4_turn_breakdown.png (if turn-logs present) tag rate per turn-class vs L
  results.tex            tabular summary, ready to \\input{}
  results.md             same in markdown
  results.json           machine-readable numbers for inline citation in paper text

Each figure renders L=0 as "∞" on the x-axis (it's the native baseline,
not literally zero context). See render_L convention in
rlm_paged.shell.shell_runner_cell.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Mirror the project's L convention without importing the package
# (lets us run on data even if the package isn't installed locally).
L_NATIVE_SENTINEL = 0


def render_L(L: int) -> str:
    return "∞ (native)" if L == L_NATIVE_SENTINEL else str(L)


def sort_key_L(L: int) -> float:
    """Sort L values so native (L=0) lands at the FAR RIGHT of the axis
    (it's the upper bound, not a small value)."""
    return float("inf") if L == L_NATIVE_SENTINEL else float(L)


def load_cells(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open() as h:
        for line in h:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_turns(path: Path) -> list[dict]:
    return load_cells(path)


def cell_L(r: dict) -> int:
    """JSONL has nested cell{L:...} from dataclasses.asdict."""
    return int(r["cell"]["L"])


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def aggregate(rows: list[dict]) -> dict[int, dict]:
    """Return {L: {n, solved, solve_rate, median_tokens, failure_modes}}."""
    by_L: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_L[cell_L(r)].append(r)
    out: dict[int, dict] = {}
    for L, group in by_L.items():
        n = len(group)
        solved = sum(1 for r in group if r.get("solved"))
        toks = [
            (r.get("input_tokens", 0) or 0)
            + (r.get("output_tokens", 0) or 0)
            + (r.get("thinking_tokens", 0) or 0)
            for r in group
        ]
        fail_modes = Counter(
            (r.get("failure_reason") or "unscored") if not r.get("solved") else "solved"
            for r in group
        )
        out[L] = {
            "n": n,
            "solved": solved,
            "solve_rate": solved / n if n else 0.0,
            "median_tokens": statistics.median(toks) if toks else 0,
            "p90_tokens": sorted(toks)[max(0, int(len(toks) * 0.9) - 1)] if toks else 0,
            "failure_modes": dict(fail_modes),
        }
    return out


def aggregate_turns(rows: list[dict]) -> dict[int, dict]:
    """Per-L tag rates for turn-classification: how often does the
    model re-cat instructions, hit length-cap, write to notes.md, etc.
    """
    import re

    _re = {
        "re_cat_instructions": re.compile(r"\bcat instructions\.txt\b"),
        "notes_md": re.compile(r"\bnotes\.md\b"),
        "git_diff": re.compile(r"\bgit (?:-C \w+ )?diff\b"),
        "export": re.compile(r"\bexport\b"),
        "pytest": re.compile(r"\bpytest\b"),
        "length_cap": None,  # via finish_reason
        "zero_out_deep_think": None,  # via output_tokens/thinking_tokens
    }
    by_L: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_L[int(r["L"])].append(r)
    out: dict[int, dict] = {}
    for L, group in by_L.items():
        n = len(group)
        tags = Counter()
        for r in group:
            resp = r.get("response") or ""
            for tag, rx in _re.items():
                if rx is not None and rx.search(resp):
                    tags[tag] += 1
            if r.get("finish_reason") == "length":
                tags["length_cap"] += 1
            if (r.get("output_tokens") or 0) == 0 and (
                r.get("thinking_tokens") or 0
            ) > 1500:
                tags["zero_out_deep_think"] += 1
        out[L] = {"n": n, "tag_rate": {t: tags[t] / n for t in tags}}
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def plot_solve_rate(
    baseline: dict[int, dict],
    scratchpad: dict[int, dict],
    out_path: Path,
) -> None:
    Ls = sorted(set(baseline) | set(scratchpad), key=sort_key_L)
    xs = list(range(len(Ls)))
    xlabels = [render_L(L) for L in Ls]
    b_rates = [baseline.get(L, {}).get("solve_rate", 0.0) for L in Ls]
    s_rates = [scratchpad.get(L, {}).get("solve_rate", 0.0) for L in Ls]
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        [x - width / 2 for x in xs],
        b_rates,
        width=width,
        label="baseline",
        color="#888",
    )
    ax.bar(
        [x + width / 2 for x in xs],
        s_rates,
        width=width,
        label="scratchpad",
        color="#2a7",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("L (context-window cap, tokens)")
    ax.set_ylabel("solve rate")
    ax.set_ylim(0, 1)
    ax.set_title("Solve rate vs L — baseline vs scratchpad protocol")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    # Annotate n on each bar.
    for x, L in zip(xs, Ls):
        bn = baseline.get(L, {}).get("n", 0)
        sn = scratchpad.get(L, {}).get("n", 0)
        ax.text(x - width / 2, 0.02, f"n={bn}", ha="center", fontsize=8, color="white")
        ax.text(x + width / 2, 0.02, f"n={sn}", ha="center", fontsize=8, color="white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_tokens(
    baseline: dict[int, dict],
    scratchpad: dict[int, dict],
    out_path: Path,
) -> None:
    Ls = sorted(set(baseline) | set(scratchpad), key=sort_key_L)
    xs = list(range(len(Ls)))
    xlabels = [render_L(L) for L in Ls]
    b = [baseline.get(L, {}).get("median_tokens", 0) for L in Ls]
    s = [scratchpad.get(L, {}).get("median_tokens", 0) for L in Ls]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, b, "o-", label="baseline", color="#888")
    ax.plot(xs, s, "s-", label="scratchpad", color="#2a7")
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("L")
    ax.set_ylabel("median tokens per cell")
    ax.set_title("Cost per cell vs L")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_failure_modes(
    baseline: dict[int, dict],
    scratchpad: dict[int, dict],
    out_path: Path,
) -> None:
    Ls = sorted(set(baseline) | set(scratchpad), key=sort_key_L)
    # Collect every failure_reason that appears
    modes: set[str] = set()
    for d in (baseline, scratchpad):
        for L in Ls:
            modes |= d.get(L, {}).get("failure_modes", {}).keys()
    # Stable ordering: solved first, then named failures alphabetically
    modes_order = ["solved"] + sorted(m for m in modes if m != "solved")
    palette = {
        "solved": "#2a7",
        "max_turns_reached": "#c4a",
        "cost_cap": "#e90",
        "unscored": "#999",
        "scorer_failed": "#d22",
    }

    def color_for(m: str) -> str:
        return palette.get(m, "#39c")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, label, agg in (
        (axes[0], "baseline", baseline),
        (axes[1], "scratchpad", scratchpad),
    ):
        xs = list(range(len(Ls)))
        bottoms = [0.0] * len(Ls)
        for m in modes_order:
            vals = [agg.get(L, {}).get("failure_modes", {}).get(m, 0) for L in Ls]
            ax.bar(xs, vals, bottom=bottoms, label=m, color=color_for(m))
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_xticks(xs)
        ax.set_xticklabels([render_L(L) for L in Ls])
        ax.set_xlabel("L")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("# cells")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Failure-mode breakdown vs L")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_turn_tags(
    baseline_turns: dict[int, dict],
    scratchpad_turns: dict[int, dict],
    out_path: Path,
) -> None:
    Ls = sorted(set(baseline_turns) | set(scratchpad_turns), key=sort_key_L)
    if not Ls:
        return
    tags_of_interest = [
        ("re_cat_instructions", "re-cat instructions"),
        ("notes_md", "notes.md mentioned"),
        ("length_cap", "LENGTH-CAP"),
        ("zero_out_deep_think", "zero-out deep-think"),
        ("export", "export"),
        ("git_diff", "git diff"),
    ]
    fig, axes = plt.subplots(
        1, len(tags_of_interest), figsize=(3 * len(tags_of_interest), 3.5),
        sharey=True,
    )
    xs = list(range(len(Ls)))
    width = 0.36
    for ax, (tag, label) in zip(axes, tags_of_interest):
        b = [baseline_turns.get(L, {}).get("tag_rate", {}).get(tag, 0.0) for L in Ls]
        s = [scratchpad_turns.get(L, {}).get("tag_rate", {}).get(tag, 0.0) for L in Ls]
        ax.bar([x - width / 2 for x in xs], b, width=width, color="#888", label="baseline")
        ax.bar([x + width / 2 for x in xs], s, width=width, color="#2a7", label="scratchpad")
        ax.set_xticks(xs)
        ax.set_xticklabels([render_L(L) for L in Ls], rotation=30)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("fraction of turns")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Per-turn behavior tags by L and variant")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Tables / numbers
# --------------------------------------------------------------------------


def emit_tables(
    baseline: dict[int, dict],
    scratchpad: dict[int, dict],
    out_dir: Path,
) -> None:
    Ls = sorted(set(baseline) | set(scratchpad), key=sort_key_L)
    rows = []
    for L in Ls:
        b = baseline.get(L, {})
        s = scratchpad.get(L, {})
        rows.append(
            {
                "L": render_L(L),
                "baseline_solved": f"{b.get('solved',0)}/{b.get('n',0)}",
                "baseline_rate": f"{b.get('solve_rate',0):.0%}",
                "scratchpad_solved": f"{s.get('solved',0)}/{s.get('n',0)}",
                "scratchpad_rate": f"{s.get('solve_rate',0):.0%}",
                "delta_pp": f"{100*(s.get('solve_rate',0) - b.get('solve_rate',0)):+.0f}",
            }
        )

    # markdown
    md_lines = [
        "| L | baseline | rate | scratchpad | rate | Δ (pp) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['L']} | {r['baseline_solved']} | {r['baseline_rate']} | "
            f"{r['scratchpad_solved']} | {r['scratchpad_rate']} | {r['delta_pp']} |"
        )
    (out_dir / "results.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # latex
    tex_lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"$L$ & baseline & rate & scratchpad & rate & $\Delta$ (pp) \\",
        r"\midrule",
    ]
    for r in rows:
        tex_lines.append(
            f"{r['L']} & {r['baseline_solved']} & {r['baseline_rate']} & "
            f"{r['scratchpad_solved']} & {r['scratchpad_rate']} & {r['delta_pp']} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "results.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    # raw numbers for inline citation
    (out_dir / "results.json").write_text(
        json.dumps({"baseline": baseline, "scratchpad": scratchpad}, default=str, indent=2),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--scratchpad", type=Path, required=True)
    p.add_argument("--turns-baseline", type=Path, default=None)
    p.add_argument("--turns-scratchpad", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    b_cells = load_cells(args.baseline)
    s_cells = load_cells(args.scratchpad)
    print(f"baseline cells: {len(b_cells)}")
    print(f"scratchpad cells: {len(s_cells)}")

    b_agg = aggregate(b_cells)
    s_agg = aggregate(s_cells)

    plot_solve_rate(b_agg, s_agg, args.out / "fig1_solve_rate.png")
    plot_tokens(b_agg, s_agg, args.out / "fig2_tokens.png")
    plot_failure_modes(b_agg, s_agg, args.out / "fig3_failure_modes.png")
    emit_tables(b_agg, s_agg, args.out)

    if args.turns_baseline and args.turns_scratchpad:
        bt = aggregate_turns(load_turns(args.turns_baseline))
        st = aggregate_turns(load_turns(args.turns_scratchpad))
        plot_turn_tags(bt, st, args.out / "fig4_turn_tags.png")

    print(f"figures written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
