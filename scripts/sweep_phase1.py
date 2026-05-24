"""Drive the Phase 1 TTC L-sweep across providers, schemes, benchmarks.

Iterates the Cartesian product of (provider, scheme, L, benchmark, task)
and writes one JSONL row per cell to the configured output path.

Resumability: skips cells whose `(cell_key)` already appears in the output
file. Safe to ctrl-c and re-run.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from rlm_paged.bench import build_suite
from rlm_paged.client import build_client
from rlm_paged.harness import SweepCell, build_scheme, run_cell
from rlm_paged.utils.config import load_config


def _cell_key(cell: SweepCell) -> str:
    return f"{cell.provider}|{cell.scheme}|{cell.L}|{cell.benchmark}|{cell.task_id}|{cell.seed}"


def _read_done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                done.add(row.get("cell_key", ""))
            except json.JSONDecodeError:
                continue
    done.discard("")
    return done


def _expand(config: dict) -> Iterable[tuple[str, str, int, str, dict]]:
    """Yield (provider_spec, scheme, L, benchmark_name, suite_kwargs) tuples."""
    providers = config["providers"]
    schemes = config["schemes"]
    L_values = list(config["sweep"]["L_values"])
    if config["sweep"].get("include_native_baseline", False):
        # Native is encoded as L=0 + scheme=native; only emit once per provider/bench.
        pass
    benches = config["benchmarks"]
    for prov in providers:
        for bench_name, bench_kwargs in benches.items():
            for scheme in schemes:
                for L in L_values:
                    yield prov, scheme, L, bench_name, bench_kwargs or {}
            # Native baseline once per (provider, bench), L=0.
            if config["sweep"].get("include_native_baseline", False):
                yield prov, "native", 0, bench_name, bench_kwargs or {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sweep/phase1.yaml")
    parser.add_argument("--output", default="runs/phase1.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-tasks", type=int, default=None,
                        help="Cap tasks per benchmark (for smoke tests).")
    parser.add_argument("--only-provider", default=None,
                        help="Run only one provider spec (for debugging).")
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = _read_done_keys(output_path)

    cells_planned = 0
    cells_done = 0
    cells_skipped = 0
    cells_failed = 0

    for prov, scheme_name, L, bench_name, bench_kwargs in _expand(config):
        if args.only_provider and prov != args.only_provider:
            continue

        suite_kwargs = dict(bench_kwargs)
        if args.limit_tasks is not None:
            suite_kwargs["limit"] = args.limit_tasks

        if args.dry_run:
            # In dry-run, print one line per (prov, scheme, L, bench) and skip
            # task-level expansion so we don't need network/datasets installed.
            print(f"would run cells: {prov} | {scheme_name} | L={L} | {bench_name}")
            cells_planned += 1
            continue

        try:
            suite = build_suite(bench_name, **suite_kwargs)
        except Exception as exc:
            print(f"  ! suite {bench_name} build failed: {exc}", file=sys.stderr)
            continue

        try:
            tasks = suite.tasks()
        except Exception as exc:
            print(f"  ! suite {bench_name} task load failed: {exc}", file=sys.stderr)
            continue

        scheme = build_scheme(scheme_name)
        client = build_client(prov)
        summarizer = build_client(prov) if scheme_name == "summarized" else None

        for task in tasks:
            cells_planned += 1
            cell = SweepCell(
                provider=prov,
                scheme=scheme_name,
                L=L,
                benchmark=bench_name,
                task_id=task.task_id,
                cost_cap_tokens=int(config.get("cost_cap_tokens", 100_000)),
            )
            key = _cell_key(cell)
            if key in done:
                cells_skipped += 1
                continue

            try:
                result = run_cell(
                    cell,
                    client=client,
                    suite=suite,
                    task=task,
                    scheme=scheme,
                    summarizer=summarizer,
                )
            except Exception as exc:
                cells_failed += 1
                print(f"  ! {key} -> {type(exc).__name__}: {exc}", file=sys.stderr)
                continue

            row: dict[str, Any] = dataclasses.asdict(result)
            row["cell_key"] = key
            row["ts"] = time.time()
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
            cells_done += 1
            print(
                f"  {key} solved={result.solved} "
                f"tokens={result.input_tokens + result.output_tokens} "
                f"turns={result.turns}"
            )

    print(
        f"\nsweep: planned={cells_planned} done={cells_done} "
        f"skipped={cells_skipped} failed={cells_failed}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
