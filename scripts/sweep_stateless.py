"""Drive the Phase 1 L-sweep under the stateless-turn architecture.

Same shape as scripts/sweep_phase1.py (the legacy paged-window sweep),
but uses StatelessCell + run_stateless_cell. The 'scheme' axis is gone —
there's only one architecture here, parameterized by L.

For ablating against legacy schemes (truncated, summarized, native), see
scripts/sweep_phase1.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from rlm_paged.bench import build_suite
from rlm_paged.client import build_client
from rlm_paged.harness import StatelessCell, run_stateless_cell
from rlm_paged.utils.config import load_config


def _provider_spec_and_kwargs(entry: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, dict) and "spec" in entry:
        return entry["spec"], dict(entry.get("kwargs") or {})
    raise ValueError(f"unrecognized provider entry: {entry!r}")


def _cell_key(cell: StatelessCell) -> str:
    return (
        f"{cell.provider}|stateless|{cell.L}|{cell.benchmark}|"
        f"{cell.task_id}|{cell.seed}"
    )


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


def _expand(
    config: dict,
) -> Iterable[tuple[str, dict[str, Any], int, str, dict]]:
    """Yield (spec, kwargs, L, benchmark, suite_kwargs)."""
    L_values = list(config["sweep"]["L_values"])
    include_native = config["sweep"].get("include_native_baseline", False)
    for entry in config["providers"]:
        spec, kwargs = _provider_spec_and_kwargs(entry)
        for bench_name, bench_kwargs in config["benchmarks"].items():
            for L in L_values:
                yield spec, kwargs, L, bench_name, bench_kwargs or {}
            if include_native:
                yield spec, kwargs, 0, bench_name, bench_kwargs or {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sweep/phase1.yaml")
    parser.add_argument("--output", default="runs/phase1_stateless.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument("--only-provider", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = _read_done_keys(output_path)

    cells_planned = 0
    cells_done = 0
    cells_skipped = 0
    cells_failed = 0

    for spec, prov_kwargs, L, bench_name, bench_kwargs in _expand(config):
        if args.only_provider and spec != args.only_provider:
            continue

        suite_kwargs = dict(bench_kwargs)
        if args.limit_tasks is not None:
            suite_kwargs["limit"] = args.limit_tasks

        if args.dry_run:
            print(f"would run: {spec} | stateless | L={L} | {bench_name}")
            cells_planned += 1
            continue

        try:
            suite = build_suite(bench_name, **suite_kwargs)
            tasks = suite.tasks()
        except Exception as exc:
            print(
                f"  ! suite {bench_name} failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue

        # Stateless architecture is provider-agnostic; we don't need
        # special variants like +interleaved. Just build a plain client.
        client = build_client(spec, **prov_kwargs)

        for task in tasks:
            cells_planned += 1
            cell = StatelessCell(
                provider=spec,
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
                result = run_stateless_cell(
                    cell, client=client, suite=suite, task=task
                )
            except Exception as exc:
                cells_failed += 1
                print(
                    f"  ! {key} -> {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
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
        f"\nstateless sweep: planned={cells_planned} done={cells_done} "
        f"skipped={cells_skipped} failed={cells_failed}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
