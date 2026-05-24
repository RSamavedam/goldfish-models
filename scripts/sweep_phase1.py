"""Drive the Phase 1 TTC L-sweep across providers, schemes, benchmarks.

Iterates the Cartesian product of (provider, scheme, L, benchmark, task)
and writes one JSONL row per cell to the configured output path.

Resumability: skips cells whose `(cell_key)` already appears in the output
file. Safe to ctrl-c and re-run.
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
from rlm_paged.harness import SweepCell, build_scheme, run_cell
from rlm_paged.utils.config import load_config


def _provider_spec_and_kwargs(entry: Any) -> tuple[str, dict[str, Any]]:
    """A provider entry can be a bare string spec or a {spec, kwargs} mapping."""
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, dict) and "spec" in entry:
        return entry["spec"], dict(entry.get("kwargs") or {})
    raise ValueError(f"unrecognized provider entry: {entry!r}")


def _build_client_for_cell(spec: str, kwargs: dict[str, Any], L: int) -> Any:
    """Resolve sweep-time substitutions (e.g. thinking_budget_to_L) then build."""
    resolved = dict(kwargs)
    if resolved.pop("thinking_budget_to_L", False):
        # L=0 means "no cap" → fall back to a reasonable default thinking budget.
        resolved["thinking_budget"] = max(1024, L) if L > 0 else 4096
    if resolved.get("interleaved_thinking"):
        from rlm_paged.tools import ANTHROPIC_TOOLS

        resolved.setdefault("tools", ANTHROPIC_TOOLS)
    return build_client(spec, **resolved)


def _provider_label(spec: str, kwargs: dict[str, Any]) -> str:
    """Stable label including interleaved-thinking variant so cell_keys differ."""
    if kwargs.get("interleaved_thinking"):
        return f"{spec}+interleaved"
    return spec


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


def _expand(
    config: dict,
) -> Iterable[tuple[str, dict[str, Any], str, int, str, dict]]:
    """Yield (spec, provider_kwargs, scheme, L, benchmark_name, suite_kwargs)."""
    L_values = list(config["sweep"]["L_values"])
    include_native = config["sweep"].get("include_native_baseline", False)
    benches = config["benchmarks"]
    schemes = config["schemes"]
    for entry in config["providers"]:
        spec, kwargs = _provider_spec_and_kwargs(entry)
        for bench_name, bench_kwargs in benches.items():
            for scheme in schemes:
                for L in L_values:
                    yield spec, kwargs, scheme, L, bench_name, bench_kwargs or {}
            if include_native:
                yield spec, kwargs, "native", 0, bench_name, bench_kwargs or {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sweep/phase1.yaml")
    parser.add_argument("--output", default="runs/phase1.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-tasks", type=int, default=None,
                        help="Cap tasks per benchmark (for smoke tests).")
    parser.add_argument("--only-provider", default=None,
                        help="Run only one provider label (e.g. 'anthropic:claude-opus-4-7+interleaved').")
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = _read_done_keys(output_path)

    cells_planned = 0
    cells_done = 0
    cells_skipped = 0
    cells_failed = 0

    for spec, prov_kwargs, scheme_name, L, bench_name, bench_kwargs in _expand(config):
        label = _provider_label(spec, prov_kwargs)
        if args.only_provider and label != args.only_provider:
            continue

        suite_kwargs = dict(bench_kwargs)
        if args.limit_tasks is not None:
            suite_kwargs["limit"] = args.limit_tasks

        if args.dry_run:
            print(f"would run cells: {label} | {scheme_name} | L={L} | {bench_name}")
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
        client = _build_client_for_cell(spec, prov_kwargs, L)
        # Summarizer always uses the plain (non-interleaved) client variant
        # since interleaved+tools doesn't apply to a summarization role.
        summarizer = build_client(spec) if scheme_name == "summarized" else None

        for task in tasks:
            cells_planned += 1
            cell = SweepCell(
                provider=label,
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
