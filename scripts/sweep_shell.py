"""Drive the Phase 1 L-sweep under the shell-harness architecture.

Mirrors `sweep_stateless.py` but uses `ShellCell` + `run_shell_cell`.
JSONL output schema is structurally compatible with the stateless-turn
output (same cell.provider / cell.benchmark / cell.L fields, same
solved/score/turns/tokens), so analyze.py works against either.

Two routing modes per benchmark:

  - `--use-swe-bench-cell` (default: auto)  Use the SWE-bench wrapper
    that bootstraps the repo into the agent filesystem and extracts the
    final patch. Auto-enabled when the benchmark name starts with
    "swe_bench"; can be forced on/off explicitly.

Optional S3 sync: set --s3-bucket or GOLDFISH_S3_BUCKET to enable. The
JSONL is uploaded every --s3-sync-every (default 10) completed cells.
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
from rlm_paged.shell import ShellCell, run_shell_cell, run_swe_bench_cell
from rlm_paged.utils.config import load_config
from rlm_paged.utils.s3_sync import S3Syncer


def _provider_spec_and_kwargs(entry: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, dict) and "spec" in entry:
        return entry["spec"], dict(entry.get("kwargs") or {})
    raise ValueError(f"unrecognized provider entry: {entry!r}")


def _cell_key(cell: ShellCell) -> str:
    return (
        f"{cell.provider}|shell|{cell.L}|{cell.benchmark}|"
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
    L_values = list(config["sweep"]["L_values"])
    include_native = config["sweep"].get("include_native_baseline", False)
    for entry in config["providers"]:
        spec, kwargs = _provider_spec_and_kwargs(entry)
        for bench_name, bench_kwargs in config["benchmarks"].items():
            for L in L_values:
                yield spec, kwargs, L, bench_name, bench_kwargs or {}
            if include_native:
                yield spec, kwargs, 0, bench_name, bench_kwargs or {}


def _is_swe_bench(bench_name: str, override: str) -> bool:
    if override == "yes":
        return True
    if override == "no":
        return False
    return bench_name.startswith("swe_bench")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sweep/phase1.yaml")
    parser.add_argument("--output", default="runs/phase1_shell.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument("--only-provider", default=None)
    parser.add_argument("--only-benchmark", default=None)
    parser.add_argument(
        "--keep-agent-dirs",
        action="store_true",
        help="Keep per-trajectory agent directories on disk for inspection.",
    )
    parser.add_argument(
        "--use-swe-bench-cell",
        choices=("auto", "yes", "no"),
        default="auto",
        help="Route SWE-bench trajectories through run_swe_bench_cell.",
    )
    parser.add_argument(
        "--repo-cache-dir",
        default=None,
        help="Local cache of cloned repos shared across SWE-bench tasks.",
    )
    parser.add_argument(
        "--turn-log",
        default=None,
        help=(
            "If set, write a per-turn transcript JSONL here. One row per "
            "model call with system_prompt + user_prompt + response + "
            "token counts. Lets downstream analysis see exactly what the "
            "model saw and wrote."
        ),
    )
    parser.add_argument(
        "--prompt-variant",
        choices=("baseline", "scratchpad"),
        default=None,
        help=(
            "Override the system-prompt variant for every cell in the "
            "sweep. 'scratchpad' adds the paged-memory protocol that "
            "mandates notes.md maintenance. CLI flag overrides any "
            "value the config might set."
        ),
    )
    parser.add_argument(
        "--scorer-diag-dir",
        default=None,
        help=(
            "Persist per-instance scorer stdout/stderr/exit_code/report "
            "into this directory (bug 14 visibility). Survives the "
            "scorer's tempdir cleanup so silent failures stay diagnosable."
        ),
    )
    # ---------- S3 sync flags
    parser.add_argument("--s3-bucket", default=None)
    parser.add_argument("--s3-prefix", default=None)
    parser.add_argument("--s3-sync-every", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = _read_done_keys(output_path)

    syncer = S3Syncer.from_env_or_args(
        bucket=args.s3_bucket,
        prefix=args.s3_prefix,
        sync_every=args.s3_sync_every,
    )
    if syncer.enabled:
        try:
            syncer.start()
            print(
                f"S3 sync enabled: s3://{syncer.bucket}/{syncer.prefix} "
                f"every {syncer.sync_every} cells",
                file=sys.stderr,
            )
        except RuntimeError as exc:
            print(
                f"warning: S3 sync requested but disabled: {exc}",
                file=sys.stderr,
            )
            syncer.enabled = False

    cells_planned = 0
    cells_done = 0
    cells_skipped = 0
    cells_failed = 0

    repo_cache = Path(args.repo_cache_dir) if args.repo_cache_dir else None

    # Per-turn transcript logger. Append-only JSONL, one row per turn.
    # Closure captures the file handle and serializes each row as a
    # standalone JSON line; the harness invokes this callback after each
    # API call. Best-effort: an exception in the callback never aborts
    # the trajectory.
    turn_log_fh = None
    turn_logger = None
    if args.turn_log:
        turn_log_path = Path(args.turn_log)
        turn_log_path.parent.mkdir(parents=True, exist_ok=True)
        turn_log_fh = turn_log_path.open("a", encoding="utf-8")

        def _turn_logger(row: dict) -> None:
            turn_log_fh.write(json.dumps(row, default=str) + "\n")
            turn_log_fh.flush()

        turn_logger = _turn_logger
        print(
            f"Per-turn transcripts: {turn_log_path}",
            file=sys.stderr,
        )

    for spec, prov_kwargs, L, bench_name, bench_kwargs in _expand(config):
        if args.only_provider and spec != args.only_provider:
            continue
        if args.only_benchmark and bench_name != args.only_benchmark:
            continue

        suite_kwargs = dict(bench_kwargs)
        if args.limit_tasks is not None:
            suite_kwargs["limit"] = args.limit_tasks
        # CLI scorer-diag-dir wins over per-suite config; bug-14 forensics.
        if args.scorer_diag_dir:
            suite_kwargs["scorer_diag_dir"] = args.scorer_diag_dir

        if args.dry_run:
            print(f"would run: {spec} | shell | L={L} | {bench_name}")
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

        client = build_client(spec, **prov_kwargs)
        use_swe_cell = _is_swe_bench(bench_name, args.use_swe_bench_cell)

        for task in tasks:
            cells_planned += 1
            # Prompt variant: CLI flag overrides config; config defaults
            # to baseline. This is the knob the paper sweep flips.
            prompt_variant = (
                args.prompt_variant
                or config.get("prompt_variant", "baseline")
            )
            cell = ShellCell(
                provider=spec,
                L=L,
                benchmark=bench_name,
                task_id=task.task_id,
                cost_cap_tokens=int(config.get("cost_cap_tokens", 100_000)),
                max_turns=int(config.get("max_turns", 16)),
                command_timeout_s=float(config.get("command_timeout_s", 10.0)),
                prompt_variant=prompt_variant,
            )
            key = _cell_key(cell)
            if key in done:
                cells_skipped += 1
                continue

            try:
                if use_swe_cell:
                    result = run_swe_bench_cell(
                        cell,
                        client=client,
                        suite=suite,
                        task=task,
                        repo_cache_dir=repo_cache,
                        keep_agent_dir=args.keep_agent_dirs,
                        turn_logger=turn_logger,
                    )
                else:
                    result = run_shell_cell(
                        cell,
                        client=client,
                        suite=suite,
                        task=task,
                        keep_agent_dir=args.keep_agent_dirs,
                        turn_logger=turn_logger,
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
                f"turns={result.turns} "
                f"cmds={result.commands_executed}"
            )

            sync_result = syncer.maybe_sync([output_path])
            if sync_result and sync_result.files_uploaded:
                print(
                    f"  s3-sync: {sync_result.files_uploaded} files, "
                    f"{sync_result.bytes_uploaded} bytes",
                    file=sys.stderr,
                )

    # Final upload before exit (covers anything since the last periodic sync).
    final = syncer.stop_and_final_sync([output_path]) if syncer.enabled else None
    if final and final.files_uploaded:
        print(
            f"final s3-sync: {final.files_uploaded} files, "
            f"{final.bytes_uploaded} bytes",
            file=sys.stderr,
        )

    if turn_log_fh is not None:
        turn_log_fh.close()

    print(
        f"\nshell sweep: planned={cells_planned} done={cells_done} "
        f"skipped={cells_skipped} failed={cells_failed}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
