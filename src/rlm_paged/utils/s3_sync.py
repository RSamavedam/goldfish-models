"""Periodic S3 sync helper for long-running sweeps.

Designed to be optional and zero-friction:
  - When no bucket is configured, all methods are no-ops.
  - When `boto3` isn't installed, the constructor raises a clear error.
  - One bucket/prefix combination per syncer; one syncer per process.

Usage in a sweep loop:

    syncer = S3Syncer.from_env_or_args(
        bucket=args.s3_bucket, prefix=args.s3_prefix,
        sync_every=args.s3_sync_every,
    )
    syncer.start()                       # no-op if disabled
    ...
    for cell in cells:
        result = run_cell(...)
        emit_jsonl_row(...)
        syncer.maybe_sync(["runs/phase1_shell.jsonl"])
    syncer.stop_and_final_sync(["runs/phase1_shell.jsonl"])

Files passed to sync are uploaded under
    s3://<bucket>/<prefix>/<path_relative_to_cwd>
which preserves the sweep's local layout.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class S3SyncResult:
    files_uploaded: int = 0
    bytes_uploaded: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class S3Syncer:
    bucket: Optional[str] = None
    prefix: str = ""
    sync_every: int = 10            # sync after this many maybe_sync() calls
    enabled: bool = False
    _calls_since_last_sync: int = 0
    _last_sync_at: float = 0.0
    _client: Any = None

    @classmethod
    def from_env_or_args(
        cls,
        *,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
        sync_every: Optional[int] = None,
    ) -> "S3Syncer":
        """Resolve config from explicit args, then env vars, then defaults."""
        bucket = bucket or os.environ.get("GOLDFISH_S3_BUCKET")
        prefix = (
            prefix
            if prefix is not None
            else os.environ.get("GOLDFISH_S3_PREFIX", "goldfish/")
        )
        sync_every = sync_every if sync_every is not None else int(
            os.environ.get("GOLDFISH_S3_SYNC_EVERY", "10")
        )
        if not bucket:
            return cls(enabled=False)
        return cls(
            bucket=bucket,
            prefix=prefix or "",
            sync_every=max(1, sync_every),
            enabled=True,
        )

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "S3 sync enabled but boto3 is not installed; "
                "`pip install boto3`"
            ) from exc
        self._client = boto3.client("s3")
        self._last_sync_at = time.time()

    def maybe_sync(self, paths: list[str | Path]) -> Optional[S3SyncResult]:
        """Increment the per-cell counter; sync iff we've hit `sync_every`."""
        if not self.enabled:
            return None
        self._calls_since_last_sync += 1
        if self._calls_since_last_sync < self.sync_every:
            return None
        return self.force_sync(paths)

    def force_sync(self, paths: list[str | Path]) -> S3SyncResult:
        """Upload each path immediately. Resets the cell counter."""
        result = S3SyncResult()
        if not self.enabled:
            return result
        for raw in paths:
            p = Path(raw)
            if not p.exists():
                continue
            if p.is_dir():
                for child in p.rglob("*"):
                    if child.is_file():
                        self._upload_one(child, result)
            else:
                self._upload_one(p, result)
        self._calls_since_last_sync = 0
        self._last_sync_at = time.time()
        return result

    def stop_and_final_sync(self, paths: list[str | Path]) -> S3SyncResult:
        return self.force_sync(paths)

    # ----------------------------------------------------- internals

    def _upload_one(self, p: Path, result: S3SyncResult) -> None:
        rel = self._object_key(p)
        try:
            self._client.upload_file(str(p), self.bucket, rel)
            result.files_uploaded += 1
            result.bytes_uploaded += p.stat().st_size
        except Exception as exc:
            result.errors.append(f"{p}: {type(exc).__name__}: {exc}")

    def _object_key(self, p: Path) -> str:
        # Relative to cwd; falls back to basename if outside.
        try:
            rel = p.resolve().relative_to(Path.cwd().resolve())
            key_path = str(rel)
        except ValueError:
            key_path = p.name
        prefix = self.prefix.rstrip("/")
        return f"{prefix}/{key_path}" if prefix else key_path
