"""SWE-bench Verified loader. Stub — Phase 1 implementation."""

from __future__ import annotations

from rlm_paged.bench.base import BenchSuite, BenchTask


class SweBenchSuite(BenchSuite):
    def __init__(self, root: str, *, subset_size: int = 50) -> None:
        self.root = root
        self.subset_size = subset_size

    @property
    def family(self) -> str:
        return "coding"

    def tasks(self) -> list[BenchTask]:
        raise NotImplementedError

    def score(self, task: BenchTask, response: str) -> tuple[bool, float]:
        raise NotImplementedError
