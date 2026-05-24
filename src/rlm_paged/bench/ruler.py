"""RULER NIAH-family loader. Stub — Phase 1 implementation."""

from __future__ import annotations

from rlm_paged.bench.base import BenchSuite, BenchTask


class RulerSuite(BenchSuite):
    def __init__(self, root: str) -> None:
        self.root = root

    @property
    def family(self) -> str:
        return "long_doc"

    def tasks(self) -> list[BenchTask]:
        raise NotImplementedError

    def score(self, task: BenchTask, response: str) -> tuple[bool, float]:
        raise NotImplementedError
