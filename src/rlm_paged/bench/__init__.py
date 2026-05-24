from rlm_paged.bench.base import (
    FAMILY_CODING,
    FAMILY_LONG_DOC,
    FAMILY_MEMORY_DIALOGUE,
    FAMILY_TTC,
    BenchSuite,
    BenchTask,
)

__all__ = [
    "BenchSuite",
    "BenchTask",
    "FAMILY_CODING",
    "FAMILY_LONG_DOC",
    "FAMILY_MEMORY_DIALOGUE",
    "FAMILY_TTC",
    "build_suite",
]


def build_suite(name: str, **kwargs) -> BenchSuite:
    """Factory for benchmark suites by short name."""
    if name == "gpqa_diamond":
        from rlm_paged.bench.gpqa import GpqaDiamondSuite
        return GpqaDiamondSuite(**kwargs)
    if name == "math500":
        from rlm_paged.bench.math500 import Math500Suite
        return Math500Suite(**kwargs)
    if name in {"aime_2024", "aime"}:
        from rlm_paged.bench.aime import AimeSuite
        return AimeSuite(**kwargs)
    if name == "aime_2025":
        from rlm_paged.bench.aime import AimeSuite
        return AimeSuite(
            dataset_id=kwargs.pop("dataset_id", "MathArena/aime_2025"),
            year_label="2025",
            **kwargs,
        )
    if name == "hle":
        from rlm_paged.bench.hle import HleSuite
        return HleSuite(**kwargs)
    if name == "arc_agi":
        from rlm_paged.bench.arc_agi import ArcAgiSuite
        return ArcAgiSuite(**kwargs)
    raise ValueError(f"unknown benchmark: {name!r}")
