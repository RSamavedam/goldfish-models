"""MATH-500 loader (curated 500-problem subset of the MATH benchmark).

Canonical HF: HuggingFaceH4/MATH-500. Each row has a `problem`, `solution`,
and `answer` (already extracted from the boxed form). The model is asked
to produce a boxed final answer; we compare via `equal_numeric` because
the answer space is mostly numeric / simple symbolic.
"""

from __future__ import annotations

from rlm_paged.bench.answer_extraction import (
    equal_numeric,
    extract_boxed,
    extract_final_answer_line,
    normalize_numeric,
)
from rlm_paged.bench.base import FAMILY_TTC, BenchSuite, BenchTask


MATH_PROMPT_TEMPLATE = """{question}

Solve the problem. Put your final answer in \\boxed{{...}}."""


class Math500Suite(BenchSuite):
    def __init__(self, *, split: str = "test", limit: int | None = None) -> None:
        self.split = split
        self.limit = limit
        self._tasks: list[BenchTask] | None = None

    @property
    def family(self) -> str:
        return FAMILY_TTC

    @property
    def name(self) -> str:
        return "math500"

    def tasks(self) -> list[BenchTask]:
        if self._tasks is not None:
            return self._tasks
        from datasets import load_dataset  # type: ignore[import-not-found]

        ds = load_dataset("HuggingFaceH4/MATH-500", split=self.split)
        out: list[BenchTask] = []
        for idx, row in enumerate(ds):
            out.append(
                BenchTask(
                    task_id=f"math500-{idx:04d}",
                    family=FAMILY_TTC,
                    payload={"question": row["problem"]},
                    expected={"answer": row["answer"], "level": row.get("level")},
                )
            )
        if self.limit:
            out = out[: self.limit]
        self._tasks = out
        return out

    def task_prompt(self, task: BenchTask) -> str:
        return MATH_PROMPT_TEMPLATE.format(**task.payload)

    def score(self, task: BenchTask, response: str) -> tuple[bool, float]:
        predicted = extract_boxed(response) or extract_final_answer_line(response)
        gold = task.expected["answer"]
        if predicted is None:
            return False, 0.0
        ok = equal_numeric(predicted, gold) or (
            normalize_numeric(predicted) == normalize_numeric(gold)
        )
        if not ok:
            # Last-ditch: stripped string equality (handles symbolic answers
            # like 'x^2 + 1' that don't reduce to a number).
            ok = predicted.strip().rstrip(".") == gold.strip().rstrip(".")
        return ok, 1.0 if ok else 0.0
