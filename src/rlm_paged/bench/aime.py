"""AIME loader (30 problems per year, integer answers 0-999).

Defaults to AIME 2024 from Maxwell-Jia/AIME_2024. AIME 2025 can be loaded
via the `dataset_id` override once the canonical HF release stabilizes.
"""

from __future__ import annotations

from rlm_paged.bench.answer_extraction import (
    equal_numeric,
    extract_boxed,
    extract_final_answer_line,
)
from rlm_paged.bench.base import FAMILY_TTC, BenchSuite, BenchTask


AIME_PROMPT_TEMPLATE = """{question}

Solve the problem. The answer is an integer between 0 and 999.
Put your final answer in \\boxed{{...}}."""


class AimeSuite(BenchSuite):
    def __init__(
        self,
        *,
        dataset_id: str = "Maxwell-Jia/AIME_2024",
        split: str = "train",
        problem_col: str = "Problem",
        answer_col: str = "Answer",
        limit: int | None = None,
        year_label: str = "2024",
    ) -> None:
        self.dataset_id = dataset_id
        self.split = split
        self.problem_col = problem_col
        self.answer_col = answer_col
        self.limit = limit
        self.year_label = year_label
        self._tasks: list[BenchTask] | None = None

    @property
    def family(self) -> str:
        return FAMILY_TTC

    @property
    def name(self) -> str:
        return f"aime_{self.year_label}"

    def tasks(self) -> list[BenchTask]:
        if self._tasks is not None:
            return self._tasks
        from datasets import load_dataset  # type: ignore[import-not-found]

        ds = load_dataset(self.dataset_id, split=self.split)
        out: list[BenchTask] = []
        for idx, row in enumerate(ds):
            out.append(
                BenchTask(
                    task_id=f"aime{self.year_label}-{idx:02d}",
                    family=FAMILY_TTC,
                    payload={"question": row[self.problem_col]},
                    expected={"answer": str(row[self.answer_col])},
                )
            )
        if self.limit:
            out = out[: self.limit]
        self._tasks = out
        return out

    def task_prompt(self, task: BenchTask) -> str:
        return AIME_PROMPT_TEMPLATE.format(**task.payload)

    def score(self, task: BenchTask, response: str) -> tuple[bool, float]:
        predicted = extract_boxed(response) or extract_final_answer_line(response)
        if predicted is None:
            return False, 0.0
        ok = equal_numeric(predicted, task.expected["answer"])
        return ok, 1.0 if ok else 0.0
