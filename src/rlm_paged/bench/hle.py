"""HLE (Humanity's Last Exam) loader, text-only subset.

Canonical HF: cais/hle. The full set is multimodal; Phase 1 filters to
text-only rows. HLE scoring uses an LLM-as-judge per the official rubric;
Phase 1's score here is the strict-match baseline (extracted answer ==
gold answer), which under-counts correct but flexibly-phrased responses.
Plug in the LLM judge once Phase 1 numbers stabilize.
"""

from __future__ import annotations

from rlm_paged.bench.answer_extraction import (
    extract_boxed,
    extract_final_answer_line,
    normalize_numeric,
)
from rlm_paged.bench.base import FAMILY_TTC, BenchSuite, BenchTask


HLE_PROMPT_TEMPLATE = """{question}

Provide your final answer in the format: "Final answer: ..."."""


class HleSuite(BenchSuite):
    def __init__(
        self,
        *,
        split: str = "test",
        limit: int | None = None,
        text_only: bool = True,
    ) -> None:
        self.split = split
        self.limit = limit
        self.text_only = text_only
        self._tasks: list[BenchTask] | None = None

    @property
    def family(self) -> str:
        return FAMILY_TTC

    @property
    def name(self) -> str:
        return "hle"

    def tasks(self) -> list[BenchTask]:
        if self._tasks is not None:
            return self._tasks
        from datasets import load_dataset  # type: ignore[import-not-found]

        ds = load_dataset("cais/hle", split=self.split)
        out: list[BenchTask] = []
        for idx, row in enumerate(ds):
            if self.text_only:
                # Skip rows with image content. Field names: "image" or "image_url".
                if row.get("image") or row.get("image_url"):
                    continue
            out.append(
                BenchTask(
                    task_id=f"hle-{idx:05d}",
                    family=FAMILY_TTC,
                    payload={
                        "question": row["question"],
                        "category": row.get("category", ""),
                    },
                    expected={"answer": row["answer"]},
                )
            )
        if self.limit:
            out = out[: self.limit]
        self._tasks = out
        return out

    def task_prompt(self, task: BenchTask) -> str:
        return HLE_PROMPT_TEMPLATE.format(**task.payload)

    def score(self, task: BenchTask, response: str) -> tuple[bool, float]:
        predicted = extract_final_answer_line(response) or extract_boxed(response)
        gold = task.expected["answer"]
        if predicted is None:
            return False, 0.0
        # Try numeric match, then case-insensitive substring containment as a
        # lax fallback. Real HLE scoring uses an LLM judge; this is the
        # placeholder strict baseline.
        if normalize_numeric(predicted) == normalize_numeric(gold):
            return True, 1.0
        if predicted.strip().lower() == gold.strip().lower():
            return True, 1.0
        return False, 0.0
