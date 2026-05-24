"""GPQA Diamond loader (graduate-level science MC, 198 questions).

Canonical HF: Idavidrein/gpqa, config 'gpqa_diamond'. Some access requires
gated approval — the loader bubbles up the load error if so.
"""

from __future__ import annotations

import random
from typing import Any

from rlm_paged.bench.answer_extraction import extract_multiple_choice
from rlm_paged.bench.base import FAMILY_TTC, BenchSuite, BenchTask


GPQA_PROMPT_TEMPLATE = """{question}

(A) {choice_a}
(B) {choice_b}
(C) {choice_c}
(D) {choice_d}

Work through the problem step by step. End with: "Final answer: X" where X is one of A, B, C, D."""


class GpqaDiamondSuite(BenchSuite):
    def __init__(
        self,
        *,
        split: str = "train",  # the dataset only ships 'train'; we sample from it
        limit: int | None = None,
        seed: int = 0,
    ) -> None:
        self.split = split
        self.limit = limit
        self.seed = seed
        self._tasks: list[BenchTask] | None = None

    @property
    def family(self) -> str:
        return FAMILY_TTC

    @property
    def name(self) -> str:
        return "gpqa_diamond"

    def tasks(self) -> list[BenchTask]:
        if self._tasks is not None:
            return self._tasks
        from datasets import load_dataset  # type: ignore[import-not-found]

        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split=self.split)
        out: list[BenchTask] = []
        rng = random.Random(self.seed)
        for idx, row in enumerate(ds):
            question = row["Question"]
            correct = row["Correct Answer"]
            incorrect = [
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            choices = incorrect + [correct]
            rng.shuffle(choices)
            answer_letter = "ABCD"[choices.index(correct)]
            payload = {
                "question": question,
                "choice_a": choices[0],
                "choice_b": choices[1],
                "choice_c": choices[2],
                "choice_d": choices[3],
            }
            out.append(
                BenchTask(
                    task_id=f"gpqa-{idx:04d}",
                    family=FAMILY_TTC,
                    payload=payload,
                    expected={"answer": answer_letter, "correct_text": correct},
                )
            )
        if self.limit:
            out = out[: self.limit]
        self._tasks = out
        return out

    def task_prompt(self, task: BenchTask) -> str:
        return GPQA_PROMPT_TEMPLATE.format(**task.payload)

    def score(self, task: BenchTask, response: str) -> tuple[bool, float]:
        predicted = extract_multiple_choice(response)
        if predicted is None:
            return False, 0.0
        ok = predicted.upper() == task.expected["answer"]
        return ok, 1.0 if ok else 0.0
