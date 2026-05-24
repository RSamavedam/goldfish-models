"""ARC-AGI loader (visual reasoning grids).

ARC-AGI is distributed as a JSON-per-task tree on GitHub. There's no
canonical single HF dataset that captures both the 2024 (ARC-AGI-1) and
2025 (ARC-AGI-2) splits cleanly. This loader reads from a directory of
JSON files matching the public release layout.

Phase 1 priority: GPQA + MATH + AIME + HLE first; ARC-AGI is plugged in
once we have local copies of the public training/eval sets.
"""

from __future__ import annotations

import json
from pathlib import Path

from rlm_paged.bench.base import FAMILY_TTC, BenchSuite, BenchTask


ARC_PROMPT_TEMPLATE = """You are given several input/output grid pairs that demonstrate a transformation rule. Apply the same rule to the test input.

Training examples:
{train_examples}

Test input:
{test_input}

Output the test output as a grid of integers, one row per line, space-separated.
End with: "Final answer:" followed by the grid."""


def _render_grid(grid: list[list[int]]) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def _render_pair(pair: dict) -> str:
    return f"Input:\n{_render_grid(pair['input'])}\nOutput:\n{_render_grid(pair['output'])}"


class ArcAgiSuite(BenchSuite):
    def __init__(self, *, root: str, limit: int | None = None) -> None:
        self.root = Path(root)
        self.limit = limit
        self._tasks: list[BenchTask] | None = None

    @property
    def family(self) -> str:
        return FAMILY_TTC

    @property
    def name(self) -> str:
        return "arc_agi"

    def tasks(self) -> list[BenchTask]:
        if self._tasks is not None:
            return self._tasks
        if not self.root.exists():
            raise FileNotFoundError(
                f"ARC-AGI data not found at {self.root}. Clone "
                "https://github.com/fchollet/ARC-AGI into a local dir and pass "
                "the data/evaluation path as `root`."
            )
        out: list[BenchTask] = []
        for path in sorted(self.root.glob("*.json")):
            data = json.loads(path.read_text())
            train_examples = "\n\n".join(_render_pair(p) for p in data["train"])
            for ti, test_pair in enumerate(data["test"]):
                payload = {
                    "train_examples": train_examples,
                    "test_input": _render_grid(test_pair["input"]),
                    "raw_train": data["train"],
                    "raw_test_input": test_pair["input"],
                }
                out.append(
                    BenchTask(
                        task_id=f"arc-{path.stem}-{ti}",
                        family=FAMILY_TTC,
                        payload=payload,
                        expected={"grid": test_pair["output"]},
                    )
                )
        if self.limit:
            out = out[: self.limit]
        self._tasks = out
        return out

    def task_prompt(self, task: BenchTask) -> str:
        return ARC_PROMPT_TEMPLATE.format(
            train_examples=task.payload["train_examples"],
            test_input=task.payload["test_input"],
        )

    def score(self, task: BenchTask, response: str) -> tuple[bool, float]:
        # Extract the grid following "Final answer:".
        idx = response.lower().rfind("final answer")
        if idx == -1:
            return False, 0.0
        tail = response[idx:].split(":", 1)[-1].strip()
        try:
            predicted = [
                [int(c) for c in row.split()]
                for row in tail.splitlines()
                if row.strip() and all(c.lstrip("-").isdigit() for c in row.split())
            ]
        except ValueError:
            return False, 0.0
        gold = task.expected["grid"]
        ok = predicted == gold
        return ok, 1.0 if ok else 0.0
