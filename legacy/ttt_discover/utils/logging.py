from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, backend: str = "jsonl", path: str = "runs/default.jsonl", **kwargs: Any) -> None:
        self.backend = backend
        self.path = Path(path)
        self.kwargs = kwargs
        if backend == "jsonl":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        elif backend != "wandb":
            raise ValueError(f"Unsupported logging backend: {backend}")

    def log(self, event: dict[str, Any]) -> None:
        if self.backend == "jsonl":
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            return
        import wandb

        wandb.log(event)
