from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Generation:
    text: str
    logprob: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingExample:
    prompt: str
    completion: str
    reward: float
    logprob: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingBatch:
    examples: list[TrainingExample]

    def __len__(self) -> int:
        return len(self.examples)


class Policy:
    """Thin policy abstraction with stubbed backends.

    The class is shaped like a trainable LLM policy, but defaults to deterministic
    template completions so the repository works without model weights.
    """

    SUPPORTED_BACKENDS = {"vllm", "huggingface", "api"}

    def __init__(self, backend: str = "huggingface", model_name: str | None = None, **kwargs: Any) -> None:
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported policy backend: {backend}")
        self.backend = backend
        self.model_name = model_name or "stub-policy"
        self.kwargs = kwargs
        self._updates = 0

    def generate(self, prompts: list[str], temperature: float = 1.0, n: int = 1) -> list[Generation]:
        generations: list[Generation] = []
        for prompt in prompts:
            for i in range(n):
                generations.append(
                    Generation(
                        text=self._stub_completion(prompt, i),
                        logprob=-1.0,
                        metadata={"backend": self.backend, "temperature": temperature},
                    )
                )
        return generations

    def update(self, batch: TrainingBatch) -> dict[str, float]:
        self._updates += 1
        return {"policy_updates": float(self._updates), "batch_size": float(len(batch))}

    def save_checkpoint(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "policy_stub.txt").write_text(
            f"backend={self.backend}\nmodel={self.model_name}\nupdates={self._updates}\n",
            encoding="utf-8",
        )

    def load_checkpoint(self, path: str) -> None:
        marker = Path(path) / "policy_stub.txt"
        if marker.exists():
            for line in marker.read_text(encoding="utf-8").splitlines():
                if line.startswith("updates="):
                    self._updates = int(line.split("=", 1)[1])

    def _stub_completion(self, prompt: str, index: int) -> str:
        if "sort_values" in prompt or "sorting" in prompt.lower():
            return "def sort_values(xs):\n    return sorted(xs)\n"
        return f"# candidate {index}\n# TODO: replace stub completion with model output\n"
