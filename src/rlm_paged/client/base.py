from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str  # "stop" | "length" | "tool" | "error"


class LLMClient(ABC):
    """Provider-agnostic single-turn generation interface.

    The harness drives multi-turn behavior; the client is single-shot.
    The harness assembles (pinned prefix + active window) into one prompt
    string per call, so the client doesn't need to know about the window.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        stop: list[str] | None = None,
        temperature: float = 0.0,
    ) -> GenerationResult: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
