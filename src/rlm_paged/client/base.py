from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class GenerationResult:
    """One model turn's output.

    `text` is the visible content the harness operates on (parses ops out of,
    accumulates into the tail). `thinking_text` is provider-side reasoning
    that the model emitted but the wrapper cannot see/manipulate token-by-token
    (Claude extended thinking blocks; OpenAI o-series reasoning is opaque so
    only `thinking_tokens` is set there with `thinking_text=""`).
    """

    text: str
    input_tokens: int
    output_tokens: int           # excludes thinking
    finish_reason: str           # "stop" | "length" | "tool" | "error"
    thinking_text: str = ""
    thinking_tokens: int = 0
    raw: dict = field(default_factory=dict)


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
        system: str | None = None,
    ) -> GenerationResult: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def supports_visible_thinking(self) -> bool:
        """True iff the provider exposes thinking content the harness can read."""
        return False

    @property
    def supports_interleaved_thinking(self) -> bool:
        """True iff the client dispatches native tool calls mid-thinking.

        Clients that return True must also implement `generate_with_dispatcher`
        — the harness calls that instead of `generate` to preserve signed
        thinking blocks across tool calls within a single logical turn.
        """
        return False
