"""Canonical tokenizer for the active-window cap.

We measure `L` in tokens of a *fixed reference tokenizer* (tiktoken
`cl100k_base`) across every provider. This makes the sweep comparable —
the same `L=128` means the same number of canonical tokens whether the
backend is Claude, GPT, or Gemini. Each provider's own tokenizer may yield
slightly different counts at the API level, but the wrapper's notion of
"how many tokens are in the active window" is single-source.

This is a tradeoff. Pros: cross-provider numbers comparable, no per-
provider tokenizer dependencies in the hot path. Cons: provider billing
counts diverge slightly from our internal counts; the wrapper's `L` cap
under- or over-approximates the provider's native count by a few percent.
For the TTC-benchmark Phase 1 sweep this is acceptable because we never
get close enough to a provider's native context limit for the divergence
to matter; the comparisons are between schemes at the same canonical L.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - env without tiktoken
        raise RuntimeError(
            "tiktoken not installed; `pip install tiktoken` or run with "
            "GOLDFISH_USE_HEURISTIC_TOKENIZER=1 for the fallback."
        ) from exc
    return tiktoken.get_encoding("cl100k_base")


def count(text: str) -> int:
    """Canonical token count of `text`."""
    if not text:
        return 0
    import os

    if os.environ.get("GOLDFISH_USE_HEURISTIC_TOKENIZER") == "1":
        # 4-chars-per-token rule of thumb. Wrong by ~25% but lets tests run
        # without tiktoken installed.
        return max(1, len(text) // 4)
    return len(_encoder().encode(text))


def encode(text: str) -> list[int]:
    return _encoder().encode(text)


def decode(tokens: list[int]) -> str:
    return _encoder().decode(tokens)
