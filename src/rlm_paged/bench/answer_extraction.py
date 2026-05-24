"""Answer-extraction helpers for free-form responses on TTC benchmarks.

These are stricter than the prompt-side `\\boxed{}` convention; we'll accept
several common patterns. The model is *told* to put its final answer in a
specific format, but real outputs are messy.
"""

from __future__ import annotations

import re
from fractions import Fraction


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}")
_FINAL_ANS_RE = re.compile(
    r"(?:final\s+answer|answer\s*(?:is|:))\s*[:=]?\s*([^\n.]{1,200})",
    re.IGNORECASE,
)
_MC_LETTER_RE = re.compile(r"\b([A-D])\b")


def extract_boxed(text: str) -> str | None:
    """Return the contents of the *last* \\boxed{...} in `text`."""
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def extract_final_answer_line(text: str) -> str | None:
    """Return the value after the last 'Final answer:' / 'Answer is' phrase."""
    matches = _FINAL_ANS_RE.findall(text)
    return matches[-1].strip().rstrip(".") if matches else None


def extract_multiple_choice(text: str, choices: str = "ABCD") -> str | None:
    """Return the last standalone A/B/C/D mentioned in `text`."""
    # Prefer an explicit "answer is X" pattern first.
    final = extract_final_answer_line(text)
    if final:
        m = re.search(rf"\b([{choices}])\b", final)
        if m:
            return m.group(1)
    matches = _MC_LETTER_RE.findall(text)
    return matches[-1] if matches else None


def normalize_numeric(s: str) -> str | None:
    """Reduce a numeric-ish string to a canonical form for equality checks.

    Accepts: '042', '42.0', '42', '\\frac{1}{2}', '1/2'. Returns Fraction str
    or the trimmed original if not numeric.
    """
    s = s.strip().rstrip(".").replace(",", "").replace(" ", "")
    if not s:
        return None
    # \frac{a}{b}
    frac_m = re.match(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if frac_m:
        try:
            return str(Fraction(int(frac_m.group(1)), int(frac_m.group(2))))
        except (ValueError, ZeroDivisionError):
            return s
    # a/b
    if "/" in s and all(p.lstrip("-").isdigit() for p in s.split("/")):
        try:
            num, den = s.split("/")
            return str(Fraction(int(num), int(den)))
        except (ValueError, ZeroDivisionError):
            return s
    # plain int / float — route through Fraction so "0.5" and "1/2" unify
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(Fraction(f).limit_denominator(10**6))
    except (ValueError, OverflowError):
        return s


def equal_numeric(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    return normalize_numeric(a) == normalize_numeric(b)
