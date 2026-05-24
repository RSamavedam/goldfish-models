from __future__ import annotations

from rlm_paged.bench.answer_extraction import (
    equal_numeric,
    extract_boxed,
    extract_final_answer_line,
    extract_multiple_choice,
    normalize_numeric,
)


def test_extract_boxed_last_wins():
    text = r"... after computing, \boxed{12}. Wait, recompute. \boxed{42}"
    assert extract_boxed(text) == "42"


def test_extract_boxed_handles_nested_braces():
    text = r"final \boxed{\frac{1}{2}}"
    assert extract_boxed(text) == r"\frac{1}{2}"


def test_extract_boxed_none_when_missing():
    assert extract_boxed("no boxed here, sorry") is None


def test_extract_final_answer_line():
    text = "...lots of work...\nFinal answer: 137"
    assert extract_final_answer_line(text) == "137"


def test_extract_final_answer_handles_answer_is():
    text = "After much thought, the answer is C."
    assert extract_final_answer_line(text) == "C"


def test_extract_multiple_choice_from_explicit_phrase():
    text = "Working through... So the answer is B."
    assert extract_multiple_choice(text) == "B"


def test_extract_multiple_choice_falls_back_to_last_letter():
    text = "Could be A or D... eventually D wins"
    assert extract_multiple_choice(text) == "D"


def test_normalize_numeric_handles_fractions():
    assert normalize_numeric(r"\frac{1}{2}") == "1/2"
    assert normalize_numeric("2/4") == "1/2"


def test_normalize_numeric_strips_trailing_period():
    assert normalize_numeric("42.") == "42"


def test_equal_numeric_handles_form_variation():
    assert equal_numeric("042", "42")
    assert equal_numeric("0.5", "1/2")
    assert not equal_numeric("42", "43")
