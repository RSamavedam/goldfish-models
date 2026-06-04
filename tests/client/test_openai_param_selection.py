"""Regression tests for `_looks_like_reasoning_model`.

The first cloud sweep failed 250/750 cells because gpt-5 was not
recognized as a reasoning-style model, so the client sent `max_tokens`
which the API rejects with a 400. These tests pin the matcher so we
don't lose gpt-5 (or future families) on a careless edit.
"""

from __future__ import annotations

from rlm_paged.client.openai import OpenAIClient


def _looks(model: str) -> bool:
    return OpenAIClient(model, api_key="test")._looks_like_reasoning_model()


def test_o_series_recognized_as_reasoning():
    assert _looks("o1")
    assert _looks("o1-mini")
    assert _looks("o3")
    assert _looks("o3-mini")
    assert _looks("o4-mini")


def test_gpt_5_recognized_as_reasoning():
    # This was the regression: gpt-5 needs max_completion_tokens.
    assert _looks("gpt-5")
    assert _looks("gpt-5-2026-01-15")
    assert _looks("gpt-5-mini")


def test_gpt_4_family_uses_max_tokens():
    # gpt-4o, gpt-4.1 etc. still use the old max_tokens parameter.
    assert not _looks("gpt-4o")
    assert not _looks("gpt-4o-mini")
    assert not _looks("gpt-4.1")
    assert not _looks("gpt-4-turbo")


def test_case_insensitive():
    assert _looks("GPT-5")
    assert _looks("O3-MINI")
