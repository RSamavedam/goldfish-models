from __future__ import annotations

import pytest

from rlm_paged.window import ActiveWindow, WindowConfig, WindowViolation


def test_append_tail_within_budget():
    win = ActiveWindow(WindowConfig(L=32, tail_max=16))
    win.append_tail(8)
    assert win.tail == 8
    assert win.middle == 0
    assert win.free == 24


def test_append_tail_violates_tail_cap():
    win = ActiveWindow(WindowConfig(L=32, tail_max=10))
    with pytest.raises(WindowViolation):
        win.append_tail(11)


def test_freeze_tail_promotes_into_middle():
    win = ActiveWindow(WindowConfig(L=64, tail_max=32))
    win.append_tail(16)
    win.freeze_tail_into_middle()
    assert win.middle == 16
    assert win.tail == 0
    assert win.used == 16


def test_evict_head_only_touches_middle():
    win = ActiveWindow(WindowConfig(L=64, tail_max=32))
    win.append_tail(20)
    win.freeze_tail_into_middle()
    win.append_tail(8)
    assert win.middle == 20 and win.tail == 8
    freed = win.evict_head(12)
    assert freed == 12
    assert win.middle == 8
    assert win.tail == 8  # tail untouched


def test_evict_more_than_middle_is_clamped():
    win = ActiveWindow(WindowConfig(L=64, tail_max=32))
    win.append_tail(10)
    win.freeze_tail_into_middle()
    freed = win.evict_head(999)
    assert freed == 10
    assert win.middle == 0


def test_default_tail_max_is_half_L():
    win = ActiveWindow(WindowConfig(L=128))
    assert win.config.resolved_tail_max() == 64


def test_can_fit_respects_both_caps():
    win = ActiveWindow(WindowConfig(L=32, tail_max=16))
    win.append_tail(10)
    win.freeze_tail_into_middle()
    assert win.can_fit(20)        # 10 + 20 <= 32
    assert not win.can_fit(23)    # 10 + 23 > 32
