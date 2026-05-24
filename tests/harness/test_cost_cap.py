from __future__ import annotations

import pytest

from rlm_paged.harness import CostCap, CostCapExceeded


def test_charge_accumulates():
    cap = CostCap(max_tokens=1000)
    cap.charge(300)
    cap.charge(400)
    assert cap.spent == 700
    assert cap.remaining == 300


def test_charge_overshoot_raises():
    cap = CostCap(max_tokens=100)
    cap.charge(60)
    with pytest.raises(CostCapExceeded):
        cap.charge(50)
