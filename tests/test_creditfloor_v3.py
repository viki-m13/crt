"""Unit tests for CreditFloor v3 invariants.

Run:  python3 -m pytest tests/test_creditfloor_v3.py  (or python3 tests/test_creditfloor_v3.py)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "strategies", "credit_spread"))

from common import covered_options_expiry  # noqa: E402
from research import v3_cap, v3_published_buffer, HIST_CLEAR, K_SIGMA  # noqa: E402


def test_snap_down_never_exceeds_horizon():
    # The certified window must always cover the actual expiry.
    for d, h in [("2020-01-23", 21), ("2020-02-24", 21), ("2024-03-25", 21),
                 ("2026-06-11", 7), ("2026-06-11", 126), ("2026-06-11", 252),
                 ("2024-03-25", 5), ("2008-04-23", 7), ("2008-09-15", 21)]:
        snap = covered_options_expiry(d, h)
        assert snap is not None, (d, h)
        _exp, _kind, cal_days, sessions = snap
        assert 0 < sessions <= h, (d, h, snap)
        assert cal_days > 0, (d, h, snap)


def test_covid_protocol_bug_is_fixed():
    # Legacy snap-up assigned 2020-03-20 (57 cal days) to an h=21 signal
    # published 2020-01-23; snap-down must stay inside the certified window.
    exp, kind, cal_days, sessions = covered_options_expiry("2020-01-23", 21)
    assert exp == "2020-02-21" and kind == "monthly"
    assert sessions <= 21 and cal_days == 29


def test_v3_buffer_gates():
    # sigma=32% annualized, h=21: b = 2.5*(0.32/sqrt(252))*sqrt(21)+0.01 ≈ 24.1%
    sigma, h = 0.32, 21
    b = v3_published_buffer(sigma, hist_max=0.20, h=h)
    assert b is not None and 0.20 <= b <= v3_cap(h)
    # history clearance: a worst-ever move far above the sigma distance
    # must veto publication
    assert v3_published_buffer(sigma, hist_max=0.50, h=h) is None
    # cap: very high vol pushes b over the 25% short-horizon cap
    assert v3_published_buffer(1.20, hist_max=0.10, h=h) is None
    # degenerate inputs fail closed
    assert v3_published_buffer(None, hist_max=0.10, h=h) is None
    assert v3_published_buffer(0.0, hist_max=0.10, h=h) is None
    assert b >= HIST_CLEAR * 0.20 and K_SIGMA == 2.5


if __name__ == "__main__":
    test_snap_down_never_exceeds_horizon()
    test_covid_protocol_bug_is_fixed()
    test_v3_buffer_gates()
    print("all tests passed")
