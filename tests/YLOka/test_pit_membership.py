"""PIT S&P 500 membership invariants. Run: python3 tests/YLOka/test_pit_membership.py"""
import sys
sys.path.insert(0, "/home/user/crt")

import pandas as pd
from pathlib import Path

PIT = Path("/home/user/crt/experiments/monthly_dca/cache/v2/sp500_pit/sp500_membership_monthly.parquet")


def test_pit_panel_shape():
    pit = pd.read_parquet(PIT)
    assert pit.shape[1] == 2 and set(pit.columns) == {"asof", "ticker"}, pit.columns


def test_member_counts():
    pit = pd.read_parquet(PIT)
    counts = pit.groupby("asof").size()
    assert 480 <= counts.min() <= 510, f"min members per month {counts.min()}"
    assert 480 <= counts.max() <= 510, f"max members per month {counts.max()}"
    assert 495 <= counts.mean() <= 505, f"mean members per month {counts.mean()}"


def test_date_coverage():
    pit = pd.read_parquet(PIT)
    asofs = sorted(pit["asof"].unique())
    first, last = asofs[0], asofs[-1]
    assert pd.Timestamp("2003-01-01") <= first <= pd.Timestamp("2003-12-31"), first
    assert last >= pd.Timestamp("2026-01-01"), last


def test_unique_tickers_reasonable():
    pit = pd.read_parquet(PIT)
    n_unique = pit["ticker"].nunique()
    assert 900 <= n_unique <= 1100, f"expected ~985 unique tickers; got {n_unique}"


def test_known_historical_member_present():
    """LEHMAN should appear in pre-Sep-2008 membership; absent after."""
    pit = pd.read_parquet(PIT)
    pre = pit[(pit["asof"] < pd.Timestamp("2008-09-01"))]["ticker"].unique()
    post = pit[(pit["asof"] > pd.Timestamp("2009-12-31"))]["ticker"].unique()
    # MER (Merrill Lynch) was acquired by BoA in Sep 2008
    assert "MER" in pre, "MER should be in pre-2008 PIT membership"
    assert "MER" not in post, "MER should NOT be in post-2009 PIT membership"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                print(f"FAIL  {name}: {e}")
                sys.exit(1)
    print("\nAll PIT membership tests passed.")
