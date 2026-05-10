"""Phase 4 hard-constraint test: PIT S&P 500 membership.

Verifies that for every (asof, ticker) row in our composite panel:
  ticker IS a member of the S&P 500 at asof, per
  experiments/monthly_dca/cache/v2/sp500_pit/sp500_membership_monthly.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def test_pit_membership_holds():
    from experiments.multi_pillar_43Agh.strategy import selection
    panel = selection.build_composite_panel()
    panel["asof"] = pd.to_datetime(panel["asof"])
    mem = pd.read_parquet(ROOT / "experiments" / "monthly_dca" / "cache" /
                          "v2" / "sp500_pit" / "sp500_membership_monthly.parquet")
    mem["asof"] = pd.to_datetime(mem["asof"])
    mem["in_sp"] = True
    j = panel.merge(mem, on=["asof", "ticker"], how="left")
    n_violations = int(j["in_sp"].isna().sum())
    print(f"PIT membership test: panel rows = {len(panel)}, violations = {n_violations}")
    assert n_violations == 0, f"PIT violations: {n_violations}"
    print("✓ test_pit_membership PASSED")


def test_no_future_features_in_panel():
    """Spot check: panel asofs are all <= the cache's last available feature parquet."""
    from experiments.multi_pillar_43Agh.strategy import selection
    panel = selection.build_composite_panel()
    panel["asof"] = pd.to_datetime(panel["asof"])
    feat_dir = ROOT / "experiments" / "monthly_dca" / "cache" / "features"
    avail = sorted(pd.Timestamp(p.stem) for p in feat_dir.glob("*.parquet"))
    for ao in panel["asof"].unique():
        assert ao <= max(avail), f"panel asof {ao} > available features"
    print(f"✓ test_no_future_features_in_panel PASSED ({panel['asof'].nunique()} asofs)")


if __name__ == "__main__":
    test_pit_membership_holds()
    test_no_future_features_in_panel()
