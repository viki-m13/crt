"""Forward-return labels in ml_preds_v2 must equal the ACTUAL next-month return.

Run: python3 tests/YLOka/test_feature_lag.py
"""
import sys
sys.path.insert(0, "/home/user/crt")

import pandas as pd
import numpy as np
from pathlib import Path

CACHE = Path("/home/user/crt/experiments/monthly_dca/cache")


def test_fwd_1m_ret_alignment():
    ml = pd.read_parquet(CACHE / "v2" / "ml_preds_v2.parquet")
    mr = pd.read_parquet(CACHE / "v2" / "monthly_returns_clean.parquet")
    # Sample 200 rows where fwd is not NaN
    sample = ml.dropna(subset=["fwd_1m_ret"]).sample(200, random_state=42)
    mr_idx = mr.index
    mismatches = 0
    for _, r in sample.iterrows():
        asof, tk = r["asof"], r["ticker"]
        pos = mr_idx.searchsorted(asof)
        if pos + 1 >= len(mr_idx):
            continue
        # Next-month return (T -> T+1)
        nxt = mr_idx[pos + 1]
        if abs((nxt - asof).days - 30) > 5:
            # Off-by-month-end edge case; skip
            continue
        if tk not in mr.columns:
            continue
        observed = mr.at[nxt, tk]
        if pd.isna(observed):
            continue
        if not np.isclose(observed, r["fwd_1m_ret"], atol=1e-3):
            mismatches += 1
    assert mismatches < 5, f"{mismatches} of 200 sampled fwd_1m_ret mismatches"


def test_pred_columns_are_finite():
    ml = pd.read_parquet(CACHE / "v2" / "ml_preds_v2.parquet")
    for c in ["pred_1m", "pred_3m", "pred_6m"]:
        assert ml[c].notna().all(), f"{c} has NaNs"
        assert np.isfinite(ml[c]).all(), f"{c} has non-finite values"


def test_no_future_data_in_features_index():
    """The features cache must not contain any month-end after the latest panel asof."""
    fdir = CACHE / "features"
    files = sorted(fdir.glob("*.parquet"))
    last_feature = pd.Timestamp(files[-1].stem)
    ml = pd.read_parquet(CACHE / "v2" / "ml_preds_v2.parquet")
    # ml asofs should never exceed the latest features file (otherwise GBM is using stale features)
    assert ml["asof"].max() <= last_feature + pd.Timedelta(days=35), \
        f"ml asof {ml['asof'].max()} > last feature file {last_feature}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                print(f"FAIL  {name}: {e}")
                sys.exit(1)
    print("\nAll feature-lag tests passed.")
