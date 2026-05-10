"""Test that novel features at asof T do not depend on data after T.

Compute novel features at T using full panel, then again using
panel truncated to T.  They should be identical.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_engine import load_panel
from strategy.features.novel_features import compute_all_novel


def test_no_lookahead_at_2018_12_31():
    panel = load_panel()
    asof = pd.Timestamp("2018-12-31")
    # Method A: full panel, asof=2018-12-31
    feat_A = compute_all_novel(panel, asof)
    # Method B: panel truncated at 2018-12-31, asof=2018-12-31
    panel_trunc = panel.loc[panel.index <= asof]
    feat_B = compute_all_novel(panel_trunc, asof)

    # Same set of columns
    common = [c for c in feat_A.columns if c in feat_B.columns]
    assert len(common) >= 8, f"Expected at least 8 common cols, got {common}"

    # For each numeric column, the values for tickers that appear in both
    # should match (within float tol).
    differing = []
    for c in common:
        a = feat_A[c].astype(float)
        b = feat_B[c].astype(float)
        idx = a.index.intersection(b.index)
        a = a.loc[idx]
        b = b.loc[idx]
        # Both NaN counts as match
        ok = ((a.isna() & b.isna()) | (np.isclose(a.fillna(0), b.fillna(0), equal_nan=False, rtol=1e-6, atol=1e-9)))
        if not ok.all():
            n_diff = (~ok).sum()
            mismatches = a[~ok].head(3).index.tolist()
            differing.append((c, int(n_diff), mismatches))
    assert not differing, f"Look-ahead detected: {differing}"


if __name__ == "__main__":
    test_no_lookahead_at_2018_12_31()
    print("PASSED: no look-ahead in novel features at 2018-12-31")
