"""
Leakage audit harness.
For each feature function, verifies f(data[<=t]) == f(data)[t] over a check window.
Fails loudly on mismatch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable


def audit_feature(
    prices_full: pd.DataFrame,
    feature_fn: Callable,
    check_dates: list,
    feature_col: str,
    tol: float = 1e-8,
) -> dict:
    """
    Verify that computing feature at t on prices[:t] equals the same value
    computed with the full price history.

    Returns {"passed": bool, "mismatches": list_of_dates}
    """
    mismatches = []
    for asof in check_dates:
        # Full history (would leak if feature uses future data)
        val_full = feature_fn(prices_full, asof)
        # Restricted to t
        val_pit = feature_fn(prices_full.loc[:asof], asof)

        if isinstance(val_full, pd.Series) and isinstance(val_pit, pd.Series):
            diff = (val_full - val_pit).abs()
            if diff.max() > tol:
                mismatches.append(asof)
        elif abs(float(val_full) - float(val_pit)) > tol:
            mismatches.append(asof)

    return {
        "passed": len(mismatches) == 0,
        "mismatches": mismatches,
        "n_checked": len(check_dates),
    }


def run_standard_audits(prices: pd.DataFrame, check_dates: list) -> None:
    """Run audit for all standard features. Raises AssertionError on leakage."""
    from features.momentum import mom_12_1, mom_6_1, vol_12m

    for name, fn in [("mom_12_1", mom_12_1), ("mom_6_1", mom_6_1), ("vol_12m", vol_12m)]:
        result = audit_feature(prices, fn, check_dates, name)
        if not result["passed"]:
            raise AssertionError(
                f"LEAKAGE DETECTED in {name} at dates: {result['mismatches'][:5]}"
            )
        print(f"[AUDIT PASS] {name}: no leakage detected over {result['n_checked']} dates")
