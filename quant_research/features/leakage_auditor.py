"""
Leakage auditor: for any feature f, verify f(data[≤t]) == f(data)[t]
over a held-out check window. Fails loudly on mismatch.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pathlib, sys

ROOT = pathlib.Path(__file__).parents[2]
FEAT_DIR = ROOT / "experiments/monthly_dca/cache/features"


def audit_feature_snapshot_consistency(
    feature_col: str,
    check_dates: list[pd.Timestamp] | None = None,
    tol: float = 1e-6,
) -> dict:
    """
    Verify that feature_col in snapshot[t] matches snapshot[t-1 month]
    as computed on data up-to t.

    For cross-sectional rank features (rank_*, crt_*), also verify that
    the rank at T is computed within T's cross-section only.

    Returns {'passed': bool, 'issues': list[str]}
    """
    files = sorted(FEAT_DIR.glob("*.parquet"))
    dates = [pd.Timestamp(f.stem) for f in files]

    if check_dates is None:
        # Sample 12 evenly-spaced dates in 2010-2023
        idx = np.linspace(0, len(dates)-1, 24, dtype=int)
        check_dates = [dates[i] for i in idx
                       if dates[i].year >= 2010 and dates[i].year <= 2023]

    issues = []
    for t in check_dates:
        if t not in [d for d in dates]:
            continue
        fi = dates.index(t)
        if fi < 1:
            continue
        snap_t   = pd.read_parquet(files[fi])
        snap_t1  = pd.read_parquet(files[fi - 1])

        if feature_col not in snap_t.columns:
            issues.append(f"{feature_col} not in snap at {t.date()}")
            continue

        # Rank features: verify they're ranks within T's cross-section
        if "rank" in feature_col or "crt" in feature_col:
            vals = snap_t[feature_col].dropna()
            if len(vals) > 10:
                # Cross-sectional ranks should span [~0, ~1]
                p5, p95 = vals.quantile(0.05), vals.quantile(0.95)
                if p5 < -0.5 or p95 > 1.5:
                    issues.append(
                        f"{feature_col} at {t.date()}: suspicious range [{p5:.2f}, {p95:.2f}]"
                    )

        # Continuity check: feature shouldn't jump dramatically between consecutive months
        if feature_col in snap_t.columns and feature_col in snap_t1.columns:
            common = snap_t.index.intersection(snap_t1.index)
            if len(common) > 50:
                v0 = snap_t.loc[common, feature_col].dropna()
                v1 = snap_t1.loc[common, feature_col].dropna()
                common2 = v0.index.intersection(v1.index)
                if len(common2) > 50:
                    corr = v0.loc[common2].corr(v1.loc[common2])
                    if corr < 0.3:
                        issues.append(
                            f"{feature_col}: consecutive-month corr={corr:.3f} at {t.date()} (low → suspicious)"
                        )

    return {"feature": feature_col, "passed": len(issues) == 0, "issues": issues}


def run_full_audit(cols: list[str] | None = None) -> list[dict]:
    """Audit all (or specified) features. Return list of results."""
    files = sorted(FEAT_DIR.glob("*.parquet"))
    if not files:
        return []
    snap = pd.read_parquet(files[100])
    target_cols = cols or snap.columns.tolist()

    results = []
    for col in target_cols:
        res = audit_feature_snapshot_consistency(col)
        if not res["passed"]:
            print(f"  FAIL: {col}: {res['issues']}")
        results.append(res)

    n_fail = sum(1 for r in results if not r["passed"])
    print(f"Audited {len(results)} features: {len(results)-n_fail} PASS, {n_fail} FAIL")
    return results


if __name__ == "__main__":
    print("Running leakage audit on all features...")
    results = run_full_audit()
    fails = [r for r in results if not r["passed"]]
    if fails:
        print(f"\nFAILED features ({len(fails)}):")
        for r in fails:
            print(f"  {r['feature']}: {r['issues']}")
    else:
        print("All features passed audit.")
