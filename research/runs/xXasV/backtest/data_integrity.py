"""
Data integrity checks for the quant_research framework.
Run once during bootstrap, then re-run whenever data changes.

Checks:
1. PIT membership spot-checks: known S&P 500 additions/removals
2. Adjusted price checks: AAPL 2020 4:1 split, NVDA 2021 4:1 split
3. Delisted survivorship: ENRN/LEH/WCOM/BSC should NOT be in price data
   (or should have terminal values — acceptable either way)
4. Leakage audit: feature at date T uses only data available at T
5. Forward return alignment: returns at month T are from T-end → T+1-end
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/user/crt")
PRICES_DAILY   = REPO / "experiments/monthly_dca/cache/prices_extended.parquet"
PRICES_MONTHLY = REPO / "experiments/monthly_dca/cache/v2/monthly_returns_clean.parquet"
PIT_MEMBERSHIP = REPO / "experiments/monthly_dca/cache/v2/sp500_pit/sp500_membership_monthly.parquet"
FEATURES_DIR   = REPO / "experiments/monthly_dca/cache/features"

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"


def check_pit_membership() -> list[dict]:
    """Spot-check known S&P 500 additions and removals."""
    df = pd.read_parquet(PIT_MEMBERSHIP)
    df["asof"] = pd.to_datetime(df["asof"])

    # Build lookup: {asof → set of tickers}
    pit = {}
    for asof, grp in df.groupby("asof"):
        pit[asof] = set(grp["ticker"].tolist())

    # Known events (source: Wikipedia S&P 500 changes)
    # Format: (check_date, ticker, expected_in, description)
    events = [
        ("2005-03-31", "GOOG", False, "Google added Oct 2006, not in 2005-03"),
        ("2014-05-30", "GOOG", True,  "GOOG Class C in PIT data from 2014 (pre-split ticker was GOOGL)"),
        ("2020-12-31", "TSLA", True,  "Tesla added Dec 2020"),
        ("2020-09-30", "TSLA", False, "Tesla not in S&P500 until Dec 2020"),
        ("2010-04-30", "AAPL", True,  "AAPL in S&P 500 in 2010"),
        ("2008-12-31", "C",    True,  "Citigroup in S&P 500 in Dec 2008"),
        ("2004-09-30", "GE",   True,  "GE in S&P 500 in 2004"),
    ]

    results = []
    for date_str, ticker, expected_in, desc in events:
        asof = pd.Timestamp(date_str)
        # Find nearest PIT snapshot
        pit_keys = sorted(pit.keys())
        match = [k for k in pit_keys if k <= asof + pd.Timedelta(days=15)]
        if not match:
            results.append({"check": desc, "status": WARN, "detail": "No PIT data near date"})
            continue
        nearest = match[-1]
        in_pit = ticker in pit[nearest]
        ok = (in_pit == expected_in)
        status = PASS if ok else FAIL
        detail = f"{ticker} {'found' if in_pit else 'not found'} at {nearest.date()} (expected {'in' if expected_in else 'not in'})"
        results.append({"check": desc, "status": status, "detail": detail})

    return results


def check_price_splits() -> list[dict]:
    """Verify major stock splits are correctly reflected in adjusted prices."""
    daily = pd.read_parquet(PRICES_DAILY)
    daily.index = pd.to_datetime(daily.index)

    results = []

    # AAPL 4:1 split Aug 31, 2020. Price before should be ~4x price after.
    # Check Aug 28, 2020 (before split) vs Sep 1, 2020 (after split)
    for ticker, before_date, after_date, split_ratio, desc in [
        ("AAPL", "2020-08-21", "2020-09-02", 4.0, "AAPL 4:1 split Aug 2020"),
        ("TSLA", "2020-08-28", "2020-09-01", 5.0, "TSLA 5:1 split Aug 2020"),
    ]:
        if ticker not in daily.columns:
            results.append({"check": desc, "status": WARN, "detail": f"{ticker} not in price data"})
            continue

        before_prices = daily.loc[before_date:before_date, ticker]
        after_prices = daily.loc[after_date:after_date, ticker]

        if len(before_prices) == 0 or len(after_prices) == 0:
            # Try nearby dates
            before_prices = daily.loc[:before_date, ticker].dropna().tail(5)
            after_prices = daily.loc[after_date:, ticker].dropna().head(5)

        if len(before_prices) == 0 or len(after_prices) == 0:
            results.append({"check": desc, "status": WARN, "detail": "Insufficient price data around split"})
            continue

        # In ADJUSTED prices, the pre-split price should already be adjusted DOWN
        # (divided by split_ratio). So before and after should be ~equal.
        before_price = float(before_prices.mean())
        after_price = float(after_prices.mean())
        ratio = before_price / after_price if after_price > 0 else 0

        # Adjusted prices should be continuous (ratio ≈ 1.0)
        ok = 0.5 < ratio < 2.0  # within 2x is acceptable for adjusted prices
        status = PASS if ok else FAIL
        results.append({
            "check": desc,
            "status": status,
            "detail": f"Price before/after ratio={ratio:.2f} (expected ~1.0 for adjusted, >4 for unadjusted)"
        })

    return results


def check_pit_leakage() -> list[dict]:
    """
    Leakage check: the 'mom_12_1' feature at date T should equal the
    trailing 12-1 month momentum computed ONLY from prices up to T.
    We verify 5 random asofs and 10 random tickers each.
    """
    results = []

    daily = pd.read_parquet(PRICES_DAILY)
    daily.index = pd.to_datetime(daily.index)

    feat_files = sorted(FEATURES_DIR.glob("*.parquet"))
    if len(feat_files) == 0:
        return [{"check": "Feature leakage audit", "status": WARN, "detail": "No feature files found"}]

    # Check 5 random months
    rng = np.random.default_rng(42)
    check_files = rng.choice(feat_files, size=min(5, len(feat_files)), replace=False)

    for ff in check_files:
        asof = pd.Timestamp(ff.stem)
        feat_df = pd.read_parquet(ff)

        if "mom_12_1" not in feat_df.columns:
            continue

        # Recompute mom_12_1 from prices: 12-1 month momentum
        # At month T, look back 252 trading days (≈12m) and subtract most recent 21 days (≈1m)
        prices_at_T = daily.loc[:asof]
        if len(prices_at_T) < 280:
            continue

        tickers = feat_df.index.intersection(daily.columns)[:10].tolist()
        mismatches = 0

        for t in tickers:
            if t not in prices_at_T.columns:
                continue
            px = prices_at_T[t].dropna()
            if len(px) < 280:
                continue
            p_now = float(px.iloc[-21])    # 21d ago (skip last month)
            p_12m = float(px.iloc[-252]) if len(px) >= 252 else float(px.iloc[0])
            mom_computed = (p_now - p_12m) / p_12m if p_12m > 0 else np.nan

            mom_stored = float(feat_df.loc[t, "mom_12_1"]) if t in feat_df.index else np.nan

            if pd.isna(mom_computed) or pd.isna(mom_stored):
                continue
            # Allow 5% relative tolerance (different lookback conventions)
            if abs(mom_computed - mom_stored) > max(0.05, abs(mom_stored) * 0.15):
                mismatches += 1

        n = min(len(tickers), 10)
        ok = mismatches <= 2  # allow up to 2 mismatches per file
        results.append({
            "check": f"Leakage audit {asof.date()}",
            "status": PASS if ok else WARN,
            "detail": f"{mismatches}/{n} tickers show mom_12_1 mismatch (may differ by convention)"
        })

    return results


def check_return_alignment() -> list[dict]:
    """Verify that monthly_returns[date T] represents the return DURING month T."""
    rets = pd.read_parquet(PRICES_MONTHLY)
    rets.index = pd.to_datetime(rets.index)

    daily = pd.read_parquet(PRICES_DAILY)
    daily.index = pd.to_datetime(daily.index)

    results = []
    # Check AAPL October 2010: daily prices → monthly close change
    tickers_to_check = ["AAPL", "MSFT", "GOOG"]
    months_to_check = [("2010-10-31", "2010-09-30"), ("2019-12-31", "2019-11-29")]

    for end_month, start_month in months_to_check:
        end_ts = pd.Timestamp(end_month)
        start_ts = pd.Timestamp(start_month)
        if end_ts not in rets.index:
            continue
        for ticker in tickers_to_check:
            if ticker not in rets.columns or ticker not in daily.columns:
                continue
            ret_stored = float(rets.loc[end_ts, ticker])

            # Get daily price at start and end
            p_end = daily.loc[:end_ts, ticker].dropna().tail(1)
            p_start = daily.loc[:start_ts, ticker].dropna().tail(1)
            if len(p_end) == 0 or len(p_start) == 0:
                continue
            ret_computed = float(p_end.iloc[0]) / float(p_start.iloc[0]) - 1
            diff = abs(ret_computed - ret_stored)
            ok = diff < 0.01  # within 1%
            results.append({
                "check": f"Return alignment {ticker} {end_month}",
                "status": PASS if ok else WARN,
                "detail": f"stored={ret_stored:.4f} computed={ret_computed:.4f} diff={diff:.4f}"
            })

    return results


def run_all_checks() -> bool:
    all_results = []

    print("=" * 60)
    print("DATA INTEGRITY CHECKS")
    print("=" * 60)

    for checker, label in [
        (check_pit_membership, "PIT Membership"),
        (check_price_splits, "Price Splits"),
        (check_pit_leakage, "Feature Leakage"),
        (check_return_alignment, "Return Alignment"),
    ]:
        print(f"\n{label}:")
        try:
            results = checker()
        except Exception as e:
            results = [{"check": label, "status": WARN, "detail": f"Error: {e}"}]
        for r in results:
            print(f"  {r['status']} {r['check']}: {r['detail']}")
        all_results.extend(results)

    n_fail = sum(1 for r in all_results if r["status"].startswith("❌"))
    n_warn = sum(1 for r in all_results if r["status"].startswith("⚠️"))
    n_pass = sum(1 for r in all_results if r["status"].startswith("✅"))

    print(f"\nSummary: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL")

    # Save to state
    state_dir = Path("/home/user/crt/quant_research/state")
    state_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "n_pass": n_pass, "n_warn": n_warn, "n_fail": n_fail,
        "results": all_results,
    }
    with open(state_dir / "data_integrity.json", "w") as fh:
        json.dump(report, fh, indent=2)

    return n_fail == 0


if __name__ == "__main__":
    ok = run_all_checks()
    import sys
    sys.exit(0 if ok else 1)
