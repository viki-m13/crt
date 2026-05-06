"""Sweep horizons + buffer caps targeting 50%+ ROR with 95%+ OOS WR.

The existing tight tier (h<=5, buf<=3%) maxes out around 15% ROR because
short DTE = thin credit. To hit 50% ROR we need longer DTE — accepting
the larger path moves they imply by allowing wider buffers within the
stillness regime.

For each (horizon, buffer_cap, regime), this script simulates the BS
credit and resulting ROR for every (ticker, side) that passes the
walk-forward 95% gate, and reports the count of rungs at >=50% ROR.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sp_common import (
    FOLD_YEARS, WARMUP_DAYS,
    buffer_array, compute_features, fold_mask, list_tickers, load_series,
    train_mask_for_fold,
)

# Reuse credit_spread/pricing.py for BS
import importlib.util as _ilu
_pr_spec = _ilu.spec_from_file_location(
    "credit_spread_pricing",
    os.path.join(os.path.dirname(_HERE), "credit_spread", "pricing.py"),
)
_pricing = _ilu.module_from_spec(_pr_spec)
sys.modules["credit_spread_pricing"] = _pricing
_pr_spec.loader.exec_module(_pricing)
estimate_profit = _pricing.estimate_profit
realized_vol = _pricing.realized_vol


def _mask(f, **kw):
    return (
        np.isfinite(f.vol20) & np.isfinite(f.vol5) & np.isfinite(f.compression)
        & np.isfinite(f.rsi14) & np.isfinite(f.range20)
        & np.isfinite(f.trend_flat) & np.isfinite(f.move5d)
        & (f.vol20 < kw["vol20_max"])
        & (f.compression < kw["comp_max"])
        & (f.range20 < kw["range_max"])
        & (f.trend_flat < kw["flat_max"])
        & (f.rsi14 >= kw["rsi_lo"]) & (f.rsi14 <= kw["rsi_hi"])
        & (np.abs(f.move5d) < kw["recent_max"])
    )


def main():
    tickers = list_tickers()
    limit = os.environ.get("SP_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]
    print(f"Universe: {len(tickers)}")

    # Two regime gates to test
    base = dict(vol20_max=0.40, comp_max=1.05, range_max=0.15,
                flat_max=0.04, rsi_lo=25.0, rsi_hi=75.0, recent_max=0.08)
    tight = dict(vol20_max=0.30, comp_max=0.95, range_max=0.10,
                 flat_max=0.025, rsi_lo=35.0, rsi_hi=65.0, recent_max=0.05)
    regimes = {"base": base, "tight": tight}

    # Longer horizons and wider buffer caps
    horizons = [21, 30, 42, 63, 90, 126]
    bufs = [0.05, 0.08, 0.10, 0.15, 0.20]
    qs = [0.97, 0.95]
    safety = 0.005
    target_pooled = 0.95
    target_per_fold = 0.85
    min_train = 60
    min_pooled = 40
    min_folds = 4

    # results: per (regime, h, q, mb) → list of (ticker, side, ror_pct, buffer_pct, wr)
    results = {}

    print(f"{'reg':<6} {'h':<4} {'q':<5} {'maxB':<5} {'#elig':>5} "
          f"{'#>=50%':>6} {'medROR':>7} {'medBuf':>7} {'maxROR':>7}")
    print("-" * 75)

    for ti, t in enumerate(tickers, 1):
        ts = load_series(t)
        if ts is None: continue
        f = compute_features(ts.close)
        warmup = np.zeros(len(ts.dates), dtype=bool)
        warmup[WARMUP_DAYS:] = True
        sigma = f.vol20
        rv = realized_vol(ts.close)
        if rv is None or rv <= 0:
            continue
        spot = float(ts.close[-1])

        for rname, rg in regimes.items():
            sp = _mask(f, **rg)
            if int(sp.sum()) < min_train:
                continue
            for h in horizons:
                ok = np.zeros(len(ts.dates), dtype=bool)
                ok[: len(ts.dates) - h] = True
                for side in ("put", "call"):
                    buf = buffer_array(ts.close, h, side)
                    base_mask = (warmup & sp & np.isfinite(buf))
                    if int(base_mask.sum()) < min_train:
                        continue

                    # Per-fold static-conformal eligibility
                    fold_data = []
                    for year in FOLD_YEARS:
                        tr = base_mask & train_mask_for_fold(ts.dates, year, h)
                        te = base_mask & fold_mask(ts.dates, year) & ok
                        if tr.sum() < 30 or te.sum() == 0:
                            continue
                        fold_data.append((tr, te))
                    if len(fold_data) < min_folds:
                        continue

                    for q in qs:
                        pooled_w = pooled_l = 0
                        fold_wrs = []
                        for tr, te in fold_data:
                            b_hat = float(np.quantile(buf[tr], q)) + safety
                            test_buf = buf[te]
                            w = int((test_buf <= b_hat).sum())
                            l = int((test_buf > b_hat).sum())
                            pooled_w += w
                            pooled_l += l
                            fold_wrs.append(w / max(w + l, 1))
                        pooled = pooled_w + pooled_l
                        if pooled < min_pooled:
                            continue
                        wr = pooled_w / pooled
                        if wr < target_pooled:
                            continue
                        if any(fwr < target_per_fold for fwr in fold_wrs):
                            continue
                        b_final = float(np.quantile(buf[base_mask], q)) + safety

                        # Estimate ROR using BS
                        # Snap to actual options expiry calendar days
                        cal_days = max(int(h * 1.45), 1)  # rough trading->calendar
                        prof = estimate_profit(
                            side=side, spot=spot, buffer=b_final,
                            horizon_sessions=h, realized_sigma=rv,
                            calendar_days_to_expiry=cal_days,
                        )
                        if prof is None:
                            continue
                        ror = prof.return_on_risk * 100.0
                        for mb in bufs:
                            if b_final <= mb:
                                key = (rname, h, q, mb)
                                results.setdefault(key, []).append(
                                    (t, side, ror, b_final*100, wr*100)
                                )
        if ti % 200 == 0:
            print(f"# processed {ti}/{len(tickers)}")

    print()
    print(f"{'reg':<6} {'h':<4} {'q':<5} {'maxB':<5} {'#elig':>5} "
          f"{'#>=50%':>6} {'medROR':>7} {'medBuf':>7} {'maxROR':>7}")
    print("-" * 75)
    rows = sorted(results.items(), key=lambda kv: -sum(1 for r in kv[1] if r[2]>=50))
    for key, rs in rows[:60]:
        rname, h, q, mb = key
        rors = [r[2] for r in rs]
        bufs_l = [r[3] for r in rs]
        n50 = sum(1 for r in rors if r >= 50)
        if n50 == 0 and len(rs) < 10:
            continue
        print(f"{rname:<6} {h:<4} {q:<5} {mb*100:>4.1f}% {len(rs):>5} "
              f"{n50:>6} {np.median(rors):>6.2f}% {np.median(bufs_l):>6.2f}% "
              f"{max(rors):>6.2f}%")

    # Also dump top 30 ROR signals across all configs
    flat = []
    for key, rs in results.items():
        for r in rs:
            flat.append((key, *r))
    flat.sort(key=lambda x: -x[3])  # by ROR desc
    print()
    print("Top 30 individual rungs across all configs by ROR:")
    print(f"{'reg':<6} {'h':<4} {'tk':<6} {'side':<5} {'buf':>6} {'wr':>6} {'ror':>7}")
    seen = set()
    for key, t, side, ror, b, wr in flat[:200]:
        rname, h, q, mb = key
        # one row per (ticker, side, h)
        sig = (t, side, h)
        if sig in seen:
            continue
        seen.add(sig)
        print(f"{rname:<6} {h:<4} {t:<6} {side:<5} {b:>5.2f}% {wr:>5.2f}% {ror:>6.2f}%")
        if len(seen) >= 30:
            break


if __name__ == "__main__":
    main()
