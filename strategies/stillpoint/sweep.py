"""Hyperparameter sweep over Stillpoint regime thresholds and conformal q.

For each (regime threshold tightness, conformal quantile) try the engine
on a subset and report:
   - n eligible (ticker, side, horizon) variants where pooled WR >= 95%
     AND buffer <= 5%
   - their median buffer
   - their pooled WR
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import sp_common as sc
from sp_common import (
    HORIZONS, FOLD_YEARS, WARMUP_DAYS,
    buffer_array, compute_features, fold_mask, list_tickers, load_series,
    train_mask_for_fold,
)


def stillpoint_mask_custom(f, vol20_max, comp_max, range_max, flat_max,
                            rsi_lo, rsi_hi, recent_max):
    return (
        np.isfinite(f.vol20) & np.isfinite(f.vol5) & np.isfinite(f.compression)
        & np.isfinite(f.rsi14) & np.isfinite(f.range20)
        & np.isfinite(f.trend_flat) & np.isfinite(f.move5d)
        & (f.vol20 < vol20_max)
        & (f.compression < comp_max)
        & (f.range20 < range_max)
        & (f.trend_flat < flat_max)
        & (f.rsi14 >= rsi_lo) & (f.rsi14 <= rsi_hi)
        & (np.abs(f.move5d) < recent_max)
    )


def evaluate(tickers, regime_kwargs, q, max_buf, min_train_fires=60,
             min_pooled_test=40, min_folds=4, target_pooled=0.95,
             target_per_fold=0.90):
    rows = []
    for t in tickers:
        ts = load_series(t)
        if ts is None:
            continue
        f = compute_features(ts.close)
        sp = stillpoint_mask_custom(f, **regime_kwargs)
        if int(sp.sum()) < min_train_fires:
            continue
        warmup = np.zeros(len(ts.dates), dtype=bool)
        warmup[WARMUP_DAYS:] = True
        for h in HORIZONS:
            for side in ("put", "call"):
                buf = buffer_array(ts.close, h, side)
                base = warmup & sp & np.isfinite(buf)
                n_base = int(base.sum())
                if n_base < min_train_fires:
                    continue
                folds_ok_count = 0
                pooled_w = pooled_l = 0
                fold_count = 0
                fold_wrs = []
                for year in FOLD_YEARS:
                    tr = base & train_mask_for_fold(ts.dates, year, h)
                    te = base & fold_mask(ts.dates, year)
                    n = len(ts.dates)
                    ok = np.zeros(n, dtype=bool)
                    ok[: n - h] = True
                    te = te & ok
                    if tr.sum() < 30 or te.sum() == 0:
                        continue
                    b_hat = float(np.quantile(buf[tr], q)) + 0.005
                    if b_hat > max_buf:
                        continue
                    fold_count += 1
                    test_buf = buf[te]
                    w = int((test_buf <= b_hat).sum())
                    l = int((test_buf > b_hat).sum())
                    pooled_w += w
                    pooled_l += l
                    fold_wrs.append(w / max(w + l, 1))
                pooled = pooled_w + pooled_l
                if pooled < min_pooled_test or fold_count < min_folds:
                    continue
                wr = pooled_w / pooled
                if wr < target_pooled:
                    continue
                if any(fwr < target_per_fold for fwr in fold_wrs):
                    continue
                b_final = float(np.quantile(buf[base], q)) + 0.005
                if b_final > max_buf:
                    continue
                rows.append({
                    "ticker": t, "side": side, "h": h,
                    "buf": b_final, "wr": wr, "pooled": pooled,
                    "n_folds": fold_count,
                })
    return rows


def main():
    tickers = list_tickers()
    limit = os.environ.get("SP_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]
    print(f"Tickers: {len(tickers)}")

    # Regime variants
    base_regime = dict(
        vol20_max=0.30, comp_max=0.95, range_max=0.10, flat_max=0.025,
        rsi_lo=35.0, rsi_hi=65.0, recent_max=0.05,
    )
    looser_regime = dict(
        vol20_max=0.35, comp_max=1.00, range_max=0.12, flat_max=0.03,
        rsi_lo=30.0, rsi_hi=70.0, recent_max=0.06,
    )
    much_looser = dict(
        vol20_max=0.40, comp_max=1.05, range_max=0.15, flat_max=0.04,
        rsi_lo=25.0, rsi_hi=75.0, recent_max=0.08,
    )

    grid = [
        # (label, regime, q, max_buf, target_per_fold)
        ("base",         base_regime,   0.97, 0.05, 0.90),
        ("loose",        looser_regime, 0.97, 0.05, 0.90),
        ("vloose",       much_looser,   0.97, 0.05, 0.90),
        ("base",         base_regime,   0.97, 0.05, 0.85),
        ("loose",        looser_regime, 0.97, 0.05, 0.85),
        ("vloose",       much_looser,   0.97, 0.05, 0.85),
        ("loose",        looser_regime, 0.97, 0.05, 0.80),
        ("vloose",       much_looser,   0.97, 0.05, 0.80),
        ("vloose",       much_looser,   0.96, 0.05, 0.85),
        ("vloose",       much_looser,   0.95, 0.05, 0.85),
        ("vloose",       much_looser,   0.95, 0.04, 0.85),
        ("vloose",       much_looser,   0.97, 0.04, 0.85),
    ]
    print()
    print(f"{'regime':<8} {'q':<5} {'maxB':<5} {'pflr':<5} {'#elig':>6} {'medBuf':>7} {'medWR':>7} {'medPool':>8}")
    print("-" * 70)
    for name, rg, q, mxb, pflr in grid:
        rows = evaluate(tickers, rg, q, mxb, target_per_fold=pflr)
        if rows:
            mb = float(np.median([r["buf"] for r in rows])) * 100
            mw = float(np.median([r["wr"] for r in rows])) * 100
            mp = float(np.median([r["pooled"] for r in rows]))
        else:
            mb = mw = mp = 0.0
        print(f"{name:<8} {q:<5} {mxb*100:>4.1f}% {pflr:<5} {len(rows):>6} "
              f"{mb:>6.2f}% {mw:>6.2f}% {mp:>8.0f}")


if __name__ == "__main__":
    main()
