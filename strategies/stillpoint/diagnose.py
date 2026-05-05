"""Diagnose Stillpoint regime fires and buffer distributions per ticker.

Usage:
    SP_LIMIT=50 python3 diagnose.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sp_common import (
    HORIZONS, FOLD_YEARS, WARMUP_DAYS,
    SP_CONFORMAL_Q, SP_SAFETY_EPS, SP_MAX_BUFFER, SP_MIN_TRAIN_FIRES,
    buffer_array, compute_features, fold_mask, list_tickers, load_series,
    stillpoint_mask, train_mask_for_fold,
)


def main() -> int:
    tickers = list_tickers()
    limit = os.environ.get("SP_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]

    rows = []
    for t in tickers:
        ts = load_series(t)
        if ts is None:
            continue
        f = compute_features(ts.close)
        sp = stillpoint_mask(f)
        n_fires = int(sp.sum())
        if n_fires < SP_MIN_TRAIN_FIRES:
            continue
        warmup = np.zeros(len(ts.dates), dtype=bool)
        warmup[WARMUP_DAYS:] = True
        for h in HORIZONS:
            for side in ("put", "call"):
                buf = buffer_array(ts.close, h, side)
                base = warmup & sp & np.isfinite(buf)
                n_base = int(base.sum())
                if n_base < SP_MIN_TRAIN_FIRES:
                    continue
                # Walk-forward folds: simulate
                folds = []
                pooled_w = pooled_l = 0
                for year in FOLD_YEARS:
                    tr = base & train_mask_for_fold(ts.dates, year, h)
                    te = base & fold_mask(ts.dates, year)
                    n = len(ts.dates)
                    ok = np.zeros(n, dtype=bool)
                    ok[: n - h] = True
                    te = te & ok
                    if tr.sum() < 30:
                        continue
                    if te.sum() == 0:
                        continue
                    b_hat = float(np.quantile(buf[tr], SP_CONFORMAL_Q)) + SP_SAFETY_EPS
                    if b_hat > SP_MAX_BUFFER:
                        continue
                    test_buf = buf[te]
                    w = int((test_buf <= b_hat).sum())
                    l = int((test_buf > b_hat).sum())
                    pooled_w += w
                    pooled_l += l
                    folds.append((year, int(tr.sum()), int(te.sum()), b_hat, w, l))
                if not folds:
                    continue
                pooled = pooled_w + pooled_l
                pooled_wr = pooled_w / pooled
                # Final live buffer
                b_final = float(np.quantile(buf[base], SP_CONFORMAL_Q)) + SP_SAFETY_EPS
                rows.append({
                    "ticker": t, "side": side, "h": h,
                    "n_base": n_base, "pooled_w": pooled_w, "pooled_l": pooled_l,
                    "pooled_wr": pooled_wr, "b_final": b_final,
                    "n_folds": len(folds),
                })
    print(f"Total (ticker, side, horizon) variants with enough samples: {len(rows)}")
    # Top 30 by pooled win rate * n_test, with buffer < 5%
    rows.sort(key=lambda r: (r["pooled_wr"], -(r["b_final"])), reverse=True)
    print()
    print(f"{'tkr':<6} {'side':<5} {'h':<3} {'n_base':>6} "
          f"{'wr%':>6} {'pool':>5} {'buf%':>5} {'folds':>5}")
    print("-" * 60)
    n_at95 = 0
    n_at95_with_buf = 0
    for r in rows[:80]:
        marker = ""
        if r["pooled_wr"] >= 0.95: n_at95 += 1
        if r["pooled_wr"] >= 0.95 and r["b_final"] <= 0.05:
            n_at95_with_buf += 1
            marker = " *"
        if r["b_final"] <= SP_MAX_BUFFER:
            print(f"{r['ticker']:<6} {r['side']:<5} {r['h']:<3} "
                  f"{r['n_base']:>6} {r['pooled_wr']*100:>5.1f}% "
                  f"{r['pooled_l']+r['pooled_w']:>5} "
                  f"{r['b_final']*100:>4.2f}% {r['n_folds']:>5}{marker}")
    print()
    n_buffered = sum(1 for r in rows if r["b_final"] <= SP_MAX_BUFFER)
    print(f"Variants with buffer <= 5%: {n_buffered}")
    print(f"Variants with pooled win rate >= 95%: "
          f"{sum(1 for r in rows if r['pooled_wr'] >= 0.95)}")
    print(f"Variants with both: {n_at95_with_buf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
