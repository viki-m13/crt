"""Cache-efficient targeted sweep.

For each ticker we load the series ONCE, compute features ONCE, compute
buffer arrays for all horizons ONCE, and then evaluate every (regime,
horizon, q, max_buf, method) combo against the cached numpy arrays.
This drops cost from O(n_configs × n_tickers × time_per_ticker) to
O(n_tickers × time_per_ticker).

Methods compared:
  - "static"   : b_hat = quantile_q(b*) + safety       (current Stillpoint)
  - "voladapt" : b_hat = z_q × sigma_today × sqrt(h/252) + safety
                 with z = b* / (sigma × sqrt(h/252))
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sp_common import (
    FOLD_YEARS, WARMUP_DAYS,
    buffer_array, compute_features, fold_mask, list_tickers, load_series,
    train_mask_for_fold,
)


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

    base = dict(vol20_max=0.40, comp_max=1.05, range_max=0.15,
                flat_max=0.04, rsi_lo=25.0, rsi_hi=75.0, recent_max=0.08)
    tight = dict(vol20_max=0.30, comp_max=0.95, range_max=0.10,
                 flat_max=0.025, rsi_lo=35.0, rsi_hi=65.0, recent_max=0.05)
    ultra = dict(vol20_max=0.22, comp_max=0.85, range_max=0.06,
                 flat_max=0.015, rsi_lo=40.0, rsi_hi=60.0, recent_max=0.03)
    regimes = {"base": base, "tight": tight, "ultra": ultra}

    # Configs: (regime_name, horizon, q, max_buf, method)
    horizons = [2, 3, 5]
    qs = [0.95, 0.97]
    bufs = [0.015, 0.02, 0.025, 0.03]
    methods = ["static", "voladapt"]
    safety = 0.005
    min_train_fires = 60
    min_pooled_test = 40
    min_folds = 4
    target_pooled = 0.95
    target_per_fold = 0.85

    # Aggregate stats per (regime, h, q, max_buf, method)
    stats = defaultdict(lambda: {"n_elig": 0, "bufs": [], "wrs": [],
                                  "pools": []})

    for ti, t in enumerate(tickers, 1):
        ts = load_series(t)
        if ts is None:
            continue
        f = compute_features(ts.close)
        warmup = np.zeros(len(ts.dates), dtype=bool)
        warmup[WARMUP_DAYS:] = True
        sigma = f.vol20

        # Cache buffer arrays per horizon
        bufs_arr = {h: buffer_array(ts.close, h, side)
                    for h in horizons for side in ("put", "call")}
        # Above is wrong — let me redo
        bufs_arr = {}
        for h in horizons:
            for side in ("put", "call"):
                bufs_arr[(h, side)] = buffer_array(ts.close, h, side)

        for rname, rg in regimes.items():
            sp = _mask(f, **rg)
            if int(sp.sum()) < min_train_fires:
                continue
            for h in horizons:
                # Test-window mask
                ok = np.zeros(len(ts.dates), dtype=bool)
                ok[: len(ts.dates) - h] = True
                for side in ("put", "call"):
                    buf = bufs_arr[(h, side)]
                    base_mask = (warmup & sp & np.isfinite(buf) &
                                 np.isfinite(sigma) & (sigma > 0))
                    if int(base_mask.sum()) < min_train_fires:
                        continue
                    sigma_today = float(sigma[-1])
                    T = h / 252.0

                    # Compute fold-by-fold for both methods, against ALL q's
                    # all at once (since q only affects the quantile calc).
                    fold_data = []
                    for year in FOLD_YEARS:
                        tr = base_mask & train_mask_for_fold(ts.dates, year, h)
                        te = base_mask & fold_mask(ts.dates, year) & ok
                        if tr.sum() < 30 or te.sum() == 0:
                            continue
                        fold_data.append((tr, te))

                    if len(fold_data) < min_folds:
                        continue

                    # For each (q, method, mb): compute eligibility
                    for q in qs:
                        for method in methods:
                            # Compute b_hat per fold (from training)
                            # Test-set wins
                            pooled_w = pooled_l = 0
                            fold_wrs = []
                            for tr, te in fold_data:
                                if method == "static":
                                    b_hat = float(np.quantile(buf[tr], q)) + safety
                                else:
                                    z_tr = buf[tr] / (sigma[tr] * np.sqrt(T))
                                    z_q = float(np.quantile(z_tr, q))
                                    # per-row test thresholds
                                    pass
                                test_buf = buf[te]
                                if method == "static":
                                    w = int((test_buf <= b_hat).sum())
                                    l = int((test_buf > b_hat).sum())
                                else:
                                    thresh = z_q * sigma[te] * np.sqrt(T) + safety
                                    w = int((test_buf <= thresh).sum())
                                    l = int((test_buf > thresh).sum())
                                pooled_w += w
                                pooled_l += l
                                fold_wrs.append(w / max(w + l, 1))
                            pooled = pooled_w + pooled_l
                            if pooled < min_pooled_test:
                                continue
                            wr = pooled_w / pooled
                            if wr < target_pooled:
                                continue
                            if any(fwr < target_per_fold for fwr in fold_wrs):
                                continue
                            # Live buffer (today)
                            if method == "static":
                                b_final = float(np.quantile(buf[base_mask], q)) + safety
                            else:
                                z_full = buf[base_mask] / (sigma[base_mask] * np.sqrt(T))
                                z_q_f = float(np.quantile(z_full, q))
                                b_final = z_q_f * sigma_today * np.sqrt(T) + safety
                            for mb in bufs:
                                if b_final <= mb:
                                    key = (rname, h, q, mb, method)
                                    s = stats[key]
                                    s["n_elig"] += 1
                                    s["bufs"].append(b_final)
                                    s["wrs"].append(wr)
                                    s["pools"].append(pooled)
        if ti % 100 == 0:
            print(f"  {ti}/{len(tickers)} processed")

    # Print sorted by n_elig descending, filtering for max_buf <= 0.03
    rows = []
    for (rname, h, q, mb, method), s in stats.items():
        if s["n_elig"] == 0:
            continue
        rows.append((rname, h, q, mb, method, s))
    rows.sort(key=lambda r: (-r[5]["n_elig"], r[3]))

    print(f"\n{'reg':<6} {'h':<3} {'q':<5} {'maxB':<5} {'meth':<10} "
          f"{'#elig':>5} {'medB':>6} {'medWR':>6} {'minB':>6} {'medPool':>7}")
    print("-" * 75)
    for rname, h, q, mb, method, s in rows[:80]:
        medB = float(np.median(s["bufs"])) * 100
        minB = float(min(s["bufs"])) * 100
        medWR = float(np.median(s["wrs"])) * 100
        medPool = int(np.median(s["pools"]))
        print(f"{rname:<6} {h:<3} {q:<5} {mb*100:>4.1f}% {method:<10} "
              f"{s['n_elig']:>5} {medB:>5.2f}% {medWR:>5.2f}% "
              f"{minB:>5.2f}% {medPool:>7}")


if __name__ == "__main__":
    main()
