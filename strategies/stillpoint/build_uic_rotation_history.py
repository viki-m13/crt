"""Rotation history: compute UIC eligibility at multiple historical
as-of dates to demonstrate that the eligible-ticker set rotates over
time as new data becomes available.

This addresses bias concerns about a "static list" — the 26-ticker
list at any moment is the OUTPUT of running the same algorithm on
the data available up to that date. Different as-of dates produce
different lists. Tickers join when their walk-forward statistics
finally clear the bar; tickers leave when a new losing trade brings
their pooled WR below 95%.

For each as-of date in a snapshot grid, we:
  1. Truncate every ticker's series at that date.
  2. Run the same UIC validation that the production engine uses.
  3. Record the eligible set (ticker, horizon, q, ROR, WR, n_tests).

Output: strategies/stillpoint/results/uic_rotation_history.json
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from sp_common import (
    WARMUP_DAYS,
    SP_SAFETY_EPS, SP_MIN_TRAIN_FIRES, SP_MIN_POOLED_TEST, SP_MIN_FOLDS,
    SP_UIC_HORIZONS, SP_UIC_CONFORMAL_QS, SP_UIC_MAX_BUFFER,
    SP_UIC_PER_FOLD_WIN, SP_UIC_POOLED_WIN, SP_UIC_TARGET_ROR,
    SP_UIC_WIDTHS, SP_UIC_MAX_COMBINED_CREDIT_RATIO,
    close_buffer_arrays, compute_features, fold_mask, list_tickers,
    load_series, train_mask_for_fold,
)

import importlib.util as _ilu
_pr_spec = _ilu.spec_from_file_location(
    "credit_spread_pricing",
    os.path.join(os.path.dirname(_HERE), "credit_spread", "pricing.py"),
)
_pricing = _ilu.module_from_spec(_pr_spec)
sys.modules["credit_spread_pricing"] = _pricing
_pr_spec.loader.exec_module(_pricing)
bs_put = _pricing.bs_put
bs_call = _pricing.bs_call
credit_spread_mid = _pricing.credit_spread_mid  # smile-aware leg pricing
realized_vol = _pricing.realized_vol


# Quarterly snapshot grid back through 2024 to demonstrate rotation.
ASOF_DATES = [
    "2024-06-28",
    "2024-12-31",
    "2025-03-31",
    "2025-06-30",
    "2025-09-30",
    "2025-12-31",
    "2026-02-27",
    "2026-05-05",
]


def fold_years_for_asof(asof_iso: str):
    asof_year = int(asof_iso[:4])
    return list(range(2020, asof_year + 1))


def evaluate_uic_asof(t: str, asof_dt64: np.datetime64):
    ts = load_series(t)
    if ts is None: return []
    keep = ts.dates <= asof_dt64
    n_keep = int(keep.sum())
    if n_keep < WARMUP_DAYS + max(SP_UIC_HORIZONS) + 100:
        return []
    close = ts.close[keep]
    dates = ts.dates[keep]

    f = compute_features(close)
    rv = realized_vol(close)
    if rv is None or rv <= 0: return []
    spot_now = float(close[-1])
    sigma = f.vol20
    if not np.isfinite(sigma[-1]) or sigma[-1] <= 0: return []
    sigma_now = float(sigma[-1])

    fold_years = fold_years_for_asof(str(asof_dt64))
    warmup = np.zeros(len(dates), dtype=bool); warmup[WARMUP_DAYS:] = True

    eligible = []
    for h in SP_UIC_HORIZONS:
        bP, bC = close_buffer_arrays(close, h)
        base = (warmup & np.isfinite(bP) & np.isfinite(bC)
                & np.isfinite(sigma) & (sigma > 0))
        if int(base.sum()) < SP_MIN_TRAIN_FIRES: continue
        T = h / 252.0; sqrtT = math.sqrt(T)
        best = None
        for q in SP_UIC_CONFORMAL_QS:
            pw = pl = 0; fold_count = 0; fold_wrs = []
            for y in fold_years:
                tr = base & train_mask_for_fold(dates, y, h)
                te = base & fold_mask(dates, y)
                ok = np.zeros(len(dates), dtype=bool); ok[: len(dates) - h] = True
                te = te & ok
                if tr.sum() < 60 or te.sum() == 0: continue
                z_p = bP[tr] / (sigma[tr] * sqrtT)
                z_c = bC[tr] / (sigma[tr] * sqrtT)
                if not (np.isfinite(z_p).all() and np.isfinite(z_c).all()): continue
                zp_q = float(np.quantile(z_p, q))
                zc_q = float(np.quantile(z_c, q))
                fold_count += 1
                tp = zp_q * sigma[te] * sqrtT + SP_SAFETY_EPS
                tc = zc_q * sigma[te] * sqrtT + SP_SAFETY_EPS
                joint = (bP[te] <= tp) & (bC[te] <= tc)
                w = int(joint.sum()); l = int((~joint).sum())
                pw += w; pl += l; fold_wrs.append(w / max(w + l, 1))
            pooled = pw + pl
            if pooled < SP_MIN_POOLED_TEST or fold_count < SP_MIN_FOLDS: continue
            wr = pw / pooled
            if wr < SP_UIC_POOLED_WIN: continue
            if any(fwr < SP_UIC_PER_FOLD_WIN for fwr in fold_wrs): continue
            zp_full = float(np.quantile(bP[base] / (sigma[base] * sqrtT), q))
            zc_full = float(np.quantile(bC[base] / (sigma[base] * sqrtT), q))
            bp_now = zp_full * sigma_now * sqrtT + SP_SAFETY_EPS
            bc_now = zc_full * sigma_now * sqrtT + SP_SAFETY_EPS
            if bp_now > SP_UIC_MAX_BUFFER or bc_now > SP_UIC_MAX_BUFFER: continue
            iv = rv * 1.30
            cal_days = max(int(h * 1.45), 1); T_cal = cal_days / 365
            K_ps = spot_now * (1 - bp_now); K_cs = spot_now * (1 + bc_now)
            best_ror = 0; best_w = None
            for w_pct in SP_UIC_WIDTHS:
                width = spot_now * w_pct
                K_pl = K_ps - width; K_cl = K_cs + width
                if K_pl <= 0: continue
                cp = credit_spread_mid("put",  spot_now, K_ps, K_pl, T_cal, iv) * 0.80
                cc = credit_spread_mid("call", spot_now, K_cs, K_cl, T_cal, iv) * 0.80
                cred = cp + cc
                cred = min(cred, SP_UIC_MAX_COMBINED_CREDIT_RATIO * width)
                ml = max(width - cred, 0.01)
                ror = cred / ml
                if ror > best_ror: best_ror = ror; best_w = w_pct
            if best_ror < SP_UIC_TARGET_ROR: continue
            cand = {
                "horizon": h, "q": q, "width_pct": best_w * 100,
                "buf_put_pct": bp_now * 100, "buf_call_pct": bc_now * 100,
                "joint_oos_wr_pct": wr * 100, "n_oos_tests": pooled,
                "ror_pct": best_ror * 100,
            }
            if best is None or cand["ror_pct"] > best["ror_pct"]:
                best = cand
        if best is not None:
            eligible.append(best)
    return eligible


def main():
    tickers = list_tickers()
    limit = os.environ.get("SP_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]
    print(f"Universe: {len(tickers)} tickers")
    print(f"As-of grid: {ASOF_DATES}")
    print()

    history = []
    for asof in ASOF_DATES:
        asof_dt64 = np.datetime64(asof)
        print(f"\n=== As-of {asof} ===")
        eligible_per_ticker = {}
        n_qualified = 0
        for ti, t in enumerate(tickers, 1):
            try:
                elig = evaluate_uic_asof(t, asof_dt64)
            except Exception:
                continue
            if elig:
                elig.sort(key=lambda r: -r["ror_pct"])
                eligible_per_ticker[t] = elig[0]
                n_qualified += 1
            if ti % 200 == 0:
                print(f"  {ti}/{len(tickers)}  qualified_so_far={n_qualified}")
        sorted_ts = sorted(eligible_per_ticker.items(),
                           key=lambda kv: -kv[1]["ror_pct"])
        print(f"  Eligible tickers: {len(eligible_per_ticker)}")
        print(f"  Top 15: " + ", ".join(t for t, _ in sorted_ts[:15]))
        history.append({
            "as_of": asof,
            "n_eligible_tickers": len(eligible_per_ticker),
            "tickers": {t: cfg for t, cfg in sorted_ts},
        })

    all_tickers_seen = set()
    for snap in history:
        all_tickers_seen |= set(snap["tickers"].keys())
    print()
    print("=" * 70)
    print("ROTATION SUMMARY")
    print("=" * 70)
    print(f"Unique tickers across all snapshots: {len(all_tickers_seen)}")
    appearances = defaultdict(int)
    for snap in history:
        for t in snap["tickers"]:
            appearances[t] += 1
    for n in range(1, len(history) + 1):
        cnt = sum(1 for c in appearances.values() if c == n)
        if cnt:
            print(f"  Tickers in exactly {n}/{len(history)} snapshots: {cnt}")
    print()
    print("Snapshot-to-snapshot churn:")
    for i in range(1, len(history)):
        prev_set = set(history[i - 1]["tickers"].keys())
        cur_set = set(history[i]["tickers"].keys())
        added = cur_set - prev_set
        removed = prev_set - cur_set
        print(f"  {history[i-1]['as_of']} → {history[i]['as_of']}: "
              f"+{len(added)} added (e.g. {sorted(added)[:5]}), "
              f"−{len(removed)} removed (e.g. {sorted(removed)[:5]}), "
              f"{len(prev_set & cur_set)} unchanged")

    out_path = os.path.join(_HERE, "results", "uic_rotation_history.json")
    with open(out_path, "w") as fh:
        json.dump({
            "as_of_grid": ASOF_DATES,
            "n_snapshots": len(history),
            "n_unique_tickers_ever_eligible": len(all_tickers_seen),
            "snapshots": history,
            "appearances": dict(appearances),
        }, fh, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
