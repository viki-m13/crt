"""Universal IC sweep — close-at-expiry win condition instead of path.

Real-world credit spreads held to expiry settle on the close price at
expiration, not on path extrema during the holding period. The
path-min/max criterion (used by the previous Stillpoint tiers) is
strictly conservative — it counts as a loss any case where the path
TOUCHED the short strike even if the stock recovered by expiry.

For European cash-settled options (SPX, NDX, RUT) only the closing
price matters. For American-style OTM credit spreads held to expiry,
early exercise is essentially never optimal when the option is OTM, so
in practice only the close at expiry matters there too.

Joint close-at-expiry win condition:
  WIN iff K_put_short ≤ close[t+h] ≤ K_call_short

We backtest the same universal vol-adaptive joint conformal as before
but using close-at-expiry instead of path-min/max. Expected: more
tickers qualify because we no longer punish "touched and recovered"
paths.
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
    FOLD_YEARS, WARMUP_DAYS,
    SP_SAFETY_EPS, SP_MIN_TRAIN_FIRES, SP_MIN_POOLED_TEST, SP_MIN_FOLDS,
    compute_features, fold_mask, list_tickers, load_series,
    train_mask_for_fold,
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
realized_vol = _pricing.realized_vol


def close_buffer_arrays(close: np.ndarray, h: int):
    """Close-only buffer: how far close[t+h] is from close[t], signed.
    Returns (b_put_close, b_call_close) — the absolute fractional drop
    or rise at the EXPIRY close (not the path extrema).
    """
    n = len(close)
    fwd = np.full(n, np.nan, dtype="float64")
    if n > h:
        fwd[: n - h] = close[h:]
    with np.errstate(invalid="ignore", divide="ignore"):
        # Drop = max(0, 1 - close[t+h]/close[t])
        drop = np.where(np.isnan(fwd), np.nan, np.maximum(1.0 - fwd / close, 0.0))
        rise = np.where(np.isnan(fwd), np.nan, np.maximum(fwd / close - 1.0, 0.0))
    return drop, rise


HORIZONS = [21, 30, 42, 63, 90, 126]
QS = [0.96, 0.97, 0.975, 0.98, 0.985]
WIDTHS = [0.005, 0.01, 0.02, 0.03, 0.05]
MAX_PER_LEG_BUFFER = 0.30
TARGET_JOINT_WR = 0.95
TARGET_PER_FOLD = 0.80
TARGET_ROR = 0.50
# Realistic ceilings — BS at very narrow widths can produce
# combined_credit > width which is an arbitrage artifact. Filter those.
MAX_COMBINED_CREDIT_RATIO = 0.50    # combined_credit ≤ 0.50 × width
MIN_WIDTH_PCT = 0.01                 # ignore widths < 1% of spot


def main():
    tickers = list_tickers()
    limit = os.environ.get("SP_LIMIT")
    if limit:
        tickers = tickers[: int(limit)]
    print(f"Universe: {len(tickers)} tickers (NO regime gate; CLOSE-AT-EXPIRY win condition)")
    print()

    eligible = []
    for ti, t in enumerate(tickers, 1):
        ts = load_series(t)
        if ts is None: continue
        f = compute_features(ts.close)
        rv = realized_vol(ts.close)
        if rv is None or rv <= 0: continue
        spot_now = float(ts.close[-1])
        sigma_now = float(f.vol20[-1]) if np.isfinite(f.vol20[-1]) else None
        if sigma_now is None or sigma_now <= 0: continue

        warmup = np.zeros(len(ts.dates), dtype=bool)
        warmup[WARMUP_DAYS:] = True

        for h in HORIZONS:
            T = h / 252.0
            sqrtT = math.sqrt(T)
            bP, bC = close_buffer_arrays(ts.close, h)
            sigma = f.vol20
            base = (warmup & np.isfinite(bP) & np.isfinite(bC)
                    & np.isfinite(sigma) & (sigma > 0))
            if int(base.sum()) < SP_MIN_TRAIN_FIRES:
                continue

            best_for_h = None
            for q in QS:
                pw = pl = 0; fold_count = 0; fold_wrs = []
                for y in FOLD_YEARS:
                    tr = base & train_mask_for_fold(ts.dates, y, h)
                    te = base & fold_mask(ts.dates, y)
                    ok = np.zeros(len(ts.dates), dtype=bool)
                    ok[: len(ts.dates) - h] = True
                    te = te & ok
                    if tr.sum() < 60 or te.sum() == 0: continue
                    z_put_train = bP[tr] / (sigma[tr] * sqrtT)
                    z_call_train = bC[tr] / (sigma[tr] * sqrtT)
                    if not (np.isfinite(z_put_train).all() and
                             np.isfinite(z_call_train).all()):
                        continue
                    zp_q = float(np.quantile(z_put_train, q))
                    zc_q = float(np.quantile(z_call_train, q))
                    fold_count += 1
                    thresh_p = zp_q * sigma[te] * sqrtT + SP_SAFETY_EPS
                    thresh_c = zc_q * sigma[te] * sqrtT + SP_SAFETY_EPS
                    joint = (bP[te] <= thresh_p) & (bC[te] <= thresh_c)
                    w = int(joint.sum()); l = int((~joint).sum())
                    pw += w; pl += l
                    fold_wrs.append(w / max(w + l, 1))
                pooled = pw + pl
                if pooled < SP_MIN_POOLED_TEST or fold_count < SP_MIN_FOLDS:
                    continue
                wr = pw / pooled
                if wr < TARGET_JOINT_WR: continue
                if any(fwr < TARGET_PER_FOLD for fwr in fold_wrs): continue

                z_put_full = bP[base] / (sigma[base] * sqrtT)
                z_call_full = bC[base] / (sigma[base] * sqrtT)
                zp_q_f = float(np.quantile(z_put_full, q))
                zc_q_f = float(np.quantile(z_call_full, q))
                bp_now = zp_q_f * sigma_now * sqrtT + SP_SAFETY_EPS
                bc_now = zc_q_f * sigma_now * sqrtT + SP_SAFETY_EPS
                if bp_now > MAX_PER_LEG_BUFFER or bc_now > MAX_PER_LEG_BUFFER:
                    continue

                iv = rv * 1.30
                cal_days = max(int(h * 1.45), 1)
                T_cal = cal_days / 365.0
                K_ps = spot_now * (1 - bp_now)
                K_cs = spot_now * (1 + bc_now)
                best_ror = 0; best_width = None
                for w_pct in WIDTHS:
                    if w_pct < MIN_WIDTH_PCT: continue
                    width = spot_now * w_pct
                    K_pl = K_ps - width
                    K_cl = K_cs + width
                    if K_pl <= 0: continue
                    cp = max(bs_put(spot_now, K_ps, T_cal, iv) - bs_put(spot_now, K_pl, T_cal, iv), 0) * 0.80
                    cc = max(bs_call(spot_now, K_cs, T_cal, iv) - bs_call(spot_now, K_cl, T_cal, iv), 0) * 0.80
                    cred = cp + cc
                    # Realistic ceiling on combined credit/width
                    if cred > MAX_COMBINED_CREDIT_RATIO * width:
                        cred = MAX_COMBINED_CREDIT_RATIO * width
                    ml = max(width - cred, 0.01)
                    ror = cred / ml
                    if ror > best_ror:
                        best_ror = ror; best_width = w_pct
                if best_ror < TARGET_ROR:
                    continue

                cand = {
                    "ticker": t, "horizon": h, "q": q,
                    "buf_put_pct": bp_now * 100,
                    "buf_call_pct": bc_now * 100,
                    "joint_oos_wr_pct": wr * 100,
                    "n_oos_tests": pooled,
                    "n_folds": fold_count,
                    "best_width_pct": best_width * 100,
                    "ror_pct": best_ror * 100,
                    "spot": spot_now,
                    "sigma_today_pct": sigma_now * 100,
                    "K_put_short": K_ps,
                    "K_call_short": K_cs,
                }
                if best_for_h is None or cand["ror_pct"] > best_for_h["ror_pct"]:
                    best_for_h = cand
            if best_for_h is not None:
                eligible.append(best_for_h)
        if ti % 100 == 0:
            print(f"  {ti}/{len(tickers)}  eligible_so_far={len(eligible)}")

    print()
    print(f"Eligible (ticker, horizon) combos: {len(eligible)}")
    by_tk = defaultdict(list)
    for e in eligible:
        by_tk[e["ticker"]].append(e)
    print(f"Unique tickers: {len(by_tk)}")
    print()
    print("Top 50 by ROR:")
    print(f'{"tk":<7} {"h":<4} {"q":<6} {"bP":>6} {"bC":>6} {"WR":>6} {"n":>6} {"width":>6} {"ROR":>7}')
    eligible.sort(key=lambda r: -r["ror_pct"])
    for e in eligible[:50]:
        print(f'{e["ticker"]:<7} {e["horizon"]:<4} {e["q"]:<6} '
              f'{e["buf_put_pct"]:>5.2f}% {e["buf_call_pct"]:>5.2f}% '
              f'{e["joint_oos_wr_pct"]:>5.2f}% {e["n_oos_tests"]:>6} '
              f'{e["best_width_pct"]:>5.1f}% {e["ror_pct"]:>6.2f}%')

    out_path = os.path.join(_HERE, "results", "universal_ic_close_sweep.json")
    with open(out_path, "w") as fh:
        json.dump({
            "summary": {
                "n_tickers_processed": len(tickers),
                "n_eligible_combos": len(eligible),
                "n_unique_tickers": len(by_tk),
                "horizons": HORIZONS,
                "qs": QS,
                "widths_pct": [w * 100 for w in WIDTHS],
                "win_condition": "close-at-expiry",
                "max_combined_credit_ratio": MAX_COMBINED_CREDIT_RATIO,
                "min_width_pct": MIN_WIDTH_PCT * 100,
                "target_joint_wr_pct": TARGET_JOINT_WR * 100,
                "target_per_fold_pct": TARGET_PER_FOLD * 100,
                "target_ror_pct": TARGET_ROR * 100,
                "max_per_leg_buffer_pct": MAX_PER_LEG_BUFFER * 100,
            },
            "eligible": eligible,
        }, fh, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
