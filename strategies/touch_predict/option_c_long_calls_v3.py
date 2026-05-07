"""Option C — long-call v3: macro-gated.

The v2 walk-forward showed 2021 -47% and 2022 -94% on the best rule
(connors_tps h=252 otm=20%). Both were "bear after the regime fired"
periods. Layer in the macro gate Option C uses for credit spreads:

    SPY above its 200-day SMA on the FIRE DATE.

Hypothesis: requiring the broader market to be in a bullish regime
filters out fires from inside a sustained bear, which is when even
deep-mean-reversion setups fail (the rally never comes).

Compares gated vs. ungated for the top rule.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from v2_common import FOLD_YEARS, WARMUP_DAYS, compute_features, list_tickers, load_series, spy_context
from v2_regimes import CALL_REGIMES
from pricing import bs_call

IV_MULT = 1.15
PREMIUM_SLIPPAGE = 1.05
MIN_PREMIUM = 0.02

# Same shortlist as v2 — these are the rules with strongest alpha
RULES = [
    ("connors_tps", 252, 0.20),
    ("connors_tps", 252, 0.10),
    ("connors_tps", 252, 0.05),
    ("connors_tps", 252, 0.00),
    ("multi_stack", 252, 0.10),
    ("multi_stack", 180, 0.10),
    ("multi_stack", 180, 0.00),
    ("connors_tps", 180, 0.10),
]


def _premium(spot, k_otm, T, sigma):
    if sigma <= 0 or T <= 0 or spot <= 0:
        return 0.0
    K = spot * (1.0 + k_otm)
    bs = bs_call(spot, K, T, sigma)
    if bs <= 0:
        return 0.0
    if bs < 0.10:
        slip = 1.30
    elif bs < 0.25:
        slip = 1.15
    else:
        slip = PREMIUM_SLIPPAGE
    p = bs * slip
    return p if p >= MIN_PREMIUM else 0.0


def _spy_above_sma200_array():
    """Return (dates, bool array) where True = SPY[t] >= 200-day SMA at t."""
    ctx = spy_context()
    if ctx is None:
        return None, None
    spy_dates, spy_close, _ = ctx
    n = len(spy_close)
    out = np.zeros(n, dtype=bool)
    if n < 200:
        return spy_dates, out
    csum = np.cumsum(spy_close, dtype="float64")
    csum = np.concatenate(([0.0], csum))
    sma = (csum[200:] - csum[:-200]) / 200.0
    out[199:] = spy_close[199:] >= sma
    return spy_dates, out


def _spy_lookup(spy_dates, spy_above, when):
    idx = int(np.searchsorted(spy_dates, when))
    if idx >= len(spy_dates):
        return False
    return bool(spy_above[idx])


def evaluate(regime_name, horizon, k_otm, gated):
    spy_dates, spy_above = _spy_above_sma200_array()
    rfn = CALL_REGIMES[regime_name]
    fires_rows = []
    T_years = horizon * 1.4 / 365.0

    for tk in list_tickers():
        s = load_series(tk)
        if s is None:
            continue
        f = compute_features(s)
        try:
            mask = rfn(f, s.close, s.dates)
        except Exception:
            continue
        warmup = np.zeros(len(s.dates), dtype=bool)
        warmup[WARMUP_DAYS:] = True
        valid = mask & warmup & np.isfinite(f.rv60)
        if not valid.any():
            continue
        idxs = np.where(valid)[0]
        n = len(s.close)
        for i in idxs:
            j = i + horizon
            if j >= n:
                continue
            year = int(str(s.dates[i])[:4])
            if year not in FOLD_YEARS:
                continue
            if gated and not _spy_lookup(spy_dates, spy_above, s.dates[i]):
                continue
            spot = float(s.close[i])
            sig = float(f.rv60[i])
            prem = _premium(spot, k_otm, T_years, sig * IV_MULT)
            if prem <= 0:
                continue
            close_h = float(s.close[j])
            K = spot * (1.0 + k_otm)
            pnl = max(close_h - K, 0.0) - prem
            fires_rows.append({
                "year": year, "ticker": tk,
                "premium": prem, "pnl_hold": pnl,
            })

    if not fires_rows:
        return None
    n = len(fires_rows)
    total_prem = sum(r["premium"] for r in fires_rows)
    pnl = sum(r["pnl_hold"] for r in fires_rows)
    wins = sum(1 for r in fires_rows if r["pnl_hold"] > 0)

    by_year = defaultdict(list)
    for r in fires_rows:
        by_year[r["year"]].append(r)
    folds = []
    for y in sorted(by_year):
        ys = by_year[y]
        prem = sum(r["premium"] for r in ys)
        p = sum(r["pnl_hold"] for r in ys)
        folds.append({"year": y, "n": len(ys),
                      "roi": p / prem * 100 if prem > 0 else 0,
                      "win": sum(1 for r in ys if r["pnl_hold"] > 0) / len(ys) * 100,
                      "pnl": p, "prem": prem})
    return {
        "n_fires": n,
        "win_rate_pct": wins / n * 100,
        "roi_on_premium_pct": pnl / total_prem * 100 if total_prem > 0 else 0,
        "pooled_pnl": pnl,
        "pooled_premium": total_prem,
        "folds": folds,
    }


def main():
    print("=== Macro-gate (SPY≥200SMA on fire date) impact on best long-call rules ===\n")
    print(f"{'rule':<35} {'fires':>11}  {'ROI hold':>20}  {'losing fold yrs':>20}")
    print(f"{'':<35} {'ungated/gtd':>11}  {'ungated  /  gated':>20}  {'ungated /  gated':>20}")
    print('-' * 100)
    out = []
    for (regime, h, k_otm) in RULES:
        u = evaluate(regime, h, k_otm, gated=False)
        g = evaluate(regime, h, k_otm, gated=True)
        if u is None or g is None:
            continue
        u_los = sum(1 for f in u["folds"] if f["roi"] <= 0)
        g_los = sum(1 for f in g["folds"] if f["roi"] <= 0)
        head = f"{regime}-h{h}-otm{int(k_otm*100)}%"
        print(f"{head:<35} {u['n_fires']:>5}/{g['n_fires']:>5}   "
              f"{u['roi_on_premium_pct']:>+6.1f}%  /  {g['roi_on_premium_pct']:>+6.1f}%   "
              f"{u_los:>4}/7    /  {g_los:>4}/{len(g['folds'])}")
        out.append({"regime": regime, "horizon": h, "k_otm": k_otm,
                    "ungated": u, "gated": g})

    # Show fold-by-fold for the top rule
    if out:
        best = max(out, key=lambda r: r["ungated"]["roi_on_premium_pct"])
        print(f"\n=== {best['regime']} h={best['horizon']} otm={best['k_otm']*100:.0f}% — fold-by-fold ===")
        print(f"{'year':>5} {'ungated ROI':>13} {'ungated win':>13} {'gated ROI':>11} {'gated win':>11}")
        # Build joined view
        gated_by_year = {f["year"]: f for f in best["gated"]["folds"]}
        for f in best["ungated"]["folds"]:
            g = gated_by_year.get(f["year"])
            if g is None:
                print(f"{f['year']:>5} {f['roi']:>+12.1f}% {f['win']:>12.1f}% "
                      f"{'(no fires)':>11} {'-':>11}")
            else:
                print(f"{f['year']:>5} {f['roi']:>+12.1f}% {f['win']:>12.1f}% "
                      f"{g['roi']:>+10.1f}% {g['win']:>10.1f}%")

    out_path = os.path.join(_HERE, "results", "option_c_long_calls_v3.json")
    with open(out_path, "w") as fh:
        json.dump({"rules": out}, fh, separators=(",", ":"))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
