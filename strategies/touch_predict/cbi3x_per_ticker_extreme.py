"""CBI-3X v3: extreme deep-ITM per-ticker scan.

Last hope at the 90% win rate goal: does ANY (ticker, h, k_strike)
combo with very-deep-ITM strikes (-15% to -25%) clear 90%? At that
ITM depth, the call is essentially a stock proxy with capped
downside, so the win condition is "stock didn't crash 15-25% in <21
sessions." For LIQUID mega-caps that's a very high probability event.

Trade-off: ROI shrinks to single-digits because most of the premium
is intrinsic. We measure honestly.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from v2_common import FOLD_YEARS, WARMUP_DAYS, align_to_spy, compute_features, list_tickers, load_series, spy_context
from v2_regimes import CALL_REGIMES
from pricing import bs_call


HORIZONS = [5, 7, 10, 14, 21]
K_STRIKE_GRID = [-0.30, -0.25, -0.20, -0.15, -0.12, -0.10, -0.08, -0.05]
OVERSOLD_REGIMES = ["connors_tps", "multi_stack", "panic_day", "deep_oversold"]

# Realistic slippage for DEEP-ITM liquid options: tight relative spread
DEEP_ITM_SLIP = 1.005    # 0.5% over BS for deeply-ITM monthly options
SHALLOW_ITM_SLIP = 1.02  # 2% for shallower
IV_MULT = 1.15
MIN_PREMIUM = 0.05


def _premium(spot, k_strike, T_years, sigma):
    if sigma <= 0 or T_years <= 0 or spot <= 0:
        return 0.0
    K = spot * (1.0 + k_strike)
    if K <= 0:
        return 0.0
    bs = bs_call(spot, K, T_years, sigma)
    if bs <= 0:
        return 0.0
    if k_strike <= -0.10:
        slip = DEEP_ITM_SLIP
    else:
        slip = SHALLOW_ITM_SLIP
    p = bs * slip
    return p if p >= MIN_PREMIUM else 0.0


def gather_confirmed_fires():
    ctx = spy_context()
    if ctx is None:
        return {}
    spy_dates, spy_close, _ = ctx
    spy_r1 = np.full_like(spy_close, np.nan, dtype="float64")
    spy_r1[1:] = spy_close[1:] / spy_close[:-1] - 1.0

    fires_by_tk = {}
    for tk in list_tickers():
        s = load_series(tk)
        if s is None:
            continue
        f = compute_features(s)
        n = len(s.close)
        warmup = np.zeros(n, dtype=bool)
        warmup[WARMUP_DAYS:] = True

        any_oversold = np.zeros(n, dtype=bool)
        for rname in OVERSOLD_REGIMES:
            try:
                m = CALL_REGIMES[rname](f, s.close, s.dates)
            except Exception:
                continue
            any_oversold |= m

        spy_r1_aligned = align_to_spy(s.dates, spy_dates, spy_r1)

        fires = []
        for i in range(2, n):
            if not warmup[i]:
                continue
            if not any_oversold[i - 1]:
                continue
            if not (s.close[i] > s.close[i - 1]):
                continue
            if not np.isfinite(f.vol_z20[i]) or f.vol_z20[i] < 1.5:
                continue
            day_range = s.high[i] - s.low[i]
            if day_range <= 0:
                continue
            if s.close[i] < (s.low[i] + day_range * 0.5):
                continue
            stock_r = s.close[i] / s.close[i - 1] - 1.0
            spy_r = spy_r1_aligned[i] if np.isfinite(spy_r1_aligned[i]) else 0.0
            if spy_r < 0 and (stock_r - spy_r) < 0.02:
                continue
            if not np.isfinite(f.rv60[i]) or f.rv60[i] <= 0:
                continue
            fires.append({"idx": i, "spot": float(s.close[i]),
                          "sigma": float(f.rv60[i]),
                          "date": str(s.dates[i])})
        if fires:
            fires_by_tk[tk] = fires
    return fires_by_tk


def main():
    t0 = time.time()
    fires_by_tk = gather_confirmed_fires()
    n_total = sum(len(v) for v in fires_by_tk.values())
    print(f"[1/3] Fires: {n_total} across {len(fires_by_tk)} tickers")

    # Resolve trades, per-ticker
    per_ticker_cells = defaultdict(list)
    for tk, fires in fires_by_tk.items():
        s = load_series(tk)
        for fi in fires:
            year = int(fi["date"][:4])
            if year not in FOLD_YEARS:
                continue
            for h in HORIZONS:
                j = fi["idx"] + h
                if j >= len(s.close):
                    continue
                T = h * 1.4 / 365.0
                iv = fi["sigma"] * IV_MULT
                close_h = float(s.close[j])
                for k in K_STRIKE_GRID:
                    K = fi["spot"] * (1.0 + k)
                    if K <= 0:
                        continue
                    prem = _premium(fi["spot"], k, T, iv)
                    if prem <= 0:
                        continue
                    pnl = max(close_h - K, 0.0) - prem
                    per_ticker_cells[(tk, h, k)].append({
                        "year": year, "pnl": pnl, "premium": prem,
                    })

    print(f"[2/3] Resolved trades ({time.time()-t0:.1f}s)")

    # Filter to ≥10 trades, ≥2 folds, ≥90% win, ROI > 0
    rows = []
    for (tk, h, k), trades in per_ticker_cells.items():
        if len(trades) < 10:
            continue
        folds = {t["year"] for t in trades}
        if len(folds) < 2:
            continue
        wins = sum(1 for t in trades if t["pnl"] > 0)
        win_rate = wins / len(trades) * 100
        pnl = sum(t["pnl"] for t in trades)
        prem = sum(t["premium"] for t in trades)
        roi = pnl / prem * 100 if prem > 0 else 0
        rows.append({
            "ticker": tk, "horizon": h, "k_strike": k,
            "n_trades": len(trades), "wins": wins,
            "win_rate_pct": win_rate, "roi_on_premium_pct": roi,
            "n_folds": len(folds), "pooled_pnl": pnl, "pooled_premium": prem,
        })

    rows.sort(key=lambda r: (-r["win_rate_pct"], -r["roi_on_premium_pct"]))

    # Top 30 by win-rate
    print()
    print(f"[3/3] Per-ticker results — top 40 by win-rate")
    print(f"{'tkr':<6} {'h':>2} {'kS%':>5} {'n':>3} {'wins':>4} "
          f"{'win%':>5} {'ROI%':>6} {'fld':>3} {'avgPrem$':>9}")
    print("-" * 60)
    n_90 = 0
    n_90_pos_roi = 0
    for r in rows[:40]:
        avg_prem = r["pooled_premium"] / r["n_trades"]
        flag = " *" if r["win_rate_pct"] >= 90 and r["roi_on_premium_pct"] > 0 else ""
        if r["win_rate_pct"] >= 90:
            n_90 += 1
            if r["roi_on_premium_pct"] > 0:
                n_90_pos_roi += 1
        print(f"{r['ticker']:<6} {r['horizon']:>2} {r['k_strike']*100:>+4.1f} "
              f"{r['n_trades']:>3} {r['wins']:>4} "
              f"{r['win_rate_pct']:>4.1f} {r['roi_on_premium_pct']:>+5.1f} "
              f"{r['n_folds']:>3} {avg_prem:>8.2f}{flag}")
    print()
    n_full_90 = sum(1 for r in rows if r["win_rate_pct"] >= 90)
    n_full_90_pos = sum(1 for r in rows
                         if r["win_rate_pct"] >= 90 and r["roi_on_premium_pct"] > 0)
    print(f"Total combos with ≥90% win rate:           {n_full_90}")
    print(f"Total combos with ≥90% win + positive ROI: {n_full_90_pos}")

    out_path = os.path.join(_HERE, "results", "cbi3x_per_ticker_extreme.json")
    with open(out_path, "w") as fh:
        json.dump({"rows": rows}, fh, separators=(",", ":"))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
