"""CBI-3X v2: "Confirmed Reversal" entry on oversold setups.

Pivot from v1: the per-ticker conformal floor on min-low blocked
everything because oversold stocks legitimately drop more before they
bounce. Instead of trying to bound the drawdown, we WAIT for the
bounce to confirm before firing.

Five filters (all must hold):

  1. PRIOR-DAY OVERSOLD FIRE.
     Yesterday (T-1) at least one of {connors_tps, multi_stack,
     panic_day, deep_oversold} fired.

  2. TODAY: GREEN-DAY CONFIRMATION.
     Today's close > yesterday's close (the bounce starts).

  3. TODAY: VOLUME-CONFIRMED.
     Today's volume > 1.5 × 20-day average (institutional money
     coming in, not just noise).

  4. TODAY: STRONG INTRADAY CLOSE.
     Today's close > today's intraday midpoint (low + range/2).
     Stock closed in upper half of its day's range — buyers won
     the session.

  5. TODAY: SPY UP, OR STOCK STRENGTH.
     Either SPY closed positive today OR the stock outperformed SPY
     by ≥ 2%. Avoids "stock-specific bounce on a market-down day"
     setups which often fail.

Trade: buy a 3%-ITM call at today's close, expiring 5/7/10/14 sessions
out. Hold to expiry — we already waited for the bounce, no need for a
fancy exit ladder.

Win condition: pnl > 0. We measure pooled win rate by (h, k_strike).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from v2_common import (FOLD_YEARS, WARMUP_DAYS, align_to_spy, compute_features,
                       list_tickers, load_series, spy_context)
from v2_regimes import CALL_REGIMES
from pricing import bs_call


# Short-dated only
HORIZONS = [5, 7, 10, 14, 21]
K_STRIKE_GRID = [-0.10, -0.07, -0.05, -0.03, -0.02, -0.01, 0.00, +0.01, +0.02]

OVERSOLD_REGIMES = ["connors_tps", "multi_stack", "panic_day", "deep_oversold"]

VOLUME_Z_THRESHOLD = 1.5
SPY_OR_OUTPERF_THRESHOLD = 0.02       # stock − SPY return must clear +2% if SPY is red

IV_MULT = 1.15
PREMIUM_SLIPPAGE = 1.05
MIN_PREMIUM = 0.02

MIN_FIRES_PER_CELL = 30
MIN_FOLDS_PER_CELL = 3


def _premium(spot, k_strike_frac, T_years, sigma):
    if sigma <= 0 or T_years <= 0 or spot <= 0:
        return 0.0
    K = spot * (1.0 + k_strike_frac)
    if K <= 0:
        return 0.0
    bs = bs_call(spot, K, T_years, sigma)
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


def gather_confirmed_fires():
    """For each ticker, find every (T) where filters 1–5 hold."""
    ctx = spy_context()
    if ctx is None:
        return {}
    spy_dates, spy_close, _ = ctx
    spy_ret_1d = np.full_like(spy_close, np.nan, dtype="float64")
    spy_ret_1d[1:] = spy_close[1:] / spy_close[:-1] - 1.0

    fires_by_tk = {}
    for tk in list_tickers():
        s = load_series(tk)
        if s is None:
            continue
        f = compute_features(s)
        n = len(s.close)
        warmup = np.zeros(n, dtype=bool)
        warmup[WARMUP_DAYS:] = True

        # Per-day, did ANY oversold regime fire?
        any_oversold = np.zeros(n, dtype=bool)
        for rname in OVERSOLD_REGIMES:
            rfn = CALL_REGIMES[rname]
            try:
                m = rfn(f, s.close, s.dates)
            except Exception:
                continue
            any_oversold = any_oversold | m

        # Per-day SPY return aligned
        spy_r1_aligned = align_to_spy(s.dates, spy_dates,
                                       np.concatenate(([0.0], spy_close[1:]/spy_close[:-1] - 1.0)))
        # Note: spy_close lookup of 1d return uses align_to_spy convention

        fires = []
        for i in range(2, n):
            if not warmup[i]:
                continue
            if not any_oversold[i - 1]:
                continue
            # Today: green-day confirmation
            if not (s.close[i] > s.close[i - 1]):
                continue
            # Volume-confirmed
            if not np.isfinite(f.vol_z20[i]) or f.vol_z20[i] < VOLUME_Z_THRESHOLD:
                continue
            # Strong intraday close (upper half of range)
            day_range = s.high[i] - s.low[i]
            if day_range <= 0:
                continue
            if s.close[i] < (s.low[i] + day_range * 0.5):
                continue
            # SPY direction or outperf
            stock_r = s.close[i] / s.close[i - 1] - 1.0
            spy_r = spy_r1_aligned[i] if np.isfinite(spy_r1_aligned[i]) else 0.0
            if spy_r < 0 and (stock_r - spy_r) < SPY_OR_OUTPERF_THRESHOLD:
                continue
            if not np.isfinite(f.rv60[i]) or f.rv60[i] <= 0:
                continue
            fires.append({
                "ticker": tk, "idx": i, "date": str(s.dates[i]),
                "spot": float(s.close[i]),
                "sigma": float(f.rv60[i]),
            })
        if fires:
            fires_by_tk[tk] = fires
    return fires_by_tk


def main():
    t0 = time.time()
    fires_by_tk = gather_confirmed_fires()
    n_total = sum(len(v) for v in fires_by_tk.values())
    print(f"[1/3] Confirmed-reversal fires: {n_total} across "
          f"{len(fires_by_tk)} tickers ({time.time()-t0:.1f}s)")

    # Resolve every (h, k_strike, fire) trade
    by_cell = defaultdict(list)
    for tk, fires in fires_by_tk.items():
        s = load_series(tk)
        if s is None:
            continue
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
                for k_strike in K_STRIKE_GRID:
                    K = fi["spot"] * (1.0 + k_strike)
                    if K <= 0:
                        continue
                    prem = _premium(fi["spot"], k_strike, T, iv)
                    if prem <= 0:
                        continue
                    pnl = max(close_h - K, 0.0) - prem
                    by_cell[(h, k_strike)].append({
                        "ticker": tk, "year": year,
                        "pnl": pnl, "premium": prem,
                    })

    print(f"[2/3] Resolved trades across {len(by_cell)} cells "
          f"({time.time()-t0:.1f}s)")

    # Stage A: pooled
    print()
    print("=== STAGE A: Pooled universe ===")
    print(f"{'h':>2} {'kS%':>5} {'n':>4} {'wins':>4} {'win%':>5} "
          f"{'ROI%':>6} {'avgPrem$':>9} {'pooledPnl$':>11}")
    print("-" * 60)
    pooled_rows = []
    for (h, k), trades in sorted(by_cell.items()):
        if len(trades) < MIN_FIRES_PER_CELL:
            continue
        folds = {t["year"] for t in trades}
        if len(folds) < MIN_FOLDS_PER_CELL:
            continue
        wins = sum(1 for t in trades if t["pnl"] > 0)
        win_rate = wins / len(trades) * 100
        pnl = sum(t["pnl"] for t in trades)
        prem = sum(t["premium"] for t in trades)
        roi = pnl / prem * 100 if prem > 0 else 0
        avg_prem = prem / len(trades)
        print(f"{h:>2} {k*100:>+4.1f} {len(trades):>4} {wins:>4} "
              f"{win_rate:>4.1f} {roi:>+5.1f} {avg_prem:>8.2f} {pnl:>10.2f}")
        pooled_rows.append({
            "horizon": h, "k_strike": k, "n_trades": len(trades),
            "win_rate_pct": win_rate, "roi_on_premium_pct": roi,
            "avg_premium": avg_prem,
        })

    # Stage B: per-ticker for cells that hit ≥ 80% pooled
    print()
    print("=== STAGE B: Per-ticker drill-down (cells ≥80% pooled win-rate) ===")
    candidate_cells = {(r["horizon"], r["k_strike"])
                       for r in pooled_rows if r["win_rate_pct"] >= 80.0}
    print(f"{'tkr':<6} {'h':>2} {'kS%':>5} {'n':>3} {'wins':>4} "
          f"{'win%':>5} {'ROI%':>6} {'fld':>3}")
    print("-" * 50)
    eligible_combos = []
    for (h, k), trades in sorted(by_cell.items()):
        if (h, k) not in candidate_cells:
            continue
        by_tk = defaultdict(list)
        for t in trades:
            by_tk[t["ticker"]].append(t)
        for tk, ts in by_tk.items():
            if len(ts) < 10:
                continue
            folds = {t["year"] for t in ts}
            if len(folds) < 2:
                continue
            wins = sum(1 for t in ts if t["pnl"] > 0)
            win_rate = wins / len(ts) * 100
            pnl = sum(t["pnl"] for t in ts)
            prem = sum(t["premium"] for t in ts)
            roi = pnl / prem * 100 if prem > 0 else 0
            if win_rate >= 90.0 and roi > 0:
                row = {
                    "ticker": tk, "horizon": h, "k_strike": k,
                    "n": len(ts), "wins": wins, "win_rate_pct": win_rate,
                    "roi_on_premium_pct": roi, "n_folds": len(folds),
                }
                eligible_combos.append(row)
                print(f"{tk:<6} {h:>2} {k*100:>+4.1f} {len(ts):>3} {wins:>4} "
                      f"{win_rate:>4.1f} {roi:>+5.1f} {len(folds):>3}")

    # Save
    out = {
        "n_total_fires": n_total,
        "pooled_rows": pooled_rows,
        "eligible_combos_per_ticker": eligible_combos,
    }
    out_path = os.path.join(_HERE, "results", "cbi3x_confirmed.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"\n{len(eligible_combos)} per-ticker combos clear 90% win rate AND positive ROI")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
