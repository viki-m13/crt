"""Calendar-spread engine on confirmed-reversal touch-prediction regimes.

Why this might clear 95%
------------------------
A long-call calendar at strike K is profitable when:
  (a) Stock at K (or near it) at short-leg expiry → max profit (long
      leg retains TV; short leg expires worthless).
  (b) Stock far below K → minor loss (both options decay; long stays
      with more TV, partially offsetting).
  (c) Stock far above K → minor loss (both options gain ~equal
      intrinsic; net change is the TV differential).

The "tent" payoff peaks at K and stays bounded on both sides,
giving very high win rates if K is placed sensibly.

For confirmed oversold-bounce setups, we expect stock to recover
~3-7% within the short leg's lifetime. Placing K at spot * (1 +
expected bounce) puts the tent peak at the most-likely outcome.

Trade structure
---------------
At fire date T0:
  Sell call at K, expires T0 + T_short days   (collect credit S)
  Buy call at K, expires T0 + T_long days     (pay premium L)
  Net debit = L - S  (always positive)

At short expiry T_short:
  Short call payoff = -max(spot[T_short] - K, 0)
  Long call value   = BS(spot[T_short], K, T_long - T_short,
                          σ at T_short)  × slippage_haircut
  Net P&L = (long_value + short_payoff) - net_debit

Win = P&L > 0.

Sweep
-----
  K placement: 1.00, 1.02, 1.05, 1.08, 1.10 × spot.
  T_short:     5, 7, 10, 14 sessions.
  T_long:      30, 45, 60, 90, 120 sessions.

For each cell we report pooled win-rate, ROI on debit, and
per-ticker.

Output: results/calendar_spread.json
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


# ----- config ---------------------------------------------------------

OVERSOLD_REGIMES = ["connors_tps", "multi_stack", "panic_day", "deep_oversold"]

K_OFFSETS  = [0.00, 0.02, 0.05, 0.08, 0.10]
T_SHORTS   = [5, 7, 10, 14]
T_LONGS    = [30, 45, 60, 90, 120]

# Slippage model on calendar entry: net debit is paid 5% above
# theoretical mid (each leg has bid-ask, calendar cost is wider).
ENTRY_SLIPPAGE = 1.05
# Exit haircut on long-leg value: collect 95% of theoretical BS value
# (selling into bid).
EXIT_HAIRCUT = 0.95

IV_MULT = 1.15
MIN_DEBIT = 0.10            # minimum net debit per share to be tradeable

MIN_FIRES_PER_CELL = 30
MIN_FOLDS_PER_CELL = 3


# ----- helpers --------------------------------------------------------

def _bs(spot, K, T, sigma):
    if T <= 0 or sigma <= 0 or spot <= 0 or K <= 0:
        return max(spot - K, 0.0)
    return bs_call(spot, K, T, sigma)


def calendar_entry_debit(spot, K, T_short_y, T_long_y, sigma):
    """Cost of entering a long-call calendar at strike K."""
    iv = sigma * IV_MULT
    long_p = _bs(spot, K, T_long_y, iv)
    short_p = _bs(spot, K, T_short_y, iv)
    debit = (long_p - short_p) * ENTRY_SLIPPAGE
    return max(debit, 0.0)


def calendar_exit_pnl(spot_at_short_expiry, K,
                      T_remaining_y, sigma_at_short_expiry,
                      net_debit):
    """P&L of unwinding the calendar at short-leg expiry."""
    iv = sigma_at_short_expiry * IV_MULT
    long_remaining = _bs(spot_at_short_expiry, K, T_remaining_y, iv) * EXIT_HAIRCUT
    short_payoff = -max(spot_at_short_expiry - K, 0.0)
    return long_remaining + short_payoff - net_debit


# ----- fires ----------------------------------------------------------

def gather_confirmed_fires():
    """Confirmed-reversal trigger from CBI-3X v2."""
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
            if not warmup[i] or not any_oversold[i - 1]:
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


# ----- main -----------------------------------------------------------

def main():
    t0 = time.time()
    fires_by_tk = gather_confirmed_fires()
    n_fires = sum(len(v) for v in fires_by_tk.values())
    print(f"[1/3] Confirmed-reversal fires: {n_fires} across "
          f"{len(fires_by_tk)} tickers ({time.time()-t0:.1f}s)")

    # Cache series + rv60 for sigma-at-T_short lookups
    series_cache = {}
    for tk in fires_by_tk:
        s = load_series(tk)
        if s is None:
            continue
        f = compute_features(s)
        series_cache[tk] = (s, f)

    by_cell = defaultdict(list)
    skipped = 0
    for tk, fires in fires_by_tk.items():
        if tk not in series_cache:
            continue
        s, f = series_cache[tk]
        n = len(s.close)
        for fi in fires:
            year = int(fi["date"][:4])
            if year not in FOLD_YEARS:
                continue
            spot0 = fi["spot"]
            sigma0 = fi["sigma"]
            for T_s in T_SHORTS:
                idx_short = fi["idx"] + T_s
                if idx_short >= n:
                    continue
                spot_s = float(s.close[idx_short])
                # Sigma at short expiry: use rv60 at that date if
                # finite, else fall back to sigma at fire date.
                sigma_s = float(f.rv60[idx_short]) if np.isfinite(f.rv60[idx_short]) else sigma0
                for T_l in T_LONGS:
                    if T_l <= T_s:
                        continue
                    T_short_y = T_s * 1.4 / 365.0
                    T_long_y  = T_l * 1.4 / 365.0
                    T_remain_y = (T_l - T_s) * 1.4 / 365.0
                    for k_off in K_OFFSETS:
                        K = spot0 * (1.0 + k_off)
                        debit = calendar_entry_debit(spot0, K, T_short_y, T_long_y, sigma0)
                        if debit < MIN_DEBIT:
                            skipped += 1
                            continue
                        pnl = calendar_exit_pnl(spot_s, K, T_remain_y, sigma_s, debit)
                        by_cell[(T_s, T_l, k_off)].append({
                            "ticker": tk, "year": year,
                            "pnl": pnl, "debit": debit,
                            "spot0": spot0, "spot_s": spot_s,
                        })

    print(f"[2/3] Resolved trades across {len(by_cell)} cells "
          f"(skipped {skipped} too-cheap; {time.time()-t0:.1f}s)")

    # Stage A: pooled
    print()
    print("=== STAGE A: Pooled universe (calendar-spread cells) ===")
    print(f"{'Tshort':>6} {'Tlong':>5} {'K_off%':>6} "
          f"{'n':>5} {'win%':>5} {'ROI%':>6} "
          f"{'avgDebit$':>9} {'avgPnL$':>8} {'maxPnL$':>8}")
    print("-" * 75)
    pooled_rows = []
    for (Ts, Tl, k), trades in sorted(by_cell.items()):
        if len(trades) < MIN_FIRES_PER_CELL:
            continue
        folds = {t["year"] for t in trades}
        if len(folds) < MIN_FOLDS_PER_CELL:
            continue
        wins = sum(1 for t in trades if t["pnl"] > 0)
        win_rate = wins / len(trades) * 100
        pnl = sum(t["pnl"] for t in trades)
        debit = sum(t["debit"] for t in trades)
        roi = pnl / debit * 100 if debit > 0 else 0
        avg_debit = debit / len(trades)
        avg_pnl = pnl / len(trades)
        max_pnl = max(t["pnl"] for t in trades)
        pooled_rows.append({
            "T_short": Ts, "T_long": Tl, "k_offset": k,
            "n": len(trades), "wins": wins,
            "win_rate_pct": win_rate, "roi_on_debit_pct": roi,
            "avg_debit": avg_debit, "avg_pnl": avg_pnl,
            "max_pnl": max_pnl,
            "n_folds": len(folds),
        })

    # Sort by win rate desc, then ROI desc
    pooled_rows.sort(key=lambda r: (-r["win_rate_pct"], -r["roi_on_debit_pct"]))
    for r in pooled_rows[:50]:
        print(f"{r['T_short']:>6} {r['T_long']:>5} "
              f"{r['k_offset']*100:>+5.1f}% "
              f"{r['n']:>5} {r['win_rate_pct']:>4.1f} "
              f"{r['roi_on_debit_pct']:>+5.1f} "
              f"{r['avg_debit']:>8.2f} {r['avg_pnl']:>+7.2f} "
              f"{r['max_pnl']:>+7.2f}")

    # Stage B: per-ticker drill-down on cells ≥ 90% pooled
    print()
    print("=== STAGE B: Per-ticker drill-down on top pooled cells ===")
    candidate = {(r["T_short"], r["T_long"], r["k_offset"])
                 for r in pooled_rows[:15] if r["win_rate_pct"] >= 80}
    print(f"{'tkr':<6} {'Ts':>3} {'Tl':>3} {'K%':>5} "
          f"{'n':>3} {'wn':>3} {'win%':>5} {'ROI%':>6} {'avg$':>7}")
    print("-" * 55)
    eligible = []
    for cell in candidate:
        trades = by_cell.get(cell, [])
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
            debit = sum(t["debit"] for t in ts)
            roi = pnl / debit * 100 if debit > 0 else 0
            avg_pnl = pnl / len(ts)
            row = {
                "ticker": tk, "T_short": cell[0], "T_long": cell[1],
                "k_offset": cell[2], "n": len(ts), "wins": wins,
                "win_rate_pct": win_rate, "roi_on_debit_pct": roi,
                "avg_pnl": avg_pnl, "n_folds": len(folds),
            }
            if win_rate >= 95:
                eligible.append(row)
                print(f"{tk:<6} {cell[0]:>3} {cell[1]:>3} "
                      f"{cell[2]*100:>+4.1f}% {len(ts):>3} {wins:>3} "
                      f"{win_rate:>4.1f} {roi:>+5.1f} {avg_pnl:>+6.2f}")

    print(f"\n[3/3] Per-ticker combos clearing 95% win-rate: {len(eligible)}")
    n_pool_95 = sum(1 for r in pooled_rows if r["win_rate_pct"] >= 95)
    print(f"Pooled cells clearing 95% win-rate: {n_pool_95}")

    out = {
        "n_total_fires": n_fires,
        "config": {
            "k_offsets": K_OFFSETS,
            "T_shorts": T_SHORTS,
            "T_longs": T_LONGS,
            "entry_slippage": ENTRY_SLIPPAGE,
            "exit_haircut": EXIT_HAIRCUT,
            "iv_mult": IV_MULT,
            "min_debit": MIN_DEBIT,
        },
        "pooled_rows": pooled_rows,
        "eligible_per_ticker_95pct": eligible,
    }
    out_path = os.path.join(_HERE, "results", "calendar_spread.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
