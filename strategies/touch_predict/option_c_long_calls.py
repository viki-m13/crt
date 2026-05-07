"""Option C — long-call variant.

Same event-triggered logic as the short-put credit-spread engine
(option_c_research.py), but instead of SELLING a put-credit-spread
when an oversold/bounce regime fires, we BUY an OTM call.

Why this might be interesting
-----------------------------
The short-put-spread engine collects a small credit on each fire, with
high win rate but capped upside (the credit). The mean-reversion
regimes (panic_day, multi_stack, deep_oversold, connors_tps,
spy_rel_weak, dip_in_uptrend) often precede rallies that are *much*
larger than the strike buffer, so a long call gets convex payoff on
exactly those names. A modest hit rate on a 10x-credit-multiple call
can dominate a 90% win rate at 10% credit.

What we measure
---------------
For each (regime × horizon × strike-OTM%):
  - Fires of CALL-side regimes only (oversold → expect rally).
  - At fire date t, BUY one OTM call at K = spot * (1 + k_otm).
    Premium = bs_call(spot, K, T, σ_iv) * slippage.
  - Resolve at t+h with two models:
      (M1) Hold to expiry:
           payoff = max(close[t+h] - K, 0)
           pnl    = payoff - premium
      (M2) Touch-target-and-exit:
           target = spot * (1 + buffer)  (where buffer == k_otm + 5%)
           If high path crosses target before expiry:
             realised payoff = (target - K) + (small) residual TV
                            = (target - K)  (we keep it conservative)
           Else:
             payoff = max(close[t+h] - K, 0)
  - Walk-forward by year (same FOLD_YEARS as Option C).
  - Pooled win rate, ROI on premium, % of fires that are >+100% trades.

Output
------
results/option_c_long_calls.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from v2_common import (
    FOLD_YEARS, WARMUP_DAYS,
    OhlcvSeries, V2Features,
    actual_options_expiry, compute_features, list_tickers,
    load_series, spy_context,
)
from v2_regimes import CALL_REGIMES
from pricing import bs_call


# ------------------------------------------------------------------ config

# Strike OTM grid (call premium becomes very cheap further OTM, but the
# probability of finishing ITM also drops). 1% → 25% covers the range
# from "near-the-money lottery" to "deep OTM tail bet".
K_OTM_GRID = [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]

# Horizons (sessions). Same grid as Option C — ATM weeklies all the way
# out to LEAPS-style annual, so we can see where convexity peaks.
HORIZONS = [5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 252]

# Touch-target buffer: how far above the strike we ASSUME we sell into
# strength. If stock touches `spot*(1+strike+TARGET_BUFFER)` during the
# life of the trade, we close out at that price.
TARGET_BUFFER = 0.05

IV_MULT = 1.15            # IV = realized × this (matches credit-spread engine)
PREMIUM_SLIPPAGE = 1.05   # pay 5% more than BS-mid (1.30 if BS<$0.10)
MIN_PREMIUM = 0.02        # floor — too cheap to be real fills

MIN_FIRES_PER_RULE = 100  # need this many pooled fires to publish a rule
MIN_FOLDS = 5             # need this many fold-years


# ------------------------------------------------------------------ pricing

def _premium(spot: float, k_otm: float, T_years: float, sigma: float) -> float:
    """Black-Scholes call premium (post-slippage) for an OTM call at
    K = spot * (1 + k_otm). Returns 0.0 if too cheap to fill."""
    if sigma <= 0 or T_years <= 0 or spot <= 0:
        return 0.0
    K = spot * (1.0 + k_otm)
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
    if p < MIN_PREMIUM:
        return 0.0
    return p


# ------------------------------------------------------------------ fires

@dataclass
class CallFire:
    ticker: str
    date: np.datetime64
    spot: float
    sigma: float
    close_at_expiry: float
    max_high_in_window: float   # max(high) over (t+1 .. t+h)


def _gather_call_fires(regime_name: str, regime_fn, horizon: int) -> list[CallFire]:
    """Collect (ticker, day) fires of a CALL-side regime, with the σ on
    the fire day, the close at t+h, AND the highest high reached during
    (t+1 .. t+h]. The high path is what makes touch-and-exit possible."""
    _ = spy_context()
    fires: list[CallFire] = []
    for t in list_tickers():
        s = load_series(t)
        if s is None:
            continue
        f = compute_features(s)
        try:
            mask = regime_fn(f, s.close, s.dates)
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] regime {regime_name} on {t}: {exc}", file=sys.stderr)
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
            mx = float(np.max(s.high[i + 1 : j + 1]))
            fires.append(CallFire(
                ticker=t, date=s.dates[i],
                spot=float(s.close[i]),
                sigma=float(f.rv60[i]),
                close_at_expiry=float(s.close[j]),
                max_high_in_window=mx,
            ))
    return fires


# ------------------------------------------------------------------ trade pnl

def _trade_pnl_hold(spot: float, close_at_expiry: float, k_otm: float,
                    premium: float) -> float:
    """Hold-to-expiry: payoff = max(close - K, 0) - premium."""
    K = spot * (1.0 + k_otm)
    payoff = max(close_at_expiry - K, 0.0)
    return payoff - premium


def _trade_pnl_touch(spot: float, max_high: float, close_at_expiry: float,
                     k_otm: float, premium: float,
                     target_buffer: float = TARGET_BUFFER) -> float:
    """Touch-target-and-exit:
       If stock touches spot*(1 + k_otm + target_buffer) during window,
       realize intrinsic (= target - K = target_buffer * spot) on exit.
       Else, payoff = max(close - K, 0) like hold-to-expiry.
    """
    K = spot * (1.0 + k_otm)
    target = spot * (1.0 + k_otm + target_buffer)
    if max_high >= target:
        return (target - K) - premium
    payoff = max(close_at_expiry - K, 0.0)
    return payoff - premium


# ------------------------------------------------------------------ walk-forward

@dataclass
class CallRuleResult:
    regime: str
    horizon: int
    k_otm: float
    folds: list[dict] = field(default_factory=list)
    n_fires: int = 0
    n_wins: int = 0
    n_big_wins: int = 0       # P&L >= 100% of premium
    n_huge_wins: int = 0      # P&L >= 300% of premium
    pooled_pnl: float = 0.0
    pooled_premium: float = 0.0
    avg_premium: float = 0.0
    win_rate: float = 0.0
    avg_roi_on_premium: float = 0.0
    median_pnl: float = 0.0
    p90_pnl: float = 0.0
    pnl_per_share_max: float = 0.0
    losing_folds: int = 0
    eligible: bool = False
    # Touch-and-exit metrics (computed in parallel)
    touch_n_wins: int = 0
    touch_pooled_pnl: float = 0.0
    touch_win_rate: float = 0.0
    touch_avg_roi_on_premium: float = 0.0
    n_touched_target: int = 0


def _evaluate_call_rule(regime_name: str, horizon: int, k_otm: float,
                        fires: list[CallFire]) -> CallRuleResult | None:
    if not fires:
        return None

    rr = CallRuleResult(regime=regime_name, horizon=horizon, k_otm=k_otm)
    by_year: dict[int, list[CallFire]] = {}
    for fi in fires:
        y = int(str(fi.date)[:4])
        by_year.setdefault(y, []).append(fi)

    # Calendar-day approximation: most expiries in the regimes are
    # standard 3rd-Friday monthlies (h≥21) or weeklies (h<21). We use
    # the simple session→calendar approximation T = h * 1.4 / 365.
    # The bs_call call below uses calendar T_years.
    T_years = horizon * 1.4 / 365.0

    pnl_list_hold: list[float] = []

    for year in FOLD_YEARS:
        ffires = by_year.get(year, [])
        if not ffires:
            continue
        wins = 0; big_wins = 0; huge_wins = 0
        n = 0
        pnl_sum = 0.0
        prem_sum = 0.0
        touch_wins = 0
        touch_pnl_sum = 0.0
        n_touched = 0
        for fi in ffires:
            iv = fi.sigma * IV_MULT
            premium = _premium(fi.spot, k_otm, T_years, iv)
            if premium <= 0:
                continue   # too cheap to be tradeable
            pnl_h = _trade_pnl_hold(fi.spot, fi.close_at_expiry, k_otm, premium)
            pnl_t = _trade_pnl_touch(
                fi.spot, fi.max_high_in_window, fi.close_at_expiry,
                k_otm, premium,
            )
            target = fi.spot * (1.0 + k_otm + TARGET_BUFFER)
            if fi.max_high_in_window >= target:
                n_touched += 1

            pnl_sum += pnl_h
            prem_sum += premium
            n += 1
            pnl_list_hold.append(pnl_h / max(premium, 1e-9))
            if pnl_h > 0:
                wins += 1
            if pnl_h >= premium:           # ≥100% return
                big_wins += 1
            if pnl_h >= 3 * premium:        # ≥300% return
                huge_wins += 1

            touch_pnl_sum += pnl_t
            if pnl_t > 0:
                touch_wins += 1

        if n == 0:
            continue
        rr.folds.append({
            "year": year, "n_fires": n,
            "wins": wins, "big_wins": big_wins, "huge_wins": huge_wins,
            "pnl_hold": pnl_sum,
            "pnl_touch": touch_pnl_sum,
            "premium_paid": prem_sum,
            "roi_hold": pnl_sum / prem_sum if prem_sum > 0 else 0.0,
            "roi_touch": touch_pnl_sum / prem_sum if prem_sum > 0 else 0.0,
            "n_touched_target": n_touched,
        })
        rr.n_fires += n
        rr.n_wins += wins
        rr.n_big_wins += big_wins
        rr.n_huge_wins += huge_wins
        rr.pooled_pnl += pnl_sum
        rr.pooled_premium += prem_sum
        rr.touch_n_wins += touch_wins
        rr.touch_pooled_pnl += touch_pnl_sum
        rr.n_touched_target += n_touched

    if rr.n_fires < MIN_FIRES_PER_RULE or len(rr.folds) < MIN_FOLDS:
        return None

    rr.avg_premium = rr.pooled_premium / rr.n_fires
    rr.win_rate = rr.n_wins / rr.n_fires
    rr.avg_roi_on_premium = (rr.pooled_pnl / rr.pooled_premium
                             if rr.pooled_premium > 0 else 0.0)
    rr.touch_win_rate = rr.touch_n_wins / rr.n_fires
    rr.touch_avg_roi_on_premium = (rr.touch_pooled_pnl / rr.pooled_premium
                                   if rr.pooled_premium > 0 else 0.0)
    if pnl_list_hold:
        a = np.array(pnl_list_hold, dtype=float)
        rr.median_pnl = float(np.median(a))
        rr.p90_pnl = float(np.percentile(a, 90))
        rr.pnl_per_share_max = float(np.max(a))
    rr.losing_folds = sum(1 for f in rr.folds if f["pnl_hold"] <= 0)
    # Eligible if expected ROI per trade > some threshold (we publish it
    # regardless of "win rate" — convex bets often have <50% hit rate
    # but +EV due to fat right tail).
    rr.eligible = bool(rr.avg_roi_on_premium > 0
                       and len(rr.folds) >= MIN_FOLDS)
    return rr


# ------------------------------------------------------------------ main

def main() -> int:
    t0 = time.time()
    results: list[CallRuleResult] = []
    print(f"Sweeping {len(CALL_REGIMES)} call-side regimes × "
          f"{len(HORIZONS)} horizons × {len(K_OTM_GRID)} OTM strikes…")

    for rname, rfn in CALL_REGIMES.items():
        for h in HORIZONS:
            fires = _gather_call_fires(rname, rfn, h)
            if not fires:
                continue
            for k in K_OTM_GRID:
                rr = _evaluate_call_rule(rname, h, k, fires)
                if rr is None:
                    continue
                results.append(rr)
        print(f"  {rname}: {sum(1 for r in results if r.regime==rname)} eligible rules; "
              f"elapsed {time.time()-t0:.1f}s")

    # Rank by pooled ROI on premium under hold-to-expiry — most robust
    # measure (touch-and-exit can be optimistic).
    results.sort(key=lambda r: -r.avg_roi_on_premium)

    print(f"\nFound {len(results)} eligible (regime × horizon × OTM-strike) rules\n")
    print(f"{'regime':<14} {'h':>3} {'OTM%':>5} {'fires':>5} "
          f"{'win%':>5} {'big%':>5} {'huge%':>5} "
          f"{'avgPrem':>7} {'ROI(hold)':>9} {'ROI(touch)':>10} "
          f"{'medROI':>7} {'p90ROI':>7} {'maxROI':>7} {'losYrs':>6}")
    print("-" * 130)
    for r in results[:30]:
        big_pct = r.n_big_wins / r.n_fires * 100 if r.n_fires else 0
        huge_pct = r.n_huge_wins / r.n_fires * 100 if r.n_fires else 0
        print(f"{r.regime:<14} {r.horizon:>3} {r.k_otm*100:>4.1f}% "
              f"{r.n_fires:>5} {r.win_rate*100:>4.1f} "
              f"{big_pct:>4.1f}% {huge_pct:>4.1f}% "
              f"{r.avg_premium:>7.2f} {r.avg_roi_on_premium*100:>8.1f}% "
              f"{r.touch_avg_roi_on_premium*100:>9.1f}% "
              f"{r.median_pnl*100:>6.0f}% {r.p90_pnl*100:>6.0f}% "
              f"{r.pnl_per_share_max*100:>6.0f}% {r.losing_folds:>6}")

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "horizons": HORIZONS,
            "k_otm_grid": K_OTM_GRID,
            "target_buffer": TARGET_BUFFER,
            "iv_mult": IV_MULT,
            "premium_slippage": PREMIUM_SLIPPAGE,
            "min_fires_per_rule": MIN_FIRES_PER_RULE,
            "min_folds": MIN_FOLDS,
        },
        "n_eligible_rules": len(results),
        "rules": [
            {
                "regime": r.regime,
                "horizon": r.horizon,
                "k_otm": r.k_otm,
                "n_fires": r.n_fires,
                "n_folds": len(r.folds),
                "win_rate_pct": r.win_rate * 100,
                "big_win_pct": (r.n_big_wins / r.n_fires * 100) if r.n_fires else 0,
                "huge_win_pct": (r.n_huge_wins / r.n_fires * 100) if r.n_fires else 0,
                "avg_premium_per_share": r.avg_premium,
                "pooled_pnl": r.pooled_pnl,
                "pooled_premium": r.pooled_premium,
                "avg_roi_on_premium_pct": r.avg_roi_on_premium * 100,
                "touch_avg_roi_on_premium_pct": r.touch_avg_roi_on_premium * 100,
                "touch_win_rate_pct": r.touch_win_rate * 100,
                "n_touched_target": r.n_touched_target,
                "pct_touched_target": r.n_touched_target / r.n_fires * 100,
                "median_roi_per_trade": r.median_pnl,
                "p90_roi_per_trade": r.p90_pnl,
                "max_roi_per_trade": r.pnl_per_share_max,
                "losing_fold_years": r.losing_folds,
                "folds": r.folds,
            }
            for r in results
        ],
    }
    out_path = os.path.join(_HERE, "results", "option_c_long_calls.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
