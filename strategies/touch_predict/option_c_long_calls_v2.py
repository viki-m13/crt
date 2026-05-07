"""Option C — long-call variant v2 (modifications).

v1 (option_c_long_calls.py) showed:
  * connors_tps and multi_stack at 180-252d horizons deliver +50% to +95%
    ROI on premium with +30% to +56% alpha vs unconditional plain calls.
  * The "touch-and-exit at +5% above strike" rule actually HURT
    expectancy because it capped the right tail.

This v2 tries three modifications:

  M1. Bigger touch-target buffers (10%, 15%, 25%) — capture meaningful
      mid-trade rallies without clipping the long tail.
  M2. "Half-and-half" exit: sell half at touch of target, let the rest
      ride to expiry. Captures some realized P&L while preserving the
      fat-right-tail expectancy of holding.
  M3. Per-ticker breakdown for the top rules — which specific names
      drive the alpha?

Output: results/option_c_long_calls_v2.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from v2_common import FOLD_YEARS, WARMUP_DAYS, compute_features, list_tickers, load_series, spy_context
from v2_regimes import CALL_REGIMES
from pricing import bs_call


# Best-rule short-list discovered from v1 (sorted by alpha vs plain)
TOP_RULES = [
    ("connors_tps", 252, 0.20),
    ("connors_tps", 252, 0.10),
    ("connors_tps", 252, 0.05),
    ("connors_tps", 252, 0.00),
    ("multi_stack", 252, 0.10),
    ("multi_stack", 252, 0.00),
    ("multi_stack", 180, 0.10),
    ("multi_stack", 180, 0.00),
    ("connors_tps", 180, 0.10),
    ("panic_day",   252, 0.10),
    ("spy_rel_weak", 252, 0.10),
    ("deep_oversold", 252, 0.10),
]

TOUCH_BUFFERS = [0.05, 0.10, 0.15, 0.25, 0.50]   # in addition to plain hold-to-expiry
IV_MULT = 1.15
PREMIUM_SLIPPAGE = 1.05
MIN_PREMIUM = 0.02


@dataclass
class CallFire:
    ticker: str
    date: np.datetime64
    spot: float
    sigma: float
    close_at_expiry: float
    max_high_in_window: float
    second_max_high: float        # max(high) AFTER first touch of +20% — for half-half model

# ------------------------------------------------------------------ pricing

def _premium(spot: float, k_otm: float, T: float, sigma: float) -> float:
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


# ------------------------------------------------------------------ fires

def _gather_fires(regime_name: str, regime_fn, horizon: int) -> list[CallFire]:
    _ = spy_context()
    fires: list[CallFire] = []
    for tk in list_tickers():
        s = load_series(tk)
        if s is None:
            continue
        f = compute_features(s)
        try:
            mask = regime_fn(f, s.close, s.dates)
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
            mx = float(np.max(s.high[i + 1 : j + 1]))
            fires.append(CallFire(
                ticker=tk, date=s.dates[i],
                spot=float(s.close[i]),
                sigma=float(f.rv60[i]),
                close_at_expiry=float(s.close[j]),
                max_high_in_window=mx,
                second_max_high=mx,   # placeholder
            ))
    return fires


# ------------------------------------------------------------------ exit models

def _pnl_hold(spot, close_h, k_otm, premium):
    K = spot * (1.0 + k_otm)
    return max(close_h - K, 0.0) - premium

def _pnl_touch(spot, max_high, close_h, k_otm, premium, target_buffer):
    """Exit at FIRST touch of spot*(1 + k_otm + target_buffer); else hold."""
    K = spot * (1.0 + k_otm)
    target = spot * (1.0 + k_otm + target_buffer)
    if max_high >= target:
        return (target - K) - premium
    return max(close_h - K, 0.0) - premium

def _pnl_half_half(spot, max_high, close_h, k_otm, premium, target_buffer):
    """Half-and-half: sell HALF at first touch of target, let the other
    half ride to expiry. Meaningfully smoother P&L without capping the
    fat tail."""
    K = spot * (1.0 + k_otm)
    target = spot * (1.0 + k_otm + target_buffer)
    payoff_half_at_touch = (target - K) if max_high >= target else max(close_h - K, 0.0)
    payoff_half_at_expiry = max(close_h - K, 0.0)
    avg_payoff = 0.5 * payoff_half_at_touch + 0.5 * payoff_half_at_expiry
    return avg_payoff - premium


# ------------------------------------------------------------------ evaluate

def evaluate_rule(regime_name: str, horizon: int, k_otm: float):
    """Evaluate the rule under (a) hold-to-expiry, (b) touch at each
    TOUCH_BUFFERS value, (c) half-half at each TOUCH_BUFFERS value.
    Reports pooled across all FOLD_YEARS and per-ticker."""
    fires = _gather_fires(regime_name, CALL_REGIMES[regime_name], horizon)
    fires = [f for f in fires if int(str(f.date)[:4]) in FOLD_YEARS]
    if not fires:
        return None

    T_years = horizon * 1.4 / 365.0

    rows = []
    for fi in fires:
        iv = fi.sigma * IV_MULT
        prem = _premium(fi.spot, k_otm, T_years, iv)
        if prem <= 0:
            continue
        rec = {
            "ticker": fi.ticker,
            "year": int(str(fi.date)[:4]),
            "date": str(fi.date),
            "spot": fi.spot,
            "premium": prem,
            "max_high": fi.max_high_in_window,
            "close_h": fi.close_at_expiry,
            "pnl_hold": _pnl_hold(fi.spot, fi.close_at_expiry, k_otm, prem),
        }
        for tb in TOUCH_BUFFERS:
            rec[f"pnl_touch_{int(tb*100)}"] = _pnl_touch(
                fi.spot, fi.max_high_in_window, fi.close_at_expiry, k_otm, prem, tb)
            rec[f"pnl_half_{int(tb*100)}"] = _pnl_half_half(
                fi.spot, fi.max_high_in_window, fi.close_at_expiry, k_otm, prem, tb)
        rows.append(rec)

    if not rows:
        return None

    n = len(rows)
    total_prem = sum(r["premium"] for r in rows)
    summary = {
        "regime": regime_name, "horizon": horizon, "k_otm": k_otm,
        "n_fires": n, "total_premium": total_prem,
        "avg_premium": total_prem / n,
    }

    # Pooled stats per exit model
    def stats(key):
        pnls = [r[key] for r in rows]
        wins = sum(1 for p in pnls if p > 0)
        big = sum(1 for r in rows if r[key] >= r["premium"])
        huge = sum(1 for r in rows if r[key] >= 3 * r["premium"])
        a = np.array(pnls, dtype=float)
        roi_per = np.array([r[key] / max(r["premium"], 1e-9) for r in rows])
        return {
            "win_rate": wins / n,
            "big_win_pct": big / n * 100,
            "huge_win_pct": huge / n * 100,
            "pooled_pnl": float(a.sum()),
            "roi_on_premium_pct": float(a.sum() / total_prem * 100) if total_prem > 0 else 0,
            "median_roi_per_trade": float(np.median(roi_per)),
            "p90_roi_per_trade": float(np.percentile(roi_per, 90)),
            "max_roi_per_trade": float(np.max(roi_per)),
        }

    summary["hold"] = stats("pnl_hold")
    for tb in TOUCH_BUFFERS:
        summary[f"touch_{int(tb*100)}"] = stats(f"pnl_touch_{int(tb*100)}")
        summary[f"half_{int(tb*100)}"]  = stats(f"pnl_half_{int(tb*100)}")

    # Per-fold breakdown (hold-to-expiry only — that's the headline model)
    by_year = defaultdict(list)
    for r in rows:
        by_year[r["year"]].append(r)
    folds = []
    for y in sorted(by_year):
        ys = by_year[y]
        prem = sum(r["premium"] for r in ys)
        pnl = sum(r["pnl_hold"] for r in ys)
        folds.append({
            "year": y, "n": len(ys),
            "pnl_hold": pnl, "prem": prem,
            "roi_hold_pct": pnl / prem * 100 if prem > 0 else 0,
            "win_pct": sum(1 for r in ys if r["pnl_hold"] > 0) / len(ys) * 100,
        })
    summary["folds"] = folds

    # Per-ticker breakdown
    by_tk = defaultdict(list)
    for r in rows:
        by_tk[r["ticker"]].append(r)
    per_ticker = []
    for tk in sorted(by_tk):
        ts = by_tk[tk]
        prem = sum(r["premium"] for r in ts)
        pnl_h = sum(r["pnl_hold"] for r in ts)
        per_ticker.append({
            "ticker": tk, "n": len(ts),
            "pnl_hold": pnl_h, "prem": prem,
            "roi_hold_pct": pnl_h / prem * 100 if prem > 0 else 0,
            "win_pct": sum(1 for r in ts if r["pnl_hold"] > 0) / len(ts) * 100,
            "max_pnl_dollars": max(r["pnl_hold"] for r in ts),
        })
    per_ticker.sort(key=lambda x: -x["roi_hold_pct"])
    summary["per_ticker"] = per_ticker

    return summary


# ------------------------------------------------------------------ main

def main() -> int:
    t0 = time.time()
    out_rules = []
    print(f"Evaluating {len(TOP_RULES)} top rules with multiple exit models…")
    for (regime, h, k_otm) in TOP_RULES:
        rr = evaluate_rule(regime, h, k_otm)
        if rr is None:
            continue
        out_rules.append(rr)
        print(f"  {regime:<14} h={h:>3} OTM={k_otm*100:>4.1f}%  "
              f"n={rr['n_fires']:>4}  hold ROI={rr['hold']['roi_on_premium_pct']:>+5.1f}%  "
              f"  elapsed {time.time()-t0:.1f}s")

    print()
    print("EXIT-MODEL COMPARISON (per top-rule, ROI on premium):")
    print(f"{'rule':<35} {'hold':>7} "
          f"{'touch5':>7} {'touch10':>7} {'touch15':>7} {'touch25':>7} "
          f"{'half10':>7} {'half15':>7} {'half25':>7}")
    print("-" * 105)
    for r in out_rules:
        head = f"{r['regime']}-h{r['horizon']}-otm{int(r['k_otm']*100)}%"
        print(f"{head:<35} {r['hold']['roi_on_premium_pct']:>6.1f}% "
              f"{r['touch_5']['roi_on_premium_pct']:>6.1f}% "
              f"{r['touch_10']['roi_on_premium_pct']:>6.1f}% "
              f"{r['touch_15']['roi_on_premium_pct']:>6.1f}% "
              f"{r['touch_25']['roi_on_premium_pct']:>6.1f}% "
              f"{r['half_10']['roi_on_premium_pct']:>6.1f}% "
              f"{r['half_15']['roi_on_premium_pct']:>6.1f}% "
              f"{r['half_25']['roi_on_premium_pct']:>6.1f}%")

    # Print the best rule's fold-by-fold story
    best = max(out_rules, key=lambda r: r['hold']['roi_on_premium_pct'])
    print(f"\n=== BEST RULE WALK-FORWARD: {best['regime']} h={best['horizon']} otm={best['k_otm']*100:.0f}% ===")
    print(f"{'year':>5} {'n':>5} {'roi%':>7} {'win%':>6} {'pnl':>10} {'prem':>10}")
    for f in best['folds']:
        print(f"{f['year']:>5} {f['n']:>5} {f['roi_hold_pct']:>+6.1f}% "
              f"{f['win_pct']:>5.1f}% {f['pnl_hold']:>9.1f} {f['prem']:>9.1f}")

    # Top tickers under best rule
    print(f"\nTop tickers (by ROI on premium) for {best['regime']} h={best['horizon']} otm={best['k_otm']*100:.0f}%:")
    print(f"{'ticker':<6} {'n':>4} {'roi%':>7} {'win%':>6} {'maxPnL$':>10}")
    for t in best['per_ticker'][:15]:
        print(f"{t['ticker']:<6} {t['n']:>4} {t['roi_hold_pct']:>+6.1f}% "
              f"{t['win_pct']:>5.1f}% {t['max_pnl_dollars']:>9.2f}")

    out_path = os.path.join(_HERE, "results", "option_c_long_calls_v2.json")
    with open(out_path, "w") as fh:
        json.dump({"rules": out_rules,
                   "config": {"top_rules": TOP_RULES,
                              "touch_buffers": TOUCH_BUFFERS,
                              "iv_mult": IV_MULT}}, fh, separators=(",", ":"))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
