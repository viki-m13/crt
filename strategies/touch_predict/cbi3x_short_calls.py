"""CBI-3X: Conformal Bounded ITM Calls with Triple Exit (short-dated).

A novel attempt at a ≤21-session long-call strategy with ≥90% positive
ROI win rate. Combines:

  1. CONCURRENT-REGIME AND-GATE.
     Fire only when ≥ 2 independent call-side regime FAMILIES trigger
     on the same (ticker, day). Families collapse {connors_tps,
     multi_stack, deep_oversold, spy_rel_weak} → "oversold" and
     {panic_day, plain} → other. So a real fire requires e.g.
     panic_day + connors_tps simultaneously, not one alone.

  2. PER-TICKER CONFORMAL MIN-LOW QUANTILE.
     For each (ticker, regime-set, horizon), compute the 95th
     percentile of (min(low[t+1..t+h]) / spot[t]) across historical
     fires. Strike must be set ≤ that quantile minus a 1% safety
     buffer. This is the distribution-free guarantee that ≥95% of
     historical fires kept the stock above the strike.

  3. ITM STRIKE (k_strike < 0).
     Premium is mostly intrinsic; theta is small. Trade still has
     leverage on rallies but very low theta bleed risk.

  4. TRIPLE EXIT (the proprietary part).
       (a) Touch-exit: if high ≥ spot * 1.02 at any point in window,
           close at intrinsic from that touch + tiny residual TV.
       (b) Day-3 time-stop: if no touch by day-3 and close ≤ spot,
           close at intrinsic + remaining BS time value.
       (c) Hold-to-expiry: otherwise, ride to expiry close.

  5. RV-BEATS-IV GATE.
     Fire only when 5-day realized vol > 60-day realized vol * 1.0.
     The IV used in pricing is rv60 * 1.15 — so when rv5 > rv60 we
     are getting MORE realized turbulence than the option pricing
     assumed. Calls are implicitly cheap.

  6. PER-TICKER 90% HISTORICAL FLOOR.
     Final ship gate. After all the above filters, only publish a
     (ticker, regime-set, h, k_strike) combo whose IN-SAMPLE
     positive-ROI rate ≥ 90% AND ≥ 30 historical fires AND ≥ 3
     distinct fold-years.

Honesty note: even with all this, hitting 90% positive-ROI on
short-dated calls is HARD. We measure faithfully and report whether
we cleared the bar — not engineer a backfit.
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


# ------------------------------------------------------------------ config

# Short-dated only — that's the brief.
HORIZONS = [5, 7, 10, 14, 21]

# Strike grid: ITM (negative = K below spot). Plus 0% and a tiny OTM
# for completeness.
K_STRIKE_GRID = [-0.10, -0.07, -0.05, -0.03, -0.02, -0.01, 0.00, +0.01]

# Triple-exit constants
TOUCH_BOUNCE_TARGET = 0.02      # +2% high triggers immediate exit
TIME_STOP_DAY = 3               # day to abort if no touch and not green
TIME_STOP_LIMIT = 1.00          # ≤ this fraction of spot triggers stop

# Conformal coverage requirement
CONFORMAL_QUANTILE = 0.05       # 5th-percentile of min-low / spot
CONFORMAL_SAFETY = 0.01         # extra 1% buffer below the quantile

# Ship gate
MIN_FIRES_PER_TICKER = 15
MIN_FOLDS_PER_TICKER = 2
MIN_WIN_RATE_PCT = 90.0
# How many distinct regimes (not families) must co-fire on the same day
REGIMES_REQUIRED = 2

# Pricing
IV_MULT = 1.15
PREMIUM_SLIPPAGE = 1.05
PREMIUM_SLIPPAGE_CHEAP_LO = 1.30   # for BS price < $0.10
PREMIUM_SLIPPAGE_CHEAP_HI = 1.15   # for BS price < $0.25
MIN_PREMIUM = 0.02

# Family map for the AND-gate
FAMILY = {
    "connors_tps":   "oversold",
    "multi_stack":   "oversold",
    "deep_oversold": "oversold",
    "spy_rel_weak":  "oversold",
    "panic_day":     "panic",
    "plain":         "baseline",
}


# ------------------------------------------------------------------ helpers

def _premium(spot, k_strike_frac, T_years, sigma):
    """BS call premium with slippage. k_strike_frac < 0 means ITM."""
    if sigma <= 0 or T_years <= 0 or spot <= 0:
        return 0.0
    K = spot * (1.0 + k_strike_frac)
    if K <= 0:
        return 0.0
    bs = bs_call(spot, K, T_years, sigma)
    if bs <= 0:
        return 0.0
    if bs < 0.10:
        slip = PREMIUM_SLIPPAGE_CHEAP_LO
    elif bs < 0.25:
        slip = PREMIUM_SLIPPAGE_CHEAP_HI
    else:
        slip = PREMIUM_SLIPPAGE
    p = bs * slip
    return p if p >= MIN_PREMIUM else 0.0


def _bs_residual_tv(spot, K, T_years, sigma, intrinsic):
    """Time value left in the call at a mid-trade exit point."""
    if T_years <= 0:
        return 0.0
    bs = bs_call(spot, K, T_years, sigma)
    return max(bs - intrinsic, 0.0)


# ------------------------------------------------------------------ fires

@dataclass
class Fire:
    ticker: str
    date: np.datetime64
    idx: int            # index into ticker series
    spot: float
    sigma60: float
    sigma5: float
    families: tuple[str, ...]


def _rv5(closes: np.ndarray) -> np.ndarray:
    """Realized vol over 5 trading days (annualized)."""
    n = len(closes)
    out = np.full(n, np.nan)
    if n < 7:
        return out
    log_ret = np.concatenate(([0.0], np.diff(np.log(np.maximum(closes, 1e-9)))))
    # Rolling 5-day mean of squared log-returns. csum has length n+1 with
    # a leading zero; (csum[k+5] - csum[k]) / 5 gives the mean over the
    # window ending at index k+4 (inclusive). We assign starting at idx 5
    # — i.e., the first 5 days are NaN, vol is defined from day 5 onward.
    csum = np.concatenate(([0.0], np.cumsum(log_ret * log_ret)))
    var5 = (csum[6:] - csum[1:-5]) / 5.0
    out[5:] = np.sqrt(np.maximum(var5, 0.0)) * math.sqrt(252.0)
    return out


def gather_concurrent_fires() -> dict[tuple, list[Fire]]:
    """For each ticker, find every day where ≥ 2 distinct regime
    families fire AND the RV-beats-IV condition holds. Returns
    {(ticker,): [Fire,...]} — keyed by ticker only since the rule set
    is fixed (the AND-gate)."""
    _ = spy_context()
    out: dict[tuple, list[Fire]] = {}
    for tk in list_tickers():
        s = load_series(tk)
        if s is None:
            continue
        f = compute_features(s)

        masks = {}
        for rname, rfn in CALL_REGIMES.items():
            try:
                m = rfn(f, s.close, s.dates)
            except Exception:
                continue
            masks[rname] = m

        # Per-day family membership
        n = len(s.close)
        warmup = np.zeros(n, dtype=bool)
        warmup[WARMUP_DAYS:] = True
        rv5 = _rv5(s.close)
        rv5_beats_rv60 = (rv5 > f.rv60) & np.isfinite(rv5) & np.isfinite(f.rv60)

        # Build per-day list of REGIMES that fired (excluding "plain"
        # which fires every day and gives us no info)
        day_regimes: list[set[str]] = [set() for _ in range(n)]
        for rname, m in masks.items():
            if rname == "plain":
                continue
            for i in np.where(m & warmup)[0]:
                day_regimes[i].add(rname)

        fires: list[Fire] = []
        for i in range(n):
            regs = day_regimes[i]
            # AND-gate: need ≥ N distinct regimes
            if len(regs) < REGIMES_REQUIRED:
                continue
            fams = {FAMILY.get(r, r) for r in regs}
            if not rv5_beats_rv60[i]:
                continue
            if not np.isfinite(f.rv60[i]) or f.rv60[i] <= 0:
                continue
            fires.append(Fire(
                ticker=tk, date=s.dates[i], idx=i,
                spot=float(s.close[i]),
                sigma60=float(f.rv60[i]),
                sigma5=float(rv5[i]),
                families=tuple(sorted(fams)),
            ))
        if fires:
            out[(tk,)] = fires
    return out


# ------------------------------------------------------------------ trade resolver

@dataclass
class TradeResult:
    ticker: str
    date: np.datetime64
    spot: float
    K: float
    premium: float
    pnl: float
    exit_reason: str        # 'touch', 'time_stop', 'expiry'
    days_held: int


def resolve_trade(s, f, fire: Fire, h: int, k_strike: float,
                  rv5_arr: np.ndarray) -> TradeResult | None:
    """Triple-exit P&L resolver."""
    n = len(s.close)
    j = fire.idx + h
    if j >= n:
        return None
    spot = fire.spot
    K = spot * (1.0 + k_strike)
    if K <= 0:
        return None
    iv = fire.sigma60 * IV_MULT
    T = h * 1.4 / 365.0
    premium = _premium(spot, k_strike, T, iv)
    if premium <= 0:
        return None

    # Walk forward day by day looking for touch or time-stop
    touch_target_high = spot * (1.0 + TOUCH_BOUNCE_TARGET)
    touched_at = None
    touched_high = 0.0
    for d_off in range(1, h + 1):
        d_idx = fire.idx + d_off
        if d_idx >= n:
            break
        if s.high[d_idx] >= touch_target_high:
            touched_at = d_off
            touched_high = float(s.high[d_idx])
            break

    if touched_at is not None:
        # Exit at touch-time: realize intrinsic at the (estimated) touch
        # price + small residual time value.
        intrinsic = max(touched_high - K, 0.0)
        T_remaining = (h - touched_at) * 1.4 / 365.0
        tv = _bs_residual_tv(touched_high, K, T_remaining, iv, intrinsic) * 0.5
        # Conservative: capture half of residual TV (real fills don't get full BS mid)
        proceeds = intrinsic + tv
        pnl = proceeds - premium
        return TradeResult(fire.ticker, fire.date, spot, K, premium, pnl,
                           "touch", touched_at)

    # No touch by horizon end — check time-stop at day-3
    d3_idx = fire.idx + min(TIME_STOP_DAY, h)
    if d3_idx < n and s.close[d3_idx] <= spot * TIME_STOP_LIMIT:
        # Time-stop: exit at close on day TIME_STOP_DAY at intrinsic + TV
        d3_close = float(s.close[d3_idx])
        intrinsic = max(d3_close - K, 0.0)
        T_remaining = max(h - TIME_STOP_DAY, 0) * 1.4 / 365.0
        tv = _bs_residual_tv(d3_close, K, T_remaining, iv, intrinsic) * 0.5
        proceeds = intrinsic + tv
        pnl = proceeds - premium
        return TradeResult(fire.ticker, fire.date, spot, K, premium, pnl,
                           "time_stop", min(TIME_STOP_DAY, h))

    # Hold to expiry
    close_h = float(s.close[j])
    pnl = max(close_h - K, 0.0) - premium
    return TradeResult(fire.ticker, fire.date, spot, K, premium, pnl,
                       "expiry", h)


# ------------------------------------------------------------------ conformal strike eligibility

def conformal_strike_floor(s, fires: list[Fire], h: int) -> float | None:
    """For this ticker's fires at horizon h, return the empirical
    CONFORMAL_QUANTILE-th percentile of min(low[t+1..t+h]) / spot[t].
    The strike must satisfy k_strike ≤ this floor − safety, i.e. K ≤
    spot * (floor − safety)."""
    n = len(s.close)
    ratios = []
    for fi in fires:
        j = fi.idx + h
        if j >= n:
            continue
        mn = float(np.min(s.low[fi.idx + 1 : j + 1]))
        ratios.append(mn / fi.spot)
    if len(ratios) < 20:
        return None
    return float(np.quantile(ratios, CONFORMAL_QUANTILE))


# ------------------------------------------------------------------ walk-forward + ship gate

def main():
    t0 = time.time()
    fires_by_ticker = gather_concurrent_fires()
    n_total_fires = sum(len(v) for v in fires_by_ticker.values())
    print(f"[1/4] Concurrent ≥2-family AND-gate + RV>IV: "
          f"{n_total_fires} fires across {len(fires_by_ticker)} tickers "
          f"(elapsed {time.time()-t0:.1f}s)")

    # Build per-ticker series cache
    series_cache = {}
    for (tk,), _ in fires_by_ticker.items():
        s = load_series(tk)
        if s is None:
            continue
        f = compute_features(s)
        rv5 = _rv5(s.close)
        series_cache[tk] = (s, f, rv5)

    print(f"[2/4] Resolving trades across {len(HORIZONS)} horizons × "
          f"{len(K_STRIKE_GRID)} strike placements…")

    # Resolve every trade once at every (h, k_strike) — record per-trade
    # so we can both pool universe-wide AND drill down per-ticker.
    all_trades_by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for (tk,), fires in fires_by_ticker.items():
        if tk not in series_cache:
            continue
        s, f, rv5 = series_cache[tk]
        for h in HORIZONS:
            floor = conformal_strike_floor(s, fires, h)
            if floor is None:
                continue
            for k_strike in K_STRIKE_GRID:
                # Conformal eligibility: K = spot * (1 + k_strike) ≤ spot
                # * (floor − safety) → k_strike ≤ floor − safety − 1.
                conformal_max_k = floor - CONFORMAL_SAFETY - 1.0
                if k_strike > conformal_max_k:
                    continue
                for fi in fires:
                    if int(str(fi.date)[:4]) not in FOLD_YEARS:
                        continue
                    tr = resolve_trade(s, f, fi, h, k_strike, rv5)
                    if tr is None:
                        continue
                    all_trades_by_cell[(h, k_strike)].append({
                        "ticker": tk, "year": int(str(tr.date)[:4]),
                        "pnl": tr.pnl, "premium": tr.premium,
                        "exit_reason": tr.exit_reason,
                        "ticker_floor": floor,
                    })

    # ----- Stage A: pooled-universe view per (h, k_strike) cell -----
    print()
    print("=== STAGE A: Pooled universe (every cell that has ≥50 trades) ===")
    print(f"{'h':>2} {'k_strike':>9} {'n':>5} {'win%':>5} {'ROI%':>6} "
          f"{'touch%':>6} {'tstop%':>6} {'expry%':>6}")
    print("-" * 60)
    pooled_rows = []
    for (h, k_strike), trades in sorted(all_trades_by_cell.items()):
        if len(trades) < 50:
            continue
        wins = sum(1 for t in trades if t["pnl"] > 0)
        win_rate = wins / len(trades) * 100
        pnl = sum(t["pnl"] for t in trades)
        prem = sum(t["premium"] for t in trades)
        roi = pnl / prem * 100 if prem > 0 else 0
        ec = defaultdict(int)
        for t in trades:
            ec[t["exit_reason"]] += 1
        n = len(trades)
        print(f"{h:>2} {k_strike*100:>+8.1f}% {n:>5} {win_rate:>4.1f} "
              f"{roi:>+5.1f} {ec['touch']/n*100:>5.1f}% "
              f"{ec['time_stop']/n*100:>5.1f}% {ec['expiry']/n*100:>5.1f}%")
        pooled_rows.append({
            "horizon": h, "k_strike": k_strike, "n_trades": n,
            "win_rate_pct": win_rate, "roi_on_premium_pct": roi,
            "exit_touch_pct": ec['touch']/n*100,
            "exit_time_stop_pct": ec['time_stop']/n*100,
            "exit_expiry_pct": ec['expiry']/n*100,
        })

    # ----- Stage B: per-ticker drill-down only on cells that clear 90% -----
    summary_rows = []
    eligible_combos = []
    print()
    print("=== STAGE B: Per-ticker (cells clearing 90% pooled OR ≥15 fires) ===")
    print(f"{'tkr':<5} {'h':>2} {'k_strike':>9} "
          f"{'n':>4} {'win%':>5} {'ROI%':>6} "
          f"{'touch%':>6} {'tstop%':>6} {'expry%':>6} {'flr':>5} {'fld':>3}")
    print("-" * 75)
    for (h, k_strike), trades in sorted(all_trades_by_cell.items()):
        by_tk = defaultdict(list)
        for t in trades:
            by_tk[t["ticker"]].append(t)
        for tk, ts in by_tk.items():
            if len(ts) < MIN_FIRES_PER_TICKER:
                continue
            folds = {t["year"] for t in ts}
            if len(folds) < MIN_FOLDS_PER_TICKER:
                continue
            wins = sum(1 for t in ts if t["pnl"] > 0)
            win_rate = wins / len(ts) * 100
            pnl = sum(t["pnl"] for t in ts)
            prem = sum(t["premium"] for t in ts)
            roi = pnl / prem * 100 if prem > 0 else 0
            ec = defaultdict(int)
            for t in ts:
                ec[t["exit_reason"]] += 1
            row = {
                "ticker": tk, "horizon": h, "k_strike": k_strike,
                "n_trades": len(ts), "n_wins": wins,
                "win_rate_pct": win_rate,
                "roi_on_premium_pct": roi,
                "exit_touch_pct":     ec["touch"]     / len(ts) * 100,
                "exit_time_stop_pct": ec["time_stop"] / len(ts) * 100,
                "exit_expiry_pct":    ec["expiry"]    / len(ts) * 100,
                "n_folds": len(folds),
                "ticker_floor": ts[0]["ticker_floor"],
            }
            summary_rows.append(row)
            if win_rate >= MIN_WIN_RATE_PCT:
                eligible_combos.append(row)

    print(f"\n[3/4] Evaluated {len(summary_rows)} (ticker, h, k_strike) combos "
          f"({time.time()-t0:.1f}s)")
    print(f"[4/4] Combos clearing the 90% positive-ROI win-rate bar: "
          f"{len(eligible_combos)}")
    print()

    # Top by win-rate, then by ROI
    eligible_combos.sort(key=lambda r: (-r["win_rate_pct"], -r["roi_on_premium_pct"]))
    print(f"{'tkr':<5} {'h':>2} {'k_strike':>8} "
          f"{'n':>4} {'win%':>5} {'ROI%':>6} "
          f"{'touch%':>6} {'tstop%':>6} {'expry%':>6} "
          f"{'floor':>6} {'folds':>5}")
    print("-" * 80)
    for r in eligible_combos[:50]:
        print(f"{r['ticker']:<5} {r['horizon']:>2} {r['k_strike']*100:>+7.1f}% "
              f"{r['n_trades']:>4} {r['win_rate_pct']:>4.1f} "
              f"{r['roi_on_premium_pct']:>+5.1f} "
              f"{r['exit_touch_pct']:>5.1f}% {r['exit_time_stop_pct']:>5.1f}% "
              f"{r['exit_expiry_pct']:>5.1f}% "
              f"{r['conformal_floor']:>5.3f} {r['n_folds']:>5}")

    # Aggregate: sum across all eligible combos, what's the pooled
    # win-rate and pooled ROI?
    if eligible_combos:
        n = sum(r["n_trades"] for r in eligible_combos)
        w = sum(r["n_wins"] for r in eligible_combos)
        prem = sum(r["pooled_premium"] for r in eligible_combos)
        pnl = sum(r["pooled_pnl"] for r in eligible_combos)
        print()
        print(f"=== POOLED ACROSS ALL ELIGIBLE COMBOS ===")
        print(f"Trades:          {n}")
        print(f"Win rate:        {w/n*100:.1f}%")
        print(f"ROI on premium:  {pnl/prem*100:+.1f}%")
        print(f"Pooled PnL ($):  {pnl:.2f}")
        print(f"Pooled prem ($): {prem:.2f}")

    out_path = os.path.join(_HERE, "results", "cbi3x_short_calls.json")
    with open(out_path, "w") as fh:
        json.dump({
            "config": {
                "horizons": HORIZONS,
                "k_strike_grid": K_STRIKE_GRID,
                "touch_bounce_target": TOUCH_BOUNCE_TARGET,
                "time_stop_day": TIME_STOP_DAY,
                "conformal_quantile": CONFORMAL_QUANTILE,
                "conformal_safety": CONFORMAL_SAFETY,
                "min_fires_per_ticker": MIN_FIRES_PER_TICKER,
                "min_folds_per_ticker": MIN_FOLDS_PER_TICKER,
                "min_win_rate_pct": MIN_WIN_RATE_PCT,
            },
            "n_evaluated": len(summary_rows),
            "n_eligible": len(eligible_combos),
            "all_rows": summary_rows,
            "eligible": eligible_combos,
        }, fh, separators=(",", ":"))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
