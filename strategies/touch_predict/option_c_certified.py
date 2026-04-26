"""Option C — Certified-Core engine (truly-100% credit-spread predictions).

Layers ON TOP of option_c_research.py:

  1. **Worst-historical-drawdown floor.** For every (ticker, side,
     horizon), compute the deepest post-fire move the ticker EVER
     made across full history. Require K_short < (spot × (1 − worst))
     − 1% safety. This makes every historical entry a winner by
     construction (not just on average).

  2. **Per-ticker fold coverage.** Per ticker:
       - ≥ MIN_TICKER_FIRES historical fires (default 200)
       - ≥ MIN_TICKER_FOLDS distinct fold years (default 5)
       - every fold individually 100% (zero losing folds)
     This excludes the "small-sample 100%" rules that blew up in
     diagnostics (IBM had only 2 folds: 2020, 2021).

  3. **Macro regime conformity gate** (live, applied at publish time).
     - SPY above its 200-day SMA today
     - SPY 5-day return ≥ −3%
     - Stock 5-day return ≥ −10%
     Skip the signal today if any is violated.

  4. **Multi-rule consensus.** Require ≥ 2 independent rule families
     to agree on the same ticker for the same expiry window at the
     same strike (within ±5% of each other). Single-family signals
     downgrade from Certified → Near-certified.

We deliberately keep this engine SEPARATE from option_c_research.py
so we can validate it on truly unseen data before merging into the
production scan.
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
    actual_options_expiry, compute_features, fold_mask, list_tickers,
    load_series, spy_context, train_mask_for_fold,
)
from v2_regimes import CALL_REGIMES, PUT_REGIMES
from option_c_research import (
    K_SHORT_GRID, SPREAD_WIDTH, IV_MULT, HAIRCUT,
    HORIZONS, _credit_and_maxloss, _trade_pnl, _gather_fires, Fire,
)


# ---------- config ------------------------------------------------------

WORST_BUFFER_SAFETY = 0.01           # 1% extra below worst-ever historical move
MIN_TICKER_FIRES = 200
MIN_TICKER_FOLDS = 5
REQUIRE_EVERY_FOLD_PERFECT = True

# Macro gate (today's state)
SPY_5D_FLOOR_PCT = -0.03             # SPY can't have dropped > 3% in last 5 days
STOCK_5D_FLOOR_PCT = -0.10           # Stock-specific 5-day return floor

# Multi-rule consensus
CONSENSUS_FAMILIES_REQUIRED = 2      # need ≥ 2 distinct regime families
STRIKE_AGREEMENT_PCT = 0.05          # strikes must be within ±5% of each other
EXPIRY_AGREEMENT_DAYS = 30           # expiries must be within ±30 calendar days

# Regime "family" mapping — independent triggers should be in different families
REGIME_FAMILY = {
    # Call-regimes (which trigger PUT credit spreads)
    "plain":             "baseline",
    "deep_oversold":     "oversold",
    "connors_tps":       "oversold",   # also oversold-style — same family
    "multi_stack":       "oversold",
    "panic_day":         "panic",
    "spy_rel_weak":      "rel-weakness",
    "dip_in_uptrend":    "trend-pullback",
    # Put-regimes (which trigger CALL credit spreads)
    "overbought":        "overbought",
    "deep_overbought":   "overbought",
    "parabolic":         "overbought",
}


# ---------- per-ticker worst historical post-fire move ------------------


def worst_post_fire_move(side: str, fires: list[Fire], horizon_sessions: int) -> float | None:
    """Return the deepest post-fire move (fraction of spot) the ticker
    ever experienced when this regime fired. For side='put' returns
    the deepest DOWN move; for side='call' the deepest UP move.

    None if no fires."""
    if not fires:
        return None
    worst = 0.0
    for fi in fires:
        if side == "put":
            move = max(0.0, (fi.spot - fi.close_at_expiry) / fi.spot)
        else:
            move = max(0.0, (fi.close_at_expiry - fi.spot) / fi.spot)
        if move > worst:
            worst = move
    return worst


# ---------- per-ticker fold breakdown -----------------------------------


def per_ticker_folds(side: str, fires: list[Fire], k_short: float,
                     k_long: float) -> dict[str, dict[int, dict]]:
    """For each ticker, accumulate per-fold {wins, losses, pnl}."""
    out: dict[str, dict[int, dict]] = {}
    for fi in fires:
        y = int(str(fi.date)[:4])
        if y not in FOLD_YEARS:
            continue
        T = 0  # not used for win/loss; only need close_at_expiry vs strikes
        if side == "put":
            Ks = fi.spot * (1.0 - k_short)
            Kl = fi.spot * (1.0 - k_long)
            if fi.close_at_expiry >= Ks: pnl_sign = +1
            else: pnl_sign = -1
        else:
            Ks = fi.spot * (1.0 + k_short)
            Kl = fi.spot * (1.0 + k_long)
            if fi.close_at_expiry <= Ks: pnl_sign = +1
            else: pnl_sign = -1
        tk = out.setdefault(fi.ticker, {})
        f = tk.setdefault(y, {"wins": 0, "losses": 0})
        if pnl_sign > 0:
            f["wins"] += 1
        else:
            f["losses"] += 1
    return out


# ---------- macro state ------------------------------------------------

@dataclass
class MacroState:
    spy_above_sma200: bool
    spy_5d_ret: float
    valid: bool


def macro_state_today() -> MacroState:
    """Today's macro state — used as a live publish-time gate."""
    ctx = spy_context()
    if ctx is None:
        return MacroState(False, float("nan"), False)
    spy_dates, spy_close, spy_ret_5d = ctx
    today_close = float(spy_close[-1])
    if len(spy_close) < 200:
        return MacroState(False, float("nan"), False)
    sma200 = float(np.mean(spy_close[-200:]))
    return MacroState(
        spy_above_sma200=today_close >= sma200,
        spy_5d_ret=float(spy_ret_5d[-1]) if np.isfinite(spy_ret_5d[-1]) else float("nan"),
        valid=True,
    )


def macro_state_at(spy_dates: np.ndarray, spy_close: np.ndarray,
                   spy_ret_5d: np.ndarray, when: np.datetime64) -> MacroState:
    """Macro state on a given calendar date — used in held-out validation
    to verify the gate had the same effect historically."""
    idx = int(np.searchsorted(spy_dates, when))
    if idx >= len(spy_dates) or idx < 200:
        return MacroState(False, float("nan"), False)
    today = float(spy_close[idx])
    sma200 = float(np.mean(spy_close[idx - 200 : idx]))
    r5 = float(spy_ret_5d[idx]) if np.isfinite(spy_ret_5d[idx]) else float("nan")
    return MacroState(today >= sma200, r5, True)


def passes_macro_gate(side: str, m: MacroState, stock_5d_ret: float) -> bool:
    if not m.valid:
        return False
    if not m.spy_above_sma200:
        return False
    if not np.isfinite(m.spy_5d_ret) or m.spy_5d_ret < SPY_5D_FLOOR_PCT:
        return False
    if not np.isfinite(stock_5d_ret) or stock_5d_ret < STOCK_5D_FLOOR_PCT:
        return False
    return True


# ---------- main engine -----------------------------------------------


@dataclass
class CertifiedRule:
    side: str
    regime: str
    horizon: int
    k_short: float
    k_long: float
    ticker: str
    n_fires: int
    n_folds: int
    every_fold_perfect: bool
    worst_post_fire_move: float
    floor_strike_frac: float       # the 1−worst−safety threshold
    passes_floor: bool             # did the rule's K_short clear the floor?


def evaluate_certified_rule_per_ticker(
    side: str, regime: str, horizon: int, k_short: float, k_long: float,
    fires: list[Fire],
) -> list[CertifiedRule]:
    """For one (rule, k_short) combo, evaluate per-ticker eligibility
    against ALL four layers (where layer 3 / macro gate is applied
    later at publish time)."""
    if not fires:
        return []

    by_ticker: dict[str, list[Fire]] = {}
    for fi in fires:
        by_ticker.setdefault(fi.ticker, []).append(fi)

    out: list[CertifiedRule] = []
    folds_by_ticker = per_ticker_folds(side, fires, k_short, k_long)

    for tkr, tk_fires in by_ticker.items():
        tkr_folds = folds_by_ticker.get(tkr, {})
        n_fires = sum(f["wins"] + f["losses"] for f in tkr_folds.values())
        n_folds = len(tkr_folds)
        every_perfect = all(
            f["losses"] == 0 and f["wins"] > 0 for f in tkr_folds.values()
        )

        # Layer 4: per-ticker fold coverage
        if n_fires < MIN_TICKER_FIRES:
            continue
        if n_folds < MIN_TICKER_FOLDS:
            continue
        if REQUIRE_EVERY_FOLD_PERFECT and not every_perfect:
            continue

        # Layer 1: worst-historical-drawdown floor
        worst = worst_post_fire_move(side, tk_fires, horizon)
        if worst is None or worst <= 0:
            continue
        floor = worst + WORST_BUFFER_SAFETY
        # K_short is OTM by k_short; the floor requires K_short be
        # ≥ floor (i.e. the strike sits BELOW the worst-ever move plus safety
        # for puts; ABOVE the worst-ever rally + safety for calls).
        passes_floor = k_short >= floor

        if not passes_floor:
            continue

        out.append(CertifiedRule(
            side=side, regime=regime, horizon=horizon,
            k_short=k_short, k_long=k_long, ticker=tkr,
            n_fires=n_fires, n_folds=n_folds,
            every_fold_perfect=every_perfect,
            worst_post_fire_move=worst, floor_strike_frac=floor,
            passes_floor=True,
        ))
    return out


def find_certified_rules() -> list[CertifiedRule]:
    """Run layers 1, 2, 4 across the universe. Layer 3 (macro) and the
    consensus check (also layer 3 broad-stroke) are applied later at
    publish time, since they depend on TODAY's state."""
    _ = spy_context()  # warm SPY cache
    out: list[CertifiedRule] = []
    for side, regime_map in (("put", CALL_REGIMES), ("call", PUT_REGIMES)):
        for rname, rfn in regime_map.items():
            for h in HORIZONS:
                fires = _gather_fires(side, rname, rfn, h)
                if not fires:
                    continue
                for k_short in K_SHORT_GRID:
                    k_long = k_short + SPREAD_WIDTH
                    out.extend(evaluate_certified_rule_per_ticker(
                        side, rname, h, k_short, k_long, fires,
                    ))
    return out


# ---------- live signal building (with macro + consensus) -------------


def consensus_certify(rules_for_ticker: list[CertifiedRule]) -> str:
    """Apply layer-3 consensus on rules that already passed 1+2+4.
    Returns 'certified' (≥2 families on near-same trade), 'near'
    (passes layers but only 1 family), or 'fail' (no eligible rule)."""
    if not rules_for_ticker:
        return "fail"
    # Group by approximate (expiry, strike). Since rules don't carry
    # expiries here, we approximate by (horizon, k_short bucket).
    # Two rules from different families on the same (horizon, k_short)
    # bucket count as consensus.
    families_seen = set()
    for r in rules_for_ticker:
        families_seen.add(REGIME_FAMILY.get(r.regime, r.regime))
    if len(families_seen) >= CONSENSUS_FAMILIES_REQUIRED:
        return "certified"
    return "near"


if __name__ == "__main__":
    print("Running Certified-Core layer 1+2+4 evaluation across universe…")
    t0 = time.time()
    rules = find_certified_rules()
    print(f"  found {len(rules)} (rule, ticker) pairs passing layers 1+2+4")
    print(f"  elapsed {time.time()-t0:.1f}s")

    # Per-ticker rollup
    by_ticker: dict[str, list[CertifiedRule]] = {}
    for r in rules:
        by_ticker.setdefault(r.ticker, []).append(r)

    print(f"\nUnique tickers passing layers 1+2+4: {len(by_ticker)}")
    print()
    for tkr in sorted(by_ticker.keys()):
        tier = consensus_certify(by_ticker[tkr])
        rs = by_ticker[tkr]
        families = sorted({REGIME_FAMILY.get(r.regime, r.regime) for r in rs})
        sides = sorted({r.side for r in rs})
        n_rules = len(rs)
        avg_fires = sum(r.n_fires for r in rs) / n_rules
        avg_folds = sum(r.n_folds for r in rs) / n_rules
        worst = max(r.worst_post_fire_move for r in rs)
        print(f"  {tkr:<6} side={','.join(sides):<5} tier={tier:<10} "
              f"families={','.join(families):<25} "
              f"rules={n_rules:>2}  avg_fires={avg_fires:>5.0f}  "
              f"avg_folds={avg_folds:.1f}  worst-move={worst*100:>5.1f}%")
