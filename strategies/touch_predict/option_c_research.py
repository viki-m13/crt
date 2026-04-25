"""Option C — event-triggered conditional credit spreads.

Built on top of the touch-prediction regimes in v2_regimes.py. For
each regime × horizon × strike-placement, we simulate selling a
short-dated credit spread at every historical fire and measure
dollar P&L across walk-forward folds.

Trade structure (put side — triggered by oversold / touch-up signals):
    Sell put at K_short = spot * (1 - k_short)        (e.g. 2% OTM)
    Buy  put at K_long  = spot * (1 - k_long)          (further OTM)
    width = (k_long - k_short) * spot                  (max risk/share)
    credit = bs_put(spot, K_short, T, σ) - bs_put(spot, K_long, T, σ)
             (with a small haircut for bid-ask / liquidity)

At expiry close C:
    close >= K_short         → win: collect full credit
    K_long < close < K_short → partial loss: (K_short - close) - credit
    close <= K_long          → max loss: width - credit

Call side is mirror-symmetric.

The walk-forward protocol matches credit_spread's: train on pre-fold
data to set (k_short, k_long) parameters (we fix them up-front here
and just measure win/loss per fold), test on the fold-year fires, and
require (a) positive $ P&L in every fold, (b) win rate >= MIN_WIN_RATE,
(c) EV > 0.

Output:
    strategies/touch_predict/results/option_c_signals.json
    {
      "short_puts":   [{…}, …],   # rules eligible for short-put deployment
      "short_calls":  [{…}, …],   # rules eligible for short-call
      "recommended":  [{…}, …],   # combined, ranked by ROI on max-loss
      "live_fires":   {regime: [{…}, …]}   # rules firing today
    }
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
    load_series, spy_context,
)
from v2_regimes import CALL_REGIMES, PUT_REGIMES
from pricing import bs_call, bs_put


# -------- config ---------------------------------------------------------

# Spread design — strike placement as fractions of spot (signed by side).
# For puts: K_short = spot*(1 - k_short_frac); K_long  = spot*(1 - k_long_frac)
# For calls: K_short = spot*(1 + k_short_frac); K_long = spot*(1 + k_long_frac)
# We grid-search over these.
K_SHORT_GRID = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25,
                0.30, 0.40, 0.50]            # 1-50% OTM
SPREAD_WIDTH = 0.03                       # long leg k_long = k_short + 3%

# Horizons span short-dated weeklies to LEAPS-style annual. Longer
# tenors with deeper OTM strikes give mean-reversion the maximum
# possible cushion — this is what unlocks 100% win rates.
HORIZONS = [5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 252]

IV_MULT = 1.15                            # IV = realized × this
HAIRCUT = 0.80                            # keep 80% of BS credit after slippage

MIN_POOLED_TEST = 30                      # need this many pooled OOS test fires
MIN_WIN_RATE = 0.83                       # empirical close-at-expiry win rate
MIN_AGG_ROI = 0.005                       # 0.5% (low: keep 100%-certified deep-OTM rules)
MAX_LOSING_FOLDS = 1                      # allow up to this many years with net loss
MIN_TOTAL_FOLDS = 5                       # but must have been tested in >= this many years


# -------- helpers --------------------------------------------------------

def _credit_and_maxloss(side: str, spot: float, sigma: float,
                        k_short: float, k_long: float,
                        T_years: float) -> tuple[float, float]:
    """Return (credit, max_loss) per share for a credit spread with the
    given short/long strike OFFSETS (fractions of spot)."""
    if sigma <= 0 or T_years <= 0:
        return 0.0, 0.0
    if side == "put":
        Ks = spot * (1.0 - k_short)
        Kl = spot * (1.0 - k_long)
        cs = bs_put(spot, Ks, T_years, sigma)
        cl = bs_put(spot, Kl, T_years, sigma)
    else:  # call
        Ks = spot * (1.0 + k_short)
        Kl = spot * (1.0 + k_long)
        cs = bs_call(spot, Ks, T_years, sigma)
        cl = bs_call(spot, Kl, T_years, sigma)
    credit = max(cs - cl, 0.0) * HAIRCUT
    width = abs(Kl - Ks)
    max_loss = max(width - credit, 0.01)
    return credit, max_loss


def _trade_pnl(side: str, spot: float, close_at_expiry: float,
               k_short: float, k_long: float, credit: float) -> float:
    """Realized P&L per share on a credit spread, given the actual
    close at expiry. credit is already post-haircut."""
    if side == "put":
        Ks = spot * (1.0 - k_short)
        Kl = spot * (1.0 - k_long)
        if close_at_expiry >= Ks:
            return credit                             # win
        if close_at_expiry <= Kl:
            return credit - (Ks - Kl)                 # max loss
        return credit - (Ks - close_at_expiry)        # partial loss
    # call
    Ks = spot * (1.0 + k_short)
    Kl = spot * (1.0 + k_long)
    if close_at_expiry <= Ks:
        return credit
    if close_at_expiry >= Kl:
        return credit - (Kl - Ks)
    return credit - (close_at_expiry - Ks)


# -------- gather fires per regime ----------------------------------------

@dataclass
class Fire:
    """One (ticker, day) regime fire, with all inputs needed to price
    and resolve a spread."""
    ticker: str
    date: np.datetime64
    spot: float
    sigma: float           # rv60 at fire date
    close_at_expiry: float # close at day t+h (may be NaN if off-end)


def _gather_fires(side: str, regime_name: str, regime_fn,
                  horizon: int) -> list[Fire]:
    """Collect every (ticker, day) fire of this regime across the
    liquid universe, with fire-date σ and close-at-expiry attached."""
    _ = spy_context()
    fires = []
    for t in list_tickers():
        s = load_series(t)
        if s is None:
            continue
        f = compute_features(s)
        try:
            mask = regime_fn(f, s.close, s.dates)
        except Exception as exc:
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
            if i + horizon >= n:
                continue  # forward window off the end
            fires.append(Fire(
                ticker=t,
                date=s.dates[i],
                spot=float(s.close[i]),
                sigma=float(f.rv60[i]),
                close_at_expiry=float(s.close[i + horizon]),
            ))
    return fires


# -------- walk-forward ---------------------------------------------------

@dataclass
class RuleResult:
    side: str
    regime: str
    horizon: int
    k_short: float
    k_long: float
    folds: list[dict] = field(default_factory=list)
    pooled_wins: int = 0
    pooled_losses: int = 0
    pooled_pnl: float = 0.0       # sum of $ P&L per share across fires
    pooled_premium: float = 0.0   # sum of $ max_loss per share (capital at risk)
    avg_credit: float = 0.0
    avg_max_loss: float = 0.0
    win_rate: float = 0.0
    avg_roi_maxloss: float = 0.0  # pooled_pnl / pooled_premium (aka "return on risk")
    eligible: bool = False


def _evaluate(side: str, regime_name: str, horizon: int,
              k_short: float, k_long: float,
              fires: list[Fire]) -> tuple[RuleResult, dict[str, list[dict]]] | None:
    """Returns (pooled_rule, per_ticker_folds) so each ticker can be
    shown its OWN historical performance on this rule (rather than the
    universe-pooled aggregate)."""
    rr = RuleResult(side=side, regime=regime_name, horizon=horizon,
                    k_short=k_short, k_long=k_long)
    if not fires:
        return None

    # Per-year × per-ticker bucket
    by_year: dict[int, list[Fire]] = {}
    for fi in fires:
        y = int(str(fi.date)[:4])
        by_year.setdefault(y, []).append(fi)

    # Collect per-ticker fold rows alongside the pooled folds.
    per_ticker: dict[str, list[dict]] = {}

    for year in FOLD_YEARS:
        fold_fires = by_year.get(year, [])
        if not fold_fires:
            continue
        wins = losses = 0
        pnl_sum = 0.0
        mxloss_sum = 0.0
        credit_sum = 0.0
        T = horizon * 1.4 / 365.0   # session→calendar rough conversion
        # per-ticker tally for this year
        tk_year: dict[str, dict] = {}
        for fi in fold_fires:
            credit, max_loss = _credit_and_maxloss(
                side, fi.spot, fi.sigma * IV_MULT, k_short, k_long, T
            )
            pnl = _trade_pnl(side, fi.spot, fi.close_at_expiry,
                             k_short, k_long, credit)
            pnl_sum += pnl
            mxloss_sum += max_loss
            credit_sum += credit
            if pnl > 0:
                wins += 1
            else:
                losses += 1
            tk = tk_year.setdefault(fi.ticker, {
                "year": year, "n_test": 0, "wins": 0, "losses": 0, "pnl": 0.0,
            })
            tk["n_test"] += 1
            if pnl > 0:
                tk["wins"] += 1
            else:
                tk["losses"] += 1
            tk["pnl"] += pnl
        rr.folds.append({
            "year": year,
            "n_fires": len(fold_fires),
            "wins": wins,
            "losses": losses,
            "pnl": pnl_sum,
            "max_loss_sum": mxloss_sum,
            "credit_sum": credit_sum,
        })
        for tkr, row in tk_year.items():
            per_ticker.setdefault(tkr, []).append(row)

    rr.pooled_wins = sum(f["wins"] for f in rr.folds)
    rr.pooled_losses = sum(f["losses"] for f in rr.folds)
    total = rr.pooled_wins + rr.pooled_losses
    if total < MIN_POOLED_TEST or not rr.folds:
        return None

    rr.win_rate = rr.pooled_wins / total
    rr.pooled_pnl = sum(f["pnl"] for f in rr.folds)
    rr.pooled_premium = sum(f["max_loss_sum"] for f in rr.folds)
    rr.avg_credit = sum(f["credit_sum"] for f in rr.folds) / total
    rr.avg_max_loss = rr.pooled_premium / total
    if rr.pooled_premium > 0:
        rr.avg_roi_maxloss = rr.pooled_pnl / rr.pooled_premium
    losing_folds = sum(1 for f in rr.folds if f["pnl"] <= 0)
    rr.eligible = bool(
        rr.win_rate >= MIN_WIN_RATE
        and rr.avg_roi_maxloss >= MIN_AGG_ROI
        and len(rr.folds) >= MIN_TOTAL_FOLDS
        and losing_folds <= MAX_LOSING_FOLDS
    )
    return rr, per_ticker


# -------- live signal lookup ---------------------------------------------

def _find_live(side: str, regime_fn, horizon: int,
               k_short: float, k_long: float) -> list[dict]:
    """For each ticker whose regime fires TODAY, build a deployable
    signal with strike, credit, max-loss, projected Friday expiry."""
    out = []
    for t in list_tickers():
        s = load_series(t)
        if s is None:
            continue
        f = compute_features(s)
        try:
            mask = regime_fn(f, s.close, s.dates)
        except Exception:
            continue
        if not mask[-1]:
            continue
        spot = float(s.close[-1])
        sigma = float(f.rv60[-1]) if np.isfinite(f.rv60[-1]) else None
        if sigma is None or sigma <= 0:
            continue
        exp_iso, exp_type, cal_days = actual_options_expiry(str(s.dates[-1]), horizon)
        T = cal_days / 365.0
        credit, max_loss = _credit_and_maxloss(side, spot, sigma * IV_MULT,
                                               k_short, k_long, T)
        if side == "put":
            Ks = spot * (1 - k_short); Kl = spot * (1 - k_long)
        else:
            Ks = spot * (1 + k_short); Kl = spot * (1 + k_long)
        out.append({
            "ticker": t,
            "spot": spot,
            "as_of": str(s.dates[-1]),
            "realized_vol": sigma,
            "expiry": exp_iso,
            "expiry_type": exp_type,
            "cal_days": cal_days,
            "k_short_frac": k_short,
            "k_long_frac":  k_long,
            "strike_short": Ks,
            "strike_long":  Kl,
            "est_credit":   credit,
            "est_max_loss": max_loss,
            "est_roi":      credit / max_loss if max_loss > 0 else 0.0,
        })
    return out


# -------- main driver ----------------------------------------------------

def main() -> int:
    t0 = time.time()
    results: list[RuleResult] = []
    live_by_rule: dict[str, list[dict]] = {}

    # Per-ticker folds keyed by rule_id, so each ticker card can show
    # its OWN historical track record on the rule (not the universe pool).
    per_ticker_folds_by_rule: dict[str, dict[str, list[dict]]] = {}

    for side, regime_map in (("put", CALL_REGIMES),    # puts triggered by touch-UP signals (call regimes)
                             ("call", PUT_REGIMES)):    # calls triggered by touch-DOWN signals (put regimes)
        for rname, rfn in regime_map.items():
            for h in HORIZONS:
                fires = _gather_fires(side, rname, rfn, h)
                if not fires:
                    continue
                for k_short in K_SHORT_GRID:
                    k_long = k_short + SPREAD_WIDTH
                    res = _evaluate(side, rname, h, k_short, k_long, fires)
                    if res is None:
                        continue
                    r, per_ticker = res
                    if not r.eligible:
                        continue
                    results.append(r)
                    rule_key = f"{side}|{rname}|{h}|{k_short:.3f}"
                    live_by_rule[rule_key] = _find_live(side, rfn, h, k_short, k_long)
                    per_ticker_folds_by_rule[rule_key] = per_ticker
        print(f"  {side} side: {sum(1 for r in results if r.side==side)} rules so far  "
              f"elapsed={time.time()-t0:.1f}s")

    # Rank by ROI-on-max-loss
    results.sort(key=lambda r: -r.avg_roi_maxloss)

    # Report
    short_puts  = [r for r in results if r.side == "put"]
    short_calls = [r for r in results if r.side == "call"]
    print(f"\nTotal Option C rules: {len(results)}")
    print(f"  short puts:  {len(short_puts)}")
    print(f"  short calls: {len(short_calls)}")

    print(f"\n{'side':<5} {'regime':<14} {'h':>3} {'ks%':>4} {'kl%':>4} "
          f"{'win%':>5} {'nTst':>5} {'credit$':>7} {'mxLoss$':>7} {'ROI':>5}")
    print("-"*82)
    for r in results[:40]:
        print(f"{r.side:<5} {r.regime:<14} {r.horizon:>3} "
              f"{r.k_short*100:>4.1f} {r.k_long*100:>4.1f} "
              f"{r.win_rate*100:>4.1f} {r.pooled_wins+r.pooled_losses:>5} "
              f"{r.avg_credit:>7.2f} {r.avg_max_loss:>7.2f} "
              f"{r.avg_roi_maxloss*100:>4.1f}%")

    # Live signals total
    n_live = sum(len(v) for v in live_by_rule.values())
    print(f"\nTotal LIVE fires across all eligible rules: {n_live}")
    flat = [(r, fi) for r in results
            for fi in live_by_rule.get(f"{r.side}|{r.regime}|{r.horizon}|{r.k_short:.3f}", [])]
    flat.sort(key=lambda p: -(p[1]["est_roi"]))
    print("\nTop 15 live signals:")
    for r, fi in flat[:15]:
        print(f"  {fi['ticker']:<6} {r.side:<5} {r.regime:<14} h={r.horizon}d "
              f"ks=${fi['strike_short']:.2f} kl=${fi['strike_long']:.2f} "
              f"credit=${fi['est_credit']:.2f} ml=${fi['est_max_loss']:.2f} "
              f"ROI={fi['est_roi']*100:.0f}% win%={r.win_rate*100:.0f}%")

    # Serialize
    def rule_dict(r: RuleResult) -> dict:
        return {
            "side": r.side, "regime": r.regime, "horizon": r.horizon,
            "k_short_frac": r.k_short, "k_long_frac": r.k_long,
            "spread_width_pct": (r.k_long - r.k_short) * 100.0,
            "pooled_wins": r.pooled_wins, "pooled_losses": r.pooled_losses,
            "win_rate_pct": r.win_rate * 100.0,
            "avg_credit_per_share": r.avg_credit,
            "avg_max_loss_per_share": r.avg_max_loss,
            "avg_roi_on_max_loss_pct": r.avg_roi_maxloss * 100.0,
            "pooled_pnl": r.pooled_pnl,
            "pooled_premium": r.pooled_premium,
            "folds": r.folds,
            "live_fires": live_by_rule.get(
                f"{r.side}|{r.regime}|{r.horizon}|{r.k_short:.3f}", []
            ),
        }

    # ---------------------------------------------------------------
    # Restructure rule-keyed output → ticker-keyed signals so the UI
    # can present these the same way as the credit_spread engine
    # (one ticker per card, multi-rung ladder).
    # ---------------------------------------------------------------

    REGIME_LABELS = {
        "plain":           "All stocks (unconditional)",
        "connors_tps":     "Connors TPS setup",
        "multi_stack":     "Multi-indicator mean-reversion",
        "panic_day":       "Panic-drop day",
        "spy_rel_weak":    "Underperforming S&P 500",
        "deep_oversold":   "Deep oversold",
        "dip_in_uptrend":  "Dip in uptrend",
        "overbought":      "Overbought",
        "deep_overbought": "Deep overbought",
        "parabolic":       "Parabolic move",
    }

    def _ticker_key(side: str, ticker: str) -> str:
        return f"{side}|{ticker}"

    by_ticker: dict[str, dict] = {}
    for r in results:
        rule_id = f"{r.side}|{r.regime}|{r.horizon}|{r.k_short:.3f}"
        for fi in live_by_rule.get(rule_id, []):
            tk_key = _ticker_key(r.side, fi["ticker"])
            entry = by_ticker.setdefault(tk_key, {
                "ticker":      fi["ticker"],
                "today_close": fi["spot"],
                "end_date":    fi["as_of"],
                "side":        r.side,
                "realized_vol_pct": fi["realized_vol"] * 100.0,
                "ladder":      [],
            })
            # Per-ticker fold breakdown for THIS ticker's track record on
            # this rule (rather than the pool-wide aggregate). May be
            # empty if the ticker had no fires under this rule (rare).
            tk_folds = per_ticker_folds_by_rule.get(rule_id, {}).get(fi["ticker"], [])
            tk_wins = sum(f["wins"] for f in tk_folds)
            tk_losses = sum(f["losses"] for f in tk_folds)
            tk_total = tk_wins + tk_losses
            tk_win_rate = (tk_wins / tk_total) if tk_total > 0 else 0.0
            # Tier badge classifies the ticker's OWN historical accuracy
            # on this specific rule. (The rule's overall pooled win rate
            # is reported separately as `pool_win_rate_pct`.)
            tier = ("certified" if tk_win_rate >= 0.999 else
                    "near"      if tk_win_rate >= 0.98  else
                    "high"      if tk_win_rate >= 0.95  else
                    "standard")
            entry["ladder"].append({
                "regime":         r.regime,
                "regime_label":   REGIME_LABELS.get(r.regime, r.regime),
                "horizon":        r.horizon,
                "tier":           tier,
                "k_short_frac":   r.k_short,
                "k_long_frac":    r.k_long,
                "spread_width_pct": (r.k_long - r.k_short) * 100.0,
                "expiry_date":    fi["expiry"],
                "expiry_type":    fi["expiry_type"],
                "calendar_days_to_expiry": fi["cal_days"],
                "strike_short":   fi["strike_short"],
                "strike_long":    fi["strike_long"],
                "buffer_pct":     r.k_short * 100.0,           # OTM distance shown as "buffer"
                # Per-ticker stats (what THIS ticker actually did on this rule)
                "win_rate_pct":   tk_win_rate * 100.0,
                "n_test":         tk_total,
                "n_folds":        len(tk_folds),
                "pooled_wins":    tk_wins,
                "pooled_losses":  tk_losses,
                # Rule-level pooled stats (the rule's universe-wide accuracy)
                "pool_win_rate_pct": r.win_rate * 100.0,
                "pool_n_test":       r.pooled_wins + r.pooled_losses,
                "profit": {
                    "est_credit_per_share":   fi["est_credit"],
                    "est_max_loss_per_share": fi["est_max_loss"],
                    "return_on_risk_pct":     fi["est_roi"] * 100.0,
                    "annualized_ror_pct":     fi["est_roi"] * (365.0 / max(fi["cal_days"], 1)) * 100.0,
                    "implied_vol_pct":        fi["realized_vol"] * 100.0 * IV_MULT,
                    "spread_width":           fi["est_credit"] + fi["est_max_loss"],
                },
                "folds": tk_folds,    # this ticker's per-year breakdown
            })

    # Per ticker: dedupe rungs that resolve to the SAME real options
    # trade. Two rungs with different session-horizons can snap to the
    # same monthly Friday expiry and produce identical (expiry, K_short,
    # K_long) — they're testing the same instrument with different
    # backtest windows. Keep the one with the highest empirical win
    # rate as the canonical rung.
    LADDER_CAP_PER_TICKER = 12   # cap to keep payload small for the webapp

    put_signals_grouped = []
    call_signals_grouped = []
    for tk_key, entry in by_ticker.items():
        per_trade: dict[tuple, dict] = {}
        for r in entry["ladder"]:
            key = (r["expiry_date"],
                   round(r["strike_short"], 4),
                   round(r["strike_long"], 4))
            prev = per_trade.get(key)
            if (prev is None
                or r["win_rate_pct"] > prev["win_rate_pct"]
                or (r["win_rate_pct"] == prev["win_rate_pct"]
                    and r["n_test"] > prev["n_test"])):
                per_trade[key] = r
        # Stratified cap: keep the highest-ROI rung in each
        # (horizon, tier) bucket. This guarantees that whatever
        # filter combination the user picks (e.g., 90d + 98%-tier)
        # will surface a result if one exists in the data — no rule
        # gets dropped just because its ROI didn't make a flat top-N.
        # Skip rungs with no per-ticker history (tk_total == 0).
        per_bucket: dict[tuple, dict] = {}
        for r in per_trade.values():
            if r["n_test"] < 5:
                # Too-thin per-ticker sample (< 5 historical fires) —
                # the ticker hasn't been in this regime often enough
                # to make a personal track-record claim.
                continue
            bkey = (r["horizon"], r["tier"])
            prev = per_bucket.get(bkey)
            if (prev is None
                or r["profit"]["return_on_risk_pct"] > prev["profit"]["return_on_risk_pct"]):
                per_bucket[bkey] = r
        # Final ladder display order: nearest expiry first, then richer credit
        entry["ladder"] = sorted(
            per_bucket.values(),
            key=lambda x: (x["expiry_date"], -x["strike_short"]),
        )
        if not entry["ladder"]:
            continue
        # Promote the best rung to top-level fields (mirrors credit_spread shape)
        best = max(entry["ladder"], key=lambda x: x["profit"]["return_on_risk_pct"])
        entry["best_tier"]         = max(
            (r["tier"] for r in entry["ladder"]),
            key=lambda t: {"certified": 3, "near": 2, "high": 1}.get(t, 0),
        )
        entry["regime"]            = best["regime"]
        entry["regime_label"]      = best["regime_label"]
        entry["horizon"]           = best["horizon"]
        entry["expiry_date"]       = best["expiry_date"]
        entry["expiry_type"]       = best["expiry_type"]
        entry["strike"]            = best["strike_short"]
        entry["strike_long"]       = best["strike_long"]
        entry["buffer_pct"]        = best["buffer_pct"]
        entry["win_rate_pct"]      = best["win_rate_pct"]
        entry["profit"]            = best["profit"]
        entry["n_test"]            = best["n_test"]
        entry["n_folds"]           = best["n_folds"]
        entry["calendar_days_to_expiry"] = best["calendar_days_to_expiry"]
        if entry["side"] == "put":
            put_signals_grouped.append(entry)
        else:
            call_signals_grouped.append(entry)

    # Sort by best-ladder ROI desc
    put_signals_grouped.sort(key=lambda e: -e["profit"]["return_on_risk_pct"])
    call_signals_grouped.sort(key=lambda e: -e["profit"]["return_on_risk_pct"])
    recommended = sorted(
        put_signals_grouped + call_signals_grouped,
        key=lambda e: -e["profit"]["return_on_risk_pct"],
    )[:30]

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "horizons": HORIZONS, "k_short_grid": K_SHORT_GRID,
            "spread_width_pct": SPREAD_WIDTH * 100,
            "iv_mult": IV_MULT, "haircut": HAIRCUT,
            "min_win_rate": MIN_WIN_RATE,
            "min_agg_roi_pct": MIN_AGG_ROI * 100,
            "max_losing_folds": MAX_LOSING_FOLDS,
            "min_total_folds": MIN_TOTAL_FOLDS,
        },
        "summary": {
            "n_eligible_rules":  len(results),
            "n_short_puts":      len(put_signals_grouped),
            "n_short_calls":     len(call_signals_grouped),
            "n_live_fires":      n_live,
            "overall_win_rate_pct": (
                sum(r.pooled_wins for r in results)
                / max(1, sum(r.pooled_wins + r.pooled_losses for r in results)) * 100
            ),
            "overall_roi_on_max_loss_pct": (
                sum(r.pooled_pnl for r in results)
                / max(1e-9, sum(r.pooled_premium for r in results)) * 100
            ),
        },
        "put_signals":  put_signals_grouped,
        "call_signals": call_signals_grouped,
        "recommended":  recommended,
    }

    out_path = os.path.join(_HERE, "results", "option_c_signals.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        # No indent — keeps the published JSON small for the webapp.
        json.dump(out, fh, separators=(",", ":"))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
