"""Build v5 webapp data.json for the PIT-S&P-500 strategy with Chronos confidence filter.

v5 = v3 baseline (ml_3plus6, tight gate, h=6) + two additions:
  1. Chronos-bolt-tiny zero-shot p70 forecast filter (require rank >= 0.45)
  2. Inverse-volatility weighting with cap=0.4 per pick

Backtest (PIT S&P 500 2003-2025):
  - Full CAGR: 43.86% (vs v3 39.77%)
  - Walk-forward mean OOS CAGR: 49.39% on augmented PIT panel (was 47.16% K=3 on biased v2 panel)
  - WF min OOS CAGR: 23.08% (vs v3 14.49%)
  - All 10/10 splits beat SPY (vs v3's 9/10)
  - Sharpe 1.06 (vs 0.96); MaxDD -48.4% (vs -49.8%)

Generalization (same config on other universes):
  - Broader 1833-ticker: 57.82% WF mean (+6pp vs v3)
  - Non-S&P 500 PIT: 62.72% WF mean (+12pp vs v3)
  - Random 500 subsets: 46-67% WF mean

Schema is backward-compatible with `docs/monthly_dca.js`.

Run from repo root:
    python3 -m experiments.monthly_dca.v5.build_webapp_v5_pit
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.monthly_dca.v2.ml_strategy import EXCLUDE  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"
WEBAPP_OUT = ROOT / "experiments" / "docs" / "monthly-dca"
WEBAPP_OUT.mkdir(parents=True, exist_ok=True)

STRATEGY_SPEC = {
    "scorer": "ml_3plus6 + chronos_p70_filter",
    "scorer_description": (
        "v2 GBM 3m+6m forward-rank ensemble, gated by HuggingFace Chronos-bolt-tiny "
        "zero-shot probabilistic p70 forecast (must rank top 55% of cross-section)"
    ),
    "K_normal": 2,
    "K_recovery": 2,
    "K_bull": 2,
    "weighting": "inverse-volatility",
    "cap_per_pick": 0.40,
    "regime_gate": "tight",
    "regime_gate_rule": (
        "crash if SPY 21d <= -8% OR (SPY 6m <= -5% AND SPY 21d <= -3%); "
        "recovery if SPY below 200dma streak >= 40d AND SPY just back above 200dma AND SPY 21d > 0; "
        "bull if SPY 12m >= 10% AND above 200dma; else normal."
    ),
    "min_hold_months": 6,
    "max_hold_months": 24,
    "rebalance_mode": "rule_based_score_drift",
    "cost_bps": 10,
    "universe": "PIT S&P 500 members at each rebalance month-end (iShares IVV current holdings used for live universe)",
    "rebalance_rule": "Hold each basket at least 6 months. After 6 months, rebalance ONLY when neither current pick is still in the new top-K eligible pool (i.e., the picker has discovered they're no longer best). Force rebalance at 24 months max. Always rebalance on regime crash transition. Within each rebalance, weights = 1/vol_1y of each pick, capped at 40% per name and re-normalized. This rule-based 'min hold 6m + score_drift' schedule beats fixed h=6 on the augmented PIT panel: WF mean 54.9% vs 49.4%, Sharpe 1.10 vs 1.04, Max DD -34.5% vs -52.5%, 10/10 splits beat SPY, +45.3pp edge in 2024 vs -10.2pp for fixed h=6.",
    "chronos_filter": {
        "model": "amazon/chronos-bolt-tiny (HuggingFace)",
        "model_size": "9M params, zero-shot foundation model",
        "input": "trailing 252-day daily prices",
        "horizon": "64 trading days (~3 months)",
        "metric": "70th percentile of probabilistic forecast distribution",
        "filter_quantile": 0.45,
        "rule": (
            "At each rebalance, the candidate pool is restricted to stocks where the "
            "Chronos p70 3m-forecast cross-sectional rank is >= 0.45 (i.e., the upper 55%). "
            "Then v3 picks the top 2 by ml_3plus6 score from that filtered pool. "
            "This is universe-agnostic alpha that complements the GBM's tabular signal — "
            "verified on broader 1833-ticker, non-S&P 500 PIT, and random subset universes."
        ),
    },
}

WINNER_NAME = "v5_pit_sp500_ml_3plus6_chronos_p70_k2_invvol_cap0.4_minhold6_scoredrift"

# v5 strategy hyperparameters.
# K_PICKS was updated 2026-05-12: K=3 -> K=2 after the augmented-PIT
# parameter sweep (`experiments/monthly_dca/v5/spx_pit/IMPROVEMENTS.md`)
# confirmed K=2 dominates K=3 on every metric (WF mean 49.4% vs 32.7%,
# 10/10 vs 8/10 splits beating SPY, Sharpe 1.04 vs 0.92, identical Max DD)
# under PIT correction AND under MC delisting overlay across all alpha.
CHRONOS_FILTER_Q = 0.45          # filter quantile: require Chronos p70 rank >= 0.45
CAP_PER_PICK = 0.40              # cap inverse-vol weights at 40% per pick
HOLD_MONTHS = 6                  # legacy fixed-hold (back-compat for v3/v5 fixed sims)
K_PICKS = 2
# Rebalance-rule config (Phase 10, May 2026): rule-based dominates fixed h=6 on
# every risk-adjusted metric on augmented PIT (see IMPROVEMENTS.md Phase 10).
# Hold at least MIN_HOLD_MONTHS; after that, rebalance EARLIER if neither
# current pick is still in the new top-K eligible pool (score_drift trigger).
# Force rebalance at MAX_HOLD_MONTHS regardless. Always rebalance on regime
# change crash↔non-crash (preserves crash protection).
REBALANCE_MODE = "rule_based"   # 'fixed' or 'rule_based'
MIN_HOLD_MONTHS = 6
MAX_HOLD_MONTHS = 24
CHRONOS_MODEL = "amazon/chronos-bolt-tiny"
CHRONOS_CONTEXT_DAYS = 252
CHRONOS_HORIZON_DAYS = 64


def _calc_invvol_weights(top: pd.DataFrame, monthly_returns: pd.DataFrame,
                        asof: pd.Timestamp, cap: float = CAP_PER_PICK) -> np.ndarray:
    """Compute inverse-volatility weights for the top picks at asof.

    Uses trailing 12-month volatility of monthly returns. Caps each weight at
    `cap` (default 40%), renormalises.
    """
    k = len(top)
    vols = []
    # Find latest month-end <= asof in monthly_returns
    mr_idx = monthly_returns.index
    pos = mr_idx.searchsorted(asof, side="right") - 1
    if pos < 12:
        return np.ones(k) / k  # not enough history, fall back to EW
    window = monthly_returns.iloc[pos - 11: pos + 1]
    for tk in top["ticker"]:
        if tk in window.columns:
            v = window[tk].std()
            if pd.isna(v) or v <= 0:
                v = 0.10  # default annualised-ish monthly vol
            vols.append(v)
        else:
            vols.append(0.10)
    vols = np.array(vols)
    invv = 1.0 / np.maximum(vols, 0.01)
    w = invv / invv.sum()
    # Apply cap
    if cap < 1.0:
        w = np.minimum(w, cap)
        w = w / w.sum()
    return w


# ---------------------------------------------------------------------------
def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (pd.Timestamp,)):
        return str(x.date())
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return f if np.isfinite(f) else None
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, np.ndarray):
        return [to_jsonable(v) for v in x.tolist()]
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x


# ---------------------------------------------------------------------------
def _spy_below_200_streak_1y(daily_spy: pd.Series, asof: pd.Timestamp) -> float:
    """Max consecutive trading days SPY closed below its 200dma in the past
    12 months (~252 trading days) ending at `asof`. Used by the regime
    classifier — replaces the legacy 5y-window streak which never re-zeros
    after a single bear market.
    """
    s = daily_spy.loc[:asof]
    if len(s) < 200 + 21:
        return 0.0
    sma200 = s.rolling(200, min_periods=200).mean()
    below = (s < sma200).astype(int).iloc[-252:]
    m = 0
    cur = 0
    for v in below.values:
        if v == 1:
            cur += 1
            if cur > m:
                m = cur
        else:
            cur = 0
    return float(m)


def load_spy_features() -> pd.DataFrame:
    feat_dir = CACHE / "features"
    # Load daily SPY for on-the-fly 12-month streak (legacy 5y feature stays
    # in the parquets for backwards compat but is no longer consulted).
    daily = pd.read_parquet(CACHE / "prices_extended.parquet")
    daily_spy = daily["SPY"].dropna() if "SPY" in daily.columns else pd.Series(dtype=float)
    rows = []
    for f in sorted(feat_dir.glob("*.parquet")):
        d = pd.Timestamp(f.stem)
        df = pd.read_parquet(f)
        if "SPY" not in df.index:
            continue
        spy = df.loc["SPY"]
        streak_1y = _spy_below_200_streak_1y(daily_spy, d) if len(daily_spy) else 0.0
        rows.append({
            "asof": d,
            "spy_dsma200": float(spy.get("d_sma200", 0.0)),
            "spy_rsi14": float(spy.get("rsi_14", 50.0)),
            "spy_mom_12_1": float(spy.get("mom_12_1", 0.0)),
            "spy_mom_6_1": float(spy.get("mom_6_1", 0.0)),
            "spy_ret_21d": float(spy.get("ret_21d", 0.0)),
            "spy_below_200_streak": streak_1y,
        })
    return pd.DataFrame(rows).set_index("asof")


def classify_regime_tight(s: dict) -> str:
    r21 = s.get("spy_ret_21d", 0.0)
    r6m = s.get("spy_mom_6_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    dsma = s.get("spy_dsma200", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0)
    if r21 <= -0.08 or (r6m <= -0.05 and r21 <= -0.03):
        return "crash"
    if streak >= 40 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


# ---------------------------------------------------------------------------
@dataclass
class Sim:
    """Tracker for the v3 simulation across the live + historical window.

    Holds the current basket, last rebalance date, equity, and cash flag.
    """
    equity: float = 1.0
    cur_picks: list = None
    cur_weights: np.ndarray = None
    last_rebalance: Optional[pd.Timestamp] = None
    cash: bool = False
    held_for: int = 0


def run_full_sim(
    members_g: dict[pd.Timestamp, set],
    preds_wf: pd.DataFrame,
    preds_live: pd.DataFrame,
    spy_features: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    monthly_prices: pd.DataFrame,
    chronos_preds: dict | None = None,
    cost_bps: float = 10.0,
    hold_months: int = HOLD_MONTHS,
    K: int = K_PICKS,
) -> tuple[list, pd.DataFrame, dict]:
    """Run the v3 strategy honestly across the full window.

    Uses ``preds_wf`` (annual-retrain walk-forward predictions with 7-month
    embargo) for any month where it has data, and falls back to ``preds_live``
    only for the latest month(s) past the WF cutoff (i.e., the LIVE picks
    that are still "live" out-of-sample).  This prevents look-ahead in the
    historical backtest while still showing the up-to-date current basket.

    Returns:
        rets_log:    list of {date, regime, ret_m, n_picks, picks, basket_id, equity}
        trade_log:   per-pick records with entry/exit/return
        live_state:  current basket info (last_rebalance, next_rebalance, picks, weights, regime)
    """
    wf_max_asof = preds_wf["asof"].max()
    # Months: union of WF and LIVE asofs (LIVE extends past WF cutoff for the latest month)
    months = sorted(set(pd.to_datetime(preds_wf["asof"].unique())).union(
        set(pd.to_datetime(preds_live["asof"].unique()))
    ))
    months = [pd.Timestamp(m) for m in months]
    print(f"  WF preds end at {wf_max_asof}; LIVE extends to {preds_live['asof'].max()}")
    print(f"  Total months in sim: {len(months)}")
    cf = cost_bps / 10000.0

    cur_picks: list[str] = []
    cur_weights: np.ndarray = np.array([])
    cash = False
    held_for = 0
    equity = 1.0
    last_rebalance: Optional[pd.Timestamp] = None
    basket_id = 0
    last_rebalance_to_hold: list[str] = []
    last_rebalance_to_buy: list[str] = []
    last_rebalance_to_sell: list[str] = []
    prev_basket_for_live: list[str] = []

    rets_log = []
    trade_log_open: list[dict] = []
    trade_log_closed: list[dict] = []
    mr_idx = monthly_returns.index

    def _compute_candidate_top(m_):
        """Peek at the would-be top-K at month-end m (PIT members + Chronos
        filter applied). Returns the top-K DataFrame, or None if no eligible
        basket can be formed. Used by the rule-based score_drift trigger AND
        the actual basket-forming branch when do_reb=True."""
        if m_ <= wf_max_asof:
            sub_ = preds_wf[preds_wf["asof"] == m_].copy()
        else:
            sub_ = preds_live[preds_live["asof"] == m_].copy()
        sub_ = sub_[~sub_["ticker"].isin(EXCLUDE)]
        sp_set_ = members_g.get(m_, set())
        sub_pit_ = sub_[sub_["ticker"].isin(sp_set_)].copy()
        if len(sub_pit_) == 0:
            return None
        sub_pit_["score"] = (sub_pit_["pred_3m"] + sub_pit_["pred_6m"]) / 2
        if chronos_preds is not None and m_ in chronos_preds:
            chronos_at_m_ = chronos_preds[m_]
            sub_pit_["chr_p70"] = sub_pit_["ticker"].map(chronos_at_m_)
            sub_pit_["chr_p70_rk"] = sub_pit_["chr_p70"].rank(pct=True)
            sub_pit_ = sub_pit_[sub_pit_["chr_p70_rk"] >= CHRONOS_FILTER_Q]
        top_ = sub_pit_.sort_values("score", ascending=False).head(K)
        return top_ if len(top_) >= K else None

    for i, m in enumerate(months):
        spy_now = spy_features.loc[m].to_dict() if m in spy_features.index else {}
        regime = classify_regime_tight(spy_now)
        # Rule-based rebalance (Phase 10):
        #   - Always reb on first month, regime crash transition, or MAX_HOLD reached.
        #   - After MIN_HOLD, reb EARLIER if neither current pick is still in the
        #     new top-K eligible pool (score_drift trigger).
        if REBALANCE_MODE == "fixed":
            do_reb = (i == 0) or (held_for >= hold_months) or cash
        else:
            do_reb = (i == 0) or (cash != (regime == "crash"))
            if held_for >= MAX_HOLD_MONTHS:
                do_reb = True
            elif held_for >= MIN_HOLD_MONTHS and cur_picks and regime != "crash":
                # Peek at would-be top-K and check overlap
                _candidate = _compute_candidate_top(m)
                if _candidate is not None:
                    new_top_set = set(_candidate["ticker"])
                    if not (set(cur_picks) & new_top_set):
                        do_reb = True

        if do_reb:
            # Close out the OLD basket: book actual entry/exit prices, realised
            # returns, and the SPY benchmark return over the same window.
            if cur_picks and last_rebalance is not None:
                pos1 = mr_idx.searchsorted(m)
                cands = []
                for j in (pos1 - 1, pos1):
                    if 0 <= j < len(mr_idx):
                        cands.append((j, abs((mr_idx[j] - m).days)))
                cands.sort(key=lambda x: x[1])
                close_d = mr_idx[cands[0][0]] if cands and cands[0][1] <= 7 else m
                # SPY return over the holding window
                spy_entry_px = (float(monthly_prices.at[trade_log_open[0]["entry_date_ts"], "SPY"])
                                if trade_log_open and "SPY" in monthly_prices.columns
                                and trade_log_open[0]["entry_date_ts"] in monthly_prices.index
                                else None)
                spy_exit_px = (float(monthly_prices.at[close_d, "SPY"])
                               if "SPY" in monthly_prices.columns and close_d in monthly_prices.index
                               else None)
                spy_ret_window = ((spy_exit_px / spy_entry_px - 1)
                                  if (spy_entry_px and spy_exit_px) else None)
                for trade in trade_log_open:
                    tk = trade["ticker"]
                    entry_d_ts = trade.pop("entry_date_ts")
                    if tk in monthly_prices.columns and entry_d_ts in monthly_prices.index:
                        entry_px_real = float(monthly_prices.at[entry_d_ts, tk])
                        if pd.isna(entry_px_real):
                            entry_px_real = None
                    else:
                        entry_px_real = None
                    if tk in monthly_prices.columns and close_d in monthly_prices.index:
                        exit_px_real = float(monthly_prices.at[close_d, tk])
                        if pd.isna(exit_px_real):
                            exit_px_real = None
                    else:
                        exit_px_real = None
                    pick_ret = ((exit_px_real / entry_px_real - 1)
                                if (entry_px_real and exit_px_real) else None)
                    trade["entry_px"] = entry_px_real
                    trade["exit_date"] = str(close_d.date())
                    trade["exit_px"] = exit_px_real
                    trade["return"] = pick_ret
                    trade["spy_entry_px"] = spy_entry_px
                    trade["spy_exit_px"] = spy_exit_px
                    trade["spy_return"] = spy_ret_window
                    trade["beat_spy"] = (pick_ret is not None and spy_ret_window is not None
                                          and pick_ret > spy_ret_window)
                    trade["status"] = "exited"
                    trade_log_closed.append(trade)
                trade_log_open = []

            if regime == "crash":
                cur_picks, cur_weights, cash = [], np.array([]), True
                held_for = 0
            else:
                # Use WF preds for historical months, LIVE preds for the most
                # recent month past the WF cutoff (current basket only).
                if m <= wf_max_asof:
                    sub = preds_wf[preds_wf["asof"] == m].copy()
                else:
                    sub = preds_live[preds_live["asof"] == m].copy()
                sub = sub[~sub["ticker"].isin(EXCLUDE)]
                sp_set = members_g.get(m, set())
                sub_pit = sub[sub["ticker"].isin(sp_set)].copy()
                if len(sub_pit) == 0:
                    cur_picks, cur_weights, cash = [], np.array([]), True
                else:
                    sub_pit["score"] = (sub_pit["pred_3m"] + sub_pit["pred_6m"]) / 2
                    # v5: apply Chronos confidence filter (rank >= 0.45)
                    if chronos_preds is not None and m in chronos_preds:
                        chronos_at_m = chronos_preds[m]  # dict ticker -> chronos_p70_3m
                        sub_pit["chr_p70"] = sub_pit["ticker"].map(chronos_at_m)
                        # Cross-sectional rank within S&P 500 cohort
                        sub_pit["chr_p70_rk"] = sub_pit["chr_p70"].rank(pct=True)
                        sub_pit = sub_pit[sub_pit["chr_p70_rk"] >= CHRONOS_FILTER_Q]
                    # Pick top-K by ml_3plus6 score from filtered pool
                    top = sub_pit.sort_values("score", ascending=False).head(K)
                    if len(top) < K:
                        cur_picks, cur_weights, cash = [], np.array([]), True
                    else:
                        prev_basket = list(cur_picks)
                        cur_picks = top["ticker"].tolist()
                        # v5: inverse-volatility weighting with cap=0.40
                        cur_weights = _calc_invvol_weights(top, monthly_returns,
                                                           m, cap=CAP_PER_PICK)
                        cash = False
                        last_rebalance = m
                        basket_id += 1
                        held_for = 0
                        # Compute hold / buy / sell deltas for the live display.
                        prev_set = set(prev_basket)
                        cur_set = set(cur_picks)
                        last_rebalance_to_hold = sorted(prev_set & cur_set)
                        last_rebalance_to_buy = sorted(cur_set - prev_set)
                        last_rebalance_to_sell = sorted(prev_set - cur_set)
                        prev_basket_for_live = list(prev_basket)
                        # Log entry trades for this basket using REAL prices
                        # at the entry month-end.
                        pos = mr_idx.searchsorted(m)
                        cands_entry = []
                        for j in (pos - 1, pos):
                            if 0 <= j < len(mr_idx):
                                cands_entry.append((j, abs((mr_idx[j] - m).days)))
                        cands_entry.sort(key=lambda x: x[1])
                        entry_d = (mr_idx[cands_entry[0][0]]
                                   if cands_entry and cands_entry[0][1] <= 7 else m)
                        for tk in cur_picks:
                            entry_px_real = None
                            if (tk in monthly_prices.columns
                                    and entry_d in monthly_prices.index):
                                v = monthly_prices.at[entry_d, tk]
                                entry_px_real = float(v) if not pd.isna(v) else None
                            trade_log_open.append({
                                "ticker": tk,
                                "entry_date": str(entry_d.date()),
                                "entry_date_ts": entry_d,  # private — popped on close
                                "entry_px": entry_px_real,
                                "regime": regime,
                                "basket_id": basket_id,
                                "status": "open",
                            })

        # Compute monthly return on current basket.
        # NaN returns for tickers that ARE in the panel mean "data not yet
        # available" (e.g. current open month) — treat as 0% rather than
        # booking a -100% wipe.  Tickers MISSING from the panel are treated
        # as genuinely delisted (-100%).
        if cash or len(cur_picks) == 0:
            ret_m = 0.0
        else:
            pos1 = mr_idx.searchsorted(m)
            cands = []
            for j in (pos1 - 1, pos1):
                if 0 <= j < len(mr_idx):
                    cands.append((j, abs((mr_idx[j] - m).days)))
            cands.sort(key=lambda x: x[1])
            if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(mr_idx):
                ret_m = 0.0
            else:
                next_d = mr_idx[cands[0][0] + 1]
                pick_rets = []
                any_data = False
                for tk in cur_picks:
                    if tk in monthly_returns.columns:
                        rr = monthly_returns.at[next_d, tk]
                        if pd.isna(rr):
                            pick_rets.append(0.0)  # data not yet available
                        else:
                            pick_rets.append(float(rr))
                            any_data = True
                    else:
                        pick_rets.append(-1.0)  # genuine delisting
                pick_rets = np.array(pick_rets)
                # If ZERO data points were available for this month, treat as
                # not-yet-realised and keep equity flat.
                if not any_data:
                    ret_m = 0.0
                else:
                    ret_m = float((pick_rets * cur_weights).sum())

        if not cash and len(cur_picks) > 0:
            if do_reb:
                equity *= (1 + ret_m) * (1 - cf)
            else:
                equity *= (1 + ret_m)
        held_for += 1

        rets_log.append({
            "date": str(m.date()), "regime": regime,
            "ret_m": ret_m, "n_picks": len(cur_picks),
            "picks": list(cur_picks),
            "basket_id": basket_id,
            "equity": float(equity),
            "rebalanced": do_reb,
        })

    # Build live state from last rebalance
    next_rebalance = None
    if last_rebalance is not None:
        # next rebalance is hold_months after the last rebalance
        nr = last_rebalance + pd.DateOffset(months=hold_months)
        # Find nearest month-end >= nr in our months list
        future = [m for m in months if m >= nr]
        next_rebalance = future[0] if future else (last_rebalance + pd.DateOffset(months=hold_months))
    months_since_rebalance = held_for

    live_state = {
        "last_rebalance_date": str(last_rebalance.date()) if last_rebalance else None,
        "next_rebalance_date": str(next_rebalance.date()) if next_rebalance else None,
        "months_since_rebalance": int(months_since_rebalance),
        "current_basket_picks": list(cur_picks),
        "current_basket_weights": cur_weights.tolist() if len(cur_weights) else [],
        "previous_basket_picks": list(prev_basket_for_live),
        # At the LAST rebalance, these were the actions a real investor would have taken:
        "last_rebalance_to_hold": list(last_rebalance_to_hold),
        "last_rebalance_to_buy": list(last_rebalance_to_buy),
        "last_rebalance_to_sell": list(last_rebalance_to_sell),
        "cash_position": cash,
        "current_regime": classify_regime_tight(
            spy_features.loc[months[-1]].to_dict() if months[-1] in spy_features.index else {}
        ),
        "basket_id": basket_id,
    }
    trades = trade_log_closed + trade_log_open
    return rets_log, pd.DataFrame(trades), live_state


# ---------------------------------------------------------------------------
# Augmented PIT data redirection (May 2026):
# If the augmented PIT artefacts exist under PIT/augmented/, prefer them
# everywhere — that's the survivorship-corrected universe (1994 tickers,
# 78% PIT coverage globally, 99.7% in 2025). See data/sp500_pit/ for the
# full dataset and methodology. Falling back to the original V2 paths
# keeps the script usable on legacy checkouts.
AUG = PIT / "augmented"
USE_AUG = (AUG / "ml_preds.parquet").exists() and (AUG / "ml_preds_chronos.parquet").exists()


def _path(filename: str, prefer_aug: bool = True) -> Path:
    """Resolve a path that may have an augmented version."""
    if USE_AUG and prefer_aug:
        cand = AUG / filename
        if cand.exists():
            return cand
    # legacy fallbacks
    if filename == "prices_extended_pit.parquet":
        # canonical: PIT/prices_extended_pit.parquet
        return PIT / filename if (PIT / filename).exists() else CACHE / "prices_extended.parquet"
    if filename in ("monthly_returns_clean.parquet", "monthly_prices_clean.parquet"):
        return V2 / filename
    if filename in ("ml_preds_v2.parquet", "ml_preds_live.parquet"):
        return V2 / filename
    if filename == "ml_preds_chronos.parquet":
        return PIT / filename
    return PIT / filename


# === DCA-investor evaluation (what a monthly-contribution investor lives) ===
# Mirrors experiments/monthly_dca/v5/spx_pit/dca_investor_eval.py exactly:
# contribute 1 unit at the start of each month, earn that month's return;
# identical schedule for every stream. The v5 stream is net of 10bps.
# Fail-soft: any error returns None so the live build never breaks.

DCA_HORIZONS = [12, 24, 36, 60, 120]


def _dca_path(rets):
    v = 0.0
    vals, basis = [], []
    for t, r in enumerate(rets):
        v = (v + 1.0) * (1.0 + r)
        vals.append(v)
        basis.append(t + 1.0)
    return np.array(vals), np.array(basis)


def _irr_from_terminal(terminal, H):
    def npv(i):
        out = sum(1.0 / (1.0 + i) ** t for t in range(H))
        return terminal / (1.0 + i) ** (H - 1) - out
    lo, hi = -0.5, 0.5
    flo = npv(lo)
    mid = 0.0
    for _ in range(160):
        mid = 0.5 * (lo + hi)
        fm = npv(mid)
        if abs(fm) < 1e-10:
            break
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return (1.0 + mid) ** 12 - 1.0


def _dca_switch(v5r, mnr, spyr, th=0.25):
    """Drawdown-conditional rotation: DCA into v5; when the portfolio's
    drawdown from its running peak breaches -th, route the book + new
    contributions into the validated market-neutral sleeve; switch back
    at -th/2 recovery (hysteresis). An honest portfolio switch between two
    already-validated streams — NOT a new alpha."""
    v = peak = 0.0
    in_mn = False
    vals = []
    for t in range(len(v5r)):
        v += 1.0
        v *= (1.0 + (mnr[t] if in_mn else v5r[t]))
        peak = max(peak, v)
        dd = v / peak - 1.0
        if not in_mn and dd <= -th:
            in_mn = True
        elif in_mn and dd >= -th / 2.0:
            in_mn = False
        vals.append(v)
    return np.array(vals)


def compute_dca_investor(rets_log, monthly_returns):
    try:
        v5 = pd.Series(
            [float(r["ret_m"]) for r in rets_log],
            index=pd.PeriodIndex([pd.Timestamp(r["date"]) for r in rets_log], freq="M"),
        )
        spy = monthly_returns["SPY"].dropna().astype(float)
        spy.index = pd.to_datetime(spy.index).to_period("M")
        idx = v5.index.intersection(spy.index)
        v5, spy = v5.reindex(idx).fillna(0.0), spy.reindex(idx)

        mn = pd.Series(0.0, index=idx)
        mn_csv = AUG / "v5_mn_sleeve_returns.csv"
        if mn_csv.exists():
            m = pd.read_csv(mn_csv, index_col=0, parse_dates=True).iloc[:, 0].astype(float)
            m.index = pd.to_datetime(m.index).to_period("M")
            mn = m.reindex(idx).fillna(0.0)

        v5a, spya, mna = v5.to_numpy(), spy.to_numpy(), mn.to_numpy()
        sw = None
        if mn_csv.exists():
            sw_path = _dca_switch(v5a, mna, spya, 0.25)

        def variant_series(name):
            if name == "v5":
                return v5a, None
            if name == "mn_switch":
                return None, _dca_switch  # path-dependent
            return spya, None

        out = {
            "window": f"{idx[0]} .. {idx[-1]}",
            "n_months": int(len(idx)),
            "convention": ("contribute $1 at the start of each month, earn that "
                           "month's return; v5 net of 10bps; PIT data, no tuning"),
            "horizons": {},
            "full_history": {},
        }

        # rolling horizons
        n = len(idx)
        for H in DCA_HORIZONS:
            row = {}
            spy_terms = []
            for s in range(0, n - H + 1):
                spy_terms.append(_dca_path(spya[s:s + H])[0][-1])
            for label in (["v5", "mn_switch", "SPY"] if mn_csv.exists() else ["v5", "SPY"]):
                wins, moic = [], []
                worst = 1e18
                for j, s in enumerate(range(0, n - H + 1)):
                    if label == "mn_switch":
                        tv = _dca_switch(v5a[s:s + H], mna[s:s + H], spya[s:s + H], 0.25)[-1]
                    elif label == "v5":
                        tv = _dca_path(v5a[s:s + H])[0][-1]
                    else:
                        tv = spy_terms[j]
                    m = tv / H
                    wins.append(tv > spy_terms[j])
                    moic.append(m)
                    worst = min(worst, m)
                row[label] = {
                    "win_vs_spy_dca": (None if label == "SPY"
                                       else round(float(np.mean(wins)), 4)),
                    "n_windows": int(len(moic)),
                    "median_moic": round(float(np.median(moic)), 3),
                    "p05_moic": round(float(np.quantile(moic, 0.05)), 3),
                    "min_moic": round(float(worst), 3),
                }
            out["horizons"][f"H{H}"] = row

        # full history
        for label in (["v5", "mn_switch", "SPY"] if mn_csv.exists() else ["v5", "SPY"]):
            if label == "mn_switch":
                val = _dca_switch(v5a, mna, spya, 0.25)
                basis = np.arange(1, len(val) + 1, dtype=float)
            else:
                val, basis = _dca_path(v5a if label == "v5" else spya)
            pk = np.maximum.accumulate(val)
            uw = (val - basis) / basis
            out["full_history"][label] = {
                "months": int(len(val)),
                "terminal_moic": round(float(val[-1] / basis[-1]), 2),
                "money_weighted_irr": round(float(_irr_from_terminal(val[-1], len(val))), 4),
                "max_value_drawdown": round(float(((val - pk) / pk).min()), 4),
                "worst_underwater_vs_contrib": round(float(uw.min()), 4),
            }
        return out
    except Exception as e:  # never break the live build
        print(f"  WARNING: compute_dca_investor failed: {e}")
        return None


def main():
    print("=== Loading inputs ===")
    print(f"    USE_AUG = {USE_AUG}  ({'augmented' if USE_AUG else 'legacy V2'} data layer)")
    members = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    members["asof"] = pd.to_datetime(members["asof"])
    members_g = members.groupby("asof")["ticker"].apply(set).to_dict()

    monthly_returns = pd.read_parquet(_path("monthly_returns_clean.parquet"))
    monthly_prices = pd.read_parquet(_path("monthly_prices_clean.parquet"))
    spy_features = load_spy_features()

    # ml_preds_live: latest-month live prediction. The augmented panel's
    # walk-forward predictions extend through the latest asof, so when
    # USE_AUG is on we treat the augmented preds as the source for both
    # historical AND live (their latest asof IS the live month).
    if USE_AUG:
        preds_live = pd.read_parquet(_path("ml_preds.parquet"))
    else:
        preds_live = pd.read_parquet(V2 / "ml_preds_live.parquet")
    preds_live["asof"] = pd.to_datetime(preds_live["asof"])
    if USE_AUG:
        preds_wf = pd.read_parquet(_path("ml_preds.parquet"))
    else:
        preds_wf = pd.read_parquet(V2 / "ml_preds_v2.parquet")
    preds_wf["asof"] = pd.to_datetime(preds_wf["asof"])
    print(f"  live preds: {len(preds_live)}, asof range "
          f"{preds_live['asof'].min().date()} -> {preds_live['asof'].max().date()}")
    print(f"  WF preds:   {len(preds_wf)}, asof range "
          f"{preds_wf['asof'].min().date()} -> {preds_wf['asof'].max().date()}")

    # v5: load Chronos-bolt-tiny forecasts (augmented when available)
    chronos_path = _path("ml_preds_chronos.parquet")
    chronos_preds = None
    if chronos_path.exists():
        chr_df = pd.read_parquet(chronos_path)
        chr_df["asof"] = pd.to_datetime(chr_df["asof"])
        chronos_preds = {}
        for asof, group in chr_df.groupby("asof"):
            chronos_preds[pd.Timestamp(asof)] = dict(zip(group["ticker"], group["chronos_p70_3m"]))
        print(f"  chronos preds: {len(chr_df)} rows, {len(chronos_preds)} asof months "
              f"({chr_df['asof'].min().date()} -> {chr_df['asof'].max().date()})")
    else:
        print(f"  WARNING: {chronos_path} not found — running v3 baseline (no Chronos filter)")

    # Pull additional ticker features for the basket display (price, mom, etc.)
    feature_files = sorted((CACHE / "features").glob("*.parquet"))
    feature_by_asof = {pd.Timestamp(f.stem): f for f in feature_files}

    print("\n=== Running v5 simulation over live window ===")
    rets_log, trade_log, live_state = run_full_sim(
        members_g, preds_wf, preds_live, spy_features, monthly_returns,
        monthly_prices, chronos_preds=chronos_preds,
        cost_bps=10.0, hold_months=HOLD_MONTHS, K=K_PICKS,
    )
    print(f"  months: {len(rets_log)}, last basket id: {live_state['basket_id']}")
    print(f"  current basket: {live_state['current_basket_picks']}")
    print(f"  last rebalance: {live_state['last_rebalance_date']}, next: {live_state['next_rebalance_date']}")

    # Build pick basket display: enrich each pick with ticker features at last_rebalance
    pick_basket = []
    last_reb = live_state["last_rebalance_date"]
    if last_reb:
        last_reb_ts = pd.Timestamp(last_reb)
        feat_path = feature_by_asof.get(last_reb_ts)
        feat_df = pd.read_parquet(feat_path) if feat_path else None
        # Find the pred row for each pick at last_reb
        sub_live = preds_live[preds_live["asof"] == last_reb_ts]
        for tk in live_state["current_basket_picks"]:
            row = sub_live[sub_live["ticker"] == tk]
            score = float(row["pred_3m"].iloc[0] + row["pred_6m"].iloc[0]) / 2 if len(row) else None
            r = feat_df.loc[tk] if (feat_df is not None and tk in feat_df.index) else None
            item = {
                "ticker": tk,
                "score": score,
                "pred_1m_rank": float(row["pred_1m"].iloc[0]) if len(row) else None,
                "pred_3m_rank": float(row["pred_3m"].iloc[0]) if len(row) else None,
                "pred_6m_rank": float(row["pred_6m"].iloc[0]) if len(row) else None,
                "price": float(r["price"]) if r is not None and "price" in r else None,
                "pullback_1y": float(r["pullback_1y"]) if r is not None and "pullback_1y" in r else None,
                "trend_health_5y": float(r["trend_health_5y"]) if r is not None and "trend_health_5y" in r else None,
                "mom_3y": float(r["mom_3y"]) if r is not None and "mom_3y" in r else None,
                "mom_12_1": float(r["mom_12_1"]) if r is not None and "mom_12_1" in r else None,
                "rsi_14": float(r["rsi_14"]) if r is not None and "rsi_14" in r else None,
                "d_sma200": float(r["d_sma200"]) if r is not None and "d_sma200" in r else None,
                "recovery_rate": float(r["recovery_rate"]) if r is not None and "recovery_rate" in r else None,
                "vol_1y": float(r["vol_1y"]) if r is not None and "vol_1y" in r else None,
                "sharpe_12m": float(r["sharpe_12m"]) if r is not None and "sharpe_12m" in r else None,
                "rs_12m_spy": float(r["rs_12m_spy"]) if r is not None and "rs_12m_spy" in r else None,
            }
            pick_basket.append(item)

    # === Equity curve / growth ===
    growth = []
    spy0 = None
    months_dt = [pd.Timestamp(r["date"]) for r in rets_log]
    if "SPY" in monthly_prices.columns:
        first_idx = monthly_prices.index.searchsorted(months_dt[0])
        if first_idx < len(monthly_prices.index):
            spy0 = float(monthly_prices["SPY"].iloc[first_idx])
    for r in rets_log:
        d = pd.Timestamp(r["date"])
        spy_val = None
        if spy0 is not None:
            pos = monthly_prices.index.searchsorted(d)
            if pos < len(monthly_prices.index):
                spy_val = float(monthly_prices["SPY"].iloc[pos]) / spy0
        growth.append({
            "date": r["date"], "strat_value": float(r["equity"]),
            "spy_value": spy_val, "invested": 1.0,
        })

    # === Headline metrics ===
    n_months = len(rets_log)
    years = max(n_months / 12.0, 1 / 12.0)
    cagr_strat = float(rets_log[-1]["equity"]) ** (1.0 / years) - 1.0 if rets_log else 0.0
    cagr_spy = (growth[-1]["spy_value"] ** (1.0 / years) - 1.0) if growth and growth[-1].get("spy_value") else None
    rets_m = np.array([r["ret_m"] for r in rets_log])
    sharpe = float(rets_m.mean() / rets_m.std() * np.sqrt(12)) if rets_m.std() > 0 else None
    win_rate = float((rets_m > 0).mean())

    # === Year-by-year ===
    df_rl = pd.DataFrame(rets_log)
    df_rl["date"] = pd.to_datetime(df_rl["date"])
    df_rl["year"] = df_rl["date"].dt.year
    yr = df_rl.groupby("year")["ret_m"].apply(lambda x: float(((1 + x).prod() - 1)))
    spy_yr = {}
    if "SPY" in monthly_returns.columns:
        spy_m = monthly_returns.loc[months_dt[0]:, "SPY"].dropna()
        spy_m_df = pd.DataFrame({"date": spy_m.index, "ret": spy_m.values})
        spy_m_df["year"] = pd.to_datetime(spy_m_df["date"]).dt.year
        spy_yr = spy_m_df.groupby("year")["ret"].apply(lambda x: float(((1 + x).prod() - 1))).to_dict()
    year_rows = []
    for y in sorted(yr.index):
        cagr_p = float(yr[y])
        spy_p = spy_yr.get(y)
        edge = cagr_p - spy_p if spy_p is not None else None
        n_picks = int(df_rl[df_rl["year"] == y]["n_picks"].sum())
        wr = float((df_rl[df_rl["year"] == y]["ret_m"] > 0).mean())
        year_rows.append({
            "year": int(y),
            "cagr_dca_picks": cagr_p,
            "cagr_dca_spy": spy_p,
            "edge": edge,
            "n_picks": n_picks,
            "win_rate": wr,
        })

    # === Horizon stats: trailing N years ===
    horizon_stats = []
    if rets_log and growth:
        for years_back in (1, 2, 3, 5, 7, 10, 15, 20):
            n_needed = years_back * 12
            if len(rets_log) < n_needed:
                continue
            window = rets_log[-n_needed:]
            window_growth = growth[-n_needed:]
            since_d = pd.Timestamp(window[0]["date"])
            strat_mult = float(np.prod([1 + r["ret_m"] for r in window]))
            strat_cagr = strat_mult ** (1.0 / years_back) - 1.0
            spy_start = window_growth[0].get("spy_value")
            spy_end = window_growth[-1].get("spy_value")
            spy_mult = (spy_end / spy_start) if (spy_start and spy_end and spy_start > 0) else None
            spy_cagr = (spy_mult ** (1.0 / years_back) - 1.0) if spy_mult else None
            edge = (strat_cagr - spy_cagr) if (strat_cagr is not None and spy_cagr is not None) else None
            n_picks = sum(int(r.get("n_picks", 0)) for r in window)
            horizon_stats.append({
                "years_back": years_back,
                "since_date": str(since_d.date()),
                "cagr_strat": strat_cagr,
                "cagr_spy": spy_cagr,
                "edge_vs_spy": edge,
                "n_picks": n_picks,
                "strat_terminal": strat_mult,
                "spy_terminal": spy_mult if spy_mult is not None else 0.0,
            })

    # === Walk-forward (load from cached v3 results) ===
    wf_split_rows = []
    wf_aggregate_rows = []
    wf_path = _path("v5_winner_walkforward.csv")
    if wf_path.exists():
        wf_test = pd.read_csv(wf_path)
        for _, r in wf_test.iterrows():
            wf_split_rows.append({
                "split": r["split"], "from": str(r["from"]), "to": str(r["to"]),
                "n_months": int(r["n_m"]),
                "CAGR_pct": float(r["cagr"]) * 100,
                "SPY_CAGR_pct": float(r["spy_cagr"]) * 100,
                "Edge_pp": float(r["edge_pp"]),
                "Sharpe": float(r["sharpe"]),
                "MaxDD": float(r["max_dd"]),
                "n_cash_months": int(r["n_cash"]),
            })
        wf_cagrs = wf_test["cagr"].astype(float)
        wf_edges = wf_test["edge_pp"].astype(float)
        wf_aggregate_rows = [{
            "n_splits": int(len(wf_test)),
            "n_splits_with_test_data": int(len(wf_test)),
            "mean_test_cagr": float(wf_cagrs.mean()),
            "median_test_cagr": float(wf_cagrs.median()),
            "min_test_cagr": float(wf_cagrs.min()),
            "max_test_cagr": float(wf_cagrs.max()),
            "mean_edge_pp": float(wf_edges.mean()),
            "n_positive_splits": int((wf_cagrs > 0).sum()),
            "n_beats_spy": int((wf_cagrs > wf_test["spy_cagr"].astype(float)).sum()),
        }]

    # === Bias overlay ===
    bias_rows = []
    bias_path = _path("v5_winner_bias_sensitivity.csv")
    if bias_path.exists():
        bias = pd.read_csv(bias_path)
        for _, r in bias.iterrows():
            bias_rows.append({
                "base_rate_annual": float(r["alpha_yr"]),
                "stratified_cagr_median": float(r["median"]) / 100.0,
                "stratified_cagr_p10": float(r["p10"]) / 100.0,
                "stratified_cagr_p90": float(r["p90"]) / 100.0,
                "uniform_cagr_median": float(r["median"]) / 100.0,
            })
    stratified_4pct = {}
    bias4 = next((r for r in bias_rows if r["base_rate_annual"] == 0.04), None)
    if bias4:
        stratified_4pct = {
            "cagr_dca_median": bias4["stratified_cagr_median"],
            "cagr_dca_p10": bias4["stratified_cagr_p10"],
            "cagr_dca_p90": bias4["stratified_cagr_p90"],
            "edge_median": (bias4["stratified_cagr_median"] - (cagr_spy or 0)),
        }

    # === Sub-period CAGR ===
    sub_period_rows = []
    sp_path = _path("v5_winner_sub_periods.csv")
    if sp_path.exists():
        sp = pd.read_csv(sp_path)
        for _, r in sp.iterrows():
            sub_period_rows.append({
                "period": r["period"],
                "from": str(r["from"]), "to": str(r["to"]),
                "cagr_strat": float(r["cagr"]),
                "cagr_spy": float(r["spy_cagr"]),
                "edge_pp": float(r["edge_pp"]),
                "n_months": int(r["n_m"]),
            })

    # === Multi-universe generalisation ===
    gen_rows = []
    gen_path = _path("v5_winner_generalize.csv")
    if gen_path.exists():
        gen = pd.read_csv(gen_path)
        for _, r in gen.iterrows():
            gen_rows.append({
                "universe": r["universe"],
                "n_pool": int(r["n_picks_universe"]),
                "cagr_full": float(r["cagr_full"]),
                "sharpe": float(r["sharpe"]),
                "max_dd": float(r["max_dd"]),
                "wf_mean_cagr": float(r["wf_mean_cagr"]),
                "wf_min_cagr": float(r["wf_min_cagr"]),
                "wf_max_cagr": float(r["wf_max_cagr"]),
                "wf_mean_edge_pp": float(r["wf_mean_edge_pp"]),
                "wf_n_positive": int(r["wf_n_pos"]),
                "wf_n_beats_spy": int(r["wf_n_beats"]),
            })

    # === Sensitivity ===
    sens_rows = []
    sens_path = _path("v5_winner_sensitivity.csv")
    if sens_path.exists():
        sn = pd.read_csv(sens_path)
        for _, r in sn.iterrows():
            sens_rows.append({
                "param": r["param"],
                "value": str(r["value"]),
                "cagr_full": float(r["cagr_full"]),
                "wf_mean_cagr": float(r["wf_mean_cagr"]),
                "wf_min_cagr": float(r["wf_min_cagr"]),
                "wf_mean_edge_pp": float(r["wf_mean_edge_pp"]),
                "wf_n_beats_spy": int(r["wf_n_beats"]),
                "max_dd": float(r["max_dd"]),
            })

    # === Most-picked tickers (concentration audit) ===
    most_picked = []
    mp_path = _path("v5_winner_most_picked.csv")
    if mp_path.exists():
        mp = pd.read_csv(mp_path).head(20)
        for _, r in mp.iterrows():
            most_picked.append({"ticker": str(r["ticker"]),
                                "n_months_picked": int(r["n_months_picked"])})

    # === Drawdown ledger ===
    drawdowns = []
    dd_path = _path("v5_winner_drawdowns.csv")
    if dd_path.exists():
        dd = pd.read_csv(dd_path).head(10)
        for _, r in dd.iterrows():
            drawdowns.append({
                "start": str(r["start"]),
                "trough": str(r["trough"]),
                "end": str(r["end"]),
                "depth_pct": float(r["depth_pct"]),
            })

    # === Regime history (last 24m) ===
    regime_history = []
    last_24 = months_dt[-24:] if len(months_dt) >= 24 else months_dt
    for d in last_24:
        s_d = spy_features.loc[d].to_dict() if d in spy_features.index else {}
        regime_history.append({
            "date": str(d.date()),
            "regime": classify_regime_tight(s_d),
        })

    # === Universe coverage ===
    coverage_rows = []
    cov_path = _path("sp500_pit_filter_coverage.csv")
    if cov_path.exists():
        cov = pd.read_csv(cov_path)
        for _, r in cov.iterrows():
            coverage_rows.append({"year": int(r.iloc[0]),
                                  "panel_coverage_pct": float(r.iloc[1]) * 100})

    # === Pick log (for trade history display) ===
    pick_log_rows = []
    for trade in trade_log.to_dict(orient="records") if len(trade_log) else []:
        ret = trade.get("return")
        spy_ret = trade.get("spy_return")
        beat = trade.get("beat_spy")
        pick_log_rows.append({
            "asof": trade.get("entry_date"),
            "ticker": trade.get("ticker"),
            "regime": trade.get("regime"),
            "next_month_ret": ret,
            "entry_px": trade.get("entry_px"),
            "exit_date": trade.get("exit_date"),
            "exit_px": trade.get("exit_px"),
            "years": 0.5,
            "return": ret,
            "ret": ret,
            "ret_strat": ret,
            "ret_spy": spy_ret,
            "cagr": (((1 + ret) ** 2 - 1) if ret is not None else None),  # 6m hold => annl ~ (1+r)^2 -1
            "spy_return": spy_ret,
            "win": (ret is not None and ret > 0),
            "beat_spy": beat,
            "status": trade.get("status"),
            "basket_id": trade.get("basket_id"),
        })

    # === DCA-investor evaluation (the product's actual user) ===
    print("=== Computing DCA-investor outcomes (rolling horizons + MN-switch) ===")
    dca_investor = compute_dca_investor(rets_log, monthly_returns)
    if dca_investor:
        h120 = dca_investor["horizons"]["H120"]["v5"]
        print(f"  10y DCA: win vs SPY-DCA {h120['win_vs_spy_dca']*100:.0f}%  "
              f"median {h120['median_moic']:.1f}x  worst {h120['min_moic']:.1f}x")

    # === Build final data.json ===
    n_picks_total = sum(r["n_picks"] for r in rets_log)
    last_pred_month = preds_live["asof"].max()
    data = {
        "as_of": str(last_pred_month.date()) if hasattr(last_pred_month, "date") else str(last_pred_month),
        "strategy_version": "v5-pit-sp500",
        "strategy_spec": STRATEGY_SPEC,
        "panel": {
            "n_tickers": int(members["ticker"].nunique()),
            "first_date": str(monthly_prices.index.min().date()),
            "last_date": str(monthly_prices.index.max().date()),
            "universe": "PIT S&P 500",
        },
        "spy_dca_cagr": cagr_spy,
        "headline": {
            "n_picks": int(n_picks_total),
            "win_rate_raw": win_rate,
            "win_rate_bias_corr": None,
            "cagr_raw": cagr_strat,
            "cagr_total": cagr_strat,
            "cagr_bias_corr": stratified_4pct.get("cagr_dca_median") if stratified_4pct else None,
            "cagr_spy_dca": cagr_spy,
            "edge": (cagr_strat - cagr_spy) if (cagr_strat is not None and cagr_spy is not None) else None,
            "sharpe": sharpe,
        },
        "current_regime": {
            "regime": live_state["current_regime"],
            "K": 3 if live_state["current_regime"] != "crash" else 0,
            "spy_dsma200": spy_features.loc[months_dt[-1]].get("spy_dsma200") if months_dt[-1] in spy_features.index else None,
            "spy_rsi14": spy_features.loc[months_dt[-1]].get("spy_rsi14") if months_dt[-1] in spy_features.index else None,
            "spy_mom_12_1": spy_features.loc[months_dt[-1]].get("spy_mom_12_1") if months_dt[-1] in spy_features.index else None,
            "spy_mom_6_1": spy_features.loc[months_dt[-1]].get("spy_mom_6_1") if months_dt[-1] in spy_features.index else None,
            "spy_ret_21d": spy_features.loc[months_dt[-1]].get("spy_ret_21d") if months_dt[-1] in spy_features.index else None,
        },
        "regime_history_24m": regime_history,
        "pick_of_month_basket": pick_basket,
        "pick_of_month": pick_basket[0] if pick_basket else None,
        "live_state": live_state,
        "recommended_strategy": {
            "name": WINNER_NAME,
            "k": 3,
            "exit_rule": "6-month rebalance",
            "description": (
                "v5 PIT S&P 500 strategy.  Walk-forward Gradient Boosted Trees "
                "ranking model (3m+6m forward-rank ensemble) restricted to "
                "point-in-time S&P 500 constituents at each rebalance, gated by "
                "the HuggingFace Chronos-bolt-tiny zero-shot foundation model "
                "(p70 forecast rank ≥ 0.45). Top-2 picks held for 6 months; "
                "inverse-volatility weighted with 40% cap per pick; 'tight' "
                "regime gate goes 100% cash on SPY 21d <= -8% or 6m <= -5%. "
                "Walk-forward 49.39% MEAN OOS CAGR over 10 splits 2003-2025 "
                "on the augmented PIT panel, with 10/10 positive, 10/10 beating "
                "SPY. Robust under Monte-Carlo synthetic-delisting overlay: "
                "41.6% median CAGR at α=4%/yr (realistic small/mid-cap rate), "
                "still +29pp over SPY. K=2 dominates K=3 at every alpha tested."
            ),
        },
        "growth": growth,
        "year_by_year": {
            "pullback_in_winner_k1": year_rows,  # backward-compat
            WINNER_NAME: year_rows,
        },
        "walk_forward_aggregate": wf_aggregate_rows,
        "walk_forward_forced": wf_split_rows,
        "splits": [],
        "wf_explanation": {
            "headline_mean_test_cagr": (wf_aggregate_rows[0]["mean_test_cagr"] if wf_aggregate_rows else None),
            "headline_min_test_cagr": (wf_aggregate_rows[0]["min_test_cagr"] if wf_aggregate_rows else None),
            "headline_max_test_cagr": (wf_aggregate_rows[0]["max_test_cagr"] if wf_aggregate_rows else None),
            "n_splits": (wf_aggregate_rows[0]["n_splits"] if wf_aggregate_rows else 0),
            "explanation": (
                "10 walk-forward TRAIN/TEST splits over 2003-2024 on the PIT "
                "S&P 500 universe.  The GBM is fit on TRAIN data only (annual "
                "retrain, 7-month embargo), then applied to TEST.  Reported "
                "metrics are TEST-window CAGR, edge over SPY, Sharpe, and "
                "max drawdown — all out-of-sample."
            ),
        },
        "survivorship": {
            "stratified_default_4pct": stratified_4pct,
            "sensitivity": bias_rows,
            "random_baseline_k1": {"cagr_mean": cagr_spy or 0.10},
        },
        "bias_sensitivity": bias_rows,
        "sub_periods": sub_period_rows,
        "multi_universe_generalisation": gen_rows,
        "parameter_sensitivity": sens_rows,
        "most_picked": most_picked,
        "drawdowns": drawdowns,
        "panel_coverage_yearly": coverage_rows,
        "windows_comparison": [
            {"window": "Full 2003-2025", "strategy_cagr": cagr_strat, "spy_cagr": cagr_spy},
        ],
        "live_picks": [],
        "horizon_stats": horizon_stats,
        "dca_investor": dca_investor,
        "oracle": {},
        "pick_log": pick_log_rows,
        "sweep_top40": [],
    }

    out_path = WEBAPP_OUT / "data.json"
    with open(out_path, "w") as f:
        json.dump(to_jsonable(data), f, indent=1)
    print(f"\nWrote {out_path}")
    print(f"  Strategy: {WINNER_NAME}")
    print(f"  CAGR: {cagr_strat*100:.2f}%, SPY: {(cagr_spy or 0)*100:.2f}%")
    print(f"  Headline pick: {pick_basket[0]['ticker'] if pick_basket else 'none'}")
    print(f"  Current basket: {[p['ticker'] for p in pick_basket]}")
    print(f"  Last rebalance: {live_state['last_rebalance_date']}, next: {live_state['next_rebalance_date']}")


if __name__ == "__main__":
    main()
