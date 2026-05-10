"""
v8 — Custom engine with novel features:
- Per-pick stop-loss (daily resolution) using prices_extended
- Adaptive holding (6m default, but break early if losers)
- Dynamic K (concentrate when conviction high)
- Optional gross > 1.0 (leverage with risk management)
- Multi-sleeve staggered rebalance (smooths picks)
- Momentum-confirmation filter (must be above 200dma)

Designed to support proprietary scoring strategies via a `score_panel` input
with the same shape as the v6 engine: asof, ticker, score, vol_1y, ...

Engine returns a monthly equity DataFrame compatible with the v6 evaluate().
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
PIT = CACHE / "v2" / "sp500_pit"
FEATURES_DIR = CACHE / "features"

V6 = ROOT / "experiments" / "monthly_dca" / "v6"
sys.path.insert(0, str(V6))
from lib_engine import (  # type: ignore
    REGIMES, regime_tight, load_spy_features,
    cagr_monthly, sharpe_monthly, maxdd_monthly,
    WF_SPLITS, build_spy_aligned, evaluate,
)


@dataclass
class V8Config:
    name: str = "v8"
    # Selection
    k_normal: int = 3
    k_bull: int = 3
    k_recovery: int = 3
    # Adaptive K: if top-pick gap (rank-2 minus rank-1) > threshold AND top score > X, reduce K
    adaptive_k: bool = False
    # Weighting
    weighting: str = "ew"            # ew | invvol | conv
    # Holding
    hold_months: int = 6
    # Per-pick early-replace: if a pick is down more than X from entry by month-end, swap it
    pick_replace_dd: float = 0.0     # 0 disables
    # Per-pick daily stop-loss (uses daily prices)
    pick_daily_stop: float = 0.0     # 0 disables; e.g. 0.30 = -30% from entry
    # Regime gate
    regime_gate: str = "tight"
    crash_persist: int = 1
    # Cash & costs
    cost_bps: float = 10.0
    cash_yield_yr: float = 0.03
    # Filters on each pick
    filt_d_sma200_min: float = -1.0  # default permissive (-1 = no filter)
    filt_pullback_min: float = -1.0  # min pullback_1y allowed (-0.7 = drop stocks down >70% from 1y high)
    filt_min_mom_12_1: float = -1.0  # min mom_12_1 allowed
    # Leverage
    gross_target: float = 1.0        # 1.0 = no leverage; 1.5/2.0 = leverage in non-crash
    gross_in_recovery: float = -1.0  # if >0, override gross during recovery (-1 = use target)
    gross_in_bull: float = -1.0      # if >0, override gross during bull
    gross_in_normal: float = -1.0    # if >0, override gross during normal
    # Smart dynamic gross: scales by SPY DD-from-52w-high (start at gross_target,
    # de-lever linearly to floor as DD reaches `dd_full_floor`).
    # E.g., gross_target=1.5, dd_full_floor=0.10, gross_floor=0.0 →
    #   DD=0% → 1.5x; DD=5% → 0.75x; DD=10% → 0.0x (full cash)
    dd_full_floor: float = 0.0       # 0 disables; otherwise DD% at which floor reached
    gross_floor: float = 0.0         # gross when DD ≥ dd_full_floor
    # Trend-confirmation lever: only deploy gross_target if SPY > 200dma; else use gross_floor
    spy_trend_only_lever: bool = False
    # Multi-sleeve (staggered rebalance: hold but with N sleeves rotated monthly)
    n_sleeves: int = 1


def _nearest_pos(idx: pd.DatetimeIndex, target: pd.Timestamp, tol_days: int = 7) -> Optional[int]:
    pos = idx.searchsorted(target)
    cands = []
    for j in (pos - 1, pos):
        if 0 <= j < len(idx):
            cands.append((j, abs((idx[j] - target).days)))
    cands.sort(key=lambda x: x[1])
    if cands and cands[0][1] <= tol_days:
        return cands[0][0]
    return None


def _next_md(mr_idx: pd.DatetimeIndex, t: pd.Timestamp) -> Optional[pd.Timestamp]:
    p = _nearest_pos(mr_idx, t)
    if p is None or p + 1 >= len(mr_idx):
        return None
    return mr_idx[p + 1]


def simulate_v8(cfg: V8Config,
                score_panel: pd.DataFrame,
                monthly_returns: pd.DataFrame,
                spy_features: pd.DataFrame,
                daily_prices: Optional[pd.DataFrame] = None,
                starting_cash: float = 1.0) -> pd.DataFrame:
    """
    Simulate the v8 strategy. Picks are chosen at each rebalance point per
    regime and held until next rebalance, optionally replaced if breaching
    a daily stop-loss.
    """
    cls_fn = REGIMES[cfg.regime_gate]
    cf = cfg.cost_bps / 10000.0
    cash_step = (1 + cfg.cash_yield_yr) ** (1 / 12) - 1 if cfg.cash_yield_yr > 0 else 0.0

    by_asof = {pd.Timestamp(d): g.copy() for d, g in score_panel.groupby("asof")}
    months = sorted(by_asof.keys())
    mr_idx = monthly_returns.index

    # State
    equity = starting_cash
    rows = []

    # Multi-sleeve setup: each sleeve has its own state
    n_sleeves = max(1, cfg.n_sleeves)
    sleeves = [
        {
            "picks": [],            # list[str]
            "weights": np.array([]),
            "entry_prices": {},     # {tk: float}
            "stopped": set(),       # tickers that hit stop-loss
            "held_for": 0,
            "in_cash": True,
            "sleeve_capital": 1.0 / n_sleeves,
        }
        for _ in range(n_sleeves)
    ]

    crash_streak = 0
    peak_equity = equity

    def _select_picks(asof, regime, n):
        sub = by_asof.get(asof, pd.DataFrame())
        if len(sub) == 0:
            return [], np.array([])
        s = sub.copy()
        # Filters
        if cfg.filt_d_sma200_min > -1.0 and "d_sma200" in s.columns:
            s = s[s["d_sma200"].fillna(-1.0) >= cfg.filt_d_sma200_min]
        if cfg.filt_pullback_min > -1.0 and "pullback_1y" in s.columns:
            s = s[s["pullback_1y"].fillna(-1.0) >= cfg.filt_pullback_min]
        if cfg.filt_min_mom_12_1 > -1.0 and "mom_12_1" in s.columns:
            s = s[s["mom_12_1"].fillna(-1.0) >= cfg.filt_min_mom_12_1]
        if len(s) < n:
            return [], np.array([])
        top = s.sort_values("score", ascending=False).head(n)
        if cfg.weighting == "ew":
            w = np.ones(n) / n
        elif cfg.weighting == "invvol":
            vv = top["vol_1y"].values
            vv = np.where(np.isnan(vv) | (vv <= 0), 0.4, vv)
            w = (1.0 / vv) / (1.0 / vv).sum()
        elif cfg.weighting == "conv":
            sc = top["score"].values
            shifted = sc - sc.min() + 1e-6
            w = shifted / shifted.sum()
        else:
            w = np.ones(n) / n
        return top["ticker"].tolist(), w

    def _gross_for_regime(regime: str, spy_state: dict) -> float:
        g = cfg.gross_target
        if regime == "bull" and cfg.gross_in_bull > 0:
            g = cfg.gross_in_bull
        elif regime == "recovery" and cfg.gross_in_recovery > 0:
            g = cfg.gross_in_recovery
        elif regime == "normal" and cfg.gross_in_normal > 0:
            g = cfg.gross_in_normal
        # Smart dynamic: linearly de-lever as SPY DD grows
        if cfg.dd_full_floor > 0:
            dd = float(spy_state.get("spy_dd_from_52wh", 0.0))  # signed (negative)
            depth = max(0.0, -dd)
            scale = max(0.0, 1.0 - depth / cfg.dd_full_floor)
            g = cfg.gross_floor + (g - cfg.gross_floor) * scale
        if cfg.spy_trend_only_lever:
            d2 = float(spy_state.get("spy_dsma200", 0.0))
            if d2 < 0:
                g = min(g, cfg.gross_floor if cfg.gross_floor > 0 else 1.0)
        return g

    def _entry_price(asof, ticker):
        """Closest daily price at or just after the asof month-end (open of next session)."""
        if daily_prices is None or ticker not in daily_prices.columns:
            return None
        idx = daily_prices.index
        pos = idx.searchsorted(asof)
        if pos >= len(idx):
            return None
        return float(daily_prices.iloc[pos][ticker]) if not pd.isna(daily_prices.iloc[pos][ticker]) else None

    def _check_stop_breach(ticker, entry_price, start_d, end_d, stop_pct):
        if daily_prices is None or stop_pct <= 0 or entry_price is None:
            return False
        if ticker not in daily_prices.columns:
            return False
        idx = daily_prices.index
        i0 = idx.searchsorted(start_d)
        i1 = idx.searchsorted(end_d)
        seg = daily_prices.iloc[i0:i1+1][ticker]
        if seg.isna().all():
            return False
        stop_px = entry_price * (1.0 - stop_pct)
        return bool((seg <= stop_px).any())

    for i, m in enumerate(months):
        spy_now = spy_features.loc[m].to_dict() if m in spy_features.index else {}
        regime = cls_fn(spy_now)
        if regime == "crash":
            crash_streak += 1
        else:
            crash_streak = 0
        eff_regime = regime if crash_streak >= cfg.crash_persist else "normal"

        # Determine sleeve(s) to rebalance this month
        sleeves_to_reb = []
        for s_idx, sl in enumerate(sleeves):
            # Stagger: sleeve s_idx rebalances when (i % hold_months) == s_idx % hold_months
            if n_sleeves > 1:
                if (i % cfg.hold_months) == (s_idx % cfg.hold_months) or sl["in_cash"] or sl["held_for"] >= cfg.hold_months:
                    sleeves_to_reb.append(s_idx)
            else:
                if i == 0 or sl["held_for"] >= cfg.hold_months or sl["in_cash"]:
                    sleeves_to_reb.append(s_idx)

        # Determine K for this asof
        k_target = {"normal": cfg.k_normal, "bull": cfg.k_bull, "recovery": cfg.k_recovery}.get(eff_regime, cfg.k_normal)

        # Process rebalances
        for s_idx in sleeves_to_reb:
            sl = sleeves[s_idx]
            if eff_regime == "crash":
                sl["picks"] = []
                sl["weights"] = np.array([])
                sl["entry_prices"] = {}
                sl["stopped"] = set()
                sl["in_cash"] = True
                sl["held_for"] = 0
            else:
                picks, w = _select_picks(m, eff_regime, k_target)
                if len(picks) == 0:
                    sl["picks"] = []
                    sl["weights"] = np.array([])
                    sl["entry_prices"] = {}
                    sl["stopped"] = set()
                    sl["in_cash"] = True
                    sl["held_for"] = 0
                else:
                    sl["picks"] = picks
                    sl["weights"] = w
                    if daily_prices is not None and cfg.pick_daily_stop > 0:
                        sl["entry_prices"] = {tk: _entry_price(m, tk) for tk in picks}
                    else:
                        sl["entry_prices"] = {}
                    sl["stopped"] = set()
                    sl["in_cash"] = False
                    sl["held_for"] = 0

        # Compute monthly return for the portfolio
        next_d = _next_md(mr_idx, m)
        sleeve_rets = []
        for s_idx, sl in enumerate(sleeves):
            cap = sl["sleeve_capital"]
            if sl["in_cash"] or len(sl["picks"]) == 0 or next_d is None:
                # cash earns yield only
                ret_sleeve = cash_step
            else:
                pick_rets = []
                # Apply per-pick daily stop
                for tk in sl["picks"]:
                    if tk in sl["stopped"]:
                        # already stopped — capital sits in cash for remainder of this hold cycle
                        pick_rets.append(cash_step)
                        continue
                    # check if stop breached during this month
                    if cfg.pick_daily_stop > 0 and daily_prices is not None and tk in sl["entry_prices"]:
                        ep = sl["entry_prices"].get(tk)
                        if ep is not None:
                            breach = _check_stop_breach(tk, ep, m, next_d, cfg.pick_daily_stop)
                            if breach:
                                sl["stopped"].add(tk)
                                # realized return = -stop_pct (slippage 0)
                                pick_rets.append(-cfg.pick_daily_stop)
                                continue
                    if tk in monthly_returns.columns:
                        rr = monthly_returns.at[next_d, tk]
                        pick_rets.append(-1.0 if pd.isna(rr) else float(rr))
                    else:
                        pick_rets.append(-1.0)
                pick_rets = np.array(pick_rets)
                ret_sleeve = float((pick_rets * sl["weights"]).sum())
                # Apply gross factor (per-sleeve, dynamic on SPY state)
                gross = _gross_for_regime(eff_regime, spy_now)
                # leverage cost: borrow at cash_yield rate (assume same)
                if gross != 1.0:
                    ret_sleeve = gross * ret_sleeve - (gross - 1.0) * cash_step

            sleeve_rets.append(cap * ret_sleeve)
            sl["held_for"] += 1

        ret_m = float(sum(sleeve_rets))
        # Apply transaction cost on rebalances (one-time)
        if any(s in sleeves_to_reb and not sleeves[s]["in_cash"] for s in range(n_sleeves)):
            n_reb = sum(1 for s in sleeves_to_reb if not sleeves[s]["in_cash"])
            ret_m -= cf * (n_reb / max(1, n_sleeves))

        equity *= (1 + ret_m)
        peak_equity = max(peak_equity, equity)

        rows.append({
            "date": m, "equity": equity, "ret_m": ret_m,
            "regime": eff_regime if any(not s["in_cash"] for s in sleeves) else "cash",
            "n_picks": sum(len(s["picks"]) for s in sleeves),
            "gross": _gross_for_regime(eff_regime, spy_now) if any(not s["in_cash"] for s in sleeves) else 0.0,
            "picks": ";".join("|".join(s["picks"]) for s in sleeves),
        })

    return pd.DataFrame(rows)


def evaluate_v8(eq: pd.DataFrame, monthly_returns: pd.DataFrame, name: str = "") -> dict:
    spy_aligned = build_spy_aligned(eq, monthly_returns)
    return evaluate(eq, spy_aligned, name=name)


if __name__ == "__main__":
    print("v8_engine module ready")
