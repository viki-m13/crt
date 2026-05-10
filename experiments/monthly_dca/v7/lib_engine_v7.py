"""
v7 — Aggressive downside protection on top of v6.

New mechanisms (proprietary):

1. **own_dd_filter / own_vol_filter** — pre-filter picks whose own 1Y MaxDD or
   1Y vol is too extreme. Removes "death-spiral" candidates that the ML
   model identifies as deep-value but that keep falling.

2. **Conditional Drawdown Insurance (CDI)** — dynamic SH overlay that GROWS
   during market stress, sized continuously from spy_dd_from_52wh and SPY
   21d realised vol. The SH allocation comes from SCALING DOWN the alpha
   sleeve. Behaves like an embedded put without paying option premiums.

3. **Permanent core-satellite** — fixed allocation to SPY or TLT alongside
   the alpha sleeve. Diversifies single-stock concentration risk at the
   cost of some alpha capture.

4. **Per-pick monthly stop-loss** — if a pick is down >X% in a single month,
   exit it (replace with cash or SPY) for the remainder of the hold.

5. **Realised-vol throttle** — monthly check of basket realised vol; if
   above target, scale gross down to target/realised.

6. **Sortino-like weighting** — weight by inverse downside semivariance
   instead of inverse total vol. Tilts away from picks with bad LEFT-tail
   history specifically.

7. **Trend-on-equity** — Faber-style 10-month MA on the strategy's own
   equity curve. When equity dips below the MA, de-risk.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "monthly_dca" / "v6"))

from lib_engine import (  # noqa: E402
    EXCLUDE_TICKERS, V2, FEATURES_DIR, PIT, REGIMES, build_spy_aligned,
    cagr_monthly, evaluate, load_score_panel as _load_score_panel,
    load_spy_features, maxdd_monthly, sharpe_monthly,
    WF_SPLITS,
)


# ---------------------------------------------------------------------------
# Extended panel loader: also attaches own MaxDD and own vol_1y
# ---------------------------------------------------------------------------
def load_panel_v7(scorer: str = "ml_3plus6", universe: str = "sp500_pit") -> pd.DataFrame:
    """Load score panel with extra per-(asof,ticker) features:
    pullback_1y, mom_12_1, trend_health_5y, vol_1y, own_dd_1y, vol_rank.
    own_dd_1y is the same as -|pullback_1y| but normalised positive magnitude.
    """
    panel = _load_score_panel(scorer, universe, attach_pullback=True)
    # own_dd_1y: positive magnitude of 1Y peak-to-trough drawdown (= -pullback_1y)
    if "pullback_1y" in panel.columns:
        panel["own_dd_1y"] = -panel["pullback_1y"].fillna(0).clip(upper=0)
    return panel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class V7Config:
    name: str = "v7"
    scorer: str = "ml_3plus6"
    universe: str = "sp500_pit"
    regime_gate: str = "tight"
    k_normal: int = 3
    k_recovery: int = 3
    k_bull: int = 3
    weighting: str = "invvol"      # ew | invvol | sortino
    hold_months: int = 6
    cost_bps: float = 10.0
    cash_yield_yr: float = 0.03

    # === v7 downside-protection mechanisms ===
    # 1. Pre-pick filters
    own_dd_filter: float = 0.0     # drop picks with own 1Y DD > X (e.g., 0.50 = 50%)
    own_vol_filter: float = 0.0    # drop picks with own 1Y vol > X (e.g., 0.80)

    # 2. Conditional Drawdown Insurance (dynamic SH overlay)
    cdi_max_hedge: float = 0.0     # max SH allocation (e.g., 0.30 = 30%)
    cdi_dd_threshold: float = 0.10 # SPY DD level at which hedge starts to grow
    cdi_vol_threshold: float = 0.20 # SPY 1y vol level at which hedge starts to grow
    cdi_hedge_ticker: str = "SH"   # SH | TLT

    # 3. Permanent core-satellite sleeve
    perm_sleeve_ticker: str = ""   # SPY | TLT | "" for off
    perm_sleeve_weight: float = 0.0  # fraction of capital in perm sleeve

    # 4. Per-pick monthly stop-loss (drop pick if 1m return < -X%)
    pick_stop_loss: float = 0.0    # 0 disabled; 0.25 means -25% triggers

    # 5. Realised vol throttle (monthly)
    rvol_target: float = 0.0       # 0 off; e.g., 0.25 = scale gross to target/basket_rv
    rvol_window: int = 21          # days for realised-vol calc proxy (months in our panel)

    # 6. Trend-on-equity (Faber-style)
    eq_trend_window: int = 0       # 0 off; e.g., 10 = de-risk when equity < 10mo MA

    # === v6 risk controls (carry-forward) ===
    spy_dd_scale: float = 0.0
    spy_dd_floor: float = 0.5
    monthly_exposure: bool = False
    crash_persist: int = 1
    crash_fallback: str = "cash"
    fallback_ticker: str = "SPY"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _nearest_pos(idx: pd.DatetimeIndex, target: pd.Timestamp, tol_days: int = 7) -> Optional[int]:
    pos = idx.searchsorted(target)
    cands = []
    for j in (pos - 1, pos):
        if 0 <= j < len(idx):
            cands.append((j, abs((idx[j] - target).days)))
    cands.sort(key=lambda x: x[1])
    return cands[0][0] if cands and cands[0][1] <= tol_days else None


def _next_month_return(monthly_returns: pd.DataFrame, asof: pd.Timestamp,
                       ticker: str, missing_value: float = -1.0) -> float:
    pos = _nearest_pos(monthly_returns.index, asof)
    if pos is None or pos + 1 >= len(monthly_returns.index):
        return 0.0
    next_d = monthly_returns.index[pos + 1]
    if ticker not in monthly_returns.columns:
        return missing_value
    rr = monthly_returns.at[next_d, ticker]
    return missing_value if pd.isna(rr) else float(rr)


def _basket_realised_vol(basket_history: list[float]) -> float:
    """Compute realised annualised vol from list of recent monthly returns."""
    if len(basket_history) < 3:
        return 0.0
    arr = np.array(basket_history)
    return float(arr.std() * np.sqrt(12))


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
def simulate_v7(cfg: V7Config,
                panel: pd.DataFrame,
                monthly_returns: pd.DataFrame,
                spy_features: pd.DataFrame,
                starting_cash: float = 1.0) -> pd.DataFrame:
    """Pure-cash compounding simulator with v7 protection mechanisms."""
    cls = REGIMES[cfg.regime_gate]
    cf = cfg.cost_bps / 10000.0
    cash_step = (1 + cfg.cash_yield_yr) ** (1 / 12) - 1 if cfg.cash_yield_yr > 0 else 0.0

    by_asof = {pd.Timestamp(d): g.copy() for d, g in panel.groupby("asof")}
    months = sorted(by_asof.keys())
    mr_idx = monthly_returns.index

    equity = starting_cash
    cur_picks: list[str] = []
    cur_unscaled: np.ndarray = np.array([])  # raw alpha-sleeve weights summing to 1
    cur_alive: np.ndarray = np.array([], dtype=bool)  # which picks are still live (not stopped)
    held_for = 0
    in_cash = False
    crash_streak = 0
    peak_equity = equity
    rows = []
    eq_history: list[float] = []      # monthly equity history (for trend-on-eq)
    basket_history: list[float] = []  # last few monthly basket returns (for rvol)

    for i, m in enumerate(months):
        spy_now = spy_features.loc[m].to_dict() if m in spy_features.index else {}
        regime = cls(spy_now)
        if regime == "crash":
            crash_streak += 1
        else:
            crash_streak = 0
        eff_regime = regime
        if regime == "crash" and crash_streak < cfg.crash_persist:
            eff_regime = "normal"

        # Trend-on-equity gate (Faber-style on the strategy's own equity)
        eq_below_ma = False
        if cfg.eq_trend_window > 0 and len(eq_history) >= cfg.eq_trend_window:
            ma = float(np.mean(eq_history[-cfg.eq_trend_window:]))
            if equity < ma:
                eq_below_ma = True

        do_reb = (i == 0) or (held_for >= cfg.hold_months) or in_cash

        if do_reb:
            if eff_regime == "crash":
                cur_picks, cur_unscaled = [], np.array([])
                in_cash = True
                held_for = 0
                gross_alpha = 0.0
            else:
                k = {"recovery": cfg.k_recovery, "bull": cfg.k_bull,
                     "normal": cfg.k_normal, "warning": cfg.k_normal}[eff_regime]
                sub = by_asof.get(m, pd.DataFrame())
                # 1. Pre-pick filters
                if cfg.own_dd_filter > 0 and "own_dd_1y" in sub.columns:
                    sub = sub[sub["own_dd_1y"].fillna(0) <= cfg.own_dd_filter]
                if cfg.own_vol_filter > 0 and "vol_1y" in sub.columns:
                    sub = sub[sub["vol_1y"].fillna(0) <= cfg.own_vol_filter]
                if len(sub) < k:
                    cur_picks, cur_unscaled = [], np.array([])
                    in_cash = True
                    gross_alpha = 0.0
                else:
                    top = sub.sort_values("score", ascending=False).head(k)
                    cur_picks = top["ticker"].tolist()
                    if cfg.weighting == "ew":
                        w = np.ones(k) / k
                    elif cfg.weighting == "invvol":
                        vv = top["vol_1y"].values
                        vv = np.where(np.isnan(vv) | (vv <= 0), 0.4, vv)
                        invv = 1.0 / vv
                        w = invv / invv.sum()
                    elif cfg.weighting == "sortino":
                        # Approximate downside semivariance with own_dd_1y
                        # Higher own_dd_1y → larger downside, lower weight
                        dd = top["own_dd_1y"].values
                        dd = np.where(np.isnan(dd) | (dd <= 0), 0.10, dd)
                        invd = 1.0 / dd
                        w = invd / invd.sum()
                    else:
                        w = np.ones(k) / k
                    cur_unscaled = w
                    cur_alive = np.ones(len(w), dtype=bool)
                    in_cash = False
                    gross_alpha = 1.0
            held_for = 0
        else:
            gross_alpha = float(cur_unscaled.sum()) if len(cur_unscaled) else 0.0

        # === Apply gross-scaling overlays (every month, not just rebalance) ===
        # SPY DD continuous scale
        if cfg.spy_dd_scale > 0 and not in_cash:
            dd52 = float(spy_now.get("spy_dd_from_52wh", 0.0))
            if dd52 < 0:
                f = max(cfg.spy_dd_floor,
                        1.0 + (dd52 / cfg.spy_dd_scale) * (1.0 - cfg.spy_dd_floor))
                gross_alpha *= f

        # Realised vol throttle (using actual basket history)
        if cfg.rvol_target > 0 and not in_cash:
            rv = _basket_realised_vol(basket_history[-12:])  # last 12 months
            if rv > cfg.rvol_target:
                gross_alpha *= float(min(cfg.rvol_target / rv, 1.0))

        # Trend-on-equity de-risk (Faber-style)
        if eq_below_ma and not in_cash:
            gross_alpha *= 0.5

        # === CDI hedge sizing ===
        cdi_w = 0.0
        if cfg.cdi_max_hedge > 0 and cfg.cdi_hedge_ticker in monthly_returns.columns:
            dd52 = float(spy_now.get("spy_dd_from_52wh", 0.0))
            spy_vol = float(spy_now.get("spy_vol_1y", 0.15))
            stress_dd = max(0.0, -dd52 / cfg.cdi_dd_threshold) if cfg.cdi_dd_threshold > 0 else 0.0
            stress_vol = max(0.0, (spy_vol - cfg.cdi_vol_threshold) / max(cfg.cdi_vol_threshold, 1e-9))
            stress = max(stress_dd, stress_vol)
            cdi_w = float(min(stress * cfg.cdi_max_hedge, cfg.cdi_max_hedge))
            # CDI hedge eats into gross_alpha
            gross_alpha = max(0.0, gross_alpha - cdi_w)

        # === Permanent sleeve ===
        perm_w = 0.0
        if cfg.perm_sleeve_ticker and cfg.perm_sleeve_weight > 0 and cfg.perm_sleeve_ticker in monthly_returns.columns:
            perm_w = cfg.perm_sleeve_weight
            # Perm sleeve eats into alpha gross
            gross_alpha = max(0.0, gross_alpha - perm_w)

        # === Compute month return ===
        # Alpha sleeve return
        if in_cash or len(cur_picks) == 0 or gross_alpha <= 0:
            alpha_ret = 0.0
        else:
            pick_rets = []
            for tk in cur_picks:
                pick_rets.append(_next_month_return(monthly_returns, m, tk))
            pick_rets = np.array(pick_rets)
            # Per-pick monthly stop-loss (REALISTIC):
            #   - If a pick's monthly return <= -X, lock in -X loss for THAT month
            #     and mark the pick dead. Subsequent months until rebalance: 0 return
            #     (capital sits in cash for that pick, earning cash_step).
            #   - Already-dead picks earn cash_step (we record alive mask via cur_alive).
            if cfg.pick_stop_loss > 0:
                # Apply stop only to currently alive picks
                trigger = cur_alive & (pick_rets <= -cfg.pick_stop_loss)
                pick_rets = np.where(trigger, -cfg.pick_stop_loss, pick_rets)
                # Dead picks (already stopped before) earn cash, not stock return
                pick_rets = np.where(~cur_alive, cash_step, pick_rets)
                # Mark newly stopped picks as dead from next month onward
                cur_alive = cur_alive & ~trigger
            alpha_ret = float((pick_rets * cur_unscaled).sum())  # raw return on sleeve

        # CDI hedge return
        cdi_ret = 0.0
        if cdi_w > 0:
            cdi_ret = _next_month_return(monthly_returns, m, cfg.cdi_hedge_ticker, missing_value=0.0)

        # Permanent sleeve return
        perm_ret = 0.0
        if perm_w > 0:
            perm_ret = _next_month_return(monthly_returns, m, cfg.perm_sleeve_ticker, missing_value=0.0)

        # Cash bucket return (residual)
        residual = 1.0 - gross_alpha - cdi_w - perm_w
        cash_ret = cash_step * max(0.0, residual)

        ret_m = gross_alpha * alpha_ret + cdi_w * cdi_ret + perm_w * perm_ret + cash_ret

        # Apply costs at rebalance: 10bp on the alpha sleeve being deployed/rotated
        if not in_cash and len(cur_picks) > 0 and do_reb:
            equity *= (1 + ret_m) * (1 - cf * gross_alpha)
        else:
            equity *= (1 + ret_m)
        held_for += 1
        peak_equity = max(peak_equity, equity)
        eq_history.append(equity)

        # Compute the realised basket return (for vol throttle)
        if not in_cash and gross_alpha > 0:
            basket_history.append(alpha_ret)
        else:
            basket_history.append(0.0)
        basket_history = basket_history[-24:]  # keep 2y history

        rows.append({
            "date": m, "equity": equity, "ret_m": ret_m,
            "regime": eff_regime if not in_cash else "cash",
            "n_picks": len(cur_picks),
            "gross_alpha": float(gross_alpha),
            "cdi_w": float(cdi_w),
            "perm_w": float(perm_w),
            "picks": ",".join(cur_picks),
            "weights_csv": ",".join(f"{w*gross_alpha:.4f}" for w in cur_unscaled) if len(cur_unscaled) else "",
        })
    return pd.DataFrame(rows)
