"""Validate the per-pick stop-loss using DAILY prices.

The monthly engine in lib_engine_v7.py uses month-end returns to detect stops.
This is a coarse approximation: a stock that drops -15% intra-month and
recovers to -8% by month-end shows -8% in monthly data, so no stop is fired.
With daily prices, the stop WOULD fire intra-month.

This validator re-prices the v7 simulator's picks at daily resolution to
measure the difference. Stops are set at entry price (last month's close)
- X% and triggered intra-month if breached.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "monthly_dca" / "v6"))
sys.path.insert(0, str(ROOT / "experiments" / "monthly_dca" / "v7"))

from lib_engine_v7 import (V7Config, simulate_v7, load_panel_v7,
                           load_spy_features, build_spy_aligned, evaluate, V2)
from lib_engine import REGIMES

DAILY = pd.read_parquet(ROOT / "experiments/monthly_dca/cache/prices_extended.parquet")


def daily_stop_pick_returns(picks: list[str], asof: pd.Timestamp, next_d: pd.Timestamp,
                            stop: float, slippage: float = 0.0) -> tuple[list[float], list[bool]]:
    """For each pick, compute the realised monthly return assuming a stop at -stop
    set at the close of `asof`, monitored daily until `next_d`.

    Returns: (per-pick return for the month, per-pick stopped flag)
        - If stop never triggers: return = (P_next_d / P_asof) - 1
        - If stop triggers: return = -(stop + slippage), then sit in cash
    """
    rets = []
    stopped_flags = []
    daily = DAILY.loc[asof:next_d]
    if len(daily) < 2:
        # No intra-month data
        for tk in picks:
            if tk not in DAILY.columns:
                rets.append(-1.0)
                stopped_flags.append(False)
                continue
            p0 = DAILY.loc[asof:asof, tk]
            p1 = DAILY.loc[next_d:next_d, tk]
            if p0.empty or p1.empty or p0.iloc[0] == 0 or pd.isna(p0.iloc[0]) or pd.isna(p1.iloc[0]):
                rets.append(-1.0)
                stopped_flags.append(False)
            else:
                rets.append(float(p1.iloc[0] / p0.iloc[0] - 1))
                stopped_flags.append(False)
        return rets, stopped_flags

    for tk in picks:
        if tk not in DAILY.columns:
            rets.append(-1.0)
            stopped_flags.append(False)
            continue
        series = daily[tk].dropna()
        if len(series) < 2:
            rets.append(-1.0)
            stopped_flags.append(False)
            continue
        p0 = series.iloc[0]
        ret_path = series / p0 - 1.0
        # Did we breach the stop?
        breached = ret_path[ret_path <= -stop]
        if len(breached) > 0:
            # Lock in stop + slippage
            rets.append(-(stop + slippage))
            stopped_flags.append(True)
        else:
            rets.append(float(ret_path.iloc[-1]))
            stopped_flags.append(False)
    return rets, stopped_flags


def simulate_daily_stop(cfg: V7Config, panel: pd.DataFrame,
                        monthly_returns: pd.DataFrame, spy_features: pd.DataFrame,
                        starting_cash: float = 1.0) -> pd.DataFrame:
    """Like simulate_v7 but uses daily-prices to evaluate the stop loss."""
    cls = REGIMES[cfg.regime_gate]
    cf = cfg.cost_bps / 10000.0
    cash_step = (1 + cfg.cash_yield_yr) ** (1 / 12) - 1 if cfg.cash_yield_yr > 0 else 0.0

    by_asof = {pd.Timestamp(d): g.copy() for d, g in panel.groupby("asof")}
    months = sorted(by_asof.keys())
    mr_idx = monthly_returns.index

    equity = starting_cash
    cur_picks: list[str] = []
    cur_unscaled: np.ndarray = np.array([])
    cur_alive: np.ndarray = np.array([], dtype=bool)
    held_for = 0
    in_cash = False
    crash_streak = 0
    rows = []
    n_stops_total = 0

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

        do_reb = (i == 0) or (held_for >= cfg.hold_months) or in_cash

        if do_reb:
            if eff_regime == "crash":
                cur_picks, cur_unscaled, cur_alive = [], np.array([]), np.array([], dtype=bool)
                in_cash = True
                held_for = 0
                gross_alpha = 0.0
            else:
                k = {"recovery": cfg.k_recovery, "bull": cfg.k_bull,
                     "normal": cfg.k_normal, "warning": cfg.k_normal}[eff_regime]
                sub = by_asof.get(m, pd.DataFrame())
                if cfg.own_dd_filter > 0 and "own_dd_1y" in sub.columns:
                    sub = sub[sub["own_dd_1y"].fillna(0) <= cfg.own_dd_filter]
                if cfg.own_vol_filter > 0 and "vol_1y" in sub.columns:
                    sub = sub[sub["vol_1y"].fillna(0) <= cfg.own_vol_filter]
                if len(sub) < k:
                    cur_picks, cur_unscaled, cur_alive = [], np.array([]), np.array([], dtype=bool)
                    in_cash = True
                    gross_alpha = 0.0
                else:
                    top = sub.sort_values("score", ascending=False).head(k)
                    cur_picks = top["ticker"].tolist()
                    if cfg.weighting == "invvol":
                        vv = top["vol_1y"].values
                        vv = np.where(np.isnan(vv) | (vv <= 0), 0.4, vv)
                        invv = 1.0 / vv
                        w = invv / invv.sum()
                    else:
                        w = np.ones(k) / k
                    cur_unscaled = w
                    cur_alive = np.ones(k, dtype=bool)
                    in_cash = False
                    gross_alpha = 1.0
            held_for = 0
        else:
            gross_alpha = float(cur_unscaled.sum()) if len(cur_unscaled) else 0.0

        # Compute month return WITH daily-resolution stop-loss
        if in_cash or len(cur_picks) == 0 or gross_alpha <= 0:
            alpha_ret = 0.0
        else:
            # Find next month's date in daily panel (closest)
            pos = mr_idx.searchsorted(m)
            cands = []
            for j in (pos - 1, pos):
                if 0 <= j < len(mr_idx):
                    cands.append((j, abs((mr_idx[j] - m).days)))
            cands.sort(key=lambda x: x[1])
            if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(mr_idx):
                alpha_ret = 0.0
            else:
                next_d = mr_idx[cands[0][0] + 1]
                if cfg.pick_stop_loss > 0:
                    pick_rets, stopped = daily_stop_pick_returns(
                        cur_picks, m, next_d, cfg.pick_stop_loss, slippage=0.0)
                    pick_rets = np.array(pick_rets)
                    stopped_arr = np.array(stopped, dtype=bool)
                    # Only fire stop on alive picks
                    new_dead = stopped_arr & cur_alive
                    n_stops_total += int(new_dead.sum())
                    # Already-dead picks earn cash, not stock returns
                    pick_rets = np.where(~cur_alive, cash_step, pick_rets)
                    cur_alive = cur_alive & ~new_dead
                else:
                    pick_rets = []
                    for tk in cur_picks:
                        if tk in monthly_returns.columns:
                            rr = monthly_returns.at[next_d, tk]
                            pick_rets.append(-1.0 if pd.isna(rr) else float(rr))
                        else:
                            pick_rets.append(-1.0)
                    pick_rets = np.array(pick_rets)
                alpha_ret = float((pick_rets * cur_unscaled).sum())

        # CDI dynamic hedge sizing — eats from gross_alpha
        cdi_w = 0.0
        cdi_ret = 0.0
        if cfg.cdi_max_hedge > 0 and cfg.cdi_hedge_ticker in monthly_returns.columns:
            dd52 = float(spy_now.get("spy_dd_from_52wh", 0.0))
            spy_vol = float(spy_now.get("spy_vol_1y", 0.15))
            stress_dd = max(0.0, -dd52 / cfg.cdi_dd_threshold) if cfg.cdi_dd_threshold > 0 else 0.0
            stress_vol = max(0.0, (spy_vol - cfg.cdi_vol_threshold) / max(cfg.cdi_vol_threshold, 1e-9))
            stress = max(stress_dd, stress_vol)
            cdi_w = float(min(stress * cfg.cdi_max_hedge, cfg.cdi_max_hedge))
            gross_alpha = max(0.0, gross_alpha - cdi_w)
            pos = mr_idx.searchsorted(m)
            cands = []
            for j in (pos - 1, pos):
                if 0 <= j < len(mr_idx):
                    cands.append((j, abs((mr_idx[j] - m).days)))
            cands.sort(key=lambda x: x[1])
            if cands and cands[0][1] <= 7 and cands[0][0] + 1 < len(mr_idx):
                next_d = mr_idx[cands[0][0] + 1]
                rr = monthly_returns.at[next_d, cfg.cdi_hedge_ticker]
                cdi_ret = 0.0 if pd.isna(rr) else float(rr)

        # Permanent sleeve — also eats from gross_alpha
        perm_w = 0.0
        perm_ret = 0.0
        if cfg.perm_sleeve_ticker and cfg.perm_sleeve_weight > 0 and cfg.perm_sleeve_ticker in monthly_returns.columns:
            perm_w = cfg.perm_sleeve_weight
            gross_alpha = max(0.0, gross_alpha - perm_w)
            pos = mr_idx.searchsorted(m)
            cands = []
            for j in (pos - 1, pos):
                if 0 <= j < len(mr_idx):
                    cands.append((j, abs((mr_idx[j] - m).days)))
            cands.sort(key=lambda x: x[1])
            if cands and cands[0][1] <= 7 and cands[0][0] + 1 < len(mr_idx):
                next_d = mr_idx[cands[0][0] + 1]
                rr = monthly_returns.at[next_d, cfg.perm_sleeve_ticker]
                perm_ret = 0.0 if pd.isna(rr) else float(rr)

        residual = 1.0 - gross_alpha - perm_w - cdi_w
        cash_ret = cash_step * max(0.0, residual)
        ret_m = gross_alpha * alpha_ret + perm_w * perm_ret + cdi_w * cdi_ret + cash_ret

        if not in_cash and len(cur_picks) > 0 and do_reb:
            equity *= (1 + ret_m) * (1 - cf * gross_alpha)
        else:
            equity *= (1 + ret_m)
        held_for += 1

        rows.append({
            "date": m, "equity": equity, "ret_m": ret_m,
            "regime": eff_regime if not in_cash else "cash",
            "n_picks": len(cur_picks),
            "n_alive": int(cur_alive.sum()) if len(cur_alive) else 0,
            "gross_alpha": float(gross_alpha),
            "perm_w": float(perm_w),
            "picks": ",".join(cur_picks),
        })
    print(f"  total stops fired: {n_stops_total}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    panel = load_panel_v7("ml_3plus6", "sp500_pit")
    mr = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy = load_spy_features()

    print("=== Comparing monthly-stop vs daily-stop (sl=10%) ===")
    cfg_m = V7Config(weighting="invvol", cash_yield_yr=0.03, pick_stop_loss=0.10)
    eq_m = simulate_v7(cfg_m, panel, mr, spy)
    m_m = evaluate(eq_m, build_spy_aligned(eq_m, mr), "monthly_stop")
    print(f"  monthly stop (no intra-month detection):")
    print(f"    cagr={m_m['cagr_full']:.4f} sh={m_m['sharpe']:.4f} mdd={m_m['max_dd']:.4f} wf={m_m['wf_mean_cagr']:.4f}")

    eq_d = simulate_daily_stop(cfg_m, panel, mr, spy)
    m_d = evaluate(eq_d, build_spy_aligned(eq_d, mr), "daily_stop")
    print(f"  daily stop (intra-month detection, slippage=0):")
    print(f"    cagr={m_d['cagr_full']:.4f} sh={m_d['sharpe']:.4f} mdd={m_d['max_dd']:.4f} wf={m_d['wf_mean_cagr']:.4f}")

    print()
    print("=== Daily stop sweep ===")
    for sl in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        cfg = V7Config(weighting="invvol", cash_yield_yr=0.03, pick_stop_loss=sl)
        eq = simulate_daily_stop(cfg, panel, mr, spy)
        m = evaluate(eq, build_spy_aligned(eq, mr), f"daily_sl{sl}")
        print(f"  sl={sl}: cagr={m['cagr_full']:.4f} sh={m['sharpe']:.4f} mdd={m['max_dd']:.4f} wf={m['wf_mean_cagr']:.4f} wmin={m['wf_min_cagr']:.4f}")
