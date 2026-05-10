"""Daily-resolution simulator with trailing stop from running peak.

Each pick is monitored daily. The trailing stop tracks the running maximum
since entry; if the price falls X% below this peak, the position is exited.

This locks in unrealised gains while cutting losses — strictly better than
an entry-relative stop in theory because winning picks aren't stopped on
normal pullbacks from entry.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "monthly_dca" / "v6"))
sys.path.insert(0, str(ROOT / "experiments" / "monthly_dca" / "v7"))

from lib_engine_v7 import (V7Config, load_panel_v7, load_spy_features,
                           build_spy_aligned, evaluate, V2)
from lib_engine import REGIMES

DAILY = pd.read_parquet(ROOT / "experiments/monthly_dca/cache/prices_extended.parquet")


def trailing_stop_pick_returns(picks: list[str], asof: pd.Timestamp,
                               next_d: pd.Timestamp,
                               trail: float, slippage: float = 0.0
                               ) -> tuple[list[float], list[bool]]:
    """For each pick, compute realised monthly return with trailing stop from peak.

    Stop fires if price falls `trail` below running max since entry.
    On stop: lock in (peak * (1 - trail) / entry - 1) = realised_peak_return - trail - slippage.
    """
    rets = []
    stopped = []
    daily = DAILY.loc[asof:next_d]
    for tk in picks:
        if tk not in DAILY.columns:
            rets.append(-1.0)
            stopped.append(False)
            continue
        series = daily[tk].dropna()
        if len(series) < 2:
            rets.append(0.0)
            stopped.append(False)
            continue
        p0 = series.iloc[0]
        ratio = series / p0
        running_peak = ratio.cummax()
        # Stop fires when ratio drops trail*peak below the peak:
        #   ratio < running_peak * (1 - trail)
        breach = ratio < (running_peak * (1 - trail))
        if breach.any():
            first_breach = breach.idxmax()  # first True position
            # Realised return = peak at the time of breach * (1 - trail) - 1
            peak_at_breach = float(running_peak.loc[first_breach])
            realised = peak_at_breach * (1 - trail) - 1.0 - slippage
            rets.append(float(realised))
            stopped.append(True)
        else:
            rets.append(float(ratio.iloc[-1] - 1))
            stopped.append(False)
    return rets, stopped


def simulate_trailing_stop(cfg: V7Config, panel: pd.DataFrame,
                           monthly_returns: pd.DataFrame,
                           spy_features: pd.DataFrame,
                           starting_cash: float = 1.0) -> pd.DataFrame:
    """Daily-resolution sim with trailing-stop monitoring per pick during hold."""
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
    n_stops = 0

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

        # Compute month return with trailing stop
        if in_cash or len(cur_picks) == 0 or gross_alpha <= 0:
            alpha_ret = 0.0
        else:
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
                    pick_rets, was_stopped = trailing_stop_pick_returns(
                        cur_picks, m, next_d, cfg.pick_stop_loss, slippage=0.0)
                    pick_rets = np.array(pick_rets)
                    was_stopped_arr = np.array(was_stopped, dtype=bool)
                    new_dead = was_stopped_arr & cur_alive
                    n_stops += int(new_dead.sum())
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

        # Permanent sleeve
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

        residual = 1.0 - gross_alpha - perm_w
        cash_ret = cash_step * max(0.0, residual)
        ret_m = gross_alpha * alpha_ret + perm_w * perm_ret + cash_ret

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
    print(f"  total trailing stops fired: {n_stops}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    panel = load_panel_v7("ml_3plus6", "sp500_pit")
    mr = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy = load_spy_features()

    print("=== Trailing-stop sweep (entry-peak based) ===")
    for trail in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
        cfg = V7Config(weighting="invvol", cash_yield_yr=0.03, pick_stop_loss=trail)
        eq = simulate_trailing_stop(cfg, panel, mr, spy)
        m = evaluate(eq, build_spy_aligned(eq, mr), f"trail{trail}")
        print(f"  trail={trail}: cagr={m['cagr_full']:.4f} sh={m['sharpe']:.4f} mdd={m['max_dd']:.4f} wf={m['wf_mean_cagr']:.4f} wmin={m['wf_min_cagr']:.4f} npos={m['wf_n_pos']} beats={m['wf_n_beats_spy']}")

    print()
    print("=== Trailing stop + permanent TLT sleeve ===")
    for trail in [0.10, 0.15, 0.20]:
        for tlt in [0.0, 0.10, 0.20, 0.30]:
            cfg = V7Config(weighting="invvol", cash_yield_yr=0.03, pick_stop_loss=trail,
                          perm_sleeve_ticker="TLT", perm_sleeve_weight=tlt)
            eq = simulate_trailing_stop(cfg, panel, mr, spy)
            m = evaluate(eq, build_spy_aligned(eq, mr), f"trail{trail}_tlt{tlt}")
            print(f"  trail={trail} tlt={tlt}: cagr={m['cagr_full']:.4f} sh={m['sharpe']:.4f} mdd={m['max_dd']:.4f} wf={m['wf_mean_cagr']:.4f} wmin={m['wf_min_cagr']:.4f}")
