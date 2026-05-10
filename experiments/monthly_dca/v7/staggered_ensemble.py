"""Novel: staggered-rebalance ensemble.

The deployed v3 strategy holds 3 picks for 6 months, then rebalances. All
capital is exposed to the SAME 6-month decision; if those 3 picks crater
together (as in 2009-01 with MBI/GNW/THC), the entire portfolio takes the hit.

The staggered ensemble runs **6 sub-strategies in parallel**, each rebalancing
every 6 months but starting in a DIFFERENT month. At any point in time, the
portfolio holds 6 baskets at different ages: a fresh basket, a 1-month-old
basket, a 2-month-old basket, etc.

Properties:
  - Time diversification: not all capital is in any single 6-month decision.
  - Reduced concentration: at any time we have up to 6 × 3 = 18 unique picks
    (often overlapping but spread across rebalance dates).
  - Smoother equity curve: a single bad rebalance month only affects 1/6 of
    capital each subsequent month.

This is a structural change to risk distribution, not a new feature or model.
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

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def simulate_staggered(cfg: V7Config, panel: pd.DataFrame,
                       monthly_returns: pd.DataFrame,
                       spy_features: pd.DataFrame,
                       n_sleeves: int = 6) -> pd.DataFrame:
    """Run n_sleeves copies of v7 with offset start months and equal-weight blend
    their equity curves.
    """
    # Run a base simulator to get the months panel
    eq_base = simulate_v7(cfg, panel, monthly_returns, spy_features)
    months = pd.DatetimeIndex(eq_base["date"])

    # For each sleeve, create a "shifted" simulator that skips first `s` months
    # before deploying. Implementation: run the base simulation but for the
    # first `s` months, force regime='cash' (no deployment).
    sleeve_eqs = []
    by_asof = {pd.Timestamp(d): g.copy() for d, g in panel.groupby("asof")}
    months_sorted = sorted(by_asof.keys())
    cls = REGIMES[cfg.regime_gate]
    cf = cfg.cost_bps / 10000.0
    cash_step = (1 + cfg.cash_yield_yr) ** (1 / 12) - 1 if cfg.cash_yield_yr > 0 else 0.0

    for sleeve_i in range(n_sleeves):
        equity = 1.0
        cur_picks = []
        cur_unscaled = np.array([])
        held_for = 0
        in_cash = True
        crash_streak = 0
        sleeve_rows = []
        deployed_first = False
        for i, m in enumerate(months_sorted):
            spy_now = spy_features.loc[m].to_dict() if m in spy_features.index else {}
            regime = cls(spy_now)
            if regime == "crash":
                crash_streak += 1
            else:
                crash_streak = 0
            eff_regime = regime
            if regime == "crash" and crash_streak < cfg.crash_persist:
                eff_regime = "normal"

            # Force cash for the first `sleeve_i` months
            force_cash = (i < sleeve_i)

            do_reb = (i == sleeve_i) or (held_for >= cfg.hold_months) or in_cash
            do_reb = do_reb and (i >= sleeve_i)

            if do_reb and not force_cash:
                if eff_regime == "crash":
                    cur_picks, cur_unscaled = [], np.array([])
                    in_cash = True
                    held_for = 0
                else:
                    k = {"recovery": cfg.k_recovery, "bull": cfg.k_bull,
                         "normal": cfg.k_normal, "warning": cfg.k_normal}[eff_regime]
                    sub = by_asof.get(m, pd.DataFrame())
                    if len(sub) < k:
                        cur_picks, cur_unscaled = [], np.array([])
                        in_cash = True
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
                        in_cash = False
                held_for = 0

            # Compute month return
            if in_cash or force_cash or len(cur_picks) == 0:
                ret_m = cash_step
            else:
                pos = monthly_returns.index.searchsorted(m)
                cands = []
                for j in (pos - 1, pos):
                    if 0 <= j < len(monthly_returns.index):
                        cands.append((j, abs((monthly_returns.index[j] - m).days)))
                cands.sort(key=lambda x: x[1])
                if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(monthly_returns.index):
                    ret_m = 0.0
                else:
                    next_d = monthly_returns.index[cands[0][0] + 1]
                    pick_rets = []
                    for tk in cur_picks:
                        if tk in monthly_returns.columns:
                            rr = monthly_returns.at[next_d, tk]
                            pick_rets.append(-1.0 if pd.isna(rr) else float(rr))
                        else:
                            pick_rets.append(-1.0)
                    pick_rets = np.array(pick_rets)
                    ret_m = float((pick_rets * cur_unscaled).sum())

            if not in_cash and len(cur_picks) > 0 and do_reb:
                equity *= (1 + ret_m) * (1 - cf)
            else:
                equity *= (1 + ret_m)
            held_for += 1
            sleeve_rows.append({"date": m, "equity": equity, "ret_m": ret_m,
                                "regime": "cash" if in_cash else eff_regime,
                                "picks": ",".join(cur_picks)})
        sleeve_eqs.append(pd.DataFrame(sleeve_rows))

    # Blend monthly returns equally across sleeves
    blended = pd.DataFrame({"date": sleeve_eqs[0]["date"]})
    for i, eq_s in enumerate(sleeve_eqs):
        blended[f"ret_{i}"] = eq_s["ret_m"].astype(float)
    blended["ret_m"] = blended[[f"ret_{i}" for i in range(n_sleeves)]].mean(axis=1)
    blended["equity"] = (1 + blended["ret_m"]).cumprod()
    blended["regime"] = "active"  # mixed
    blended["picks"] = ""  # union too noisy
    blended["n_picks"] = 0
    return blended


if __name__ == "__main__":
    panel = load_panel_v7("ml_3plus6", "sp500_pit")
    mr = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy = load_spy_features()

    # v6 winner config
    cfg = V7Config(weighting="invvol", cash_yield_yr=0.03, hold_months=6)

    print("=== Staggered ensemble (n_sleeves) ===")
    for n in [1, 2, 3, 6, 12]:
        eq = simulate_staggered(cfg, panel, mr, spy, n_sleeves=n)
        m = evaluate(eq, build_spy_aligned(eq, mr), f"staggered_n{n}")
        print(f"  n={n}: cagr={m['cagr_full']:.4f} sh={m['sharpe']:.4f} mdd={m['max_dd']:.4f} wf={m['wf_mean_cagr']:.4f} wmin={m['wf_min_cagr']:.4f} npos={m['wf_n_pos']} beats={m['wf_n_beats_spy']}")

    # And combined with TLT
    print()
    print("=== Staggered (n=6) + TLT sleeve ===")
    for tlt in [0.0, 0.10, 0.20, 0.30]:
        cfg2 = V7Config(weighting="invvol", cash_yield_yr=0.03, hold_months=6,
                       perm_sleeve_ticker="TLT", perm_sleeve_weight=tlt)
        eq = simulate_staggered(cfg2, panel, mr, spy, n_sleeves=6)
        m = evaluate(eq, build_spy_aligned(eq, mr), f"staggered_tlt{tlt}")
        print(f"  n=6 tlt={tlt}: cagr={m['cagr_full']:.4f} sh={m['sharpe']:.4f} mdd={m['max_dd']:.4f} wf={m['wf_mean_cagr']:.4f} wmin={m['wf_min_cagr']:.4f}")
