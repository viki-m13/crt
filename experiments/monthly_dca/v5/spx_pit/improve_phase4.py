"""Phase 4: same CAGR, MORE CONSISTENTLY.

WIN1 is a Pareto win but the edge is still front-loaded and short/mid
paths are noisy. Here we test consistency levers that should NOT cost
CAGR (the opposite of vol-targeting / DD-breakers which we proved bleed
return):

  E1  WIN1 (+) WIN2 50/50 portfolio   — two strong ~same-CAGR variants
                                         with decorrelated rebalance/sel
  E2  min-hold ensemble (WIN1 @ mh 5/6/7, equal weight) — desynchronise
                                         the score-drift timing (the
                                         documented timing-luck driver)
  E3  blend-weight ensemble (WIN1 @ w 0.4/0.5/0.6, eq wt) — spec-risk
                                         diversification over the plateau
  E4  mega-ensemble (E2 x E3 grid + WIN2) — full spec ensemble
  E5  regime-conditioned K (K=2 clean trend, K=3 recovery) — cut the
                                         single-name tail only in
                                         turbulent regimes
  E6  E4 (+) E5

Scored on the full gauntlet PLUS new consistency metrics:
  - yearly-return stdev (lower = steadier)
  - worst rolling 36m / 60m annualised return
  - % of rolling 36m windows beating SPY
  - CAGR EXCLUDING 2003-2009 (forward-relevant: strips the front-load)
  - era-IRR spread (max-min across the 4 non-overlapping eras)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import experiments.monthly_dca.v5.build_webapp_v5_pit as bw  # noqa
import improve_sim_v2 as v2  # noqa
from improve_main_strategy import load_inputs, evaluate, spy_aligned  # noqa
from improve_sim_v2 import run_sim_v2  # noqa


def stream(rl):
    return np.array([r["ret_m"] for r in rl], float)


def consistency(r, dates, spv):
    r = np.asarray(r, float)
    d = pd.Series(pd.to_datetime(dates))
    yrs = d.dt.year.to_numpy()
    # yearly compounded returns
    yret = []
    for y in np.unique(yrs):
        rr = r[yrs == y]
        yret.append(np.prod(1 + rr) - 1)
    yret = np.array(yret)
    # rolling annualised
    def roll_min(H):
        v = [np.prod(1 + r[s:s + H]) ** (12 / H) - 1
             for s in range(0, len(r) - H + 1)]
        return float(np.min(v)) if v else None

    def roll_beat(H):
        w = [np.prod(1 + r[s:s + H]) > np.prod(1 + spv[s:s + H])
             for s in range(0, len(r) - H + 1)]
        return float(np.mean(w)) if w else None
    # CAGR excluding 2003-2009
    msk = (yrs >= 2010)
    rx = r[msk]
    cagr_ex = np.prod(1 + rx) ** (12 / len(rx)) - 1
    # era IRR spread
    eras = []
    for a, b in [(2003, 2009), (2010, 2015), (2016, 2020), (2021, 2030)]:
        k = (yrs >= a) & (yrs <= b)
        if k.sum() >= 6:
            eras.append(np.prod(1 + r[k]) ** (12 / k.sum()) - 1)
    return {
        "yearly_std": round(float(yret.std()), 4),
        "worst_yr": round(float(yret.min()), 4),
        "roll36_min": round(roll_min(36), 4),
        "roll60_min": round(roll_min(60), 4),
        "roll36_beat": round(roll_beat(36), 4),
        "cagr_ex0309": round(float(cagr_ex), 4),
        "era_spread": round(float(max(eras) - min(eras)), 4),
    }


def main():
    inp = load_inputs()
    mg, preds, spyf, mr, mp, chronos = inp

    def sim(trig, sel, w=0.5, mh=None, kreg=None, cost=10.0):
        v2.BLEND_W = w
        rl = run_sim_v2(mg, preds, preds, spyf, mr, mp, chronos,
                        cost_bps=cost, K=bw.K_PICKS,
                        trigger_mode=trig, select_mode=sel,
                        min_hold=mh, k_by_regime=kreg)
        v2.BLEND_W = 0.5
        return stream(rl)

    bw.SCORER_MODE = "blend"
    base_rl, _, _ = bw.run_full_sim(mg, preds, preds, spyf, mr, mp,
                                    chronos_preds=chronos, cost_bps=10.0,
                                    hold_months=bw.HOLD_MONTHS, K=bw.K_PICKS)
    dates = [r["date"] for r in base_rl]
    spv = spy_aligned(pd.to_datetime(dates), mr)

    win1 = stream(base_rl)                       # deployed
    win2 = sim("ml_3plus6", "blend")             # best-DD variant
    KREG = {"bull": 2, "normal": 2, "recovery": 3, "crash": 0}

    variants = {}
    variants["WIN1 (deployed)"] = win1
    variants["WIN2 trig=ml/sel=bl"] = win2
    variants["E1 WIN1+WIN2 50/50"] = 0.5 * win1 + 0.5 * win2
    # E2 min-hold ensemble
    mh_streams = [sim("blend", "ml_3plus6", mh=h) for h in (5, 6, 7)]
    variants["E2 WIN1 mh{5,6,7}"] = np.mean(mh_streams, axis=0)
    # E3 blend-weight ensemble
    w_streams = [sim("blend", "ml_3plus6", w=w) for w in (0.4, 0.5, 0.6)]
    variants["E3 WIN1 w{.4,.5,.6}"] = np.mean(w_streams, axis=0)
    # E4 mega-ensemble (w x mh grid + WIN2)
    grid = []
    for w in (0.4, 0.5, 0.6):
        for h in (5, 6, 7):
            grid.append(sim("blend", "ml_3plus6", w=w, mh=h))
    grid.append(win2)
    variants["E4 mega-ensemble"] = np.mean(grid, axis=0)
    # E5 regime-K
    variants["E5 WIN1 regimeK(rec=3)"] = sim("blend", "ml_3plus6", kreg=KREG)
    # E6 E4 + E5
    e5b = sim("ml_3plus6", "blend", kreg=KREG)
    variants["E6 mega + regimeK"] = np.mean(
        grid + [variants["E5 WIN1 regimeK(rec=3)"], e5b], axis=0)

    rows = {}
    for nm, r in variants.items():
        m = evaluate(r, dates, spv)
        c = consistency(r, dates, spv)
        rows[nm] = {**m, **c}

    cols1 = ("cagr", "sharpe", "max_dd", "wf_beats", "eras_beat",
             "dca_H36", "dca_H60", "dca_H120")
    print(f"\n{'variant':<24}{'CAGR':>7}{'Shrp':>6}{'MaxDD':>7}{'WF':>5}"
          f"{'era':>4}{'D3y':>6}{'D5y':>6}{'D10y':>6}")
    print("-" * 70)
    for nm, m in rows.items():
        print(f"{nm:<24}{m['cagr']*100:>6.1f}%{m['sharpe']:>6.2f}"
              f"{m['max_dd']*100:>6.0f}%{m['wf_beats']:>3}/{m['wf_n']}"
              f"{m['eras_beat']:>3}/4{str(m['dca_H36']):>6}"
              f"{str(m['dca_H60']):>6}{str(m['dca_H120']):>6}")
    print(f"\n{'variant':<24}{'CAGRx0309':>10}{'yrStd':>7}{'wrstYr':>7}"
          f"{'r36min':>8}{'r60min':>8}{'r36beat':>8}{'eraSpr':>7}")
    print("-" * 78)
    for nm, m in rows.items():
        print(f"{nm:<24}{m['cagr_ex0309']*100:>9.1f}%{m['yearly_std']*100:>6.0f}%"
              f"{m['worst_yr']*100:>6.0f}%{m['roll36_min']*100:>7.1f}%"
              f"{m['roll60_min']*100:>7.1f}%{m['roll36_beat']*100:>7.0f}%"
              f"{m['era_spread']*100:>6.0f}%")

    out_p = bw.CACHE / "v2" / "sp500_pit" / "augmented" / "improve_phase4.json"
    out_p.write_text(json.dumps(rows, indent=2, default=bw.to_jsonable))
    print(f"\nsaved -> {out_p}")


if __name__ == "__main__":
    main()
