"""Driver: sweep the new stock-picking levers vs WIN1 on the full gauntlet
plus a per-year consistency metric (the explicit user goal)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from improve_main_strategy import load_inputs, evaluate, spy_aligned  # noqa
from improve_sim_v2 import run_sim_v2  # noqa
from improve_pick_v3 import run_sim_v3  # noqa
import experiments.monthly_dca.v5.build_webapp_v5_pit as bw  # noqa


def stream(rl):
    return np.array([r["ret_m"] for r in rl], float)


def yearly(r, dates, spv):
    """Per-calendar-year CAGR and edge vs SPY; consistency stats."""
    d = pd.to_datetime(dates)
    df = pd.DataFrame({"y": d.year, "r": r, "s": spv})
    rows, beats = {}, 0
    for y, g in df.groupby("y"):
        cy = float((1 + g["r"]).prod() - 1)
        sy = float((1 + g["s"]).prod() - 1)
        rows[int(y)] = round(cy, 3)
        beats += cy > sy
    vals = np.array(list(rows.values()))
    full_years = [y for y in rows if df[df.y == y].shape[0] >= 11]
    fv = np.array([rows[y] for y in full_years])
    return {
        "by_year": rows,
        "years_beat_spy": int(beats),
        "n_years": len(rows),
        "worst_year": round(float(vals.min()), 3),
        "neg_years": int((fv < 0).sum()),
        "yr_cagr_mean": round(float(fv.mean()), 3),
        "yr_cagr_std": round(float(fv.std()), 3),
        # downside-only dispersion: penalise bad years, not good ones
        "yr_downside_std": round(
            float(np.sqrt(np.mean(np.minimum(fv - fv.mean(), 0) ** 2))), 3),
    }


def show(nm, m, y):
    print(f"{nm:<30} CAGR {m['cagr']*100:5.1f}%  Sh {m['sharpe']:.2f}  "
          f"DD {m['max_dd']*100:6.1f}%  WF {m['wf_beats']}/{m['wf_n']}  "
          f"era {m['eras_beat']}/4  D10y {m['dca_H120']}  "
          f"yBeat {y['years_beat_spy']}/{y['n_years']}  "
          f"wYr {y['worst_year']*100:6.1f}%  negY {y['neg_years']}  "
          f"yStd {y['yr_cagr_std']:.2f}")


def main():
    inp = load_inputs()
    members_g, preds, spyf, mr, mp, chronos = inp
    dates = None

    def gauntlet(rl):
        nonlocal dates
        dates = [r["date"] for r in rl]
        spv = spy_aligned(pd.to_datetime(dates), mr)
        r = stream(rl)
        return evaluate(r, dates, spv), yearly(r, dates, spv)

    out = {}

    # --- fidelity: WIN1 via v2 and via v3 must match exactly ---
    rl_v2 = run_sim_v2(members_g, preds, preds, spyf, mr, mp, chronos,
                       cost_bps=10.0, K=2, trigger_mode="blend",
                       select_mode="ml_3plus6")
    rl_v3 = run_sim_v3(members_g, preds, preds, spyf, mr, mp, chronos,
                       cost_bps=10.0, K=2, trigger_mode="blend",
                       select_mode="ml_3plus6")
    d = max(abs(a["ret_m"] - b["ret_m"]) for a, b in zip(rl_v2, rl_v3))
    print(f"[fidelity] max|Δ ret_m| v3 vs v2 (WIN1) = {d:.2e}\n")
    m0, y0 = gauntlet(rl_v2)
    out["WIN1 (deployed)"] = {**m0, **y0}
    show("WIN1 (deployed)", m0, y0)

    def run(nm, **kw):
        rl = run_sim_v3(members_g, preds, preds, spyf, mr, mp, chronos,
                        cost_bps=10.0, K=2, **kw)
        m, y = gauntlet(rl)
        out[nm] = {**m, **y}
        show(nm, m, y)
        return m, y

    print("\n--- A. combined blend trigger + blend select ---")
    run("trig=blend sel=blend", trigger_mode="blend", select_mode="blend")

    print("\n--- B. decorrelated 2nd pick (rho_max sweep) ---")
    for rho in (0.4, 0.5, 0.6, 0.7):
        run(f"decorr2 rho<={rho}", trigger_mode="blend",
            select_mode="ml_3plus6", decorr2=True, rho_max=rho)

    print("\n--- C. falling-knife screen (knife_q sweep) ---")
    for q in (0.10, 0.20, 0.30):
        run(f"knife q={q}", trigger_mode="blend",
            select_mode="ml_3plus6", knife_q=q)

    print("\n--- D. conviction-adaptive breadth (k_hi sweep) ---")
    for (lo, hi, khi) in [(0.06, 0.15, 3), (0.08, 0.18, 3),
                          (0.08, 0.18, 4), (0.10, 0.22, 4),
                          (0.06, 0.15, 5)]:
        run(f"adaptK lo{lo} hi{hi} kHi{khi}", trigger_mode="blend",
            select_mode="ml_3plus6", adaptive_k=True,
            conv_lo=lo, conv_hi=hi, k_hi=khi, k_mid=3, k_lo=2)

    print("\n--- E. best combos ---")
    run("adaptK(.08/.18/4)+decorr2 .6", trigger_mode="blend",
        select_mode="ml_3plus6", adaptive_k=True, conv_lo=0.08,
        conv_hi=0.18, k_hi=4, decorr2=True, rho_max=0.6)
    run("adaptK(.08/.18/4)+knife .2", trigger_mode="blend",
        select_mode="ml_3plus6", adaptive_k=True, conv_lo=0.08,
        conv_hi=0.18, k_hi=4, knife_q=0.2)
    run("decorr2 .6 + knife .2", trigger_mode="blend",
        select_mode="ml_3plus6", decorr2=True, rho_max=0.6, knife_q=0.2)

    p = (bw.CACHE / "v2" / "sp500_pit" / "augmented" / "improve_pick_v3.json")
    p.write_text(json.dumps(out, indent=2, default=bw.to_jsonable))
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
