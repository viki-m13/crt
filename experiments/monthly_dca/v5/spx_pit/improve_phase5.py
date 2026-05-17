"""Phase 5: can a THIRD decorrelated sleeve push consistency past E1?

E1 = 50/50 {A: sel=ml/trig=blend, B: sel=blend/trig=ml}.
Tested here (all vs the deployed E1, same gauntlet + consistency block):

  C1  3-way 1/3 {A, B, C=sel=ml/trig=consensus}
  C2  3-way 1/3 {A, B, D=sel=consensus/trig=ml}
  C3  3-way 1/3 {A, B, E=sel=blend/trig=consensus}
  C4  4-way 1/4 {A, B, C, D}
  P1  E1 with a 1-month phase-offset twin (A,B + A,B started 1m later)
      — true rebalance-luck time-diversification, equal weight
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
from improve_phase4 import consistency  # noqa


def S(trig, sel):
    v2.BLEND_W = 0.5
    rl = run_sim_v2(MG, P, P, SF, MR, MP, CH, cost_bps=10.0, K=bw.K_PICKS,
                    trigger_mode=trig, select_mode=sel)
    return np.array([r["ret_m"] for r in rl], float)


def phase_offset(stream, k=1):
    """Approximate a k-month-later twin: shift the return stream by k
    (the same rule started k months later sees a different basket cadence).
    Conservative proxy — pads the head with the early mean."""
    s = np.asarray(stream, float)
    out = np.empty_like(s)
    out[k:] = s[:-k]
    out[:k] = s[:k].mean()
    return out


def main():
    global MG, P, SF, MR, MP, CH
    MG, P, SF, MR, MP, CH = load_inputs()
    A = S("blend", "ml_3plus6")        # WIN1
    B = S("ml_3plus6", "blend")        # WIN2
    C = S("consensus", "ml_3plus6")
    D = S("ml_3plus6", "consensus")
    E = S("consensus", "blend")
    bw.SCORER_MODE = "blend"
    rlbase, _, _ = bw.run_full_sim(MG, P, P, SF, MR, MP, chronos_preds=CH,
                                   cost_bps=10.0, hold_months=bw.HOLD_MONTHS,
                                   K=bw.K_PICKS)
    dates = [r["date"] for r in rlbase]
    spv = spy_aligned(pd.to_datetime(dates), MR)

    E1 = 0.5 * A + 0.5 * B
    variants = {
        "E1 (deployed)": E1,
        "C1 1/3 {A,B,C=cons/ml}": (A + B + C) / 3,
        "C2 1/3 {A,B,D=ml/cons}": (A + B + D) / 3,
        "C3 1/3 {A,B,E=cons/bl}": (A + B + E) / 3,
        "C4 1/4 {A,B,C,D}": (A + B + C + D) / 4,
        "P1 E1 + 1m-offset twin": 0.5 * E1 + 0.5 * (
            0.5 * phase_offset(A) + 0.5 * phase_offset(B)),
    }
    rows = {}
    for nm, r in variants.items():
        m = evaluate(r, dates, spv)
        c = consistency(r, dates, spv)
        rows[nm] = {**m, **c}

    print(f"\n{'variant':<26}{'CAGR':>7}{'Shrp':>6}{'MaxDD':>7}{'WF':>6}"
          f"{'era':>5}{'D3y':>6}{'D5y':>6}{'r60min':>8}{'r36beat':>8}"
          f"{'CAGRx':>7}")
    print("-" * 86)
    for nm, m in rows.items():
        print(f"{nm:<26}{m['cagr']*100:>6.1f}%{m['sharpe']:>6.2f}"
              f"{m['max_dd']*100:>6.0f}%{m['wf_beats']:>3}/{m['wf_n']}"
              f"{m['eras_beat']:>3}/4{str(m['dca_H36']):>6}"
              f"{str(m['dca_H60']):>6}{m['roll60_min']*100:>7.1f}%"
              f"{m['roll36_beat']*100:>7.0f}%{m['cagr_ex0309']*100:>6.1f}%")
    out_p = bw.CACHE / "v2" / "sp500_pit" / "augmented" / "improve_phase5.json"
    out_p.write_text(json.dumps(rows, indent=2, default=bw.to_jsonable))
    print(f"\nsaved -> {out_p}")


if __name__ == "__main__":
    main()
