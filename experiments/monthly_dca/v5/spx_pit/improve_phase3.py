"""Phase 3: stress-test the winner `trigger=blend, select=ml_3plus6`.

Overfit screens:
  1. cost sensitivity (0/10/20/30 bps)
  2. blend-weight plateau (BLEND_W 0.2..0.8) — must be a plateau, not a peak
  3. TRUE OOS: design 2003-2012 vs untouched holdout 2013-2026
  4. Monte-Carlo synthetic-delisting overlay (per-pick monthly hazard)
  5. light catastrophic DD-breaker on the winner stream

Compares to the deployed `consensus` baseline throughout.
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
from improve_main_strategy import (load_inputs, evaluate, spy_aligned,  # noqa
                                   overlay_ddbreak)
from improve_sim_v2 import run_sim_v2  # noqa


def stream(rl):
    return np.array([r["ret_m"] for r in rl], float)


def submetrics(r, dates, spv, lo, hi):
    d = pd.Series(pd.to_datetime(dates))
    msk = ((d >= lo) & (d <= hi)).to_numpy()
    rr, ss = r[msk], spv[msk]
    n = len(rr)
    cagr = np.prod(1 + rr) ** (12 / n) - 1
    sh = rr.mean() / max(rr.std(), 1e-9) * np.sqrt(12)
    e = np.cumprod(1 + rr)
    mdd = float(((e - np.maximum.accumulate(e)) / np.maximum.accumulate(e)).min())
    spy_c = np.prod(1 + ss) ** (12 / n) - 1
    return dict(cagr=round(float(cagr), 4), sharpe=round(float(sh), 3),
                max_dd=round(mdd, 4), spy_cagr=round(float(spy_c), 4),
                n=int(n))


def mc_delist(rl, mr, alpha, n_iter=25, seed=0):
    rng = np.random.default_rng(seed)
    mr_idx = mr.index
    out = []
    for _ in range(n_iter):
        eq, dead, cur_bid = 1.0, set(), None
        for row in rl:
            bid = row.get("basket_id")
            if bid != cur_bid:
                dead, cur_bid = set(), bid
            picks = row.get("picks") or []
            if not picks:
                continue
            m = pd.Timestamp(row["date"])
            pos = mr_idx.searchsorted(m)
            if pos + 1 >= len(mr_idx):
                continue
            nxt = mr_idx[pos + 1]
            rr = []
            for tk in picks:
                if tk in dead:
                    rr.append(-1.0); continue
                if rng.random() < alpha / 12.0:
                    dead.add(tk); rr.append(-1.0); continue
                v = mr.at[nxt, tk] if (tk in mr.columns and nxt in mr_idx) else 0.0
                rr.append(0.0 if pd.isna(v) else float(v))
            eq *= (1 + float(np.mean(rr)))
        n = len(rl)
        out.append(eq ** (12.0 / n) - 1.0)
    return round(float(np.median(out)), 4)


def main():
    inp = load_inputs()
    members_g, preds, spyf, mr, mp, chronos = inp

    def sim(trig, sel, cost=10.0):
        return run_sim_v2(members_g, preds, preds, spyf, mr, mp, chronos,
                          cost_bps=cost, K=bw.K_PICKS,
                          trigger_mode=trig, select_mode=sel)

    bw.SCORER_MODE = "consensus"
    base_rl, _, _ = bw.run_full_sim(members_g, preds, preds, spyf, mr, mp,
                                    chronos_preds=chronos, cost_bps=10.0,
                                    hold_months=bw.HOLD_MONTHS, K=bw.K_PICKS)
    dates = [r["date"] for r in base_rl]
    spv = spy_aligned(pd.to_datetime(dates), mr)
    rbase = stream(base_rl)

    win_rl = sim("blend", "ml_3plus6")
    rwin = stream(win_rl)
    out = {}

    print("=== Headline (10bps) ===")
    for nm, r in (("deployed consensus", rbase), ("WIN trig=blend/sel=ml", rwin)):
        m = evaluate(r, dates, spv)
        out[nm] = m
        print(f"  {nm:<26} CAGR {m['cagr']*100:5.1f}%  Sh {m['sharpe']:.2f}"
              f"  DD {m['max_dd']*100:6.1f}%  WF {m['wf_beats']}/{m['wf_n']}"
              f"  era {m['eras_beat']}/4  D10y {m['dca_H120']}")

    # 1. cost sensitivity
    print("\n=== 1. Cost sensitivity (winner) ===")
    out["cost"] = {}
    for c in (0.0, 10.0, 20.0, 30.0):
        m = evaluate(stream(sim("blend", "ml_3plus6", cost=c)), dates, spv)
        out["cost"][f"{int(c)}bps"] = m
        print(f"  {int(c):>3} bps  CAGR {m['cagr']*100:5.1f}%  Sh {m['sharpe']:.2f}"
              f"  DD {m['max_dd']*100:6.1f}%  WF {m['wf_beats']}/{m['wf_n']}")

    # 2. blend-weight plateau
    print("\n=== 2. Blend-weight plateau (consensus-rank weight) ===")
    out["blend_w"] = {}
    for w in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        v2.BLEND_W = w
        m = evaluate(stream(sim("blend", "ml_3plus6")), dates, spv)
        out["blend_w"][f"{w:.1f}"] = m
        print(f"  w={w:.1f}  CAGR {m['cagr']*100:5.1f}%  Sh {m['sharpe']:.2f}"
              f"  DD {m['max_dd']*100:6.1f}%  WF {m['wf_beats']}/{m['wf_n']}"
              f"  era {m['eras_beat']}/4")
    v2.BLEND_W = 0.5

    # 3. TRUE OOS design/holdout
    print("\n=== 3. TRUE OOS (design 2003-2012 | holdout 2013-2026) ===")
    out["oos"] = {}
    for nm, r in (("deployed", rbase), ("winner", rwin)):
        d = submetrics(r, dates, spv, "2003-01-01", "2012-12-31")
        h = submetrics(r, dates, spv, "2013-01-01", "2026-12-31")
        out["oos"][nm] = {"design": d, "holdout": h}
        print(f"  {nm:<9} design  CAGR {d['cagr']*100:5.1f}% Sh {d['sharpe']:.2f}"
              f" DD {d['max_dd']*100:6.1f}% (SPY {d['spy_cagr']*100:.1f}%)")
        print(f"  {nm:<9} holdout CAGR {h['cagr']*100:5.1f}% Sh {h['sharpe']:.2f}"
              f" DD {h['max_dd']*100:6.1f}% (SPY {h['spy_cagr']*100:.1f}%)")

    # 4. MC delisting
    print("\n=== 4. MC synthetic-delisting median CAGR ===")
    out["mc_delist"] = {}
    for a in (0.0, 0.04, 0.08):
        b = mc_delist(base_rl, mr, a)
        w = mc_delist(win_rl, mr, a)
        out["mc_delist"][f"a{int(a*100)}"] = {"deployed": b, "winner": w}
        print(f"  alpha {int(a*100):>2}%  deployed {b*100:6.1f}%   "
              f"winner {w*100:6.1f}%")

    # 5. light catastrophic DD-breaker on the winner
    print("\n=== 5. Light DD-breaker on winner (park=spy) ===")
    out["ddbrk"] = {}
    for cut in (-0.45, -0.50, -0.55):
        rr = overlay_ddbreak(rwin, spv, dd_cut=cut, e_low=0.35,
                             dd_resume=cut * 0.5, park="spy")
        m = evaluate(rr, dates, spv)
        out["ddbrk"][f"{int(cut*100)}"] = m
        print(f"  cut {int(cut*100):>3}%  CAGR {m['cagr']*100:5.1f}%  "
              f"Sh {m['sharpe']:.2f}  DD {m['max_dd']*100:6.1f}%  "
              f"WF {m['wf_beats']}/{m['wf_n']}  era {m['eras_beat']}/4")

    out_p = (bw.CACHE / "v2" / "sp500_pit" / "augmented"
             / "improve_phase3.json")
    out_p.write_text(json.dumps(out, indent=2, default=bw.to_jsonable))
    print(f"\nsaved -> {out_p}")


if __name__ == "__main__":
    main()
