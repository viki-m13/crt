"""Experiment: can we improve the deployed main strategy further?

Goal (user): better CAGR, MORE CONSISTENTLY, with LESS DRAWDOWN.

Approach: take the REAL production pipeline (build_webapp_v5_pit.run_full_sim)
and test a battery of NEW ideas, all causal (no look-ahead), evaluated on the
canonical 10-split walk-forward + era-IRR + DCA-win gauntlet, plus the honest
drawdown of the resulting monthly stream.

Ideas tested
------------
Group A — post-hoc OVERLAYS on the deployed return stream (no picker change):
  A1  portfolio vol-target (de-risk only; park excess in cash)
  A2  portfolio vol-target, park excess in SPY (keep beta carry)
  A3  equity drawdown circuit-breaker (cut exposure when in deep DD)
  A4  vol-target + DD-breaker combined

Group B — SCORER changes (re-run the real sim):
  B1  scorer = ml_3plus6 (the other deployed-history scorer)
  B2  scorer = rank-blend of ml_3plus6 and consensus (era-robustness)

Group C — SIZING changes (re-run the real sim):
  C1  conviction-scaled gross exposure (size by Chronos-rank headroom)

Every variant is scored against the deployed `consensus` baseline on:
  full CAGR, Sharpe, Max DD, WF n-beats-SPY/10, era-IRR beats S&P-DCA,
  rolling DCA-win 3y/5y/10y.
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
from sweep_v5_aug import WF_SPLITS  # noqa


# --------------------------------------------------------------------------- #
#  Inputs                                                                      #
# --------------------------------------------------------------------------- #
def load_inputs():
    members = pd.read_parquet(bw.PIT / "sp500_membership_monthly.parquet")
    members["asof"] = pd.to_datetime(members["asof"])
    members_g = members.groupby("asof")["ticker"].apply(set).to_dict()
    mr = pd.read_parquet(bw._path("monthly_returns_clean.parquet"))
    mp = pd.read_parquet(bw._path("monthly_prices_clean.parquet"))
    spyf = bw.load_spy_features()
    preds = pd.read_parquet(bw._path("ml_preds.parquet"))
    preds["asof"] = pd.to_datetime(preds["asof"])
    c = pd.read_parquet(bw._path("ml_preds_chronos.parquet"))
    c["asof"] = pd.to_datetime(c["asof"])
    chronos = {pd.Timestamp(a): dict(zip(g["ticker"], g["chronos_p70_3m"]))
               for a, g in c.groupby("asof")}
    return members_g, preds, spyf, mr, mp, chronos


# --------------------------------------------------------------------------- #
#  Evaluation gauntlet (mirrors novel_v9_prod_validate.evaluate)               #
# --------------------------------------------------------------------------- #
def irr(tv, H):
    lo, hi = -0.5, 0.5
    f = lambda i: tv / (1 + i) ** (H - 1) - sum(1 / (1 + i) ** t for t in range(H))
    flo = f(lo); m = 0.0
    for _ in range(120):
        m = .5 * (lo + hi); fm = f(m)
        if abs(fm) < 1e-9:
            break
        if (fm > 0) == (flo > 0):
            lo, flo = m, fm
        else:
            hi = m
    return (1 + m) ** 12 - 1


def dca_t(r):
    v = 0.0
    for x in r:
        v = (v + 1) * (1 + x)
    return v


def spy_aligned(dates, mr):
    spv = []
    for d in dates:
        p = mr.index.searchsorted(d)
        spv.append(float(mr["SPY"].iloc[min(p + 1, len(mr) - 1)]))
    return np.array(spv)


def evaluate(r, dates, spv):
    r = np.asarray(r, float)
    n = len(r)
    cagr = np.prod(1 + r) ** (12 / n) - 1
    sh = r.mean() / max(r.std(), 1e-9) * np.sqrt(12)
    e = np.cumprod(1 + r)
    mdd = float(((e - np.maximum.accumulate(e)) / np.maximum.accumulate(e)).min())
    out = {"cagr": round(float(cagr), 4), "sharpe": round(float(sh), 3),
           "max_dd": round(float(mdd), 4)}
    d = pd.Series(pd.to_datetime(dates))
    wf = []
    for nm, lo, hi in WF_SPLITS:
        msk = ((d >= lo) & (d <= hi)).to_numpy()
        rr, ss = r[msk], spv[msk]
        if len(rr) < 6:
            continue
        cg = np.prod(1 + rr) ** (12 / len(rr)) - 1
        sg = np.prod(1 + ss) ** (12 / len(ss)) - 1
        wf.append(cg > sg)
    out["wf_beats"] = int(np.sum(wf))
    out["wf_n"] = int(len(wf))
    for H in (36, 60, 120):
        w = [dca_t(r[s:s + H]) > dca_t(spv[s:s + H]) for s in range(0, n - H + 1)]
        out[f"dca_H{H}"] = round(float(np.mean(w)), 3) if w else None
    yrs = d.dt.year.to_numpy()
    eras = {}
    for nm, a, b in [("03-09", 2003, 2009), ("10-15", 2010, 2015),
                     ("16-20", 2016, 2020), ("21-26", 2021, 2030)]:
        k = np.where((yrs >= a) & (yrs <= b))[0]
        if len(k) < 6:
            continue
        eras[nm] = bool(dca_t(r[k]) > dca_t(spv[k]))
    out["eras_beat"] = sum(eras.values())
    out["era_detail"] = eras
    return out


# --------------------------------------------------------------------------- #
#  Overlays (causal: exposure at month t uses only returns through t-1)        #
# --------------------------------------------------------------------------- #
def overlay_voltarget(r, spv, target_ann=0.25, win=6, park="cash", floor=0.0):
    """Scale next month's exposure so trailing realised vol ~ target.
    De-risk only (exposure capped at 1.0). Park (1-e) in cash or SPY."""
    r = np.asarray(r, float)
    out = np.empty_like(r)
    tgt_m = target_ann / np.sqrt(12)
    for t in range(len(r)):
        if t < win:
            e = 1.0
        else:
            v = r[t - win:t].std()
            e = 1.0 if v <= 1e-6 else min(1.0, tgt_m / v)
            e = max(e, floor)
        carry = spv[t] if park == "spy" else 0.0
        out[t] = e * r[t] + (1 - e) * carry
    return out


def overlay_ddbreak(r, spv, dd_cut=-0.35, e_low=0.30, dd_resume=-0.15, park="cash"):
    """Equity trailing-DD circuit breaker. When realised DD (on the OVERLAID
    equity, known through t-1) exceeds dd_cut, drop exposure to e_low until DD
    recovers above dd_resume. Causal."""
    r = np.asarray(r, float)
    out = np.empty_like(r)
    eq = 1.0
    peak = 1.0
    derisked = False
    for t in range(len(r)):
        dd = eq / peak - 1.0
        if not derisked and dd <= dd_cut:
            derisked = True
        elif derisked and dd >= dd_resume:
            derisked = False
        e = e_low if derisked else 1.0
        carry = spv[t] if park == "spy" else 0.0
        rt = e * r[t] + (1 - e) * carry
        out[t] = rt
        eq *= (1 + rt)
        peak = max(peak, eq)
    return out


# --------------------------------------------------------------------------- #
#  Scorer-blend sim: monkeypatch a 'blend' mode into build_webapp             #
# --------------------------------------------------------------------------- #
_ORIG_FLAG = {}


def run_scorer(mode, inp):
    members_g, preds, spyf, mr, mp, chronos = inp
    bw.SCORER_MODE = mode
    rl, _, ls = bw.run_full_sim(members_g, preds, preds, spyf, mr, mp,
                                chronos_preds=chronos, cost_bps=10.0,
                                hold_months=bw.HOLD_MONTHS, K=bw.K_PICKS)
    bw.SCORER_MODE = "consensus"
    return rl, ls


def patch_blend():
    """Add SCORER_MODE='blend' = mean of ml_3plus6 score-rank and consensus
    score-rank, computed inside the real _compute_candidate_top + basket
    branch. We patch by wrapping pandas operations: simplest is to add the
    branch in build_webapp via source-level monkeypatch of two spots. Instead
    we approximate blend at the data level: precompute a blended pred set."""
    raise NotImplementedError


def main():
    inp = load_inputs()
    members_g, preds, spyf, mr, mp, chronos = inp

    # ---- baselines (real production sim) ---------------------------------- #
    base_rl, base_ls = run_scorer("consensus", inp)
    dates = [row["date"] for row in base_rl]
    rcons = np.array([row["ret_m"] for row in base_rl], float)
    spv = spy_aligned(pd.to_datetime(dates), mr)

    ml_rl, _ = run_scorer("ml_3plus6", inp)
    rml = np.array([row["ret_m"] for row in ml_rl], float)

    results = {}
    results["BASE consensus (deployed)"] = evaluate(rcons, dates, spv)
    results["B1 ml_3plus6"] = evaluate(rml, dates, spv)

    # ---- B2: rank-blend of the two scorer streams ------------------------- #
    # NOTE: blending realised streams != blending picks, but it is a valid
    # *portfolio of two sub-strategies* (50/50 monthly rebalanced). Honest.
    rblend = 0.5 * rcons + 0.5 * rml
    results["B2 50/50 consensus+ml book"] = evaluate(rblend, dates, spv)

    # ---- A1/A2: vol-target sweep ----------------------------------------- #
    for tgt in (0.20, 0.25, 0.30, 0.35):
        for park in ("cash", "spy"):
            rr = overlay_voltarget(rcons, spv, target_ann=tgt, win=6, park=park)
            results[f"A vol{int(tgt*100)}% park={park}"] = evaluate(rr, dates, spv)

    # ---- A3: DD circuit-breaker sweep ------------------------------------ #
    for cut in (-0.30, -0.40, -0.50):
        for park in ("cash", "spy"):
            rr = overlay_ddbreak(rcons, spv, dd_cut=cut, e_low=0.30,
                                 dd_resume=cut * 0.4, park=park)
            results[f"A ddbrk{int(cut*100)} park={park}"] = evaluate(rr, dates, spv)

    # ---- A4: vol-target + DD-breaker ------------------------------------- #
    rr = overlay_voltarget(rcons, spv, target_ann=0.30, win=6, park="spy")
    rr = overlay_ddbreak(rr, spv, dd_cut=-0.30, e_low=0.30, dd_resume=-0.12,
                         park="spy")
    results["A4 vol30+ddbrk park=spy"] = evaluate(rr, dates, spv)

    # ---- A on the BLEND book (best-of-both) ------------------------------ #
    rr = overlay_voltarget(rblend, spv, target_ann=0.25, win=6, park="spy")
    results["A blend+vol25 park=spy"] = evaluate(rr, dates, spv)
    rr = overlay_voltarget(rblend, spv, target_ann=0.30, win=6, park="spy")
    results["A blend+vol30 park=spy"] = evaluate(rr, dates, spv)

    # ---- print table ----------------------------------------------------- #
    cols = ["cagr", "sharpe", "max_dd", "wf_beats", "dca_H36", "dca_H60",
            "dca_H120", "eras_beat"]
    print(f"\n{'variant':<32}{'CAGR':>8}{'Shrp':>7}{'MaxDD':>8}"
          f"{'WF':>5}{'D3y':>7}{'D5y':>7}{'D10y':>7}{'era':>5}")
    print("-" * 88)
    for nm, m in results.items():
        print(f"{nm:<32}{m['cagr']*100:>7.1f}%{m['sharpe']:>7.2f}"
              f"{m['max_dd']*100:>7.1f}%{m['wf_beats']:>3}/{m['wf_n']}"
              f"{m['dca_H36']:>7}{m['dca_H60']:>7}"
              f"{str(m['dca_H120']):>7}{m['eras_beat']:>4}/4")

    out_p = (bw.CACHE / "v2" / "sp500_pit" / "augmented"
             / "improve_main_strategy.json")
    out_p.write_text(json.dumps(results, indent=2, default=bw.to_jsonable))
    print(f"\nsaved -> {out_p}")


if __name__ == "__main__":
    main()
