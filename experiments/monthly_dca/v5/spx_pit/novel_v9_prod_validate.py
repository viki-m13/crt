"""Novel-v9 PRODUCTION-HARNESS validation.

Runs the REAL pipeline (build_webapp_v5_pit.run_full_sim: PIT membership
+ Chronos p70 filter + inv-vol cap + production tight regime gate +
rule-based min-6m/score-drift rebalance + 10bps) for SCORER_MODE in
{ml_3plus6 (deployed), consensus (novel-v9)} on identical inputs, then
clears the repo's gauntlet on the resulting return streams:

  - canonical 10-split walk-forward (sweep_v5_aug.WF_SPLITS)
  - non-overlapping era IRRs vs S&P-DCA
  - rolling DCA-vs-S&P-DCA win rates (3/5/10y)
  - Monte-Carlo synthetic-delisting overlay (per-pick monthly hazard)

Honest: deployed default is unchanged in the module; this only flips
the global for the experiment. Negatives reported as negatives.
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


def load_inputs():
    members = pd.read_parquet(bw.PIT / "sp500_membership_monthly.parquet")
    members["asof"] = pd.to_datetime(members["asof"])
    members_g = members.groupby("asof")["ticker"].apply(set).to_dict()
    mr = pd.read_parquet(bw._path("monthly_returns_clean.parquet"))
    mp = pd.read_parquet(bw._path("monthly_prices_clean.parquet"))
    spy_features = bw.load_spy_features()
    preds = pd.read_parquet(bw._path("ml_preds.parquet"))
    preds["asof"] = pd.to_datetime(preds["asof"])
    cpath = bw._path("ml_preds_chronos.parquet")
    chronos = None
    if cpath.exists():
        c = pd.read_parquet(cpath)
        c["asof"] = pd.to_datetime(c["asof"])
        chronos = {pd.Timestamp(a): dict(zip(g["ticker"], g["chronos_p70_3m"]))
                   for a, g in c.groupby("asof")}
    return members_g, preds, spy_features, mr, mp, chronos


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


def mc_delist(rets_log, mr, alpha, n_iter=20, seed=0):
    """Per-pick monthly hazard alpha/yr: a held pick wiped to -100% that
    month and removed for the rest of its basket. Median full CAGR."""
    rng = np.random.default_rng(seed)
    mr_idx = mr.index
    out = []
    for _ in range(n_iter):
        eq = 1.0
        dead = set()
        cur_bid = None
        for row in rets_log:
            bid = row.get("basket_id")
            if bid != cur_bid:
                dead = set(); cur_bid = bid
            picks = row.get("picks") or []
            if not picks or row.get("cash"):
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
        n = len([r for r in rets_log])
        out.append(eq ** (12.0 / n) - 1.0)
    return float(np.median(out))


def evaluate(rets_log, mr):
    eq = pd.DataFrame(rets_log)
    eq["date"] = pd.to_datetime(eq["date"])
    r = eq["ret_m"].astype(float).to_numpy()
    n = len(r)
    spv = []
    for d in eq["date"]:
        p = mr.index.searchsorted(d)
        spv.append(float(mr["SPY"].iloc[min(p + 1, len(mr) - 1)]))
    spv = np.array(spv)
    cagr = np.prod(1 + r) ** (12 / n) - 1
    sh = r.mean() / max(r.std(), 1e-9) * np.sqrt(12)
    e = np.cumprod(1 + r)
    mdd = float(((e - np.maximum.accumulate(e)) / np.maximum.accumulate(e)).min())
    out = {"cagr": round(float(cagr), 4), "sharpe": round(float(sh), 3),
           "max_dd": round(float(mdd), 4)}
    # canonical WF splits
    wf = []
    for nm, lo, hi in WF_SPLITS:
        msk = (eq["date"] >= lo) & (eq["date"] <= hi)
        rr = r[msk.to_numpy()]
        ss = spv[msk.to_numpy()]
        if len(rr) < 6:
            continue
        cg = np.prod(1 + rr) ** (12 / len(rr)) - 1
        sg = np.prod(1 + ss) ** (12 / len(ss)) - 1
        wf.append(cg > sg)
    out["wf_n_beats_spy"] = int(np.sum(wf))
    out["wf_n_splits"] = int(len(wf))
    # rolling DCA win
    for H in (36, 60, 120):
        w = [dca_t(r[s:s + H]) > dca_t(spv[s:s + H]) for s in range(0, n - H + 1)]
        out[f"dca_win_H{H}"] = round(float(np.mean(w)), 3) if w else None
    # era
    yrs = eq["date"].dt.year.to_numpy()
    out["era"] = {}
    for nm, a, b in [("2003-2009", 2003, 2009), ("2010-2015", 2010, 2015),
                     ("2016-2020", 2016, 2020), ("2021-2026", 2021, 2030)]:
        k = np.where((yrs >= a) & (yrs <= b))[0]
        if len(k) < 6:
            continue
        out["era"][nm] = {"strat_irr": round(irr(dca_t(r[k]), len(k)), 4),
                          "spy_irr": round(irr(dca_t(spv[k]), len(k)), 4),
                          "beat": bool(dca_t(r[k]) > dca_t(spv[k]))}
    out["n_eras_beat"] = sum(v["beat"] for v in out["era"].values())
    return out


def main():
    members_g, preds, spyf, mr, mp, chronos = load_inputs()
    res = {}
    for mode in ("ml_3plus6", "consensus"):
        bw.SCORER_MODE = mode
        rl, _, _ = bw.run_full_sim(members_g, preds, preds, spyf, mr, mp,
                                   chronos_preds=chronos, cost_bps=10.0,
                                   hold_months=bw.HOLD_MONTHS, K=bw.K_PICKS)
        e = evaluate(rl, mr)
        e["mc_delist"] = {f"a{int(a*100)}": round(mc_delist(rl, mr, a), 4)
                          for a in (0.0, 0.04, 0.08)}
        res[mode] = e
    bw.SCORER_MODE = "ml_3plus6"
    out_p = bw.CACHE / "v2" / "sp500_pit" / "augmented" / "novel_v9_prod_validation.json"
    out_p.write_text(json.dumps(res, indent=2))

    print(f"\n{'metric':<22}{'ml_3plus6 (deployed)':>22}{'consensus (v9)':>18}")
    for kx in ("cagr", "sharpe", "max_dd"):
        print(f"{kx:<22}{res['ml_3plus6'][kx]:>22}{res['consensus'][kx]:>18}")
    for kx in ("wf_n_beats_spy", "dca_win_H36", "dca_win_H60", "dca_win_H120", "n_eras_beat"):
        print(f"{kx:<22}{str(res['ml_3plus6'][kx]):>22}{str(res['consensus'][kx]):>18}")
    print("era IRR (strat vs S&P-DCA):")
    for er in ("2003-2009", "2010-2015", "2016-2020", "2021-2026"):
        a = res['ml_3plus6']['era'].get(er, {}); b = res['consensus']['era'].get(er, {})
        print(f"  {er:<11} deployed {a.get('strat_irr',0)*100:+5.0f}% vs SPY "
              f"{a.get('spy_irr',0)*100:+4.0f}%   |  consensus "
              f"{b.get('strat_irr',0)*100:+5.0f}% vs SPY {b.get('spy_irr',0)*100:+4.0f}%")
    print("MC synthetic-delisting median CAGR:")
    for a in ("a0", "a4", "a8"):
        print(f"  alpha {a:<3} deployed {res['ml_3plus6']['mc_delist'][a]*100:6.1f}%   "
              f"consensus {res['consensus']['mc_delist'][a]*100:6.1f}%")


if __name__ == "__main__":
    main()
