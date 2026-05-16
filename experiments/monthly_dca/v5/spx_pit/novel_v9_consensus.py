"""Novel-v9: multi-horizon CONSENSUS picker — a variance-reduction
improvement (not a new-alpha claim), tested on CLEAN data only.

The deployed scorer is mean(pred_3m, pred_6m). The GBM also emits
pred_1m, and the three horizons are only ~0.55-0.81 correlated — i.e.
there is real disagreement, and disagreement is information about how
*robust* a pick is. Hypothesis: selecting names where ALL horizons
agree (low cross-horizon rank dispersion) yields more regime-robust
picks and specifically attacks the recent-era (2021+) fragility,
without claiming any new signal.

Clean-data protocol (post the 2026-05-16 data-integrity fix):
  - scores  : ml_preds.parquet (walk-forward GBM, unaffected by the
              corrupted equity CSV)
  - returns : monthly_returns_clean.parquet (next-month realised)
  - regime  : simple crash gate from CLEAN SPY monthly returns,
              applied identically to every variant
  - membership: PIT S&P 500
  - K=2, min-hold 6m + score-drift, 10bps, a-priori thresholds, no
    sweeping; era + DCA-vs-S&P-DCA lens; negatives reported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sweep_v5_aug import AUG, PIT, EXCLUDE  # noqa

COST = 10 / 1e4
HOLD = 6
K = 2


def load():
    ml = pd.read_parquet(AUG / "ml_preds.parquet")[
        ["asof", "ticker", "pred_1m", "pred_3m", "pred_6m"]]
    ml["asof"] = pd.to_datetime(ml["asof"])
    mr = pd.read_parquet(AUG / "monthly_returns_clean.parquet").fillna(0.0)
    if not isinstance(mr.index, pd.DatetimeIndex):
        mr.index = pd.to_datetime(mr.index)
    mem = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    mem["asof"] = pd.to_datetime(mem["asof"])
    mem_g = mem.groupby("asof")["ticker"].apply(set).to_dict()
    months = sorted(ml["asof"].unique())
    # clean crash gate from SPY monthly returns: crash next month if
    # SPY last-1m <= -8% or trailing-6m cumulative <= -5%
    spy = mr["SPY"]
    crash = {}
    for m in months:
        pos = mr.index.searchsorted(pd.Timestamp(m))
        if pos < 6:
            crash[pd.Timestamp(m)] = False
            continue
        r1 = float(spy.iloc[pos - 1]) if pos >= 1 else 0.0
        r6 = float((1 + spy.iloc[pos - 6:pos]).prod() - 1)
        crash[pd.Timestamp(m)] = (r1 <= -0.08) or (r6 <= -0.05)
    ml_by = {pd.Timestamp(a): g for a, g in ml.groupby("asof")}
    return dict(ml_by=ml_by, mr=mr, mem_g=mem_g,
                months=[pd.Timestamp(m) for m in months], crash=crash)


def pick(g, mode):
    g = g.copy()
    for h in ("pred_1m", "pred_3m", "pred_6m"):
        g[h + "_r"] = g[h].rank(pct=True)
    r1, r3, r6 = g["pred_1m_r"], g["pred_3m_r"], g["pred_6m_r"]
    if mode == "deployed_eq":
        g["score"] = (r3 + r6) / 2
    elif mode == "consensus":
        # keep only names all three horizons rank in the top 40%
        keep = (r1 >= 0.60) & (r3 >= 0.60) & (r6 >= 0.60)
        g = g[keep]
        g["score"] = (r1 + r3 + r6) / 3
    elif mode == "low_disp":
        disp = pd.concat([r1, r3, r6], axis=1).std(axis=1)
        g = g[disp <= disp.median()]
        g["score"] = (r1 + r3 + r6) / 3
    elif mode == "disp_weighted":
        mean_r = (r1 + r3 + r6) / 3
        disp = pd.concat([r1, r3, r6], axis=1).std(axis=1)
        dn = (disp - disp.min()) / (disp.max() - disp.min() + 1e-9)
        g["score"] = mean_r * (1.0 - dn)        # penalise disagreement
    else:
        g["score"] = (r3 + r6) / 2
    if len(g) < K:
        return []
    return g.sort_values("score", ascending=False).head(K)["ticker"].tolist()


def simulate(data, mode):
    mr = data["mr"]; months = data["months"]
    cur, cash, held, eq = [], False, 0, 1.0
    rows = []
    for i, m in enumerate(months):
        is_crash = data["crash"].get(m, False)
        do_reb = (i == 0) or (held >= HOLD) or (cash != is_crash) or not cur
        ret = 0.0
        if not cash and cur:
            pos = mr.index.searchsorted(m)
            if pos + 1 < len(mr.index):
                nxt = mr.index[pos + 1]
                pr = [0.0 if (t not in mr.columns or pd.isna(mr.at[nxt, t]))
                      else float(mr.at[nxt, t]) for t in cur]
                ret = float(np.mean(pr))
                eq *= (1 + ret)
        # score-drift check (rebalance early if neither pick still top-K)
        g = data["ml_by"].get(m)
        if g is not None and not is_crash and cur and not do_reb:
            sp = data["mem_g"].get(m, set())
            gg = g[g["ticker"].isin(sp) & ~g["ticker"].isin(EXCLUDE)]
            topk = pick(gg, mode)
            if topk and not any(c in topk for c in cur):
                do_reb = True
        if do_reb:
            eq *= (1 - COST)
            if is_crash or g is None:
                cur, cash = [], (is_crash)
            else:
                sp = data["mem_g"].get(m, set())
                gg = g[g["ticker"].isin(sp) & ~g["ticker"].isin(EXCLUDE)]
                cur = pick(gg, mode)
                cash = False
            held = 0
        else:
            held += 1
        rows.append({"date": m, "ret_m": ret, "equity": eq, "cash": cash,
                     "n": len(cur)})
    return pd.DataFrame(rows)


def dca_t(r):
    v = 0.0
    for x in r:
        v = (v + 1) * (1 + x)
    return v


def irr(tv, H):
    lo, hi = -0.5, 0.5
    f = lambda i: tv / (1 + i) ** (H - 1) - sum(1 / (1 + i) ** t for t in range(H))
    flo = f(lo); mid = 0.0
    for _ in range(120):
        mid = .5 * (lo + hi); fm = f(mid)
        if abs(fm) < 1e-9:
            break
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return (1 + mid) ** 12 - 1


def evaluate(eq, mr, label):
    n = len(eq)
    r = eq["ret_m"].astype(float).to_numpy()
    spy = mr["SPY"].reindex(pd.DatetimeIndex(eq["date"])).fillna(0.0)
    # align next-month SPY to same convention
    sidx = [mr.index.searchsorted(d) for d in eq["date"]]
    spv = np.array([float(mr["SPY"].iloc[min(p + 1, len(mr) - 1)]) for p in sidx])
    cagr = np.prod(1 + r) ** (12 / n) - 1
    sh = r.mean() / max(r.std(), 1e-9) * np.sqrt(12)
    e = np.cumprod(1 + r)
    mdd = float(((e - np.maximum.accumulate(e)) / np.maximum.accumulate(e)).min())
    out = {"label": label, "cagr": round(float(cagr), 4),
           "sharpe": round(float(sh), 3), "max_dd": round(float(mdd), 4)}
    for H in (36, 60, 120):
        w = [dca_t(r[s:s + H]) > dca_t(spv[s:s + H])
             for s in range(0, n - H + 1)]
        out[f"dca_win_H{H}"] = round(float(np.mean(w)), 3) if w else None
    yrs = pd.DatetimeIndex(eq["date"]).year
    out["era"] = {}
    for nm, a, b in [("2003-2009", 2003, 2009), ("2010-2015", 2010, 2015),
                     ("2016-2020", 2016, 2020), ("2021-2026", 2021, 2030)]:
        k = np.where((yrs >= a) & (yrs <= b))[0]
        if len(k) < 6:
            continue
        sv, pv = dca_t(r[k]), dca_t(spv[k])
        out["era"][nm] = {"strat_irr": round(irr(sv, len(k)), 4),
                          "spy_irr": round(irr(pv, len(k)), 4),
                          "beat": bool(sv > pv)}
    out["n_eras_beat"] = sum(v["beat"] for v in out["era"].values())
    return out


def main():
    data = load()
    res = {}
    for mode in ["deployed_eq", "consensus", "low_disp", "disp_weighted"]:
        eq = simulate(data, mode)
        res[mode] = evaluate(eq, data["mr"], mode)
    (AUG / "novel_v9_consensus.json").write_text(json.dumps(res, indent=2))
    print(f"{'variant':<16}{'CAGR':>7}{'Shrp':>6}{'MaxDD':>8}{'3y':>6}{'5y':>6}"
          f"{'10y':>6}{'erasBeat':>9}  era IRR 03-09/10-15/16-20/21-26")
    for nm, e in res.items():
        ei = "/".join(f"{e['era'].get(x,{}).get('strat_irr',0)*100:+.0f}"
                       for x in ["2003-2009", "2010-2015", "2016-2020", "2021-2026"])
        print(f"{nm:<16}{e['cagr']*100:>6.1f}%{e['sharpe']:>6.2f}{e['max_dd']*100:>7.0f}%"
              f"{e['dca_win_H36']*100:>5.0f}%{e['dca_win_H60']*100:>5.0f}%"
              f"{e['dca_win_H120']*100:>5.0f}%{e['n_eras_beat']:>6}/4   {ei}")
    sp = "/".join(f"{res['deployed_eq']['era'][x]['spy_irr']*100:+.0f}"
                  for x in ["2003-2009", "2010-2015", "2016-2020", "2021-2026"])
    print(f"{'(S&P-DCA IRR)':<16}{'':<42}{sp}")


if __name__ == "__main__":
    main()


def gauntlet():
    """Skeptic's test: is low_disp a plateau or a knife-edge? Does it
    survive a true design(<=2012)/holdout(>=2013) split? No tuning —
    just sweep to SEE the surface and split to SEE OOS."""
    data = load()
    print("\n=== ROBUSTNESS GAUNTLET (skeptical) ===")
    # 1. dispersion-quantile sensitivity for low_disp
    print("disp-quantile sensitivity (low_disp): want a smooth plateau, not a spike")
    for q in (0.30, 0.40, 0.50, 0.60, 0.70):
        def _pick(g, mode, _q=q):
            g = g.copy()
            for h in ("pred_1m", "pred_3m", "pred_6m"):
                g[h + "_r"] = g[h].rank(pct=True)
            r1, r3, r6 = g["pred_1m_r"], g["pred_3m_r"], g["pred_6m_r"]
            disp = pd.concat([r1, r3, r6], axis=1).std(axis=1)
            g = g[disp <= disp.quantile(_q)]
            g["score"] = (r1 + r3 + r6) / 3
            if len(g) < K:
                return []
            return g.sort_values("score", ascending=False).head(K)["ticker"].tolist()
        glob = sys.modules[__name__]
        orig = glob.pick
        glob.pick = _pick
        try:
            e = evaluate(simulate(data, "low_disp"), data["mr"], f"q={q}")
        finally:
            glob.pick = orig
        ei = "/".join(f"{e['era'].get(x,{}).get('strat_irr',0)*100:+.0f}"
                       for x in ["2003-2009","2010-2015","2016-2020","2021-2026"])
        print(f"  q={q:.2f}  CAGR {e['cagr']*100:5.1f}%  Sharpe {e['sharpe']:.2f}  "
              f"10yWin {e['dca_win_H120']*100:3.0f}%  erasBeat {e['n_eras_beat']}/4  era {ei}")

    # 2. consensus-threshold sensitivity
    print("consensus all-3-rank threshold sensitivity:")
    for thr in (0.50, 0.55, 0.60, 0.65, 0.70):
        def _pick(g, mode, _t=thr):
            g = g.copy()
            for h in ("pred_1m", "pred_3m", "pred_6m"):
                g[h + "_r"] = g[h].rank(pct=True)
            r1, r3, r6 = g["pred_1m_r"], g["pred_3m_r"], g["pred_6m_r"]
            g = g[(r1 >= _t) & (r3 >= _t) & (r6 >= _t)]
            g["score"] = (r1 + r3 + r6) / 3
            if len(g) < K:
                return []
            return g.sort_values("score", ascending=False).head(K)["ticker"].tolist()
        glob = sys.modules[__name__]; orig = glob.pick; glob.pick = _pick
        try:
            e = evaluate(simulate(data, "consensus"), data["mr"], f"t={thr}")
        finally:
            glob.pick = orig
        ei = "/".join(f"{e['era'].get(x,{}).get('strat_irr',0)*100:+.0f}"
                       for x in ["2003-2009","2010-2015","2016-2020","2021-2026"])
        print(f"  thr={thr:.2f}  CAGR {e['cagr']*100:5.1f}%  Sharpe {e['sharpe']:.2f}  "
              f"10yWin {e['dca_win_H120']*100:3.0f}%  erasBeat {e['n_eras_beat']}/4  era {ei}")

    # 3. true design/holdout split on low_disp vs deployed_eq
    print("design(<=2012) / holdout(>=2013) split — Sharpe & DCA-vs-SPY:")
    for mode in ("deployed_eq", "low_disp"):
        eq = simulate(data, mode)
        eq["yr"] = pd.DatetimeIndex(eq["date"]).year
        for tag, msk in [("design<=2012", eq["yr"] <= 2012),
                         ("holdout>=2013", eq["yr"] >= 2013)]:
            r = eq.loc[msk, "ret_m"].to_numpy()
            sh = r.mean() / max(r.std(), 1e-9) * (12 ** .5)
            print(f"  {mode:<12} {tag:<14} n={len(r):3d}  Sharpe {sh:5.2f}  "
                  f"cum {(np.prod(1+r)-1)*100:8.1f}%")


if __name__ == "__main__":
    gauntlet()
