"""Phase F2: build the fundamental-quality sleeve from the PIT EDGAR facts,
then validate it as an ORTHOGONAL second sleeve for E2 on the dual-benchmark
DCA objective (frozen holdout).

Signal: gross profitability (Novy-Marx) = GrossProfit / Assets, lagged to
filing date (PIT). GrossProfit falls back to Revenues - CostOfRevenue/COGS.
Driver = balance-sheet quality, not price -> expected low corr to E2.

Sleeve construction mirrors the repo's price sleeves: monthly months list,
K=2, inverse-vol weights capped 0.40, same SPY crash gate (-> cash),
min-6m hold + score-drift rebalance. No ML, no Chronos.

Validation protocol (SECOND_SLEEVE_SCOPE.md):
  * correlation to E2 per WF split (require |rho| small & STABLE)
  * blend at fixed weights; dual-benchmark win-rate on the frozen holdout
  * report honestly incl. coverage caveat (~73% of universe).
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

from improve_main_strategy import load_inputs  # noqa
from improve_consistency_dca import (build_e2, bench_aligned, dual_dca,  # noqa
                                     lump_metrics, HOLDOUT_START)
import experiments.monthly_dca.v5.build_webapp_v5_pit as bw  # noqa

PIT = bw.PIT
AUG = PIT / "augmented"
EXCLUDE = bw.EXCLUDE


def pivot_latest(facts, tag):
    """For one tag: rows (ticker, filed, end, val); annual (FY) preferred."""
    d = facts[facts.tag == tag].copy()
    return d


def build_quality_panel(asofs):
    """For each asof month, gross profitability per ticker using only facts
    filed strictly on/before asof (PIT). Annual (10-K) values, latest filed."""
    facts = pd.read_parquet(AUG / "fundamentals_pit_facts.parquet")
    facts["filed"] = pd.to_datetime(facts["filed"])
    facts["end"] = pd.to_datetime(facts["end"])
    # prefer annual filings for stock variables (Assets) and flow TTM via FY
    gp = facts[facts.tag == "GrossProfit"]
    rev = facts[facts.tag.isin(["Revenues",
               "RevenueFromContractWithCustomerExcludingAssessedTax"])]
    cogs = facts[facts.tag.isin(["CostOfRevenue", "CostOfGoodsAndServicesSold",
                                 "CostOfGoodsSold"])]
    assets = facts[facts.tag == "Assets"]

    def latest_by(df, asof, agg="last"):
        d = df[df.filed <= asof]
        if len(d) == 0:
            return {}
        # use the most-recently-FILED period per ticker
        d = d.sort_values("filed").groupby("ticker").tail(1)
        return dict(zip(d.ticker, d.val))

    # annual flows: sum the 4 most-recent quarters or take FY; simplest robust
    # = most recent FY (fp==FY) gross profit & revenue/cogs filed <= asof.
    def latest_annual(df, asof):
        d = df[(df.filed <= asof) & (df.fp == "FY")]
        if len(d) == 0:
            d = df[df.filed <= asof]
            if len(d) == 0:
                return {}
        d = d.sort_values("filed").groupby("ticker").tail(1)
        return dict(zip(d.ticker, d.val))

    rows = []
    for asof in asofs:
        A = latest_by(assets, asof)          # balance-sheet: point value
        GP = latest_annual(gp, asof)
        RV = latest_annual(rev, asof)
        CG = latest_annual(cogs, asof)
        tickers = set(A) & (set(GP) | (set(RV) & set(CG)))
        for tk in tickers:
            a = A.get(tk)
            if not a or a <= 0:
                continue
            g = GP.get(tk)
            if g is None and tk in RV and tk in CG:
                g = RV[tk] - CG[tk]
            if g is None:
                continue
            rows.append({"asof": asof, "ticker": tk, "gp_to_assets": g / a})
    panel = pd.DataFrame(rows)
    return panel


def _ret_date(mr_idx, m):
    m = pd.Timestamp(m)
    pos1 = mr_idx.searchsorted(m)
    cands = [(j, abs((mr_idx[j] - m).days)) for j in (pos1 - 1, pos1)
             if 0 <= j < len(mr_idx)]
    cands.sort(key=lambda x: x[1])
    if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(mr_idx):
        return None
    return mr_idx[cands[0][0] + 1]


def run_quality_sleeve(panel, members_g, spyf, mr, dates, K=2,
                       min_hold=6, max_hold=24, cost_bps=10.0):
    """Top-K gross-profitability sleeve, invvol cap 0.40, crash->cash,
    min-hold + score-drift rebalance. Same return convention as run_sim_v2."""
    cf = cost_bps / 10000.0
    mr_idx = mr.index
    pser = {a: g.set_index("ticker")["gp_to_assets"]
            for a, g in panel.groupby("asof")}
    cur, w, cash, held, eq, bid = [], np.array([]), False, 0, 1.0, 0
    log = []

    def topk(asof, k):
        if asof not in pser:
            return None
        s = pser[asof]
        elig = [t for t in s.index if t in members_g.get(asof, set())
                and t not in EXCLUDE and t in mr.columns]
        if len(elig) < k:
            return None
        top = s.loc[elig].sort_values(ascending=False).head(k)
        return list(top.index)

    for i, m in enumerate(dates):
        m = pd.Timestamp(m)
        spy = spyf.loc[m].to_dict() if m in spyf.index else {}
        regime = bw.classify_regime_tight(spy)
        do_reb = (i == 0) or (cash != (regime == "crash"))
        if held >= max_hold:
            do_reb = True
        elif held >= min_hold and cur and regime != "crash":
            cand = topk(m, K)
            if cand is not None and not (set(cur) & set(cand)):
                do_reb = True
        if do_reb:
            if regime == "crash":
                cur, w, cash, held = [], np.array([]), True, 0
            else:
                top = topk(m, K)
                if top is None:
                    cur, w, cash = [], np.array([]), True
                else:
                    cur = top
                    w = bw._calc_invvol_weights(pd.DataFrame({"ticker": cur}),
                                                mr, m, cap=bw.CAP_PER_PICK)
                    cash, held, bid = False, 0, bid + 1
        # realise return
        if cash or not cur:
            ret = 0.0
        else:
            nd = _ret_date(mr_idx, m)
            if nd is None:
                ret = 0.0
            else:
                pr, any_d = [], False
                for tk in cur:
                    rr = mr.at[nd, tk] if tk in mr.columns else np.nan
                    if pd.isna(rr):
                        pr.append(0.0)
                    else:
                        pr.append(float(rr)); any_d = True
                ret = 0.0 if not any_d else float((np.array(pr) * w).sum())
        if not cash and cur:
            eq *= (1 + ret) * (1 - cf) if do_reb else (1 + ret)
        held += 1
        log.append({"date": str(m.date()), "ret_m": ret, "regime": regime})
    return log


WF_SPLITS = [("2011-01", "2018-12"), ("2013-01", "2019-12"),
             ("2015-01", "2021-12"), ("2017-01", "2022-12"),
             ("2019-01", "2024-12"), ("2021-01", "2026-04")]


def main():
    inp = load_inputs()
    members_g, _, spyf, mr, _, _ = inp
    d0, rE2 = build_e2(inp)
    dates = [pd.Timestamp(x) for x in d0]
    panel = build_quality_panel(dates)
    cov = panel.groupby("asof").ticker.nunique()
    print(f"quality panel: {len(panel)} rows | avg names/mo "
          f"{cov[cov.index>='2011-01-01'].mean():.0f} | "
          f"first {cov[cov>0].index.min().date()}")

    log = run_quality_sleeve(panel, members_g, spyf, mr, dates)
    rQ = np.array([x["ret_m"] for x in log])
    fQ = lump_metrics(rQ[[i for i, d in enumerate(dates) if d >= '2011-01-01']],
                      [d for d in dates if d >= pd.Timestamp('2011-01-01')])
    print(f"quality sleeve (>=2011): CAGR {fQ['cagr']*100:.1f}% "
          f"Sharpe {fQ['sharpe']:.2f} DD {fQ['max_dd']*100:.1f}%")

    # correlation to E2, per WF split (stability bar |rho|<0.25)
    dser = pd.to_datetime(pd.Series(d0))
    print("\ncorr(quality, E2) per split:")
    corrs = []
    for lo, hi in WF_SPLITS:
        msk = ((dser >= lo) & (dser <= hi)).to_numpy()
        if msk.sum() < 6:
            continue
        c = np.corrcoef(rQ[msk], rE2[msk])[0, 1]
        corrs.append(c)
        print(f"  {lo}..{hi}: rho={c:+.2f}")
    m11 = (dser >= "2011-01-01").to_numpy()
    full_rho = float(np.corrcoef(rQ[m11], rE2[m11])[0, 1])
    print(f"  full-sample(>=2011) rho={full_rho:+.2f} | "
          f"max|rho|={max(abs(c) for c in corrs):.2f}")

    # blend on dual-benchmark holdout (only where quality sleeve is live: >=2011)
    spv = bench_aligned(d0, mr, "SPY")
    qqv = bench_aligned(d0, mr, "QQQ")
    print("\n=== E2 + fundamental-quality blend, FROZEN HOLDOUT (>=2016) ===")
    print(f"{'w_q':>5} {'CAGR>=11':>9} {'DD':>7} | {'HOLD 1/3/5y':>12} | HOLD worst1y")
    out = {"corr_splits": corrs, "blends": {}}
    for w in (0.0, 0.10, 0.15, 0.20, 0.30):
        r = (1 - w) * rE2 + w * rQ
        H = dual_dca(r, spv, qqv, d0, lo=HOLDOUT_START)
        m11 = [i for i, d in enumerate(dates) if d >= pd.Timestamp('2011-01-01')]
        f = lump_metrics(r[m11], [dates[i] for i in m11])
        out["blends"][f"w={w:.2f}"] = dict(full_ge2011=f, holdout=H)
        print(f"{w:>5.2f} {f['cagr']*100:8.1f}% {f['max_dd']*100:6.1f}% | "
              f"{H[12]['win_both']*100:3.0f}/{H[36]['win_both']*100:3.0f}/"
              f"{H[60]['win_both']*100:3.0f} | "
              f"{H[12]['worst_vs_spy']:.2f}/{H[12]['worst_vs_qqq']:.2f}")

    pd.DataFrame(log).to_csv(AUG / "fundamental_quality_sleeve_returns.csv", index=False)
    (AUG / "fundamental_quality_sleeve.json").write_text(
        json.dumps(out, indent=2, default=bw.to_jsonable))
    print(f"\nsaved -> {AUG/'fundamental_quality_sleeve.json'}")


if __name__ == "__main__":
    main()
