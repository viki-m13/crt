"""Phase 12: probe candidate SECOND alpha sleeves for orthogonality.

The honest path to materially higher blended Sharpe (Phase 11) is a
second alpha sleeve that is genuinely uncorrelated to the deployed
v5 price-momentum GBM picker. This script BUILDS several candidate
price-only sleeves from features already in the augmented panel, then
measures, for each:

  - standalone Sharpe / CAGR / Max DD (walk-forward honest: the signal
    is a fixed transform of already-PIT-clean features; no fitting)
  - **correlation of its monthly return stream to the deployed v5
    sleeve** (the decisive number)
  - the projected optimal-blend Sharpe with v5 via the 2-asset
    tangency bound

Candidate sleeves (each: top-K=2 by the signal, invvol cap 0.40,
tight regime gate, SAME min-6m + score_drift rebalance as deployed,
NO Chronos filter — Chronos is part of v5's DNA so reusing it would
re-introduce correlation):

  value_ltr     score = -mom_5y           (5-year losers; deep value /
                                            long-term reversal — the
                                            canonical momentum opposite)
  deep_pullback score = pullback_5y       (furthest below 5-year high)
  lowvol        score = -vol_1y           (low-volatility anomaly)
  quality       score = quality_score_5y  (stable 5y compounders)
  meanrev_st    score = -ret_21d          (1-month reversal)
  lowbeta       score = -beta_2y          (low-beta anomaly)

Output:
  augmented/second_sleeve_probe.csv       per-sleeve metrics + corr + blend
  augmented/second_sleeve_streams.csv     monthly return streams (audit)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_v5_aug import (  # noqa: E402
    AUG, PIT, EXCLUDE, COST_BPS, WF_SPLITS,
    classify_regime_tight, load_spy_features, calc_invvol_weights,
)

ROOT = Path(__file__).resolve().parents[4]
K = 2
CAP = 0.40
MIN_HOLD = 6
MAX_HOLD = 24

# (name, signal column, ascending?) ascending=True => pick SMALLEST values
SLEEVES = [
    ("value_ltr",     "mom_5y",            True),   # smallest 5y return
    ("deep_pullback", "pullback_5y",       True),   # most-negative pullback
    ("lowvol",        "vol_1y",            True),   # lowest vol
    ("quality",       "quality_score_5y",  False),  # highest quality
    ("meanrev_st",    "ret_21d",           True),   # 1m losers
    ("lowbeta",       "beta_2y",           True),   # lowest beta
]


def build_sleeve(panel_by_asof, members_g, mr, spy, months,
                  signal_col, ascending, k=K, cap=CAP):
    """Rule-based (min6+score_drift) sleeve picking top-K by `signal_col`.
    No Chronos. Returns a monthly return Series."""
    cf = COST_BPS / 1e4

    def _cand(m_):
        sp = panel_by_asof.get(m_)
        if sp is None:
            return None
        st = members_g.get(m_, set())
        s = sp[sp["ticker"].isin(st)]
        s = s[~s["ticker"].isin(EXCLUDE)]
        s = s.dropna(subset=[signal_col])
        if len(s) < k:
            return None
        s = s.sort_values(signal_col, ascending=ascending)
        return s.head(k)

    cur, w = [], np.array([])
    cash = False
    held = 0
    rets = {}
    for i, m in enumerate(months):
        regime = classify_regime_tight(spy.loc[m].to_dict() if m in spy.index else {})
        do_reb = (i == 0) or (cash != (regime == "crash"))
        if held >= MAX_HOLD:
            do_reb = True
        elif held >= MIN_HOLD and cur and regime != "crash":
            c = _cand(m)
            if c is not None and not (set(cur) & set(c["ticker"])):
                do_reb = True
        ret_m = 0.0
        if not cash and cur:
            pos = mr.index.searchsorted(m)
            if pos + 1 < len(mr.index):
                nd = mr.index[pos + 1]
                pr = [0.0 if pd.isna(mr.at[nd, tk]) else float(mr.at[nd, tk])
                      for tk in cur if tk in mr.columns]
                if len(pr) == len(w):
                    ret_m = float((np.array(pr) * w).sum())
        if do_reb:
            if regime == "crash":
                cur, w, cash = [], np.array([]), True
            else:
                c = _cand(m)
                if c is None:
                    cur, w = [], np.array([])
                else:
                    cur = c["ticker"].tolist()
                    w = calc_invvol_weights(cur, mr, m, cap=cap)
                cash = False
            held = 0
            ret_m = ret_m - cf
        else:
            held += 1
        rets[m] = ret_m
    return pd.Series(rets).sort_index()


def stats(r):
    r = r.dropna()
    n = len(r)
    if n < 6:
        return dict(cagr=0, vol=0, sharpe=0, mdd=0, n=n)
    cagr = (1 + r).prod() ** (12 / n) - 1
    vol = r.std() * np.sqrt(12)
    sh = (r.mean() / r.std()) * np.sqrt(12) if r.std() > 0 else 0
    ec = (1 + r).cumprod()
    mdd = float(((ec - ec.cummax()) / ec.cummax()).min())
    return dict(cagr=float(cagr), vol=float(vol), sharpe=float(sh),
                mdd=float(mdd), n=n)


def wf_mean_sharpe(r):
    out = []
    for split, lo, hi in WF_SPLITS:
        seg = r[(r.index >= pd.Timestamp(lo)) & (r.index <= pd.Timestamp(hi))].dropna()
        if len(seg) >= 6 and seg.std() > 0:
            out.append((seg.mean() / seg.std()) * np.sqrt(12))
    return float(np.mean(out)) if out else 0.0, float(np.min(out)) if out else 0.0


def tangency_sharpe(sr_a, sr_b, rho):
    """Max Sharpe of an optimally-weighted 2-asset blend."""
    denom = 1 - rho ** 2
    if denom <= 0:
        return max(sr_a, sr_b)
    return float(np.sqrt(max(0.0, (sr_a**2 + sr_b**2 - 2*rho*sr_a*sr_b) / denom)))


def main():
    t0 = time.time()
    print("Loading augmented panel ...")
    panel = pd.read_parquet(AUG / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    spy = load_spy_features()
    mr = pd.read_parquet(AUG / "monthly_returns_clean.parquet").fillna(0.0)
    if not isinstance(mr.index, pd.DatetimeIndex):
        mr.index = pd.to_datetime(mr.index)
    members = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    members["asof"] = pd.to_datetime(members["asof"])
    members_g = members.groupby("asof")["ticker"].apply(set).to_dict()
    panel_by_asof = {a: g for a, g in panel.groupby("asof")}
    months = sorted(set(panel["asof"]).intersection(set(spy.index)))
    months = [pd.Timestamp(m) for m in months]

    # The deployed v5 alpha stream (canonical, audited).
    e = pd.read_csv(AUG / "v5_winner_equity.csv")
    A = e.set_index("date")["ret_m"].astype(float)
    A.index = pd.to_datetime(A.index)
    sA = stats(A)
    print(f"\nDeployed v5 sleeve (A): Sharpe {sA['sharpe']:.2f}  "
          f"CAGR {sA['cagr']*100:.1f}%  vol {sA['vol']*100:.0f}%  "
          f"MaxDD {sA['mdd']*100:.0f}%  n={sA['n']}")

    rows = []
    streams = {"v5_A": A}
    print(f"\n{'sleeve':<14}{'Sharpe':>7}{'CAGR':>8}{'vol':>6}{'MaxDD':>7}"
          f"{'corr→A':>8}{'blendSR*':>9}{'WFmeanSR':>9}{'WFminSR':>8}")
    for name, col, asc in SLEEVES:
        B = build_sleeve(panel_by_asof, members_g, mr, spy, months, col, asc)
        # Align to A's window
        B = B.reindex(A.index).dropna()
        Aa = A.reindex(B.index)
        sB = stats(B)
        rho = Aa.corr(B)
        srstar = tangency_sharpe(sA["sharpe"], sB["sharpe"], rho)
        wfm, wfmin = wf_mean_sharpe(B)
        rows.append({
            "sleeve": name, "signal": col,
            "sharpe": sB["sharpe"], "cagr": sB["cagr"], "vol": sB["vol"],
            "mdd": sB["mdd"], "corr_to_v5": float(rho),
            "blend_tangency_sharpe": srstar,
            "wf_mean_sharpe": wfm, "wf_min_sharpe": wfmin, "n": sB["n"],
        })
        streams[name] = B
        print(f"{name:<14}{sB['sharpe']:>7.2f}{sB['cagr']*100:>7.1f}%"
              f"{sB['vol']*100:>5.0f}%{sB['mdd']*100:>6.0f}%"
              f"{rho:>+8.2f}{srstar:>9.2f}{wfm:>9.2f}{wfmin:>8.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(AUG / "second_sleeve_probe.csv", index=False)
    pd.DataFrame(streams).to_csv(AUG / "second_sleeve_streams.csv")

    # Best candidate + a 3-sleeve projection (v5 + best two ~uncorrelated)
    df_ok = df[df["sharpe"] > 0.3].sort_values("blend_tangency_sharpe", ascending=False)
    print(f"\nRanked by projected 2-asset blend Sharpe with v5:")
    print(df_ok[["sleeve", "sharpe", "corr_to_v5", "blend_tangency_sharpe",
                 "wf_mean_sharpe"]].to_string(index=False))

    if len(df_ok):
        best = df_ok.iloc[0]
        # 3-sleeve naive equal-weight projection (illustrative, not optimized)
        B = streams[best["sleeve"]]
        eqw = pd.concat([A, B], axis=1).dropna().mean(axis=1)
        s3 = stats(eqw)
        print(f"\nIllustrative equal-weight blend  v5 + {best['sleeve']}:")
        print(f"  Sharpe {s3['sharpe']:.2f}  CAGR {s3['cagr']*100:.1f}%  "
              f"vol {s3['vol']*100:.0f}%  MaxDD {s3['mdd']*100:.0f}%")
        print(f"  (tangency-optimal would be ~{best['blend_tangency_sharpe']:.2f})")

    (AUG / "second_sleeve_winner.json").write_text(
        json.dumps(df_ok.head(1).to_dict("records"), indent=2, default=str))
    print(f"\nSaved -> {AUG / 'second_sleeve_probe.csv'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
