"""Novel-v8: honest meta-allocation over the THREE already-validated
streams — v5 picker, market-neutral sleeve, S&P — does any non-overfit
combination fix the recent-era (2021+) weakness without wrecking
drawdown? No new alpha is claimed; only realized, walk-forward info is
used; thresholds are a-priori (at most a 2-point sensitivity is
reported, never optimized). Negatives reported as negatives.

Variants
  1  v5_only                      reference
  2  static_50_50_v5_mn           reference
  3  static_60_40_v5_spy          reference
  4  trend_v5_else_mn             hold v5 iff its trailing-12m cum>0 else MN
  5  trend_v5_else_spy            same, fallback SPY
  6  rel_trend_v5_else_mn         hold v5 iff v5 12m > SPY 12m else MN
  7  invvol_v5_mn                 trailing-12m inverse-vol weights, monthly
  8  minvar_3                     trailing-36m long-only min-variance {v5,mn,spy}
  9  voltgt_v5_mn                 scale v5 to 15% ann (trailing-12m vol), rest MN
All weights use ONLY data available before the month they apply to.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dca_investor_eval as dca  # validated DCA math
AUG = HERE.parents[1] / "cache" / "v2" / "sp500_pit" / "augmented"


def dca_terminal(r):
    v = 0.0
    for x in r:
        v = (v + 1.0) * (1.0 + x)
    return v


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


def weights_for(name, v5, mn, spy):
    """Return (n,3) weight matrix over [v5,mn,spy], each row using only
    info strictly before month t (shift(1) on all trailing stats)."""
    n = len(v5)
    W = np.zeros((n, 3))
    df = pd.DataFrame({"v5": v5, "mn": mn, "spy": spy})
    for t in range(n):
        if name == "v5_only":
            w = [1, 0, 0]
        elif name == "static_50_50_v5_mn":
            w = [.5, .5, 0]
        elif name == "static_60_40_v5_spy":
            w = [.6, 0, .4]
        elif name in ("trend_v5_else_mn", "trend_v5_else_spy", "rel_trend_v5_else_mn"):
            if t < 12:
                w = [1, 0, 0]
            else:
                c5 = np.prod(1 + v5[t - 12:t]) - 1
                cs = np.prod(1 + spy[t - 12:t]) - 1
                good = (c5 > cs) if name == "rel_trend_v5_else_mn" else (c5 > 0)
                if good:
                    w = [1, 0, 0]
                elif name == "trend_v5_else_spy":
                    w = [0, 0, 1]
                else:
                    w = [0, 1, 0]
        elif name == "invvol_v5_mn":
            if t < 12:
                w = [.5, .5, 0]
            else:
                s5 = v5[t - 12:t].std() + 1e-9
                sm = mn[t - 12:t].std() + 1e-9
                a, b = 1 / s5, 1 / sm
                w = [a / (a + b), b / (a + b), 0]
        elif name == "voltgt_v5_mn":
            if t < 12:
                w = [1, 0, 0]
            else:
                ann = v5[t - 12:t].std() * np.sqrt(12) + 1e-9
                s = min(0.15 / ann, 1.0)
                w = [s, 1 - s, 0]
        elif name == "minvar_3":
            if t < 36:
                w = [1, 0, 0]
            else:
                C = np.cov(df.iloc[t - 36:t].to_numpy().T) + np.eye(3) * 1e-6
                try:
                    inv = np.linalg.inv(C)
                    one = np.ones(3)
                    raw = inv @ one
                    raw = np.clip(raw, 0, None)
                    w = (raw / raw.sum()) if raw.sum() > 0 else [1, 0, 0]
                except np.linalg.LinAlgError:
                    w = [1, 0, 0]
        else:
            w = [1, 0, 0]
        W[t] = w
    return W


def stream(name, v5, mn, spy):
    W = weights_for(name, v5, mn, spy)
    R = np.column_stack([v5, mn, spy])
    return (W * R).sum(1)


def evaluate(r, spy, label):
    r = np.asarray(r)
    n = len(r)
    cagr = np.prod(1 + r) ** (12 / n) - 1
    sharpe = r.mean() / max(r.std(), 1e-9) * np.sqrt(12)
    eq = np.cumprod(1 + r)
    mdd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min())
    out = {"label": label, "cagr": round(float(cagr), 4),
           "sharpe": round(float(sharpe), 3), "max_dd": round(float(mdd), 4)}
    for H in (36, 60, 120):
        w = [dca_terminal(r[s:s + H]) > dca_terminal(spy[s:s + H])
             for s in range(0, n - H + 1)]
        out[f"dca_win_H{H}"] = round(float(np.mean(w)), 3) if w else None
    return out


def main():
    df = dca.load_streams()        # period-indexed, aligned, validated
    idx = df.index
    v5, mn, spy = df["v5"].to_numpy(), df["mn"].to_numpy(), df["SPY"].to_numpy()
    ts = [p.to_timestamp(how="end") for p in idx]
    eras = [("2003-2009", 2003, 2009), ("2010-2015", 2010, 2015),
            ("2016-2020", 2016, 2020), ("2021-2026", 2021, 2030)]

    variants = ["v5_only", "static_50_50_v5_mn", "static_60_40_v5_spy",
                "trend_v5_else_mn", "trend_v5_else_spy", "rel_trend_v5_else_mn",
                "invvol_v5_mn", "voltgt_v5_mn", "minvar_3"]
    results = {}
    for nm in variants:
        r = stream(nm, v5, mn, spy)
        e = evaluate(r, spy, nm)
        era = {}
        for en, a, b in eras:
            k = [j for j, t in enumerate(ts) if a <= t.year <= b]
            if len(k) < 6:
                continue
            sv, pv = dca_terminal(r[k]), dca_terminal(spy[k])
            era[en] = {"strat_irr": round(irr(sv, len(k)), 4),
                       "spy_irr": round(irr(pv, len(k)), 4),
                       "beat": bool(sv > pv)}
        e["era"] = era
        e["n_eras_beat_spy"] = sum(v["beat"] for v in era.values())
        results[nm] = e

    (AUG / "novel_v8_meta_alloc.json").write_text(json.dumps(results, indent=2))
    print(f"{'variant':<22}{'CAGR':>7}{'Shrp':>6}{'MaxDD':>8}"
          f"{'10yWin':>8}{'erasBeat':>9}  era IRRs (03-09/10-15/16-20/21-26)")
    for nm, e in results.items():
        ei = "/".join(f"{e['era'].get(x,{}).get('strat_irr',0)*100:+.0f}"
                       for x in ["2003-2009", "2010-2015", "2016-2020", "2021-2026"])
        print(f"{nm:<22}{e['cagr']*100:>6.1f}%{e['sharpe']:>6.2f}"
              f"{e['max_dd']*100:>7.0f}%{e['dca_win_H120']*100:>7.0f}%"
              f"{e['n_eras_beat_spy']:>6}/4   {ei}")
    sp_ir = "/".join(f"{results['v5_only']['era'][x]['spy_irr']*100:+.0f}"
                      for x in ["2003-2009", "2010-2015", "2016-2020", "2021-2026"])
    print(f"{'(S&P-DCA era IRRs)':<22}{'':<36}{sp_ir}")


if __name__ == "__main__":
    main()
