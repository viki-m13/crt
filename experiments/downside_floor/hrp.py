"""Hierarchical Risk Parity (Lopez de Prado 2016) vs predicted-covariance
minimum-variance, head to head on the same monthly loop.

HRP avoids inverting the covariance matrix (which makes Markowitz min-var
fragile out of sample). Instead it:
  1. clusters names by their correlation distance (a dendrogram),
  2. quasi-diagonalizes the covariance by the cluster order,
  3. allocates top-down by recursive inverse-variance bisection.
This keeps the diversification benefit of the covariance forecast while being
far more robust to estimation error, so it usually drops drawdown further.

Run with EW / inverse-vol / min-variance / HRP on the SAME candidate pool and
covariance each month, so the comparison is apples to apples. Broad pool by
default (min-var needs heterogeneity to add value — see REPORT).
"""
from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from floor_lib import build, HERE, ROOT

warnings.filterwarnings("ignore")

PIT = ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "sp500_pit"
AUG = PIT / "augmented"
N_CAND = int(sys.argv[1]) if len(sys.argv) > 1 else 150
W_CAP = 0.06
COV_LOOKBACK = 252
TARGET_VOL = 0.10
VOL_LOOKBACK = 6


def z(s):
    s = s.astype(float); sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 1e-12 else s * 0.0


def floor_score(g):
    return (z(g["gbm_maxdd_3m"]) - z(g["gbm_uw_frac_3m"]) - z(g["vol_3m_xs"])
            + 0.5 * z(g["trend_health_5y_xs"]) + 0.5 * z(g["chr_trough_q30_3m"]))


def min_var_weights(cov):
    n = len(cov); w0 = np.ones(n) / n
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    res = minimize(lambda w: w @ cov @ w, w0, jac=lambda w: 2 * cov @ w,
                   method="SLSQP", bounds=[(0.0, W_CAP)] * n, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum() if w.sum() > 0 else w0


# ---- HRP ----
def _quasi_diag(link):
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    n_items = link[-1, 3]
    while sort_ix.max() >= n_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= n_items]
        i, j = df0.index, df0.values - n_items
        sort_ix[i] = link[j, 0]
        sort_ix = pd.concat([sort_ix, pd.Series(link[j, 1], index=i + 1)]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _cluster_var(cov, items):
    c = cov[np.ix_(items, items)]
    ivp = 1.0 / np.diag(c); ivp /= ivp.sum()
    return float(ivp @ c @ ivp)


def hrp_weights(cov):
    d = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(d, d), -1, 1)
    dist = np.sqrt(np.clip((1 - corr) / 2.0, 0, None))
    link = linkage(squareform(dist, checks=False), method="single")
    order = _quasi_diag(link)
    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            half = len(c) // 2
            c0, c1 = c[:half], c[half:]
            v0, v1 = _cluster_var(cov, c0), _cluster_var(cov, c1)
            alpha = 1 - v0 / (v0 + v1)
            w[c0] *= alpha; w[c1] *= (1 - alpha)
            nxt += [c0, c1]
        clusters = nxt
    return w.sort_index().values


def perf(r):
    r = np.asarray(r, float); eq = np.cumprod(1 + r)
    cagr = eq[-1] ** (12 / len(r)) - 1
    vol = r.std(ddof=0) * np.sqrt(12)
    dd = eq / np.maximum.accumulate(eq) - 1
    return dict(cagr=cagr, vol=vol, sharpe=(r.mean() * 12) / vol if vol > 0 else 0,
                maxdd=dd.min(), underwater=float((dd < -1e-9).mean()),
                pos_mo=float((r > 0).mean()))


def main():
    df = build().copy()
    df["fs"] = df.groupby("asof", group_keys=False).apply(
        lambda g: floor_score(g), include_groups=False)
    daily = pd.read_parquet(PIT / "prices_extended_pit.parquet").sort_index()
    dret = daily.pct_change()
    mr = pd.read_parquet(AUG / "monthly_returns_clean.parquet").sort_index()
    spy = daily["SPY"].resample("ME").last().pct_change()
    last_seen = mr.apply(lambda c: c.last_valid_index())
    midx = list(mr.index); gdates = daily.index.values

    schemes = {"EW": [], "InvVol": [], "MinVar": [], "HRP": []}
    spy_r, months = [], []
    for t in [a for a in sorted(df["asof"].unique()) if a in mr.index]:
        pos = midx.index(t)
        if pos + 1 >= len(midx):
            break
        t_next = midx[pos + 1]
        g = df[df["asof"] == t].nlargest(N_CAND, "fs")
        gpos = np.searchsorted(gdates, np.datetime64(t), side="right") - 1
        win = dret.iloc[gpos - COV_LOOKBACK + 1: gpos + 1]
        names = [tk for tk in g["ticker"] if tk in win.columns and tk in mr.columns
                 and win[tk].notna().all()]
        if len(names) < 10:
            continue
        cov = LedoitWolf().fit(win[names].values).covariance_ * 252
        vols = np.sqrt(np.diag(cov))
        W = {"EW": np.ones(len(names)) / len(names),
             "InvVol": (1 / vols) / (1 / vols).sum(),
             "MinVar": min_var_weights(cov),
             "HRP": hrp_weights(cov)}
        rnext = mr.loc[t_next, names].astype(float)
        for nm in names:
            if pd.isna(rnext[nm]) and last_seen[nm] is not None and t_next > last_seen[nm]:
                rnext[nm] = -1.0
        rnext = rnext.fillna(0.0).values
        for k, w in W.items():
            schemes[k].append(float(w @ rnext))
        spy_r.append(float(spy.get(t_next, np.nan)))
        months.append(t_next)

    base = {k: pd.Series(v, index=months) for k, v in schemes.items()}
    base["SPY"] = pd.Series(spy_r, index=months).fillna(0.0)

    def vol_target(r):
        out = []
        for i in range(len(r)):
            e = 1.0 if i < VOL_LOOKBACK else np.clip(
                TARGET_VOL / (r.iloc[i - VOL_LOOKBACK:i].std(ddof=0) * np.sqrt(12) + 1e-9), 0, 1.0)
            out.append(e * r.iloc[i])
        return pd.Series(out, index=r.index)

    series = {
        "SPY (buy & hold)": base["SPY"],
        "equal-weight": base["EW"],
        "inverse-vol": base["InvVol"],
        "min-variance (pred cov)": base["MinVar"],
        "HRP (Lopez de Prado)": base["HRP"],
        "HRP + vol-target 10%": vol_target(base["HRP"]),
    }
    print(f"=== HRP vs min-variance (broad pool N={N_CAND}, monthly 2003-2026) ===")
    print(f"{'strategy':<30}{'CAGR':>7}{'vol':>7}{'Sharpe':>8}{'maxDD':>8}{'t-undr':>8}{'pos-mo':>8}")
    out = {}
    for name, r in series.items():
        p = perf(r.values); out[name] = p
        print(f"{name:<30}{p['cagr']*100:>6.1f}%{p['vol']*100:>6.1f}%{p['sharpe']:>8.2f}"
              f"{p['maxdd']*100:>7.0f}%{p['underwater']*100:>7.0f}%{p['pos_mo']*100:>7.0f}%")
    (HERE / "hrp_results.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {HERE/'hrp_results.json'}")


if __name__ == "__main__":
    main()
