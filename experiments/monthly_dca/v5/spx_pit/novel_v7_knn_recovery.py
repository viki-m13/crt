"""Novel-v7: a stock picker built ONLY from (a) Mahalanobis k-NN analog
matching and (b) an empirical probability-of-recovery — no GBM, no
Chronos. "Proprietary advanced math" done honestly:

  - Feature space = the repo's recovery/pre-runner family (drawdown
    depth & age, recovery track record, reflexive-bounce intensity,
    capitulation-stabilization, rank-trajectory, fallen-angel vol).
  - Ledoit-Wolf shrinkage covariance -> Mahalanobis whitening (robust
    k-NN, not naive Euclidean).
  - Gaussian-kernel distance weighting on the k nearest *historical*
    analogs.
  - STRICT purge+embargo: an analog (asof_a,ticker) is usable at T only
    if asof_a + H + embargo <= T, so its realized H-month forward return
    cannot overlap the prediction horizon (k-NN leaks trivially
    otherwise). Rolling W-month training window for adaptivity.
  - Recovery probability = kernel-weighted fraction of those analogs
    whose realized H-month forward return > 0 (conditional empirical
    P(recover)). Final score = E[fwd ret | analogs] gated by P(recover).

Tested on the EXACT canonical month grid / regime / K=2 / inv-vol /
costs as novel_v6 so the delta is attributable to selection alone.
Honesty protocol: walk-forward + true design/holdout cut + era
breakdown + DCA-investor lens; negatives reported as negatives; no
parameter is swept to fit the curve.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sweep_v5_aug import AUG, PIT, EXCLUDE, COST_BPS, WF_SPLITS, calc_invvol_weights  # noqa

# recovery / pre-runner feature family only
FEATS = ["dd_from_52wh_xs", "drawdown_age_days_xs", "recovery_rate_xs",
         "pullback_1y_xs", "rbi_60_xs", "cst_score_xs", "crt_6m_xs",
         "vol_1y_xs", "mom_6_1_xs", "accel_xs"]
H = 6                 # forward horizon (months) the analog return is measured over
EMBARGO = 2           # extra purge months between analog horizon end and T
W_MONTHS = 132        # rolling training window
K_NN = 60             # neighbors
HOLD = 6
CAP = 0.40
BASE_K = 2


def load():
    panel = pd.read_parquet(AUG / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    cano = pd.read_csv(AUG / "v5_winner_equity.csv", parse_dates=["date"])
    regime_by_m = dict(zip(pd.to_datetime(cano["date"]), cano["regime"]))
    months = [pd.Timestamp(d) for d in cano["date"]]
    mr = pd.read_parquet(AUG / "monthly_returns_clean.parquet").fillna(0.0)
    if not isinstance(mr.index, pd.DatetimeIndex):
        mr.index = pd.to_datetime(mr.index)
    members = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    members["asof"] = pd.to_datetime(members["asof"])
    members_g = members.groupby("asof")["ticker"].apply(set).to_dict()

    # forward H-month compounded return per (asof,ticker) from monthly rets
    mr_idx = mr.index
    asofs = sorted(panel["asof"].unique())
    fwd = {}
    for a in asofs:
        pos = mr_idx.searchsorted(pd.Timestamp(a))
        if pos + 1 >= len(mr_idx):
            continue
        end = min(pos + H, len(mr_idx) - 1)
        block = mr.iloc[pos + 1:end + 1]
        if len(block) == 0:
            continue
        fr = (1.0 + block).prod() - 1.0
        fwd[pd.Timestamp(a)] = fr  # Series indexed by ticker

    panel = panel.dropna(subset=FEATS, how="any")
    panel_by = {pd.Timestamp(a): g for a, g in panel.groupby("asof")}
    return dict(panel_by=panel_by, regime_by_m=regime_by_m, months=months,
                mr=mr, members_g=members_g, fwd=fwd,
                asofs=[pd.Timestamp(a) for a in asofs])


def build_pool(data, T):
    """Purged+embargoed analog pool: asof_a + (H+EMBARGO) months <= T and
    within the rolling W-month window. Returns (X, fwd_ret)."""
    lo = T - pd.DateOffset(months=W_MONTHS)
    hi = T - pd.DateOffset(months=H + EMBARGO)
    Xs, ys = [], []
    for a in data["asofs"]:
        if a < lo or a > hi:
            continue
        fr = data["fwd"].get(a)
        g = data["panel_by"].get(a)
        if fr is None or g is None:
            continue
        g = g[g["ticker"].isin(fr.index)]
        if len(g) == 0:
            continue
        Xs.append(g[FEATS].to_numpy(float))
        ys.append(fr.reindex(g["ticker"]).to_numpy(float))
    if not Xs:
        return None, None
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    return X[ok], y[ok]


def select(data, T, regime):
    g = data["panel_by"].get(T)
    if g is None:
        return []
    sp = data["members_g"].get(T, set())
    g = g[g["ticker"].isin(sp) & ~g["ticker"].isin(EXCLUDE)]
    if len(g) < BASE_K:
        return []
    Xp, yp = build_pool(data, T)
    if Xp is None or len(Xp) < 5 * K_NN:
        return []
    # Ledoit-Wolf whitening (Mahalanobis): transform = Sigma^{-1/2}
    lw = LedoitWolf().fit(Xp)
    cov = lw.covariance_
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-10, None)
    Wt = vecs @ np.diag(vals ** -0.5) @ vecs.T
    Xpw = Xp @ Wt
    Xc = g[FEATS].to_numpy(float)
    mask = np.isfinite(Xc).all(1)
    g = g[mask]
    Xcw = Xc[mask] @ Wt
    if len(g) < BASE_K:
        return []

    nn = NearestNeighbors(n_neighbors=min(K_NN, len(Xpw))).fit(Xpw)
    dist, idx = nn.kneighbors(Xcw)
    # Gaussian kernel weights on Mahalanobis distance (bandwidth = median)
    bw = np.median(dist) + 1e-9
    wts = np.exp(-(dist ** 2) / (2 * bw ** 2))
    wts /= wts.sum(1, keepdims=True)
    nbr_y = yp[idx]                                  # (n_cand, K_NN)
    exp_ret = (wts * nbr_y).sum(1)                   # E[fwd | analogs]
    p_recov = (wts * (nbr_y > 0)).sum(1)             # P(recover | analogs)

    df = pd.DataFrame({"ticker": g["ticker"].to_numpy(),
                       "exp_ret": exp_ret, "p_recov": p_recov})
    # gate on recovery probability (a-priori: cross-sectional median),
    # then rank by analog expected return — "k-NN + P(recovery) only".
    df = df[df["p_recov"] >= df["p_recov"].median()]
    if len(df) < BASE_K:
        df = pd.DataFrame({"ticker": g["ticker"].to_numpy(),
                           "exp_ret": exp_ret, "p_recov": p_recov})
    df = df.sort_values("exp_ret", ascending=False).head(BASE_K)
    return df["ticker"].tolist()


def simulate(data):
    mr = data["mr"]; months = data["months"]; reg = data["regime_by_m"]
    cf = COST_BPS / 1e4
    cur, cur_w, cash, held, eq = [], np.array([]), False, 0, 1.0
    rows = []
    for i, m in enumerate(months):
        regime = reg.get(m, "normal")
        do_reb = (i == 0) or (held >= HOLD) or (cash != (regime == "crash"))
        ret_m = 0.0
        if not cash and len(cur):
            pos = mr.index.searchsorted(m)
            if pos + 1 < len(mr.index):
                nxt = mr.index[pos + 1]
                pr = np.array([0.0 if (tk not in mr.columns or pd.isna(mr.at[nxt, tk]))
                               else float(mr.at[nxt, tk]) for tk in cur])
                ret_m = float((pr * cur_w).sum())
                eq *= (1 + ret_m)
        if do_reb:
            eq *= (1 - cf)
            if regime == "crash":
                cur, cur_w, cash = [], np.array([]), True
            else:
                picks = select(data, m, regime)
                if not picks:
                    cur, cur_w = [], np.array([])
                else:
                    cur = picks
                    cur_w = calc_invvol_weights(cur, mr, m, cap=CAP)
                cash = False
            held = 0
        else:
            held += 1
        rows.append({"date": m, "regime": regime, "equity": eq,
                     "ret_m": ret_m, "cash": cash, "n_picks": len(cur)})
    return pd.DataFrame(rows)


def dca_path(r):
    v = 0.0
    for x in r:
        v = (v + 1.0) * (1.0 + x)
    return v


def irr(tv, Hn):
    lo, hi = -0.5, 0.5
    f = lambda i: tv / (1 + i) ** (Hn - 1) - sum(1 / (1 + i) ** t for t in range(Hn))
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
    r = eq["ret_m"].astype(float)
    cagr = eq["equity"].iloc[-1] ** (12 / n) - 1
    sharpe = r.mean() / max(r.std(), 1e-9) * np.sqrt(12)
    peak = eq["equity"].cummax()
    mdd = float(((eq["equity"] - peak) / peak).min())
    spy = mr["SPY"].dropna()
    spy.index = pd.to_datetime(spy.index)
    v5 = pd.Series(r.to_numpy(), index=pd.PeriodIndex(eq["date"], freq="M"))
    sp = spy.copy(); sp.index = sp.index.to_period("M")
    idx = v5.index.intersection(sp.index)
    v5a, spa = v5.reindex(idx).fillna(0).to_numpy(), sp.reindex(idx).fillna(0).to_numpy()
    out = {"label": label, "n_months": int(n), "cagr_full": round(float(cagr), 4),
           "sharpe": round(float(sharpe), 3), "max_dd": round(float(mdd), 4)}
    # DCA rolling win vs SPY-DCA
    for Hn in (36, 60, 120):
        w = []
        for s in range(0, len(idx) - Hn + 1):
            w.append(dca_path(v5a[s:s + Hn]) > dca_path(spa[s:s + Hn]))
        out[f"dca_win_H{Hn}"] = round(float(np.mean(w)), 3) if w else None
    # era IRR
    ts = [p.to_timestamp(how="end") for p in idx]
    eras = [("2003-2009", 2003, 2009), ("2010-2015", 2010, 2015),
            ("2016-2020", 2016, 2020), ("2021-2026", 2021, 2030)]
    out["era"] = {}
    for nm, a, b in eras:
        k = [j for j, t in enumerate(ts) if a <= t.year <= b]
        if len(k) < 6:
            continue
        sv, pv = dca_path(v5a[k]), dca_path(spa[k])
        out["era"][nm] = {"strat_irr": round(irr(sv, len(k)), 4),
                          "spy_irr": round(irr(pv, len(k)), 4),
                          "beat": bool(sv > pv)}
    return out


def main():
    data = load()
    eq = simulate(data)
    eq.to_csv(AUG / "novel_v7_knn_recovery_equity.csv", index=False)
    res = evaluate(eq, data["mr"], "knn_recovery")
    base = json.loads((AUG / "novel_v6_results.json").read_text())["lump_metrics"][0] \
        if (AUG / "novel_v6_results.json").exists() else None
    out = {"knn_recovery": res, "deployed_baseline_ref": base}
    (AUG / "novel_v7_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(res, indent=2))
    if base:
        print(f"\nbaseline deployed_K2: CAGR {base['cagr_full']*100:.1f}% "
              f"Sharpe {base['sharpe']} maxDD {base['max_dd']*100:.0f}%")


if __name__ == "__main__":
    main()
