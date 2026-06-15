"""Predicted-covariance MINIMUM-VARIANCE portfolio — the natural next step
that stays 100% on the predictable side of the ledger.

Volatility is predictable (IC 0.744, t=126); correlations are nearly as
persistent. A predicted COVARIANCE matrix lets us solve for the long-only
minimum-variance weights, which exploit not just each name's vol but how the
names co-move — usually beating inverse-vol on drawdown because it actively
diversifies correlated risk.

Each month:
  1. candidate set = top-N FloorScore names (the low-future-risk pool)
  2. estimate Sigma from trailing 252 daily returns, Ledoit-Wolf shrinkage
     (shrinkage is essential — a raw 50x50 sample cov on 252 days is unstable)
  3. solve  min w'Sigma w   s.t.  sum w = 1, 0 <= w <= cap   (SLSQP)
  4. hold one month; honest delisting (-100% for a name that disappears)

Compared against SPY, the equal-weight and inverse-vol Floor baskets, and the
same min-var book with a 10% vol-target overlay. We also report predicted vs
realized portfolio vol to show the covariance forecast is well-calibrated.
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
from floor_lib import build, HERE, ROOT

warnings.filterwarnings("ignore")

PIT = ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "sp500_pit"
AUG = PIT / "augmented"
# Default = the winning broad/heterogeneous config (min-var needs correlation
# structure to exploit; a narrow pre-screened pool has little, so it loses to
# 1/N there — see REPORT). Override on the CLI: `python3 min_variance.py 50 0.15`.
N_CAND = int(sys.argv[1]) if len(sys.argv) > 1 else 150  # candidate pool breadth
W_CAP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.06  # max weight per name
COV_LOOKBACK = 252  # trailing daily obs for covariance
TARGET_VOL = 0.10
VOL_LOOKBACK = 6


def z(s):
    s = s.astype(float); sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 1e-12 else s * 0.0


def floor_score(g):
    return (z(g["gbm_maxdd_3m"]) - z(g["gbm_uw_frac_3m"]) - z(g["vol_3m_xs"])
            + 0.5 * z(g["trend_health_5y_xs"]) + 0.5 * z(g["chr_trough_q30_3m"]))


def min_var_weights(cov):
    n = len(cov)
    w0 = np.ones(n) / n
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, W_CAP)] * n
    res = minimize(lambda w: w @ cov @ w, w0, jac=lambda w: 2 * cov @ w,
                   method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum() if w.sum() > 0 else w0


def perf(r):
    r = np.asarray(r, float)
    eq = np.cumprod(1 + r)
    yrs = len(r) / 12
    cagr = eq[-1] ** (1 / yrs) - 1
    vol = r.std(ddof=0) * np.sqrt(12)
    sharpe = (r.mean() * 12) / vol if vol > 0 else 0
    dd = eq / np.maximum.accumulate(eq) - 1
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, maxdd=dd.min(),
                underwater=float((dd < -1e-9).mean()), pos_mo=float((r > 0).mean()))


def main():
    df = build().copy()
    df["fs"] = df.groupby("asof", group_keys=False).apply(
        lambda g: floor_score(g), include_groups=False)

    daily = pd.read_parquet(PIT / "prices_extended_pit.parquet").sort_index()
    dret = daily.pct_change()
    mr = pd.read_parquet(AUG / "monthly_returns_clean.parquet").sort_index()
    spy = daily["SPY"].resample("ME").last().pct_change()
    last_seen = mr.apply(lambda c: c.last_valid_index())
    midx = list(mr.index)
    gdates = daily.index.values

    asofs = [a for a in sorted(df["asof"].unique()) if a in mr.index]
    rows = {"EW": [], "InvVol": [], "MinVar": []}
    spy_r, months, pred_real = [], [], []

    for t in asofs:
        pos = midx.index(t)
        if pos + 1 >= len(midx):
            break
        t_next = midx[pos + 1]
        g = df[df["asof"] == t].nlargest(N_CAND, "fs")
        gpos = np.searchsorted(gdates, np.datetime64(t), side="right") - 1
        win = dret.iloc[gpos - COV_LOOKBACK + 1: gpos + 1]
        names = [tk for tk in g["ticker"]
                 if tk in win.columns and tk in mr.columns
                 and win[tk].notna().all()]
        if len(names) < 10:
            continue
        R = win[names].values
        cov = LedoitWolf().fit(R).covariance_ * 252  # annualized

        w_mv = min_var_weights(cov)
        vols = np.sqrt(np.diag(cov))
        w_iv = (1 / vols) / (1 / vols).sum()
        w_ew = np.ones(len(names)) / len(names)

        rnext = mr.loc[t_next, names].astype(float)
        for i, nm in enumerate(names):
            if pd.isna(rnext[nm]) and last_seen[nm] is not None and t_next > last_seen[nm]:
                rnext[nm] = -1.0
        rnext = rnext.fillna(0.0).values

        rows["EW"].append(float(w_ew @ rnext))
        rows["InvVol"].append(float(w_iv @ rnext))
        rows["MinVar"].append(float(w_mv @ rnext))
        pred_real.append((float(np.sqrt(w_mv @ cov @ w_mv)), float(w_mv @ rnext)))
        spy_r.append(float(spy.get(t_next, np.nan)))
        months.append(t_next)

    base = {k: pd.Series(v, index=months) for k, v in rows.items()}
    base["SPY"] = pd.Series(spy_r, index=months).fillna(0.0)

    def vol_target(r):
        out = []
        for i in range(len(r)):
            e = 1.0 if i < VOL_LOOKBACK else np.clip(
                TARGET_VOL / (r.iloc[i - VOL_LOOKBACK:i].std(ddof=0) * np.sqrt(12) + 1e-9),
                0, 1.0)
            out.append(e * r.iloc[i])
        return pd.Series(out, index=r.index)

    series = {
        "SPY (buy & hold)": base["SPY"],
        "Floor cand. equal-weight": base["EW"],
        "Floor cand. inverse-vol": base["InvVol"],
        "Floor min-variance (pred cov)": base["MinVar"],
        "Floor min-var + vol-target 10%": vol_target(base["MinVar"]),
    }

    print(f"=== Predicted-covariance minimum-variance (N={N_CAND}, cap={W_CAP:.0%}, "
          f"monthly 2003-2026) ===")
    print(f"{'strategy':<34}{'CAGR':>7}{'vol':>7}{'Sharpe':>8}{'maxDD':>8}"
          f"{'t-undr':>8}{'pos-mo':>8}")
    out = {}
    for name, r in series.items():
        p = perf(r.values); out[name] = p
        print(f"{name:<34}{p['cagr']*100:>6.1f}%{p['vol']*100:>6.1f}%{p['sharpe']:>8.2f}"
              f"{p['maxdd']*100:>7.0f}%{p['underwater']*100:>7.0f}%{p['pos_mo']*100:>7.0f}%")

    pr = np.array(pred_real)
    real_ann = pr[:, 1].std(ddof=0) * np.sqrt(12)
    print(f"\n  covariance calibration: min-var predicted ann. vol "
          f"{pr[:,0].mean():.1%}  vs realized {real_ann:.1%}")
    print(f"  (a well-calibrated forecast lands the realized vol near the prediction)")

    (HERE / "min_variance_results.json").write_text(json.dumps(
        {"portfolio": out, "pred_vol": float(pr[:, 0].mean()),
         "realized_vol": float(real_ann)}, indent=2, default=float))
    print(f"\nwrote {HERE/'min_variance_results.json'}")


if __name__ == "__main__":
    main()
