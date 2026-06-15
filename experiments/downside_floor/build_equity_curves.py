"""Equity curves vs SPY for the headline risk-built portfolios.

Recomputes the monthly returns for SPY, the equal-weight Floor basket, the
predicted-covariance min-variance book, HRP, and min-var + vol-target, then
plots growth-of-$1 (log scale) and the drawdown path. Saves equity_curves.png.
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from floor_lib import build, HERE, ROOT
from hrp import floor_score, min_var_weights, hrp_weights, N_CAND, W_CAP

warnings.filterwarnings("ignore")
PIT = ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "sp500_pit"
AUG = PIT / "augmented"
COV_LOOKBACK, TARGET_VOL, VOL_LOOKBACK = 252, 0.10, 6


def vol_target(r):
    out = []
    for i in range(len(r)):
        e = 1.0 if i < VOL_LOOKBACK else np.clip(
            TARGET_VOL / (r.iloc[i - VOL_LOOKBACK:i].std(ddof=0) * np.sqrt(12) + 1e-9), 0, 1.0)
        out.append(e * r.iloc[i])
    return pd.Series(out, index=r.index)


def stats(r):
    r = r.values; eq = np.cumprod(1 + r)
    cagr = eq[-1] ** (12 / len(r)) - 1
    vol = r.std(ddof=0) * np.sqrt(12)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return cagr, (r.mean() * 12) / vol, dd


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

    rows = {"EW": [], "MinVar": [], "HRP": []}
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
        W = {"EW": np.ones(len(names)) / len(names),
             "MinVar": min_var_weights(cov), "HRP": hrp_weights(cov)}
        rnext = mr.loc[t_next, names].astype(float)
        for nm in names:
            if pd.isna(rnext[nm]) and last_seen[nm] is not None and t_next > last_seen[nm]:
                rnext[nm] = -1.0
        rnext = rnext.fillna(0.0).values
        for k, w in W.items():
            rows[k].append(float(w @ rnext))
        spy_r.append(float(spy.get(t_next, np.nan)))
        months.append(pd.Timestamp(t_next))

    s = {k: pd.Series(v, index=months) for k, v in rows.items()}
    s["SPY"] = pd.Series(spy_r, index=months).fillna(0.0)
    s["MinVar+VT"] = vol_target(s["MinVar"])

    plot = [
        ("SPY", "SPY (buy & hold)", "#888888", 1.6),
        ("EW", "Floor basket (equal-wt)", "#1f77b4", 1.6),
        ("HRP", "HRP", "#2ca02c", 1.6),
        ("MinVar", "Min-variance (pred cov)", "#d62728", 1.8),
        ("MinVar+VT", "Min-var + vol-target 10%", "#9467bd", 1.8),
    ]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    for key, _, color, lw in plot:
        eq = (1 + s[key]).cumprod()
        cagr, sharpe, dd = stats(s[key])
        ax1.plot(eq.index, eq.values, color=color, lw=lw,
                 label=f"{dict((k,l) for k,l,_,_ in plot)[key]}  "
                       f"(${eq.values[-1]:.0f} | {cagr*100:.1f}%/yr | Sharpe {sharpe:.2f} | DD {dd*100:.0f}%)")
        ddp = (1 + s[key]).cumprod()
        ax2.plot(ddp.index, (ddp / ddp.cummax() - 1).values * 100, color=color, lw=lw)
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of $1 (log)")
    ax1.set_title("Risk-built Floor portfolios vs SPY — monthly, 2003-2026\n"
                  "(built only on predictable volatility/covariance; no direction bets)",
                  fontsize=12)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax1.grid(True, which="both", alpha=0.25)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.25)
    ax2.axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    out = HERE / "equity_curves.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    for key, label, _, _ in plot:
        c, sh, dd = stats(s[key])
        print(f"  {label:<30} final ${ (1+s[key]).cumprod().values[-1]:6.1f}  "
              f"CAGR {c*100:5.1f}%  Sharpe {sh:.2f}  maxDD {dd*100:.0f}%")


if __name__ == "__main__":
    main()
