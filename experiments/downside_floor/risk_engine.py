"""Utilise the ONE thing we predict very well: volatility / drawdown.

The IC study (ensemble_analysis.py) showed direction is unpredictable
(IC ~ 0) while risk is hugely predictable (IC t-stats to 20). So build the
whole strategy on volatility, never on a direction bet:

  Part A  measure how predictable volatility actually is (persistence IC,
          and trailing-vol -> forward-drawdown IC).
  Part B  monetise it three ways, stacked:
            (1) SELECT the low-future-risk names (FloorScore)
            (2) SIZE them inverse to predicted volatility (risk parity)
            (3) TIME total exposure to a constant target volatility
                (Moreira-Muir vol management) — hold cash when our own
                trailing realized vol is high.
          and compare equity curves vs SPY and an equal-weight basket.

Monthly rebalance, 2003-2026. Honest delisting: a held name that disappears
next month is marked -100%.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from floor_lib import build, HERE, ROOT

warnings.filterwarnings("ignore")

PIT = ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "sp500_pit"
AUG = PIT / "augmented"
K = 20
TARGET_VOL = 0.10            # annualized vol target for the managed book
VOL_LOOKBACK = 6            # months of strategy returns for the vol estimate
MAX_EXPOSURE = 1.0          # no leverage in the headline run


def z(s):
    s = s.astype(float); sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 1e-12 else s * 0.0


def floor_score(g):
    return (z(g["gbm_maxdd_3m"]) - z(g["gbm_uw_frac_3m"]) - z(g["vol_3m_xs"])
            + 0.5 * z(g["trend_health_5y_xs"]) + 0.5 * z(g["chr_trough_q30_3m"]))


# ---------------- Part A: how predictable is volatility? ----------------
def measure_predictability(df):
    daily = pd.read_parquet(PIT / "prices_extended_pit.parquet").sort_index()
    gdates = daily.index.values
    logret = np.log(daily).diff()
    asofs = sorted(df["asof"].unique())
    persist, vol_to_dd = [], []
    lab = df.set_index(["asof", "ticker"])["maxdd_3m"]
    for d in asofs:
        e = np.searchsorted(gdates, np.datetime64(d), side="right") - 1
        if e < 64 or e + 63 >= len(gdates):
            continue
        names = df[df["asof"] == d]["ticker"]
        tv, fv, dd = [], [], []
        for tk in names:
            if tk not in logret.columns:
                continue
            past = logret[tk].values[e - 62:e + 1]
            fut = logret[tk].values[e + 1:e + 64]
            if np.isnan(past).any() or np.isnan(fut).any():
                continue
            tv.append(np.nanstd(past)); fv.append(np.nanstd(fut))
            dd.append(lab.get((d, tk), np.nan))
        if len(tv) > 30:
            tv, fv, dd = np.array(tv), np.array(fv), np.array(dd)
            persist.append(pd.Series(tv).corr(pd.Series(fv), method="spearman"))
            ok = ~np.isnan(dd)
            if ok.sum() > 30:
                vol_to_dd.append(pd.Series(tv[ok]).corr(pd.Series(-dd[ok]), method="spearman"))
    persist, vol_to_dd = np.array(persist), np.array(vol_to_dd)

    def t(x): return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    print("=== Part A: predictability of risk (cross-sectional, per month) ===")
    print(f"  trailing 3m vol  -> forward 3m vol :  IC = {persist.mean():.3f}  (t={t(persist):.0f})")
    print(f"  trailing 3m vol  -> forward drawdown:  IC = {vol_to_dd.mean():.3f}  (t={t(vol_to_dd):.0f})")
    print("  (for contrast, best DIRECTION IC anywhere was 0.03) \n")
    return dict(vol_persistence_ic=float(persist.mean()),
                vol_to_drawdown_ic=float(vol_to_dd.mean()))


# ---------------- Part B: risk-targeted portfolio ----------------
def perf(eq, rets):
    yrs = len(rets) / 12
    cagr = eq[-1] ** (1 / yrs) - 1
    vol = rets.std(ddof=0) * np.sqrt(12)
    sharpe = (rets.mean() * 12) / vol if vol > 0 else 0
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    return dict(cagr=cagr, vol=vol, sharpe=sharpe, maxdd=dd.min(),
                underwater=float((dd < -1e-9).mean()), worst_mo=float(rets.min()),
                pos_mo=float((rets > 0).mean()))


def backtest(df):
    mr = pd.read_parquet(AUG / "monthly_returns_clean.parquet").sort_index()
    daily = pd.read_parquet(PIT / "prices_extended_pit.parquet").sort_index()
    spy = daily["SPY"].resample("ME").last().pct_change()

    df = df.copy()
    df["fs"] = df.groupby("asof", group_keys=False).apply(lambda g: floor_score(g),
                                                          include_groups=False)
    asofs = [a for a in sorted(df["asof"].unique()) if a in mr.index]
    midx = list(mr.index)
    last_seen = mr.apply(lambda c: c.last_valid_index())     # delisting detection

    def fwd_ret(t_next, names):
        r = mr.loc[t_next, names].astype(float)
        for nm in names:                                     # honest delisting
            if pd.isna(r[nm]) and (last_seen[nm] is not None) and (t_next > last_seen[nm]):
                r[nm] = -1.0
        return r.fillna(0.0)

    def trailing_vol(t, names):
        hist = mr.loc[:t, names].tail(12)
        v = hist.std(ddof=0) * np.sqrt(12)
        return v.replace(0, np.nan).fillna(v.median() if np.isfinite(v.median()) else 0.2)

    # raw monthly returns for each weighting scheme (pre vol-target)
    schemes = {"basket_EW": [], "basket_InvVol": []}
    spy_r = []
    months = []
    for i, t in enumerate(asofs):
        pos = midx.index(t)
        if pos + 1 >= len(midx):
            break
        t_next = midx[pos + 1]
        g = df[df["asof"] == t]
        names = list(g.nlargest(K, "fs")["ticker"])
        names = [n for n in names if n in mr.columns]
        if len(names) < 5:
            continue
        r = fwd_ret(t_next, names)
        tv = trailing_vol(t, names)
        w_iv = (1 / tv); w_iv = w_iv / w_iv.sum()
        schemes["basket_EW"].append(float(r.mean()))
        schemes["basket_InvVol"].append(float((w_iv * r).sum()))
        spy_r.append(float(spy.get(t_next, np.nan)))
        months.append(t_next)

    out = {}
    base = {k: pd.Series(v, index=months) for k, v in schemes.items()}
    base["SPY"] = pd.Series(spy_r, index=months).fillna(0.0)

    # vol-target overlay applied to the InvVol basket and to SPY
    def vol_target(rets):
        exp, managed = [], []
        for i in range(len(rets)):
            if i < VOL_LOOKBACK:
                e = 1.0
            else:
                rv = rets.iloc[i - VOL_LOOKBACK:i].std(ddof=0) * np.sqrt(12)
                e = np.clip(TARGET_VOL / rv, 0.0, MAX_EXPOSURE) if rv > 0 else MAX_EXPOSURE
            exp.append(e); managed.append(e * rets.iloc[i])
        return pd.Series(managed, index=rets.index), np.mean(exp)

    series = {
        "SPY (buy & hold)": base["SPY"],
        "Floor basket (equal wt)": base["basket_EW"],
        "Floor basket (inverse-vol)": base["basket_InvVol"],
    }
    mgr, avg_e = vol_target(base["basket_InvVol"])
    series[f"Floor invvol + vol-target {int(TARGET_VOL*100)}%"] = mgr
    spy_mgr, spy_e = vol_target(base["SPY"])
    series["SPY + vol-target 10%"] = spy_mgr

    print(f"=== Part B: risk-targeted portfolio (monthly, K={K}, 2003-2026) ===")
    print(f"{'strategy':<34}{'CAGR':>7}{'vol':>7}{'Sharpe':>8}{'maxDD':>8}"
          f"{'t-undr':>8}{'pos-mo':>8}")
    for name, r in series.items():
        eq = (1 + r).cumprod().values
        p = perf(eq, r.values)
        out[name] = p
        print(f"{name:<34}{p['cagr']*100:>6.1f}%{p['vol']*100:>6.1f}%{p['sharpe']:>8.2f}"
              f"{p['maxdd']*100:>7.0f}%{p['underwater']*100:>7.0f}%{p['pos_mo']*100:>7.0f}%")
    print(f"\n  (vol-target avg exposure: floor {avg_e:.0%}, spy {spy_e:.0%}; "
          f"cash earns 0% here, so returns are conservative)")
    return out


def main():
    df = build()
    a = measure_predictability(df)
    b = backtest(df)
    (HERE / "risk_engine_results.json").write_text(
        json.dumps({"predictability": a, "portfolio": b}, indent=2, default=float))
    print(f"\nwrote {HERE/'risk_engine_results.json'}")


if __name__ == "__main__":
    main()
