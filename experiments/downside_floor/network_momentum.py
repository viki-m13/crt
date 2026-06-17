"""Network momentum (Pu-Roberts-Zohren-Dong, L2GMOM) tied into our framework.

Our thread found: own-momentum has ~0 cross-sectional IC for direction, but the
correlation NETWORK is highly predictable. Network momentum exploits exactly
that: a stock's signal is the momentum of its network NEIGHBOURS, propagated
over the (predictable) correlation graph. So it turns something we forecast
well (the graph) into a directional signal that own-momentum couldn't provide.

Construction (a transparent, database-free version of the paper's learned net):
  - graph A   = Ledoit-Wolf correlation of 252 trailing daily returns, clipped
                to positive edges, sparsified to each name's top-k neighbours,
                row-normalised (zero diagonal).
  - own_mom   = vol-normalised 12-1 month momentum of each name.
  - net_mom_i = sum_j A_ij * own_mom_j  (neighbours' momentum, not your own).

Part A: cross-sectional IC of own-mom vs net-mom vs forward returns.
Part B: long-only top-quintile portfolios, vol-targeted to 15% (as in the
        paper's Fig 1b), net-mom vs own-mom vs the universe, plus net-mom
        selection allocated by our predicted-covariance min-variance.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from floor_lib import build, HERE, ROOT
from hrp import min_var_weights

warnings.filterwarnings("ignore")
PIT = ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "sp500_pit"
AUG = PIT / "augmented"
COV_LB, TOPK, TARGET_VOL, VOL_LB = 252, 20, 0.15, 6


def build_signals():
    panel = pd.read_parquet(AUG / "sp500_pit_panel.parquet",
                            columns=["asof", "ticker", "fwd_1m_ret", "fwd_3m_ret"])
    panel["asof"] = pd.to_datetime(panel["asof"])
    daily = pd.read_parquet(PIT / "prices_extended_pit.parquet").sort_index()
    dret = daily.pct_change()
    mr = pd.read_parquet(AUG / "monthly_returns_clean.parquet").sort_index()
    gdates = daily.index.values
    midx = list(mr.index)

    recs = []
    for t in sorted(panel["asof"].unique()):
        if t not in mr.index:
            continue
        names = list(panel[panel["asof"] == t]["ticker"])
        gpos = np.searchsorted(gdates, np.datetime64(t), side="right") - 1
        win = dret.iloc[gpos - COV_LB + 1: gpos + 1]
        # own momentum: vol-normalised 12-1m from monthly returns
        mpos = midx.index(t)
        if mpos < 13:
            continue
        hist = mr.iloc[mpos - 12: mpos]            # months t-12..t-1
        names = [n for n in names if n in win.columns and win[n].notna().all()
                 and n in mr.columns and hist[n].notna().all()]
        if len(names) < 30:
            continue
        mom = (1 + hist[names]).prod() - 1
        vol = hist[names].std(ddof=0) * np.sqrt(12) + 1e-6
        own = (mom / vol)                          # vol-normalised momentum
        own_z = (own - own.mean()) / (own.std(ddof=0) + 1e-9)

        corr = LedoitWolf().fit(win[names].values).covariance_
        d = np.sqrt(np.diag(corr)); corr = corr / np.outer(d, d)
        A = np.clip(corr, 0, None); np.fill_diagonal(A, 0.0)
        # sparsify to top-k neighbours per row
        if A.shape[0] > TOPK:
            thr = np.sort(A, axis=1)[:, -TOPK][:, None]
            A = np.where(A >= thr, A, 0.0)
        rowsum = A.sum(1, keepdims=True); rowsum[rowsum == 0] = 1.0
        A = A / rowsum
        net = A @ own_z.values                     # neighbours' momentum

        sub = panel[(panel["asof"] == t)].set_index("ticker").reindex(names)
        recs.append(pd.DataFrame({
            "asof": t, "ticker": names,
            "own_mom": own_z.values, "net_mom": net,
            "fwd_1m": sub["fwd_1m_ret"].values, "fwd_3m": sub["fwd_3m_ret"].values}))
    return pd.concat(recs, ignore_index=True)


def ic(df, sig, tgt):
    s = df.dropna(subset=[sig, tgt])
    per = s.groupby("asof").apply(
        lambda g: g[sig].corr(g[tgt], method="spearman") if len(g) > 10 else np.nan,
        include_groups=False).dropna()
    return per.mean(), per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))


def main():
    sig = build_signals()
    print("=== Part A: does the network give directional IC own-mom lacked? ===")
    print(f"{'signal':<12}{'IC vs fwd_1m':>18}{'IC vs fwd_3m':>18}")
    for name, col in [("own_mom", "own_mom"), ("net_mom", "net_mom")]:
        i1, t1 = ic(sig, col, "fwd_1m")
        i3, t3 = ic(sig, col, "fwd_3m")
        print(f"{name:<12}{i1:>9.3f}(t{t1:>4.1f}){i3:>9.3f}(t{t3:>4.1f})")

    # ---- Part B: vol-targeted long-only quintile portfolios ----
    mr = pd.read_parquet(AUG / "monthly_returns_clean.parquet").sort_index()
    daily = pd.read_parquet(PIT / "prices_extended_pit.parquet").sort_index()
    spy = daily["SPY"].resample("ME").last().pct_change()
    last_seen = mr.apply(lambda c: c.last_valid_index())
    midx = list(mr.index)

    def fwd(t_next, names):
        r = mr.loc[t_next, names].astype(float)
        for nm in names:
            if pd.isna(r[nm]) and last_seen[nm] is not None and t_next > last_seen[nm]:
                r[nm] = -1.0
        return r.fillna(0.0)

    rows = {"net_mom": [], "own_mom": [], "univ": []}
    months, spy_r = [], []
    for t, g in sig.groupby("asof"):
        pos = midx.index(t)
        if pos + 1 >= len(midx):
            break
        t_next = midx[pos + 1]
        names_all = [n for n in g["ticker"] if n in mr.columns]
        if len(names_all) < 30:
            continue
        K = max(10, len(g) // 5)                    # top quintile
        for key, col in [("net_mom", "net_mom"), ("own_mom", "own_mom")]:
            top = g.nlargest(K, col)["ticker"]
            top = [n for n in top if n in mr.columns]
            rows[key].append(float(fwd(t_next, top).mean()))
        rows["univ"].append(float(fwd(t_next, names_all).mean()))
        spy_r.append(float(spy.get(t_next, np.nan)))
        months.append(pd.Timestamp(t_next))

    base = {k: pd.Series(v, index=months) for k, v in rows.items()}
    base["SPY"] = pd.Series(spy_r, index=months).fillna(0.0)

    def vt(r):                                      # 15% vol target
        out = []
        for i in range(len(r)):
            e = 1.0 if i < VOL_LB else np.clip(
                TARGET_VOL / (r.iloc[i - VOL_LB:i].std(ddof=0) * np.sqrt(12) + 1e-9), 0, 1.5)
            out.append(e * r.iloc[i])
        return pd.Series(out, index=r.index)

    def perf(r):
        r = r.values; eq = np.cumprod(1 + r)
        cagr = eq[-1] ** (12 / len(r)) - 1
        vol = r.std(ddof=0) * np.sqrt(12)
        dd = (eq / np.maximum.accumulate(eq) - 1).min()
        return cagr, (r.mean() * 12) / vol if vol > 0 else 0, dd

    print("\n=== Part B: long-only top-quintile, 15% vol target (paper-style) ===")
    print(f"{'strategy':<26}{'CAGR':>7}{'Sharpe':>8}{'maxDD':>8}")
    out = {}
    for name, r in [("SPY (long only)", base["SPY"]),
                    ("universe avg", vt(base["univ"])),
                    ("own-momentum", vt(base["own_mom"])),
                    ("network-momentum", vt(base["net_mom"]))]:
        c, sh, dd = perf(r); out[name] = dict(cagr=c, sharpe=sh, maxdd=dd)
        print(f"{name:<26}{c*100:>6.1f}%{sh:>8.2f}{dd*100:>7.0f}%")

    (HERE / "network_momentum_results.json").write_text(json.dumps(
        {"ic": {"own_1m": ic(sig, "own_mom", "fwd_1m"),
                "net_1m": ic(sig, "net_mom", "fwd_1m"),
                "own_3m": ic(sig, "own_mom", "fwd_3m"),
                "net_3m": ic(sig, "net_mom", "fwd_3m")},
         "portfolio": out}, indent=2, default=float))
    print(f"\nwrote {HERE/'network_momentum_results.json'}")

    # ---- mirror the paper's Fig 1: raw vs 15% vol-target cumulative returns ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lines = [("SPY (long only)", base["SPY"], "#1f77b4"),
             ("universe avg", base["univ"], "#888888"),
             ("own-momentum", base["own_mom"], "#ff7f0e"),
             ("network-momentum", base["net_mom"], "#d62728")]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    for label, r, col in lines:
        axA.plot((1 + r).cumprod().index, (1 + r).cumprod().values, color=col, label=label)
        rv = vt(r)
        axB.plot((1 + rv).cumprod().index, (1 + rv).cumprod().values, color=col,
                 label=f"{label} (Sh {perf(rv)[1]:.2f})")
    axA.set_title("(a) Raw signals"); axB.set_title("(b) Rescaled to 15% volatility target")
    for ax in (axA, axB):
        ax.set_ylabel("Cumulative growth of $1"); ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=8)
    fig.suptitle("Network momentum on S&P 500 equities (our data) — mirroring "
                 "Pu et al. L2GMOM Fig 1\nnetwork helps drawdown, but gives no "
                 "directional edge in single-asset-class equities", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "network_momentum.png", dpi=130)
    print(f"wrote {HERE/'network_momentum.png'}")


if __name__ == "__main__":
    main()
