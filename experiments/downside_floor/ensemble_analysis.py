"""Correlation-neutral, multi-signal ensemble for the downside objective.

Premise (the user's): HF time-series foundation models are NOT good at
predicting which stock goes UP. So don't ask them to. Instead extract from
several models / factors the thing each is actually good at, check what each
predicts and how correlated they are, then blend them CORRELATION-NEUTRALLY
so redundant signals don't dominate and orthogonal information stacks.

Three parts:
  1. IC table   — cross-sectional Spearman IC of each signal vs
                  (a) forward RETURN [who goes up]  and
                  (b) forward DOWNSIDE [-uw_frac, shallower maxdd].
                  Demonstrates: FM *directional* signals ~ 0 IC for return,
                  while vol/risk signals carry the downside information.
  2. Corr matrix — average cross-sectional rank-correlation between signals,
                  so we can see the orthogonal blocks (momentum vs risk vs
                  quality vs reversion).
  3. Ensemble    — walk-forward optimal blend  w = (C + lambda I)^-1 * IC
                  estimated on PAST asofs only (120d embargo), applied to the
                  current cross-section. Compared against the naive equal-
                  weight blend, the best single signal, and FloorScore; and
                  re-run WITHOUT the HF-model signals to isolate their value.

Signals span independent families; the FM contributions are:
  chr_drift     Chronos-Bolt median 3m forecast return  (a "momentum" read)
  chr_downside  Chronos-Bolt 10th-pct path trough        (a "risk" read)
  ttm_trend     IBM Granite TTM point 3m forecast return (independent model)
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from floor_lib import build, HERE

warnings.filterwarnings("ignore")

TTM = HERE / "ttm_floor_preds.parquet"

# signal -> (column expression, +1 if higher=safer/better-for-downside)
def make_signals(df):
    s = {}
    # --- momentum / trend ---
    s["mom_12_1"] = df["mom_12_1_xs"]
    s["mom_6_1"] = df["mom_6_1_xs"]
    s["chr_drift"] = df["chr_p50_end_3m"]            # FM directional read
    if "ttm_trend_3m" in df:
        s["ttm_trend"] = df["ttm_trend_3m"]          # independent FM (IBM TTM)
    # --- risk / volatility ---
    s["lowvol"] = -df["vol_3m_xs"]
    s["lowbeta"] = -df["beta_2y_xs"]
    s["chr_downside"] = df["chr_trough_q10_3m"]      # FM risk read (less neg = safer)
    # --- quality / trend health ---
    s["trend_health"] = df["trend_health_5y_xs"]
    s["quality5y"] = df["quality_score_5y_xs"]
    s["recovery"] = df["recovery_rate_xs"]
    # --- reversion ---
    s["pullback"] = df["pullback_1y_xs"]
    return pd.DataFrame(s, index=df.index)


FAM = {  # for display grouping
    "mom_12_1": "mom", "mom_6_1": "mom", "chr_drift": "mom", "ttm_trend": "mom",
    "lowvol": "risk", "lowbeta": "risk", "chr_downside": "risk",
    "trend_health": "qual", "quality5y": "qual", "recovery": "qual",
    "pullback": "rev",
}
HF = {"chr_drift", "chr_downside", "ttm_trend"}  # the HF-model signals


def xs_rank(df_sig, by):
    """cross-sectional rank (per asof) -> uniform-ish, for Spearman/blend."""
    return df_sig.groupby(by).rank(pct=True)


def per_asof_ic(sig_rank, target_rank, asof):
    """mean & t-stat of cross-sectional rank corr per asof."""
    out = {}
    g = pd.DataFrame({"asof": asof})
    for c in sig_rank.columns:
        a = sig_rank[c].values
        b = target_rank.values
        tmp = pd.DataFrame({"asof": asof.values, "a": a, "b": b}).dropna()
        ics = tmp.groupby("asof").apply(
            lambda x: np.corrcoef(x["a"], x["b"])[0, 1] if len(x) > 5 else np.nan,
            include_groups=False)
        ics = ics.dropna()
        out[c] = (ics.mean(), ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))))
    return out


def main():
    df = build().copy()
    if TTM.exists():
        ttm = pd.read_parquet(TTM)
        ttm["asof"] = pd.to_datetime(ttm["asof"])
        df = df.merge(ttm, on=["asof", "ticker"], how="left")
        print(f"TTM merged ({df['ttm_trend_3m'].notna().mean():.1%} coverage)")
    else:
        print("TTM preds not found -> ensemble uses Chronos + factors only")

    sig = make_signals(df)
    asof = df["asof"]
    sig_rank = xs_rank(sig, asof)

    # targets (rank within asof)
    up = df.groupby("asof")["end_ret_3m"].rank(pct=True)        # who goes up
    safe = df.groupby("asof")["uw_frac_3m"].rank(pct=True, ascending=False)  # less underwater = high
    shallow = df.groupby("asof")["maxdd_3m"].rank(pct=True)     # less negative = high

    print("\n=== 1) Information Coefficient (mean xs Spearman, t-stat) ===")
    print(f"{'signal':<14}{'fam':>5}{'IC vs UP':>16}{'IC vs SAFE':>16}{'IC vs SHALLOW':>16}")
    ic_up = per_asof_ic(sig_rank, up, asof)
    ic_safe = per_asof_ic(sig_rank, safe, asof)
    ic_shal = per_asof_ic(sig_rank, shallow, asof)
    for c in sig.columns:
        tag = "  <-HF" if c in HF else ""
        print(f"{c:<14}{FAM[c]:>5}"
              f"{ic_up[c][0]:>9.3f}(t{ic_up[c][1]:>4.1f}){ic_safe[c][0]:>9.3f}(t{ic_safe[c][1]:>4.1f})"
              f"{ic_shal[c][0]:>9.3f}(t{ic_shal[c][1]:>4.1f}){tag}")

    # ---- 2) average cross-sectional correlation matrix ----
    cols = list(sig.columns)
    mats = []
    for _, g in sig_rank.groupby(asof.values):
        if len(g) > 20:
            mats.append(g[cols].corr().values)
    C = np.nanmean(mats, axis=0)
    print("\n=== 2) avg cross-sectional signal correlation ===")
    print("        " + "".join(f"{c[:6]:>7}" for c in cols))
    for i, c in enumerate(cols):
        print(f"{c[:7]:<8}" + "".join(f"{C[i, j]:>7.2f}" for j in range(len(cols))))

    # ---- 3) walk-forward correlation-neutral ensemble (target = SAFE) ----
    # precompute per-asof: standardized signals (z), IC_i vs SAFE, corr matrix
    asofs = np.array(sorted(df["asof"].unique()))
    sig_z = sig.groupby(asof).transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9))
    df_loc = df[["asof", "ticker"]].copy()
    for c in cols:
        df_loc[c + "_z"] = sig_z[c].values
        df_loc[c + "_r"] = sig_rank[c].values
    df_loc["safe_r"] = safe.values

    # per-asof IC vector and corr matrix
    icvec, cormat, keyasof = {}, {}, []
    for a, g in df_loc.groupby("asof"):
        if len(g) <= 20:
            continue
        R = g[[c + "_r" for c in cols]]
        b = np.array([np.corrcoef(R[c + "_r"], g["safe_r"])[0, 1] for c in cols])
        icvec[a] = b
        cormat[a] = R.corr().values
        keyasof.append(a)
    keyasof = np.array(sorted(keyasof))

    LAM = 0.5
    EMBARGO = pd.Timedelta(days=120)
    MIN_HIST = 36

    def weights(t, use_cols_idx):
        past = keyasof[keyasof <= (t - EMBARGO)]
        if len(past) < MIN_HIST:
            return None
        b = np.nanmean([icvec[a][use_cols_idx] for a in past], axis=0)
        Cmat = np.nanmean([cormat[a][np.ix_(use_cols_idx, use_cols_idx)] for a in past], axis=0)
        w = np.linalg.solve(Cmat + LAM * np.eye(len(use_cols_idx)), b)
        return w / (np.abs(w).sum() + 1e-9)

    idx_all = np.arange(len(cols))
    idx_nohf = np.array([i for i, c in enumerate(cols) if c not in HF])

    def run(label, use_idx, equal=False):
        K = 10
        recs = []
        zmat = df_loc[[c + "_z" for c in cols]].values
        for t in asofs:
            cur = df_loc["asof"].values == np.datetime64(t)
            sub = df[cur]
            if len(sub) < 2 * K:
                continue
            if equal:
                w = np.ones(len(use_idx)) / len(use_idx)
            else:
                w = weights(pd.Timestamp(t), use_idx)
                if w is None:
                    continue
            score = zmat[cur][:, use_idx] @ w
            top = sub.iloc[np.argsort(-score)[:K]]
            m = top[~top["censored_3m"]]
            if len(m):
                recs.append((m["uw_frac_3m"].mean(), m["ever_below_3m"].mean(),
                             m["end_below_3m"].mean(), m["maxdd_3m"].mean(),
                             m["end_ret_3m"].mean(), (m["uw_frac_3m"] < 0.15).mean()))
        r = np.array(recs)
        return dict(n=len(r), uw=r[:, 0].mean(), ever=r[:, 1].mean(), endb=r[:, 2].mean(),
                    dd=r[:, 3].mean(), ret=r[:, 4].mean(), safe=r[:, 5].mean())

    print("\n=== 3) ensemble backtest (top-10, 3m, walk-forward weights) ===")
    print(f"{'ensemble':<26}{'mo':>5}{'uw':>8}{'ever<':>8}{'end<':>8}{'maxdd':>8}{'meanret':>9}{'safe%':>7}")
    res = {}
    for label, fn in [
        ("equal-weight (all)", lambda: run("eq", idx_all, equal=True)),
        ("corr-neutral (factors only)", lambda: run("cn_f", idx_nohf)),
        ("corr-neutral (+ HF models)", lambda: run("cn_all", idx_all)),
    ]:
        s = fn()
        res[label] = s
        print(f"{label:<26}{s['n']:>5}{s['uw']:>8.3f}{s['ever']:>8.3f}{s['endb']:>8.3f}"
              f"{s['dd']:>8.3f}{s['ret']:>9.3f}{s['safe']*100:>6.1f}%")

    (HERE / "ensemble_results.json").write_text(json.dumps(
        {"ic_up": {k: ic_up[k][0] for k in ic_up},
         "ic_safe": {k: ic_safe[k][0] for k in ic_safe},
         "ic_shallow": {k: ic_shal[k][0] for k in ic_shal},
         "backtest": res}, indent=2, default=float))
    print(f"\nwrote {HERE/'ensemble_results.json'}")


if __name__ == "__main__":
    main()
