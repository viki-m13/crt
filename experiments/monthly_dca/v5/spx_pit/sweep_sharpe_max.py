"""Phase 11: Sharpe maximization — blended sleeves + vol-target overlay.

GOAL: honestly push portfolio Sharpe > 2.0 and validate it OOS.

The v5 K=2 rule-based picker alone is Sharpe ~1.10 (the alpha sleeve).
To raise Sharpe we combine it with low-correlation diversifiers and a
vol-target overlay — the standard, non-curve-fit portfolio-construction
levers. Higher K was already shown to LOWER Sharpe (edge is in the
top-2), so the picker config is fixed at the deployed K=2 rule-based.

HONESTY DISCIPLINE
==================
- The ONLY walk-forward-trained component is the GBM picker (already
  embargoed 7 months in ml_preds).
- Blend weights, vol-target level, defensive-sleeve rule are FIXED
  constants chosen by reasoning — NOT optimized per split. We sweep a
  small grid and report the FULL distribution; a config only "counts"
  if its **walk-forward OOS mean Sharpe > 2 AND WF min Sharpe stays
  strong** — not just the full-sample number.
- Vol-target uses TRAILING realized vol only (no look-ahead). Exposure
  is scaled DOWN only (cap = 1.0, no leverage); slack earns 0% (cash).
- All return streams are the augmented-PIT monthly_returns_clean +
  committed ETF proxies (SPY, TLT, XLP, XLU). No new data.

Sleeves:
  A  v5_k2_rule  the deployed alpha sleeve (K=2 + Chronos + score_drift)
  SPY, TLT, XLP  buy-hold ETF streams
  TREND          SPY if SPY trailing-12m momentum > 0 else TLT (dual-mom)
  6040           0.6 SPY + 0.4 TLT monthly-rebalanced

Sweeps:
  1. static blend  w_A in {0.3..0.7} x diversifier in {SPY,TLT,XLP,6040,TREND}
  2. risk-parity   A + defensive, weights ~ 1/trailing-vol
  3. vol-target    overlay on best static blends, target ann vol in {6,8,10,12}%
  4. stacked       best static blend -> vol-target

Output:
  augmented/sharpe_max_sweep.csv         all configs, full + WF metrics
  augmented/sharpe_max_winner.json       best by WF mean Sharpe (with floor)
  augmented/v5_sharpe_alpha_stream.csv   the A sleeve monthly returns (audit)
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

K = 2
CHR_Q = 0.45
CAP = 0.40
MIN_HOLD = 6
MAX_HOLD = 24


def build_v5_alpha_stream():
    """Use the CANONICAL deployed rule-based equity curve from data.json's
    growth.strat_value (produced by build_webapp_v5_pit.py with the real
    entry/exit-price simulation, NaN handling, and the score_drift rule).
    Returns the monthly return Series. This is the audited stream — do NOT
    re-derive the picker here (an independent re-sim drifts from the
    production simulator's price/cost handling)."""
    import json
    d = json.loads((ROOT / "experiments" / "docs" / "monthly-dca" / "data.json").read_text())
    g = pd.DataFrame(d["growth"])
    g["date"] = pd.to_datetime(g["date"])
    g = g.sort_values("date").set_index("date")
    sv = g["strat_value"].astype(float)
    r = sv.pct_change()
    # First row: equity starts at strat_value[0] from $1 deposited that month.
    r.iloc[0] = sv.iloc[0] - 1.0
    return r.dropna() if r.isna().any() else r


ROOT = Path(__file__).resolve().parents[4]


# ------------------------- portfolio helpers -------------------------
def ann_stats(r: pd.Series):
    r = r.dropna()
    n = len(r)
    if n < 6:
        return dict(cagr=0, vol=0, sharpe=0, mdd=0, n=n)
    cagr = (1 + r).prod() ** (12 / n) - 1
    vol = r.std() * np.sqrt(12)
    sharpe = (r.mean() / r.std()) * np.sqrt(12) if r.std() > 0 else 0
    ec = (1 + r).cumprod()
    mdd = float(((ec - ec.cummax()) / ec.cummax()).min())
    return dict(cagr=float(cagr), vol=float(vol), sharpe=float(sharpe),
                mdd=float(mdd), n=n)


def wf_sharpes(r: pd.Series):
    """Sharpe per walk-forward split (OOS-honest: blend params fixed)."""
    out = []
    for split, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        seg = r[(r.index >= lo) & (r.index <= hi)].dropna()
        if len(seg) < 6:
            continue
        sh = (seg.mean() / seg.std()) * np.sqrt(12) if seg.std() > 0 else 0
        cg = (1 + seg).prod() ** (12 / len(seg)) - 1
        out.append({"split": split, "sharpe": float(sh), "cagr": float(cg)})
    return pd.DataFrame(out)


def voltarget(r: pd.Series, target_vol: float, lookback: int = 12, cap: float = 1.0):
    """Scale next-month exposure to hit target ann vol using trailing realized
    vol (no look-ahead). Slack earns 0% (cash). Exposure capped at `cap`."""
    out = {}
    vals = r.dropna()
    idx = vals.index
    for i, d in enumerate(idx):
        if i < lookback:
            out[d] = vals.iloc[i]  # not enough history -> unscaled
            continue
        trail = vals.iloc[i - lookback:i]
        realized = trail.std() * np.sqrt(12)
        if realized <= 0:
            scale = cap
        else:
            scale = min(cap, target_vol / realized)
        out[d] = vals.iloc[i] * scale
    return pd.Series(out).sort_index()


def main():
    t0 = time.time()
    print("Building v5 K=2 rule-based alpha stream ...")
    A = build_v5_alpha_stream()
    A.to_csv(AUG / "v5_sharpe_alpha_stream.csv")
    aS = ann_stats(A)
    print(f"  alpha sleeve: CAGR {aS['cagr']*100:.1f}% vol {aS['vol']*100:.1f}% "
          f"Sharpe {aS['sharpe']:.2f} MaxDD {aS['mdd']*100:.1f}% (n={aS['n']})")

    mr = pd.read_parquet(AUG / "monthly_returns_clean.parquet")
    if not isinstance(mr.index, pd.DatetimeIndex):
        mr.index = pd.to_datetime(mr.index)
    # Align everything to the alpha-stream months
    idx = A.index
    SPY = mr["SPY"].reindex(idx).fillna(0.0)
    TLT = mr["TLT"].reindex(idx).fillna(0.0)
    XLP = mr["XLP"].reindex(idx).fillna(0.0)
    XLU = mr["XLU"].reindex(idx).fillna(0.0)

    # SPY trailing-12m momentum for the dual-momentum TREND sleeve
    spy_cum = (1 + mr["SPY"].fillna(0)).cumprod()
    def trend_stream():
        out = {}
        for d in idx:
            pos = spy_cum.index.searchsorted(d)
            if pos < 13:
                out[d] = SPY.get(d, 0.0); continue
            mom = spy_cum.iloc[pos-1] / spy_cum.iloc[pos-13] - 1
            out[d] = SPY.get(d, 0.0) if mom > 0 else TLT.get(d, 0.0)
        return pd.Series(out).reindex(idx).fillna(0.0)
    TREND = trend_stream()
    S6040 = 0.6 * SPY + 0.4 * TLT

    # Correlations to the alpha sleeve (the key Sharpe lever)
    print("\n  Correlation of diversifiers to alpha sleeve:")
    for nm, s in [("SPY", SPY), ("TLT", TLT), ("XLP", XLP), ("XLU", XLU),
                  ("TREND", TREND), ("6040", S6040)]:
        c = A.corr(s)
        st = ann_stats(s)
        print(f"    {nm:<6} corr={c:+.2f}  Sharpe={st['sharpe']:.2f}  vol={st['vol']*100:.0f}%")

    DIVS = {"SPY": SPY, "TLT": TLT, "XLP": XLP, "XLU": XLU,
            "TREND": TREND, "6040": S6040}

    results = []

    def record(name, r, extra=None):
        full = ann_stats(r)
        wf = wf_sharpes(r)
        rec = {
            "config": name,
            "full_cagr": full["cagr"], "full_vol": full["vol"],
            "full_sharpe": full["sharpe"], "full_mdd": full["mdd"],
            "wf_mean_sharpe": float(wf["sharpe"].mean()) if len(wf) else 0,
            "wf_min_sharpe": float(wf["sharpe"].min()) if len(wf) else 0,
            "wf_max_sharpe": float(wf["sharpe"].max()) if len(wf) else 0,
            "wf_mean_cagr": float(wf["cagr"].mean()) if len(wf) else 0,
            "wf_n_splits": int(len(wf)),
        }
        if extra:
            rec.update(extra)
        results.append(rec)
        return rec

    # baseline
    record("A_only (deployed v5 K=2 rule)", A)

    # 1. static blends
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        for dn, ds in DIVS.items():
            blend = w * A + (1 - w) * ds
            record(f"static w={w:.1f} A + {1-w:.1f} {dn}", blend)

    # 2. risk-parity: weight A and defensive ~ 1/trailing-12m-vol
    for dn in ("TLT", "XLP", "TREND", "6040"):
        ds = DIVS[dn]
        out = {}
        for i, d in enumerate(idx):
            if i < 12:
                out[d] = 0.5 * A.iloc[i] + 0.5 * ds.iloc[i]; continue
            va = A.iloc[i-12:i].std() or 1e-6
            vd = ds.iloc[i-12:i].std() or 1e-6
            wa = (1/va) / (1/va + 1/vd)
            out[d] = wa * A.iloc[i] + (1-wa) * ds.iloc[i]
        record(f"riskparity A/{dn}", pd.Series(out).sort_index())

    # 3. vol-target overlay on A and on the best-looking static blends
    for tv in (0.06, 0.08, 0.10, 0.12):
        record(f"voltgt {int(tv*100)}% on A_only", voltarget(A, tv),
               {"target_vol": tv})
        for dn in ("TLT", "TREND", "6040"):
            blend = 0.5 * A + 0.5 * DIVS[dn]
            record(f"voltgt {int(tv*100)}% on (0.5A+0.5{dn})",
                   voltarget(blend, tv), {"target_vol": tv})

    # 4. stacked: 0.5A+0.5TREND -> vol-target 8/10, plus 0.6A+0.4TLT -> vt
    for tv in (0.08, 0.10):
        record(f"STACK 0.5A+0.5TREND -> vt{int(tv*100)}",
               voltarget(0.5*A + 0.5*TREND, tv), {"target_vol": tv})
        record(f"STACK 0.6A+0.4TLT -> vt{int(tv*100)}",
               voltarget(0.6*A + 0.4*TLT, tv), {"target_vol": tv})
        record(f"STACK 0.5A+0.5_6040 -> vt{int(tv*100)}",
               voltarget(0.5*A + 0.5*S6040, tv), {"target_vol": tv})

    df = pd.DataFrame(results)
    df.to_csv(AUG / "sharpe_max_sweep.csv", index=False)
    print(f"\nSaved -> {AUG / 'sharpe_max_sweep.csv'}  ({time.time()-t0:.0f}s)")

    # Honest winner: WF mean Sharpe > 2 AND WF min Sharpe > 1
    honest = df[(df["wf_mean_sharpe"] > 2.0) & (df["wf_min_sharpe"] > 1.0)]
    print("\n=== Configs with WF mean Sharpe > 2.0 AND WF min Sharpe > 1.0 ===")
    if len(honest):
        cols = ["config", "full_sharpe", "wf_mean_sharpe", "wf_min_sharpe",
                "full_cagr", "full_mdd", "wf_mean_cagr"]
        print(honest.sort_values("wf_mean_sharpe", ascending=False)[cols].to_string(index=False))
    else:
        print("  NONE met the strict honest bar. Top 10 by WF mean Sharpe:")
        cols = ["config", "full_sharpe", "wf_mean_sharpe", "wf_min_sharpe",
                "full_cagr", "full_mdd"]
        print(df.sort_values("wf_mean_sharpe", ascending=False).head(10)[cols].to_string(index=False))

    best = df.sort_values("wf_mean_sharpe", ascending=False).iloc[0].to_dict()
    (AUG / "sharpe_max_winner.json").write_text(json.dumps(best, indent=2, default=str))
    print(f"\nTop by WF mean Sharpe: {best['config']}  "
          f"(full Sharpe {best['full_sharpe']:.2f}, WF mean {best['wf_mean_sharpe']:.2f}, "
          f"WF min {best['wf_min_sharpe']:.2f})")


if __name__ == "__main__":
    main()
