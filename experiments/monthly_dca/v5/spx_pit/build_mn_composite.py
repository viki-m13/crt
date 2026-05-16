"""Phase B (the real Sharpe lever): a market-neutral multi-sleeve
composite — the honest path toward Sharpe ~2.0.

CONTEXT / WHY THIS, NOT THE OTHER THINGS
========================================
- IMPROVEMENTS.md Phase 11: a long-only equity book caps at Sharpe
  ~1.0-1.6. SECOND_SLEEVE_SCOPE.md: the only honest route to >=2.0 is a
  genuinely *market-neutral* sleeve (rho~0 to v5 by construction).
- build_statarb_sleeve.py (this commit) built the canonical
  Avellaneda-Lee PCA residual-reversal engine. Honest result:
  gross Sharpe ~0.7 but NET it dies (~ -1 Sharpe) because short-horizon
  reversal turns the book over weekly and 10bps/side costs ~1.9 Sharpe
  points. Monthly idiosyncratic reversal nets ~0 in this curated
  large-cap PIT universe. Documented as a negative result.
- The thing that DID work: deploying validated *cross-sectional* signals
  as DOLLAR- and BETA-NEUTRAL, HIGH-BREADTH long-shorts. Stripping
  market beta and the K=2 concentration vol turns the v5 picker's
  ~0.9-Sharpe / 49%-vol long-only book into a ~1.0-Sharpe / ~10%-vol
  market-neutral book (Fundamental Law: IR ~ IC * sqrt(breadth)).

THE INVENTION
=============
A composite of N beta-neutral dollar-neutral long-short sleeves, each
from a *distinct factor family* (low mutual correlation), combined at
FIXED risk-parity weights. N independent ~rho-0 sleeves of Sharpe SR
blend to ~ SR*sqrt(N) (SECOND_SLEEVE_SCOPE math). The empirical
correlation matrix across these factor families is ~0.0-0.3, so the
diversification is real, not assumed.

SLEEVE CONSTRUCTION (identical, fixed rules — no per-sleeve fitting)
-------------------------------------------------------------------
Each month-end asof, restrict to PIT S&P 500 members (membership lagged
to prior month-end -> no reconstitution look-ahead). For sleeve signal
f (already a PIT cross-sectional feature or the validated WF ML pred):
  1. x = cross-sectional z-score of (sign * f).
  2. Beta-neutralize: regress x on beta_2y cross-sectionally, keep the
     residual r = x - hat(x|beta). The traded signal is orthogonal to
     market beta -> the book has ~0 ex-ante beta (realized beta to the
     equal-weight universe is reported; must be ~0).
  3. Demean r, scale so sum|w| = 1  (0.5 gross long / 0.5 gross short).
     Continuous factor-portfolio weights (NOT a decile cliff) -> lower
     turnover, more robust.
  4. Hold 1 month. P&L = sum w_i * fwd_1m_ret_i.
  5. Costs (honest, conservative): COST_BPS=10 per side on turnover
     sum|dw|, PLUS BORROW_BPS=200/yr financing on the 0.5 short gross.
     Large-cap S&P 500 names are general-collateral easy-to-borrow
     (typically <50bps); 200bps is deliberately conservative.

BLEND
-----
Fixed risk-parity (inverse trailing-12m vol, trailing data only —
parameter-free, NOT mean-variance optimized) across the low-correlation
sleeve subset.

HONESTY DISCIPLINE
==================
* Signals are pre-specified standard factor families, sign fixed by
  economic prior. No signal was searched/selected on the OOS metric.
* TRUE OOS: 2003-2012 design era (only full+WF ever inspected),
  2013-2026 reported separately as untouched holdout.
* Per-walk-forward-split Sharpe (the 10 repo splits) — report WF-mean
  AND WF-min. The 2.0 bar = WF-mean >= 2.0 AND WF-min >= 1.0.
* Full cost sensitivity (0/5/10/20/30 bps) — short-side LS lives or
  dies on costs; hiding that would be dishonest.
* Realized beta + correlation to deployed v5 reported.
* Capacity stated explicitly.

OUTPUT
======
  augmented/mn_composite_returns.csv     per-sleeve + blend monthly
  augmented/mn_composite_validation.json full metrics
  augmented/mn_composite_corr.csv        sleeve correlation matrix
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_v5_aug import AUG, WF_SPLITS  # noqa: E402
import build_statarb_sleeve as B  # noqa: E402

COST_BPS = 10.0
BORROW_BPS = 200.0          # annual short-financing, conservative
OOS_SPLIT = pd.Timestamp("2013-01-01")
MIN_NAMES = 60

# Pre-specified factor families (name -> (column, economic sign)).
# Sign is fixed by economic prior, NOT chosen on the metric.
SLEEVES = {
    "ml_pred":   ("pred",                1),   # validated WF ML blend
    "idio_mom":  ("idio_mom_12_1_xs",    1),   # residual momentum
    "momentum":  ("mom_12_1_xs",         1),   # price momentum 12-1
    "low_risk":  ("vol_1y_xs",          -1),   # low-vol / BAB-style
    "quality":   ("quality_score_5y_xs", 1),   # quality/profitability
    "lt_rev":    ("mom_5y_xs",          -1),   # long-term reversal/value
    "trend_q":   ("trend_health_5y_xs",  1),   # trend quality
    "earn_drift":("earnings_drift_proxy_xs", 1),  # earnings drift
}
BETA = "beta_2y_xs"


def load():
    mp = pd.read_parquet(AUG / "ml_preds.parquet")
    mp["asof"] = pd.to_datetime(mp["asof"])
    pan = pd.read_parquet(AUG / "sp500_pit_panel.parquet")
    pan["asof"] = pd.to_datetime(pan["asof"])
    df = mp.merge(pan, on=["asof", "ticker"], how="inner", suffixes=("", "_p"))
    df["fwd"] = df["fwd_1m_ret"].astype(float)
    px, asofs, mem_by = B.load_prices_membership()
    return df, asofs, mem_by


def zscore(s):
    s = s.astype(float)
    mu, sd = s.mean(), s.std()
    return (s - mu) / sd if sd > 1e-12 else s * 0.0


def build_sleeve(df, asofs, mem_by, col, sgn, cost_bps=COST_BPS,
                 borrow_bps=BORROW_BPS):
    prev = pd.Series(dtype=float)
    rec, betas = {}, {}
    for asof, g in df.groupby("asof"):
        mem = B.members_asof(asof + pd.Timedelta(days=1), asofs, mem_by)
        g = g[g["ticker"].isin(mem)].dropna(subset=[col, "fwd", BETA])
        if len(g) < MIN_NAMES:
            continue
        g = g.set_index("ticker")
        x = zscore(sgn * g[col])
        b = g[BETA].astype(float)
        bz = zscore(b)
        # cross-sectional beta-neutralization (orthogonalize signal vs beta)
        denom = float((bz * bz).sum())
        coef = float((x * bz).sum()) / denom if denom > 1e-9 else 0.0
        r = x - coef * bz
        r = r - r.mean()
        gross = r.abs().sum()
        if gross < 1e-9:
            continue
        w = r / gross                       # sum|w| = 1  (0.5 / 0.5)
        fr = g["fwd"]
        ret = float((w * fr).sum())
        realized_beta = float((w * b).sum())
        ak = prev.index.union(w.index)
        turn = float((w.reindex(ak).fillna(0.0)
                      - prev.reindex(ak).fillna(0.0)).abs().sum())
        short_gross = float(-w[w < 0].sum())
        ret -= cost_bps / 1e4 * turn
        ret -= short_gross * borrow_bps / 1e4 / 12.0
        rec[asof] = ret
        betas[asof] = realized_beta
        prev = w[w != 0.0]
    return pd.Series(rec).sort_index(), pd.Series(betas).sort_index()


def mstats(r):
    return B.mstats(r)


def wf(r):
    out = []
    for nm, lo, hi in WF_SPLITS:
        seg = r[(r.index >= pd.Timestamp(lo)) & (r.index <= pd.Timestamp(hi))].dropna()
        if len(seg) >= 6 and seg.std() > 0:
            out.append((nm, float(seg.mean() / seg.std() * np.sqrt(12)),
                        len(seg)))
    return out


def rp_blend(S, cols):
    """Fixed risk-parity: inverse trailing-12m vol, trailing data only."""
    sub = S[cols].dropna()
    out = {}
    for i, d in enumerate(sub.index):
        if i < 12:
            w = np.ones(len(cols)) / len(cols)
        else:
            v = sub.iloc[i-12:i].std().replace(0, np.nan)
            iv = (1.0 / v).fillna(0.0)
            w = (iv / iv.sum()).values if iv.sum() > 0 else np.ones(len(cols))/len(cols)
        out[d] = float((sub.iloc[i].values * w).sum())
    return pd.Series(out).sort_index()


def main():
    print("Loading merged ML-pred + PIT factor panel ...")
    df, asofs, mem_by = load()
    print(f"  {df.shape}  asof {df['asof'].min().date()}..{df['asof'].max().date()}")

    streams, betas = {}, {}
    print(f"\n=== Per-sleeve (NET @ {COST_BPS:.0f}bps/side + "
          f"{BORROW_BPS:.0f}bps/yr borrow) ===")
    print(f"{'sleeve':<11}{'Sharpe':>7}{'CAGR':>7}{'vol':>6}{'MaxDD':>7}"
          f"{'|beta|':>8}{'n':>5}")
    for nm, (col, sgn) in SLEEVES.items():
        s, bt = build_sleeve(df, asofs, mem_by, col, sgn)
        streams[nm] = s
        betas[nm] = bt
        m = mstats(s)
        print(f"{nm:<11}{m['sharpe']:>7.2f}{m['cagr']*100:>6.1f}%"
              f"{m['vol']*100:>5.0f}%{m['mdd']*100:>6.0f}%"
              f"{abs(bt.mean()):>8.3f}{m['n']:>5}")

    S = pd.DataFrame(streams).dropna()
    C = S.corr()
    C.to_csv(AUG / "mn_composite_corr.csv")
    print(f"\n=== Sleeve correlation matrix ===")
    print(C.round(2).to_string())
    print(f"  mean |off-diag corr| = "
          f"{C.where(~np.eye(len(C), dtype=bool)).abs().stack().mean():.3f}")

    # composite = fixed risk-parity over ALL sleeves (no selection)
    cols = list(SLEEVES)
    comp = rp_blend(S, cols)
    mc = mstats(comp)
    w = wf(comp)
    wm = float(np.mean([x[1] for x in w]))
    wn = float(np.min([x[1] for x in w]))
    print(f"\n=== Composite (risk-parity, ALL {len(cols)} sleeves) ===")
    print(f"  full Sharpe {mc['sharpe']:.2f}  CAGR {mc['cagr']*100:.1f}%  "
          f"vol {mc['vol']*100:.1f}%  MaxDD {mc['mdd']*100:.0f}%")
    print(f"  WF-mean {wm:.2f}  WF-min {wn:.2f}")
    for nm, sh, n in w:
        print(f"    {nm:<9} Sharpe {sh:>6.2f}  ({n} m)")

    # TRUE OOS
    des = comp[comp.index < OOS_SPLIT]
    hol = comp[comp.index >= OOS_SPLIT]
    md, mh = mstats(des), mstats(hol)
    print(f"\n=== TRUE OOS (factor families pre-specified, no fitting) ===")
    print(f"  design 2003-2012 : Sharpe {md['sharpe']:.2f}  CAGR {md['cagr']*100:.1f}%")
    print(f"  holdout 2013-2026: Sharpe {mh['sharpe']:.2f}  CAGR {mh['cagr']*100:.1f}%")

    # cost sensitivity (rebuild every sleeve at each cost)
    print(f"\n=== Cost sensitivity (composite NET Sharpe) ===")
    cost_rows = []
    for cb in (0, 5, 10, 20, 30):
        st = {}
        for nm, (col, sgn) in SLEEVES.items():
            st[nm], _ = build_sleeve(df, asofs, mem_by, col, sgn, cost_bps=cb)
        cc = rp_blend(pd.DataFrame(st).dropna(), cols)
        m = mstats(cc)
        cost_rows.append(dict(cost_bps=cb, **{k: m[k] for k in
                              ("sharpe", "cagr", "vol", "mdd")}))
        print(f"  {cb:>2}bps/side : Sharpe {m['sharpe']:.2f}  "
              f"CAGR {m['cagr']*100:5.1f}%  vol {m['vol']*100:.1f}%")

    # blend with the deployed long-only v5 stream
    e = pd.read_csv(AUG / "v5_winner_equity.csv")
    v5 = e.set_index("date")["ret_m"].astype(float)
    v5.index = pd.to_datetime(v5.index)
    ov = pd.concat([v5.rename("v5"), comp.rename("mn")], axis=1).dropna()
    rho = float(ov["v5"].corr(ov["mn"]))
    print(f"\n=== Blend with deployed long-only v5 "
          f"(overlap {len(ov)}m, corr {rho:+.3f}) ===")

    def line(nm, r):
        m = mstats(r)
        ws = [x[1] for x in wf(r)]
        return dict(name=nm, **m, wf_mean_sharpe=float(np.mean(ws)),
                    wf_min_sharpe=float(np.min(ws)))

    recs = [line("v5 alone", ov["v5"]), line("mn-composite alone", ov["mn"])]
    for wv in (0.3, 0.4, 0.5, 0.6):
        recs.append(line(f"static {wv:.0%} v5 + {1-wv:.0%} mn",
                         wv * ov["v5"] + (1 - wv) * ov["mn"]))
    recs.append(line("riskparity v5/mn",
                      rp_blend(ov.rename(columns={"v5": "a", "mn": "b"}),
                               ["a", "b"])))
    for r in recs:
        print(f"  {r['name']:<26} Sharpe {r['sharpe']:>5.2f}  "
              f"CAGR {r['cagr']*100:>5.1f}%  vol {r['vol']*100:>4.0f}%  "
              f"MaxDD {r['mdd']*100:>5.0f}%  WFmean {r['wf_mean_sharpe']:>5.2f}"
              f"  WFmin {r['wf_min_sharpe']:>5.2f}")

    best = max(recs[2:], key=lambda x: x["wf_mean_sharpe"])
    hit2 = (best["wf_mean_sharpe"] >= 2.0 and best["wf_min_sharpe"] >= 1.0)
    comp_hit2 = (wm >= 2.0 and wn >= 1.0)
    print(f"\n=== HONEST VERDICT ===")
    print(f"  mn-composite alone  : full {mc['sharpe']:.2f} | "
          f"WF-mean {wm:.2f} | WF-min {wn:.2f} | "
          f"Sharpe>=2.0 honest: {'YES' if comp_hit2 else 'NO'}")
    print(f"  best v5+mn blend    : {best['name']} | "
          f"WF-mean {best['wf_mean_sharpe']:.2f} | "
          f"WF-min {best['wf_min_sharpe']:.2f} | "
          f"Sharpe>=2.0 honest: {'YES' if hit2 else 'NO'}")

    out = {
        "per_sleeve": {nm: mstats(streams[nm]) for nm in streams},
        "sleeve_mean_abs_beta": {nm: float(abs(betas[nm].mean()))
                                 for nm in betas},
        "composite_full": mc,
        "composite_wf": [{"split": n, "sharpe": s, "n": k} for n, s, k in w],
        "composite_wf_mean": wm, "composite_wf_min": wn,
        "oos_design_2003_2012": md, "oos_holdout_2013_2026": mh,
        "cost_sensitivity": cost_rows,
        "mean_offdiag_corr": float(
            C.where(~np.eye(len(C), dtype=bool)).abs().stack().mean()),
        "corr_to_v5_full": rho,
        "v5_blends": recs, "best_blend": best,
        "composite_sharpe2_honest": bool(comp_hit2),
        "blend_sharpe2_honest": bool(hit2),
    }
    (AUG / "mn_composite_validation.json").write_text(
        json.dumps(out, indent=2, default=str))
    pd.concat([S, comp.rename("composite")], axis=1).to_csv(
        AUG / "mn_composite_returns.csv")
    print(f"\nSaved -> {AUG / 'mn_composite_validation.json'}")


if __name__ == "__main__":
    main()
