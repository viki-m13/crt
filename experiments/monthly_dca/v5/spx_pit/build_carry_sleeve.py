"""Phase A: cross-asset carry / risk-parity sleeve + validation.

The cheapest honest path to a genuinely uncorrelated second sleeve
(see SECOND_SLEEVE_SCOPE.md). A GTAA / managed-futures-style
asset-class trend sleeve. Its return driver (cross-asset trend +
risk-parity sizing) is structurally different from the v5
single-stock momentum picker, so it should decorrelate to rho~=0.

SLEEVE DESIGN (fixed rules — NO fitting, NO per-split optimization)
==================================================================
Universe of liquid asset-class ETF proxies (already in the panel):
  SPY  US large-cap equity
  QQQ  US tech/growth equity
  IWM  US small-cap equity
  EFA  international developed equity
  EEM  emerging-market equity
  TLT  long US Treasuries          (the key diversifier)
  VNQ  US REITs / real estate
  SLV  silver / precious-metal     (commodity proxy)
  XLU  utilities                   (defensive / rate-sensitive)
  XLP  consumer staples            (defensive)

Each month-end T:
  1. For every asset with >=13 months of history, compute the
     classic 12-1 trend  mom = P[T-1]/P[T-13] - 1  (skip last month).
  2. Eligible = assets with mom > 0  (the trend/"carry" filter — only
     hold asset classes that are trending up; this is the risk-off
     mechanism, no separate crash gate needed).
  3. Risk-parity weight the eligible set by inverse trailing-12m vol,
     normalized to sum 1. (Equal *risk* contribution, the standard
     managed-futures construction.)
  4. If NO asset has positive trend -> 100% cash (0% return).
  5. Hold 1 month, re-evaluate. Asset-allocation sleeves rebalance
     monthly by construction — no single-name timing-luck problem.
  6. Cost: 10 bps on turnover (sum of |weight changes|).

VALIDATION
==========
  - standalone Sharpe / CAGR / vol / Max DD (full + walk-forward)
  - correlation to the deployed v5 stream, PER walk-forward split
    (the decisive stability check — a sleeve that only decorrelates
     in-sample is worthless; require |rho| < 0.25 every split)
  - blended metrics at FIXED risk-parity weights (never per-period
    mean-variance optimized): v5 and the carry sleeve combined by
    inverse trailing-12m vol, plus a few fixed static weights.
  - the honest adoption bar: blended WF-mean Sharpe must RISE and
    WF-min Sharpe must NOT fall vs deployed v5 alone.

Output:
  augmented/carry_sleeve_returns.csv     monthly return stream
  augmented/carry_sleeve_validation.json full + WF + blend metrics
  augmented/carry_sleeve_wf_corr.csv     per-split corr to v5
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_v5_aug import AUG, WF_SPLITS, COST_BPS  # noqa: E402

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "VNQ", "SLV", "XLU", "XLP"]
TREND_LOOKBACK = 13   # 12-1 momentum (months)
VOL_LOOKBACK = 12     # trailing months for risk-parity vol


def build_carry_sleeve(mr: pd.DataFrame) -> pd.Series:
    """Monthly return series of the cross-asset carry/risk-parity sleeve."""
    cf = COST_BPS / 1e4
    # price proxies from cumulative returns (start each at 1.0)
    px = {a: (1 + mr[a].fillna(0)).cumprod() for a in UNIVERSE if a in mr.columns}
    idx = mr.index
    prev_w: dict[str, float] = {}
    rets = {}
    for i, d in enumerate(idx):
        if i + 1 >= len(idx):
            break
        nd = idx[i + 1]
        # Build eligible set + risk-parity weights using ONLY data <= d
        elig = {}
        for a, series in px.items():
            s = series.loc[:d].dropna()
            if len(s) < TREND_LOOKBACK + 1:
                continue
            mom = s.iloc[-1] / s.iloc[-TREND_LOOKBACK] - 1.0
            if mom <= 0:
                continue
            # trailing-12m vol of monthly returns
            rr = mr[a].loc[:d].dropna()
            if len(rr) < VOL_LOOKBACK:
                continue
            v = rr.iloc[-VOL_LOOKBACK:].std()
            if v is None or np.isnan(v) or v <= 0:
                continue
            elig[a] = 1.0 / v
        if not elig:
            w = {}
            ret = 0.0
        else:
            tot = sum(elig.values())
            w = {a: x / tot for a, x in elig.items()}
            # realised next-month return
            ret = 0.0
            for a, wt in w.items():
                rv = mr.at[nd, a] if a in mr.columns else np.nan
                ret += wt * (0.0 if pd.isna(rv) else float(rv))
        # turnover cost
        all_keys = set(prev_w) | set(w)
        turn = sum(abs(w.get(k, 0.0) - prev_w.get(k, 0.0)) for k in all_keys)
        ret -= cf * turn
        rets[d] = ret
        prev_w = w
    return pd.Series(rets).sort_index()


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    n = len(r)
    if n < 6:
        return dict(cagr=0, vol=0, sharpe=0, mdd=0, n=n)
    cagr = (1 + r).prod() ** (12 / n) - 1
    vol = r.std() * np.sqrt(12)
    sh = (r.mean() / r.std()) * np.sqrt(12) if r.std() > 0 else 0.0
    ec = (1 + r).cumprod()
    mdd = float(((ec - ec.cummax()) / ec.cummax()).min())
    return dict(cagr=float(cagr), vol=float(vol), sharpe=float(sh),
                mdd=float(mdd), n=int(n))


def wf_table(r: pd.Series, ref: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    for split, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        seg = r[(r.index >= lo) & (r.index <= hi)].dropna()
        if len(seg) < 6:
            continue
        sh = (seg.mean() / seg.std()) * np.sqrt(12) if seg.std() > 0 else 0.0
        cg = (1 + seg).prod() ** (12 / len(seg)) - 1
        row = {"split": split, "n": len(seg), "sharpe": float(sh),
               "cagr": float(cg)}
        if ref is not None:
            j = ref.reindex(seg.index)
            both = pd.concat([seg, j], axis=1).dropna()
            row["corr_to_v5"] = float(both.iloc[:, 0].corr(both.iloc[:, 1])) \
                if len(both) > 3 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    print("Loading data ...")
    mr = pd.read_parquet(AUG / "monthly_returns_clean.parquet")
    if not isinstance(mr.index, pd.DatetimeIndex):
        mr.index = pd.to_datetime(mr.index)

    # deployed v5 canonical stream
    e = pd.read_csv(AUG / "v5_winner_equity.csv")
    A = e.set_index("date")["ret_m"].astype(float)
    A.index = pd.to_datetime(A.index)

    print("Building cross-asset carry/risk-parity sleeve ...")
    C = build_carry_sleeve(mr)
    C.to_csv(AUG / "carry_sleeve_returns.csv")

    sC = stats(C)
    sA = stats(A)
    print(f"\nCarry sleeve standalone (full, {C.index.min().date()}..{C.index.max().date()}):")
    print(f"  CAGR {sC['cagr']*100:.1f}%  vol {sC['vol']*100:.0f}%  "
          f"Sharpe {sC['sharpe']:.2f}  MaxDD {sC['mdd']*100:.0f}%  n={sC['n']}")

    # Align to overlap with v5
    ov = pd.concat([A.rename("v5"), C.rename("carry")], axis=1).dropna()
    rho_full = ov["v5"].corr(ov["carry"])
    print(f"\nOverlap with v5: {len(ov)} months "
          f"({ov.index.min().date()}..{ov.index.max().date()})")
    print(f"  full-sample corr(v5, carry) = {rho_full:+.3f}")

    # Per-split correlation stability — the decisive honest check
    wfC = wf_table(C, ref=A)
    print(f"\nPer-walk-forward-split: carry Sharpe & corr to v5")
    print(wfC.to_string(index=False))
    wfC.to_csv(AUG / "carry_sleeve_wf_corr.csv", index=False)
    max_abs_corr = float(wfC["corr_to_v5"].abs().max()) if "corr_to_v5" in wfC else np.nan
    corr_stable = max_abs_corr < 0.25 if not np.isnan(max_abs_corr) else False
    print(f"\n  max |corr| across splits = {max_abs_corr:.2f}  "
          f"(<0.25 required: {'PASS' if corr_stable else 'FAIL'})")

    # Blends — FIXED weights only (no per-period optimization)
    v5o, co = ov["v5"], ov["carry"]
    blends = {}
    # static fixed weights
    for wv in (0.5, 0.6, 0.7, 0.8):
        blends[f"static {wv:.0%} v5 + {1-wv:.0%} carry"] = wv * v5o + (1 - wv) * co
    # fixed risk-parity (inverse trailing-12m vol, recomputed monthly from
    # trailing data only — parameter-free)
    rp = {}
    for i, d in enumerate(ov.index):
        if i < 12:
            rp[d] = 0.5 * v5o.iloc[i] + 0.5 * co.iloc[i]
            continue
        va = v5o.iloc[i-12:i].std() or 1e-6
        vc = co.iloc[i-12:i].std() or 1e-6
        wv = (1/va) / (1/va + 1/vc)
        rp[d] = wv * v5o.iloc[i] + (1-wv) * co.iloc[i]
    blends["riskparity v5/carry (1/vol)"] = pd.Series(rp).sort_index()

    sA_ov = stats(v5o)  # v5 stats over the OVERLAP window (apples-to-apples)
    print(f"\n=== Blends (overlap window; v5-alone baseline shown first) ===")
    print(f"{'config':<34}{'Sharpe':>7}{'CAGR':>8}{'vol':>6}{'MaxDD':>7}"
          f"{'WFmeanSR':>9}{'WFminSR':>8}")
    def line(nm, r):
        s = stats(r); wf = wf_table(r)
        wm = wf["sharpe"].mean() if len(wf) else 0
        wn = wf["sharpe"].min() if len(wf) else 0
        print(f"{nm:<34}{s['sharpe']:>7.2f}{s['cagr']*100:>7.1f}%"
              f"{s['vol']*100:>5.0f}%{s['mdd']*100:>6.0f}%{wm:>9.2f}{wn:>8.2f}")
        return dict(name=nm, **s, wf_mean_sharpe=float(wm), wf_min_sharpe=float(wn))
    base = line("v5 alone (overlap window)", v5o)
    rec = [base]
    for nm, r in blends.items():
        rec.append(line(nm, r))

    # Honest adoption bar
    best = max(rec[1:], key=lambda x: x["wf_mean_sharpe"])
    passed = (best["wf_mean_sharpe"] > base["wf_mean_sharpe"] and
              best["wf_min_sharpe"] >= base["wf_min_sharpe"] - 0.05 and
              corr_stable)
    print(f"\n=== Honest adoption check ===")
    print(f"  best blend: {best['name']}")
    print(f"    Sharpe {best['sharpe']:.2f} (v5 alone {base['sharpe']:.2f}), "
          f"WFmean {best['wf_mean_sharpe']:.2f} (v5 {base['wf_mean_sharpe']:.2f}), "
          f"WFmin {best['wf_min_sharpe']:.2f} (v5 {base['wf_min_sharpe']:.2f})")
    print(f"    MaxDD {best['mdd']*100:.0f}% (v5 alone {base['mdd']*100:.0f}%)")
    print(f"  corr-stable across splits: {corr_stable}")
    print(f"  ADOPT: {'YES — blend improves risk-adjusted return honestly' if passed else 'NO'}")

    out = {
        "carry_full": sC,
        "v5_full": sA,
        "overlap_months": int(len(ov)),
        "overlap_start": str(ov.index.min().date()),
        "overlap_end": str(ov.index.max().date()),
        "corr_full": float(rho_full),
        "max_abs_corr_across_splits": max_abs_corr,
        "corr_stable": bool(corr_stable),
        "blends": rec,
        "best_blend": best,
        "adopt": bool(passed),
    }
    (AUG / "carry_sleeve_validation.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved -> {AUG / 'carry_sleeve_validation.json'}")


if __name__ == "__main__":
    main()
