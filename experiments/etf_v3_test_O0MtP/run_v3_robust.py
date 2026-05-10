"""v3-robust: targeted hardening of the deployed v3 strategy.

Why:
    The deployed v3 (K=3 EW, tight gate, hold=6m) is a hammer tuned for the
    S&P 500 cross-section. On broad ETFs the same setup picked the highest-
    vol sector/EM/commodity ETFs and held them through regime turns
    (entered 2008-01 in 'normal', held SLV/EWD/EWY into the GFC and didn't
    flip to cash until 2009-01). MDD ballooned to -67% / -84% on broad /
    combined.

What we change (4 minimal, principled tweaks):

    A. Mid-hold regime monitor — recheck SPY regime every month. If the
       gate flips to *crash*, force cash next month even mid-hold. Brings
       the regime gate closer to a true "fast circuit breaker".

    B. Vol-adjusted score — replace ml_3plus6 with
            score = pred - λ * vol_1y_xs
       (λ=0.10, same as the production 'ml_filter_winsor' variant). This
       penalises picks that win on high beta rather than skill.

    C. Inverse-vol weighting within K — re-weights the K picks by
       1/vol_1y so the booster (TQQQ etc.) doesn't crowd out steadier
       names.

    D. K=5 — modest diversification bump; still concentrated enough that
       the cross-sectional signal matters.

Cost = 10 bps, tight gate unchanged, hold=6m unchanged.

Inputs (per universe):
    cache/feat_<universe>.parquet     — already built by run_v3.py
    cache/preds_<universe>.parquet    — already built by run_v3.py
    data/prices_<universe>.parquet    — already fetched by fetch_prices.py

Outputs (per universe):
    results/<universe>_robust_equity.csv
    results/<universe>_robust_picks.csv
    results/<universe>_robust_yearly.csv
    results/<universe>_robust_walkforward.csv
    results/<universe>_robust_summary.json

Plus a top-level results/comparison_robust.csv merging baseline + robust.

Run:
    python3 experiments/etf_v3_test_O0MtP/run_v3_robust.py {broad|levered|combined}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE = HERE / "cache"
RESULTS = HERE / "results"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from experiments.monthly_dca.v2.sp500_pit_extended_sweep import classify_regime_tight  # noqa: E402

from run_v3 import (  # noqa: E402
    WF_SPLITS, evaluate_run, get_spy_features_per_month, load_panel,
    yearly_table,
)


ROBUST_SPEC = {
    "scorer": "ml_3plus6_minus_0.10*vol_xs",
    "k": 5,
    "weighting": "invvol",
    "regime_gate": "tight",
    "hold_months": 6,
    "mid_hold_crash_check": True,
    "cost_bps": 10.0,
    "vol_lambda": 0.10,
}


def simulate_robust(
    panel_prices: pd.DataFrame,
    preds: pd.DataFrame,
    feat_big: pd.DataFrame,
    spy_features: pd.DataFrame,
    cost_bps: float = 10.0,
    k: int = 5,
    hold_months: int = 6,
    vol_lambda: float = 0.10,
    mid_hold_crash_check: bool = True,
) -> pd.DataFrame:
    """Compound the v3-robust variant.

    feat_big: cross-section panel from build_features (used to pull vol_1y).
    """
    monthly = panel_prices.resample("ME").last()
    mret = monthly.pct_change().clip(lower=-1.0, upper=2.0)
    mret_idx = mret.index

    excluded = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD"}

    # Merge vol_1y from feat_big into preds and compute robust score
    fb = feat_big.reset_index() if isinstance(feat_big.index, pd.MultiIndex) else feat_big.copy()
    needed = [c for c in ("asof", "ticker", "vol_1y") if c in fb.columns]
    fb = fb[needed].copy()
    fb["asof"] = pd.to_datetime(fb["asof"])
    p = preds.copy()
    p["asof"] = pd.to_datetime(p["asof"])
    p = p.merge(fb, on=["asof", "ticker"], how="left")

    # Per-month cross-sectional ranks for vol_1y -> vol_xs in [-1, +1]
    p["vol_xs"] = p.groupby("asof")["vol_1y"].transform(
        lambda x: (x.rank(pct=True) - 0.5) * 2
    )
    # Robust score
    p["score"] = p["pred"] - vol_lambda * p["vol_xs"].fillna(0.0)

    p = p[~p["ticker"].isin(excluded)].dropna(subset=["score"]).copy()
    by_asof = {pd.Timestamp(d): g.sort_values("score", ascending=False)
               for d, g in p.groupby("asof")}
    months = sorted(by_asof.keys())

    cf = cost_bps / 10000.0
    equity = 1.0
    cur_picks: list[str] = []
    cur_w = np.array([])
    cur_vol = np.array([])
    held_for = 0
    cash = False
    rows = []

    for i, m in enumerate(months):
        # Per-month regime classification
        s = spy_features.loc[m].to_dict() if m in spy_features.index else {}
        regime_now = classify_regime_tight(s)

        # Decide rebalance:
        #   - First month
        #   - 6m hold elapsed
        #   - We were in cash last period
        #   - Mid-hold crash (if enabled)
        do_reb = (i == 0) or (held_for >= hold_months) or cash
        crash_force = (mid_hold_crash_check and regime_now == "crash" and not cash)
        do_reb = do_reb or crash_force

        regime_label = regime_now if do_reb else "hold"

        if do_reb:
            if regime_now == "crash":
                cur_picks, cur_w, cur_vol, cash = [], np.array([]), np.array([]), True
                held_for = 0
            else:
                sub = by_asof.get(m)
                if sub is None or len(sub) < k:
                    cur_picks, cur_w, cur_vol, cash = [], np.array([]), np.array([]), True
                    held_for = 0
                else:
                    top = sub.head(k)
                    cur_picks = top["ticker"].tolist()
                    vols = top["vol_1y"].values.astype(float)
                    vols = np.where(np.isnan(vols) | (vols <= 0), 0.4, vols)
                    invv = 1.0 / vols
                    w = invv / invv.sum()
                    cur_w = w
                    cur_vol = vols
                    cash = False
                    held_for = 0

        # Apply next-month return
        pos = mret_idx.searchsorted(m)
        cands = [(j, abs((mret_idx[j] - m).days)) for j in (pos - 1, pos) if 0 <= j < len(mret_idx)]
        cands.sort(key=lambda x: x[1])
        if cash or not cur_picks or not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(mret_idx):
            ret_m = 0.0
        else:
            next_d = mret_idx[cands[0][0] + 1]
            picks_r = []
            for tk in cur_picks:
                if tk in mret.columns:
                    rv = mret.at[next_d, tk]
                    picks_r.append(-1.0 if pd.isna(rv) else float(rv))
                else:
                    picks_r.append(-1.0)
            ret_m = float((np.asarray(picks_r) * cur_w).sum())

        if not cash and cur_picks:
            equity *= (1 + ret_m) * (1 - cf if do_reb else 1.0)
        held_for += 1

        rows.append({
            "date": m,
            "equity": equity,
            "ret_m": ret_m,
            "regime": "cash" if cash else regime_label,
            "n_picks": len(cur_picks),
            "picks": ",".join(cur_picks),
            "weights": ",".join(f"{w:.3f}" for w in cur_w) if len(cur_w) else "",
            "rebalance": int(do_reb),
            "crash_forced": int(crash_force),
        })
    return pd.DataFrame(rows)


def run_universe(name: str, *, variant: str = "robust",
                 vol_lambda: float = 0.10,
                 mid_hold_crash_check: bool = True,
                 k: int = 5) -> dict:
    print(f"\n=== run_universe ({variant}): {name} ===")
    panel = load_panel(name)
    feat = pd.read_parquet(CACHE / f"feat_{name}.parquet")
    preds = pd.read_parquet(CACHE / f"preds_{name}.parquet")
    spy_feats = get_spy_features_per_month(feat)

    eq = simulate_robust(
        panel_prices=panel,
        preds=preds,
        feat_big=feat,
        spy_features=spy_feats,
        cost_bps=ROBUST_SPEC["cost_bps"],
        k=k,
        hold_months=ROBUST_SPEC["hold_months"],
        vol_lambda=vol_lambda,
        mid_hold_crash_check=mid_hold_crash_check,
    )
    eq.to_csv(RESULTS / f"{name}_{variant}_equity.csv", index=False)
    eq[eq["rebalance"] == 1][["date", "regime", "n_picks", "picks", "weights", "crash_forced"]].to_csv(
        RESULTS / f"{name}_{variant}_picks.csv", index=False
    )

    yr = yearly_table(eq, panel)
    yr.to_csv(RESULTS / f"{name}_{variant}_yearly.csv", index=False)

    summary = evaluate_run(eq, panel)
    summary["universe"] = name
    summary["n_tickers_universe"] = int(panel.shape[1])
    summary["spec"] = {
        **ROBUST_SPEC,
        "k": k, "vol_lambda": vol_lambda,
        "mid_hold_crash_check": mid_hold_crash_check,
    }
    summary["variant"] = variant

    with open(RESULTS / f"{name}_{variant}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    pd.DataFrame(summary["wf_table"]).to_csv(RESULTS / f"{name}_{variant}_walkforward.csv", index=False)

    print(f"  -> CAGR={summary['cagr_full']*100:.2f}%  SPY={summary['spy_cagr_full']*100:.2f}%  "
          f"edge={summary['edge_full_pp']:+.2f}pp  MDD={summary['max_dd']*100:.2f}%  "
          f"Sharpe={summary['sharpe']:.2f}  cash_m={summary['n_cash']}")
    if summary["wf_mean_cagr"] is not None:
        print(f"  WF mean CAGR={summary['wf_mean_cagr']*100:.2f}%  "
              f"+{summary['wf_mean_edge_pp']:+.2f}pp  "
              f"({summary['wf_n_beats_spy']}/{summary['wf_n_splits']} beat SPY)")
    return summary


_FIELDS = ("cagr_full", "spy_cagr_full", "edge_full_pp", "sharpe",
           "max_dd", "n_cash", "n_rebalances", "n_months",
           "first_month", "last_month",
           "wf_mean_cagr", "wf_mean_edge_pp", "wf_n_beats_spy", "wf_n_splits")


def main():
    args = sys.argv[1:]
    targets = args if args else ["broad", "levered", "combined"]
    rob_rows = []
    for u in targets:
        # robust v1 (kitchen sink): mid-hold + vol-score + invvol + K=5
        s1 = run_universe(u, variant="robust",
                          vol_lambda=0.10,
                          mid_hold_crash_check=True, k=5)
        rob_rows.append({"universe": u, "variant": "v3_robust_v1_kitchen", **{f: s1.get(f) for f in _FIELDS}})
        # robust v2 (clean): just invvol + K=5 (no vol-score, no mid-hold)
        s2 = run_universe(u, variant="robust_v2_invvol_k5",
                          vol_lambda=0.0, mid_hold_crash_check=False, k=5)
        rob_rows.append({"universe": u, "variant": "v3_robust_v2_invvol_k5", **{f: s2.get(f) for f in _FIELDS}})

    base_rows = []
    for u in targets:
        bp = RESULTS / f"{u}_summary.json"
        if bp.exists():
            with open(bp) as f:
                b = json.load(f)
            base_rows.append({"universe": u, "variant": "v3_baseline",
                              **{f: b.get(f) for f in _FIELDS}})
    cmp = pd.DataFrame(base_rows + rob_rows)
    cmp.to_csv(RESULTS / "comparison_robust.csv", index=False)
    print("\n=== comparison_robust.csv ===")
    print(cmp.to_string(index=False))


if __name__ == "__main__":
    main()
