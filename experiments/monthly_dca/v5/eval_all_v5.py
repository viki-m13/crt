"""Evaluate all v5/v6 signals — individually and in ensembles — on PIT S&P 500.

Loads:
  - v3 baseline (ml_3plus6 from ml_preds_v2.parquet)
  - v6 (proprietary features) from ml_preds_v6.parquet
  - vertical classifier from ml_preds_vertical.parquet
  - pattern similarity from ml_preds_pattern_sim.parquet
  - chronos from ml_preds_chronos.parquet
  - cnn from ml_preds_cnn.parquet

Tests each individually, blended, and as ensembles.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "v4"))
from simulator_v4 import (Variant, simulate_variant_v4, evaluate, build_spy_aligned,
                          load_spy_features, _load_daily_prices, V2, PIT)


def main():
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy = load_spy_features()
    daily_prices = _load_daily_prices()

    panel = pd.read_parquet(PIT / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])

    # Attach all signal sources
    sources = {}
    ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred_3m", "pred_6m"]]
    ml["asof"] = pd.to_datetime(ml["asof"])
    panel = panel.merge(ml, on=["asof", "ticker"], how="left")
    panel["v3_score"] = (panel["pred_3m"] + panel["pred_6m"]) / 2
    sources["v3"] = "v3_score"

    if (PIT / "ml_preds_v6.parquet").exists():
        v6 = pd.read_parquet(PIT / "ml_preds_v6.parquet")
        v6["asof"] = pd.to_datetime(v6["asof"])
        panel = panel.merge(v6[["asof", "ticker", "pred_v6", "pred_v6_3m", "pred_v6_6m"]],
                            on=["asof", "ticker"], how="left")
        sources["v6"] = "pred_v6"
        sources["v6_6m"] = "pred_v6_6m"

    if (PIT / "ml_preds_vertical.parquet").exists():
        vert = pd.read_parquet(PIT / "ml_preds_vertical.parquet")
        vert["asof"] = pd.to_datetime(vert["asof"])
        panel = panel.merge(vert, on=["asof", "ticker"], how="left")
        sources["vertical"] = "p_vertical"

    if (PIT / "ml_preds_pattern_sim.parquet").exists():
        ps = pd.read_parquet(PIT / "ml_preds_pattern_sim.parquet")
        ps["asof"] = pd.to_datetime(ps["asof"])
        panel = panel.merge(ps, on=["asof", "ticker"], how="left")
        sources["pattern"] = "pattern_sim"

    if (PIT / "ml_preds_chronos.parquet").exists():
        ch = pd.read_parquet(PIT / "ml_preds_chronos.parquet")
        ch["asof"] = pd.to_datetime(ch["asof"])
        panel = panel.merge(ch[["asof", "ticker", "chronos_p50_3m", "chronos_p70_3m", "chronos_p90_3m"]],
                            on=["asof", "ticker"], how="left")
        sources["chronos_p50"] = "chronos_p50_3m"
        sources["chronos_p70"] = "chronos_p70_3m"

    if (PIT / "ml_preds_cnn.parquet").exists():
        cn = pd.read_parquet(PIT / "ml_preds_cnn.parquet")
        cn["asof"] = pd.to_datetime(cn["asof"])
        panel = panel.merge(cn, on=["asof", "ticker"], how="left")
        sources["cnn"] = "p_cnn"

    spy_aligned = build_spy_aligned(panel)

    # Compute cross-sectional ranks for each
    for label, col in sources.items():
        if col in panel.columns:
            panel[f"rk_{label}"] = panel.groupby("asof")[col].rank(pct=True)

    # Test individual
    rows = []
    print(f"\n=== Individual signals (k=3 h=6 EW tight) ===", flush=True)
    for label, col in sources.items():
        if col not in panel.columns: continue
        p = panel.copy()
        p["score"] = p[col]
        v = Variant(name=f"{label}_only", scorer=label,
                    k_normal=3, k_recovery=3, k_bull=3, weighting="ew",
                    regime_gate="tight", hold_months=6, cap_per_pick=1.0)
        eq = simulate_variant_v4(p.dropna(subset=["score"]).copy(), monthly_returns, spy, v, daily_prices=daily_prices)
        m = evaluate(eq, spy_aligned, v.name)
        m["signal"] = label
        rows.append(m)
        print(f"  {v.name:25s}  CAGR={m['cagr_full']*100:6.2f}%  WF_mean={m['wf_mean_cagr']*100:6.2f}%  "
              f"WF_min={m['wf_min_cagr']*100:6.2f}%  beats={m['wf_n_beats']}/{m['wf_n_splits']}  "
              f"Sh={m['sharpe']:.2f}", flush=True)

    # Test pairwise blends with v3
    print(f"\n=== Blends with v3 (rank-blended) ===", flush=True)
    for label in sources:
        if label == "v3": continue
        if f"rk_{label}" not in panel.columns: continue
        for w_v3 in (0.5, 0.7, 0.85):
            p = panel.copy()
            p["score"] = w_v3 * p["rk_v3"] + (1-w_v3) * p[f"rk_{label}"]
            v = Variant(name=f"v3_{label}_w{w_v3}", scorer="blend",
                        k_normal=3, k_recovery=3, k_bull=3, weighting="ew",
                        regime_gate="tight", hold_months=6, cap_per_pick=1.0)
            eq = simulate_variant_v4(p.dropna(subset=["score"]).copy(),
                                     monthly_returns, spy, v, daily_prices=daily_prices)
            m = evaluate(eq, spy_aligned, v.name)
            m["signal"] = f"v3+{label}@{w_v3}"
            rows.append(m)
            print(f"  {v.name:30s}  CAGR={m['cagr_full']*100:6.2f}%  WF_mean={m['wf_mean_cagr']*100:6.2f}%  "
                  f"WF_min={m['wf_min_cagr']*100:6.2f}%  beats={m['wf_n_beats']}/{m['wf_n_splits']}  "
                  f"Sh={m['sharpe']:.2f}", flush=True)

    # Test multiplicative ensemble: rank product
    print(f"\n=== Multi-signal ensembles ===", flush=True)
    for combo in [
        ["v3", "v6"],
        ["v3", "v6", "chronos_p50"],
        ["v3", "v6", "vertical"],
        ["v3", "chronos_p50"],
        ["v3", "chronos_p70"],
    ]:
        if not all(f"rk_{s}" in panel.columns for s in combo): continue
        p = panel.copy()
        score = np.zeros(len(p))
        for s in combo:
            score = score + p[f"rk_{s}"].fillna(0.5).values
        p["score"] = score / len(combo)
        v = Variant(name=f"ensemble_{'_'.join(combo)}", scorer="ensemble",
                    k_normal=3, k_recovery=3, k_bull=3, weighting="ew",
                    regime_gate="tight", hold_months=6, cap_per_pick=1.0)
        eq = simulate_variant_v4(p.dropna(subset=["score"]).copy(), monthly_returns, spy, v, daily_prices=daily_prices)
        m = evaluate(eq, spy_aligned, v.name)
        m["signal"] = '+'.join(combo)
        rows.append(m)
        print(f"  {v.name:35s}  CAGR={m['cagr_full']*100:6.2f}%  WF_mean={m['wf_mean_cagr']*100:6.2f}%  "
              f"WF_min={m['wf_min_cagr']*100:6.2f}%  beats={m['wf_n_beats']}/{m['wf_n_splits']}  "
              f"Sh={m['sharpe']:.2f}", flush=True)

    df = pd.DataFrame(rows).sort_values("wf_mean_cagr", ascending=False)
    df.to_csv(PIT / "v5_eval_all_results.csv", index=False)
    print("\n=== TOP 15 ===", flush=True)
    print(df[["name", "cagr_full", "wf_mean_cagr", "wf_min_cagr", "wf_n_beats", "sharpe", "max_dd"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
