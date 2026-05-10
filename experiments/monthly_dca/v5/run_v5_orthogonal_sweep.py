"""v5 orthogonal multi-strategy sweep.

Tests each strategy in isolation, computes per-strategy correlation, and tests
ensembles (rank-average, vol-weighted, regime-conditional union).
"""
from __future__ import annotations
import time, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "v4"))

from strategies_orthogonal import STRATS, load_panel, attach_ml
from simulator_v4 import (
    Variant, simulate_variant_v4, evaluate, build_spy_aligned,
    load_spy_features, _load_daily_prices, V2, PIT,
)


def main():
    t0 = time.time()
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_features = load_spy_features()
    daily_prices = _load_daily_prices()

    panel = load_panel()
    panel = attach_ml(panel)
    spy_aligned = build_spy_aligned(panel)

    rows = []

    # 1) Each individual strategy
    print("\n=== STAGE 1: Individual strategies (k=3 h=6 EW tight) ===", flush=True)
    indiv_panels = {}
    for name, fn in STRATS.items():
        p = panel.copy()
        p["score"] = fn(p)
        indiv_panels[name] = p
        for k in (3, 5):
            for h in (6, 12):
                v = Variant(name=f"{name}|k{k}_h{h}", scorer=name,
                            k_normal=k, k_recovery=k, k_bull=k, weighting="ew",
                            regime_gate="tight", hold_months=h, cap_per_pick=1.0)
                eq = simulate_variant_v4(p, monthly_returns, spy_features, v, daily_prices=daily_prices)
                m = evaluate(eq, spy_aligned, v.name)
                m["strat"] = name; m["k"] = k; m["h"] = h
                rows.append(m)
                print(f"  {v.name:50s}  CAGR={m['cagr_full']*100:6.2f}%  WF_mean={m['wf_mean_cagr']*100:6.2f}%  WF_min={m['wf_min_cagr']*100:6.2f}%  beats={m['wf_n_beats']}/{m['wf_n_splits']}  Sh={m['sharpe']:.2f}  MDD={m['max_dd']*100:.1f}%",
                      flush=True)

    # 2) Ensembles: blend strategy scores rank-wise
    print("\n=== STAGE 2: Rank-blended ensembles (k=3 h=6) ===", flush=True)
    # Compute per-asof rank for each strategy
    pretty = panel[["asof", "ticker"]].copy()
    for name, fn in STRATS.items():
        score = fn(panel)
        pretty[f"r_{name}"] = score.groupby(panel["asof"]).rank(pct=True)
    pretty = pretty.merge(panel[["asof", "ticker", "vol_1y"]], on=["asof","ticker"], how="left")

    # Try various ensemble combinations
    ens_configs = [
        ("E_all_eq", list(STRATS.keys()), None),
        ("E_S1_S3_S6_eq", ["S1_ml_3plus6", "S3_quality_pullback", "S6_multibagger_lottery"], None),
        ("E_S1_S3_S5_S6_eq", ["S1_ml_3plus6", "S3_quality_pullback", "S5_low_vol_quality", "S6_multibagger_lottery"], None),
        ("E_ML_heavy", ["S1_ml_3plus6", "S3_quality_pullback", "S6_multibagger_lottery"], [0.6, 0.2, 0.2]),
        ("E_ML_S3_balanced", ["S1_ml_3plus6", "S3_quality_pullback"], [0.5, 0.5]),
        ("E_ML_S6_balanced", ["S1_ml_3plus6", "S6_multibagger_lottery"], [0.5, 0.5]),
    ]
    for ens_name, strats, weights in ens_configs:
        if weights is None:
            weights = [1.0/len(strats)] * len(strats)
        w = np.array(weights)
        w = w / w.sum()
        score_arr = np.zeros(len(pretty))
        for s, ww in zip(strats, w):
            score_arr += ww * pretty[f"r_{s}"].values
        p = panel.copy()
        p["score"] = score_arr
        for k in (3, 5):
            for h in (6, 12):
                v = Variant(name=f"{ens_name}|k{k}_h{h}", scorer=ens_name,
                            k_normal=k, k_recovery=k, k_bull=k, weighting="ew",
                            regime_gate="tight", hold_months=h, cap_per_pick=1.0)
                eq = simulate_variant_v4(p, monthly_returns, spy_features, v, daily_prices=daily_prices)
                m = evaluate(eq, spy_aligned, v.name)
                m["strat"] = ens_name; m["k"] = k; m["h"] = h
                rows.append(m)
                print(f"  {v.name:50s}  CAGR={m['cagr_full']*100:6.2f}%  WF_mean={m['wf_mean_cagr']*100:6.2f}%  WF_min={m['wf_min_cagr']*100:6.2f}%  beats={m['wf_n_beats']}/{m['wf_n_splits']}  Sh={m['sharpe']:.2f}  MDD={m['max_dd']*100:.1f}%",
                      flush=True)

    # 3) Union strategies: take top-K from each, deduplicate, weight by vote count
    print("\n=== STAGE 3: Union/vote strategies (k=3 picks each, union, EW) ===", flush=True)
    # For each asof, pick top-K from each strategy, union them, weight by # of strategies voting
    # Then use the resulting per-pick weight as score (so simulator picks top-K of those)
    for k_per_strat in (3, 5):
        for strat_set, label in [
            (["S1_ml_3plus6", "S3_quality_pullback", "S6_multibagger_lottery"], "U_S1_S3_S6"),
            (list(STRATS.keys()), "U_all7"),
            (["S1_ml_3plus6", "S2_pure_momentum", "S3_quality_pullback", "S6_multibagger_lottery"], "U_S1_S2_S3_S6"),
        ]:
            # Compute votes
            votes = pd.Series(0, index=panel.index)
            for s in strat_set:
                # rank per asof
                rk = pretty[f"r_{s}"]
                # top-K per asof: rank > (1 - K / N)
                # easier: groupby asof then nlargest
                top_mask = pd.Series(False, index=panel.index)
                for asof, idx in panel.groupby("asof").indices.items():
                    sub = rk.iloc[idx]
                    top_idx = sub.nlargest(k_per_strat).index
                    top_mask.loc[top_idx] = True
                votes = votes + top_mask.astype(int)
            p = panel.copy()
            p["score"] = votes.values
            # In sim, top-K=3 picks the 3 most-voted stocks (tie-break by ML rank)
            p["score"] = p["score"].astype(float) + 0.001 * pretty["r_S1_ml_3plus6"].values  # tie-break
            for k_pick in (3, 5):
                for h in (6, 12):
                    v = Variant(name=f"{label}_k{k_per_strat}|pick{k_pick}_h{h}",
                                scorer=label, k_normal=k_pick, k_recovery=k_pick, k_bull=k_pick,
                                weighting="ew", regime_gate="tight", hold_months=h, cap_per_pick=1.0)
                    eq = simulate_variant_v4(p, monthly_returns, spy_features, v, daily_prices=daily_prices)
                    m = evaluate(eq, spy_aligned, v.name)
                    m["strat"] = label; m["k_per_strat"] = k_per_strat
                    m["k"] = k_pick; m["h"] = h
                    rows.append(m)
                    print(f"  {v.name:60s}  CAGR={m['cagr_full']*100:6.2f}%  WF_mean={m['wf_mean_cagr']*100:6.2f}%  WF_min={m['wf_min_cagr']*100:6.2f}%  beats={m['wf_n_beats']}/{m['wf_n_splits']}  Sh={m['sharpe']:.2f}",
                          flush=True)

    df = pd.DataFrame(rows).sort_values("wf_mean_cagr", ascending=False)
    df.to_csv(PIT / "v5_orthogonal_sweep_results.csv", index=False)
    print(f"\n{(time.time()-t0)/60:.1f} min total")
    print(f"Saved {len(df)} rows.")
    print("\n=== TOP 20 by WF mean CAGR ===")
    cols = ["name", "cagr_full", "wf_mean_cagr", "wf_min_cagr", "wf_n_beats", "wf_n_pos",
            "sharpe", "max_dd", "n_cash"]
    print(df[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
