"""Model-ensemble picker: average rank percentiles across 5 different ML
scorers we already have in cache. No retraining, no curve-fitting (uniform
weights), no new hyperparameters.

Scorers ensembled:
  - ml_v2.pred_3m, ml_v2.pred_6m  (GBM v2, current production)
  - ml_v6.pred_v6_3m, ml_v6.pred_v6_6m  (GBM v6, different arch / features)
  - ml_pattern.pattern_sim
  - ml_ttm.ttm_peak
  - ml_vertical.p_vertical

At each rebalance: each scorer ranks every PIT-S&P-500 stock by score (pct
rank). Average the 8 rank-percentile values per stock. Apply the same
Chronos p70 q=0.45 filter (universe-agnostic). Pick top-K by ensemble rank.

Hypothesis: model-level diversification reduces single-model bias →
narrower year-to-year edge distribution while preserving expected return.
Tested against the look-ahead-fixed harness with full WF rigor.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

from experiments.monthly_dca.v5.validations.harness import (
    load_all, evaluate, invvol_weights, CHRONOS_FILTER_Q, CAP_PER_PICK, K_PICKS,
)

RES = Path(__file__).resolve().parent / "results"


def make_model_ensemble_picker(use_models: list[str], k: int = K_PICKS,
                                 chronos_q: float = CHRONOS_FILTER_Q,
                                 cap: float = CAP_PER_PICK):
    """Factory returning a pick_fn that averages rank percentiles across
    the specified models, then picks top-k from the Chronos-passed cohort.
    """
    def pick(asof, eligible, data, regime):
        # Collect all 5 model outputs at this asof
        score_dfs = {}
        if "v2_3m" in use_models or "v2_6m" in use_models:
            v2 = data.ml_v2[data.ml_v2["asof"] == asof].copy()
            if "v2_3m" in use_models: score_dfs["v2_3m"] = v2[["ticker", "pred_3m"]].rename(columns={"pred_3m": "score"})
            if "v2_6m" in use_models: score_dfs["v2_6m"] = v2[["ticker", "pred_6m"]].rename(columns={"pred_6m": "score"})
        if "v6_3m" in use_models or "v6_6m" in use_models:
            v6 = data.ml_v6[data.ml_v6["asof"] == asof].copy()
            if "v6_3m" in use_models: score_dfs["v6_3m"] = v6[["ticker", "pred_v6_3m"]].rename(columns={"pred_v6_3m": "score"})
            if "v6_6m" in use_models: score_dfs["v6_6m"] = v6[["ticker", "pred_v6_6m"]].rename(columns={"pred_v6_6m": "score"})
        if "pattern" in use_models:
            pat = data.ml_pattern[data.ml_pattern["asof"] == asof].copy()
            score_dfs["pattern"] = pat[["ticker", "pattern_sim"]].rename(columns={"pattern_sim": "score"})
        if "ttm" in use_models:
            ttm = data.ml_ttm[data.ml_ttm["asof"] == asof].copy()
            score_dfs["ttm"] = ttm[["ticker", "ttm_peak"]].rename(columns={"ttm_peak": "score"})
        if "vertical" in use_models:
            vt = data.ml_vertical[data.ml_vertical["asof"] == asof].copy()
            score_dfs["vertical"] = vt[["ticker", "p_vertical"]].rename(columns={"p_vertical": "score"})

        if not score_dfs:
            return [], []

        # Convert each model's score to rank percentile within eligible
        rank_frames = []
        for name, df in score_dfs.items():
            df = df[df["ticker"].isin(eligible)].copy()
            if len(df) == 0:
                continue
            df[f"rk_{name}"] = df["score"].rank(pct=True)
            rank_frames.append(df[["ticker", f"rk_{name}"]])
        if not rank_frames:
            return [], []

        # Merge on ticker, average all rank columns
        merged = rank_frames[0]
        for rf in rank_frames[1:]:
            merged = merged.merge(rf, on="ticker", how="outer")
        rk_cols = [c for c in merged.columns if c.startswith("rk_")]
        # Fill missing model outputs with the model's median rank (0.5)
        # — a stock missing from one model still gets to be ranked fairly
        merged[rk_cols] = merged[rk_cols].fillna(0.5)
        merged["ensemble_rk"] = merged[rk_cols].mean(axis=1)

        # Apply Chronos p70 filter on TOP of the ensemble
        ch = data.chronos.get(asof, {})
        if ch:
            merged["chr"] = merged["ticker"].map(ch)
            merged["chr_rk"] = merged["chr"].rank(pct=True)
            merged = merged[merged["chr_rk"] >= chronos_q]

        if len(merged) < k:
            return [], []
        top = merged.sort_values("ensemble_rk", ascending=False).head(k)
        picks = top["ticker"].tolist()
        weights = invvol_weights(picks, data.mret, asof, cap=cap)
        return picks, list(weights)

    return pick


def main():
    RES.mkdir(parents=True, exist_ok=True)
    data = load_all()
    print("Loaded.")

    variants = [
        # Single models for reference
        ("v2_only_3m_6m", ["v2_3m", "v2_6m"],
            "v2 only (production scorer, both horizons)"),
        # Ensembles of growing breadth
        ("ens_v2_v6", ["v2_3m", "v2_6m", "v6_3m", "v6_6m"],
            "v2 + v6 (2 GBM models, 2 horizons each)"),
        ("ens_v2_v6_pattern", ["v2_3m", "v2_6m", "v6_3m", "v6_6m", "pattern"],
            "v2 + v6 + pattern (3 model families)"),
        ("ens_v2_v6_pattern_ttm", ["v2_3m", "v2_6m", "v6_3m", "v6_6m", "pattern", "ttm"],
            "v2 + v6 + pattern + TTM (4 model families)"),
        ("ens_all_5", ["v2_3m", "v2_6m", "v6_3m", "v6_6m", "pattern", "ttm", "vertical"],
            "all 5 model families (full ensemble)"),
    ]

    summary = []
    for name, models, desc in variants:
        print(f"\n{'='*70}\n  {name}\n  {desc}\n{'='*70}")
        try:
            res = evaluate(data, make_model_ensemble_picker(models), name)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        log = res.pop("log")
        with open(RES / f"{name}.json", "w") as f:
            json.dump(res, f, indent=2, default=str)
        pd.DataFrame(log).to_csv(RES / f"{name}_equity.csv", index=False)
        print(f"  Lump-sum CAGR: {res['cagr_lump_sum_pct']:7.2f}%  "
              f"DCA: {res['cagr_dca_pct']:7.2f}%")
        print(f"  WF: mean={res['wf_mean_pct']:.2f}% "
              f"min={res['wf_min_pct']:.2f}% "
              f"edge={res['wf_mean_edge_pp']:+.2f}pp  "
              f"beat_spy={res['wf_n_beat_spy']}/10")
        print(f"  Sharpe {res['sharpe']:.2f}  MaxDD {res['max_dd_pct']:.1f}%")

        # Compute year-edge std-dev across all years
        yby = res["year_by_year"]
        edges = np.array([y["edge_pp"] for y in yby if y["edge_pp"] is not None])
        e2024 = next((y["edge_pp"] for y in yby if y["year"] == 2024), None)
        print(f"  Year-edge std (yr-yr vol): {edges.std():.2f}pp  "
              f"Min: {edges.min():.2f}pp  2024: {e2024:+.2f}pp")
        summary.append({
            "variant": name,
            "description": desc,
            "cagr_lump_sum_pct": res["cagr_lump_sum_pct"],
            "cagr_dca_pct": res["cagr_dca_pct"],
            "wf_mean_pct": res["wf_mean_pct"],
            "wf_min_pct": res["wf_min_pct"],
            "wf_mean_edge_pp": res["wf_mean_edge_pp"],
            "wf_n_beat_spy": res["wf_n_beat_spy"],
            "sharpe": res["sharpe"],
            "max_dd_pct": res["max_dd_pct"],
            "year_edge_std_pp": float(edges.std()),
            "year_edge_min_pp": float(edges.min()),
            "y2024_edge_pp": e2024,
        })

    df = pd.DataFrame(summary)
    df.to_csv(RES / "model_ensemble_summary.csv", index=False)
    print("\n\n=== MODEL-ENSEMBLE SWEEP ===")
    cols = ["variant", "cagr_lump_sum_pct", "wf_mean_pct", "wf_min_pct",
            "wf_n_beat_spy", "sharpe", "max_dd_pct",
            "year_edge_std_pp", "year_edge_min_pp", "y2024_edge_pp"]
    print(df[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
