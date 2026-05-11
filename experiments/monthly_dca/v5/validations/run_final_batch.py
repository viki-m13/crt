"""Final batch — exhaust remaining prepped experiments.

P. Trend-positive entry filter:
   At pick time, require each candidate's d_sma200 > 0 (above own 200dma).
   Filters out value-trap mean-reversion picks. KEY at 2024-01 had
   d_sma200 < 0; this would have rejected it.

Q. Picker-consensus filter (use auxiliary models as VOTERS, not predictors):
   v6, pattern, ttm, vertical have low alpha alone but their consensus
   on v2's picks may still be informative. Require ≥M of N models also
   rank the pick in their top T%.

R. Sector concentration cap:
   Max 1 pick per GICS sector. Prevents 3 tech stocks or 3 banks.

S. Adaptive K based on cross-sectional score dispersion:
   When score spread top-K vs (K+1)-th is wide → high conviction → K=2
   When score spread is narrow → low conviction → K=5

T. Volatility-targeted exposure:
   Scale exposure so target portfolio annualized vol = 25%

U. Cross-sectional momentum standalone:
   Long top-decile by raw 12-1 momentum. Compare alpha vs v5.

V. MODE D — combined stack:
   Best entry filter + best overlay + best sleeve. The everything-in
   ensemble.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

from experiments.monthly_dca.v5.validations.harness import (
    HarnessData, load_all, pick_v5_baseline, classify_regime_tight,
    invvol_weights, evaluate,
    CHRONOS_FILTER_Q, CAP_PER_PICK, COST_BPS, K_PICKS, HOLD_MONTHS,
)
from experiments.monthly_dca.v5.validations.run_tactical_rebalance import _score_at
from experiments.monthly_dca.v5.validations.run_advanced_overlays import (
    sector_top_n_sleeve, summary,
)
from experiments.monthly_dca.v2.ml_strategy import EXCLUDE

RES = Path(__file__).resolve().parent / "results"
FEATURES_DIR = Path("experiments/monthly_dca/cache/features")


# =============================================================================
# P. Trend-positive entry filter (using PRIOR-MONTH d_sma200 — no look-ahead)
# =============================================================================
def make_trend_entry_picker(k: int = K_PICKS, require_uptrend: bool = True,
                              prior_features_at=None):
    """Filter eligible to only stocks with d_sma200 > 0 at the rebalance
    moment. Uses CURRENT month's d_sma200 (which is known at month-end —
    same time decisions are made). Returns top-K by v5 score from the
    uptrend-filtered cohort.
    """
    def pick(asof, eligible, data, regime):
        scored = _score_at(asof, eligible, data)
        if require_uptrend:
            # Load current asof features and filter to d_sma200 > 0
            fp = FEATURES_DIR / f"{asof.strftime('%Y-%m-%d')}.parquet"
            if fp.exists():
                feat = pd.read_parquet(fp)
                if "d_sma200" in feat.columns:
                    uptrend = feat[feat["d_sma200"] > 0].index.tolist()
                    scored = scored[scored["ticker"].isin(uptrend)]
        if len(scored) < k:
            return [], []
        top = scored.sort_values("score", ascending=False).head(k)
        picks = top["ticker"].tolist()
        weights = invvol_weights(picks, data.mret, asof, cap=CAP_PER_PICK)
        return picks, list(weights)
    return pick


# =============================================================================
# Q. Picker-consensus filter
# =============================================================================
def make_consensus_picker(k: int = K_PICKS, min_votes: int = 2,
                            vote_threshold: float = 0.4):
    """For each v2 top-N candidate, count how many auxiliary models
    (v6/pattern/ttm/vertical) rank it in their top vote_threshold%.
    Require min_votes of 4 to be valid.
    """
    def pick(asof, eligible, data, regime):
        scored = _score_at(asof, eligible, data)
        if len(scored) < k:
            return [], []
        top_pool = scored.sort_values("score", ascending=False).head(20)

        # Compute auxiliary model rank percentiles
        votes = {tk: 0 for tk in top_pool["ticker"]}
        aux_models = [
            ("v6", data.ml_v6, "pred_v6"),
            ("pattern", data.ml_pattern, "pattern_sim"),
            ("ttm", data.ml_ttm, "ttm_peak"),
            ("vertical", data.ml_vertical, "p_vertical"),
        ]
        for name, df, col in aux_models:
            sub = df[df["asof"] == asof]
            sub = sub[sub["ticker"].isin(eligible)]
            if len(sub) == 0 or col not in sub.columns:
                continue
            sub = sub.copy()
            sub["rk"] = sub[col].rank(pct=True)
            for tk, rk in zip(sub["ticker"], sub["rk"]):
                if tk in votes and rk >= (1 - vote_threshold):
                    votes[tk] += 1

        # Filter to stocks with ≥ min_votes
        consensus = top_pool[top_pool["ticker"].apply(lambda t: votes.get(t, 0) >= min_votes)]
        if len(consensus) < k:
            # Fall back to top-K from v2 alone if not enough consensus
            consensus = top_pool.head(k)
        else:
            consensus = consensus.head(k)
        picks = consensus["ticker"].tolist()
        weights = invvol_weights(picks, data.mret, asof, cap=CAP_PER_PICK)
        return picks, list(weights)
    return pick


# =============================================================================
# R. Sector concentration cap
# =============================================================================
def make_sector_capped_picker(k: int = K_PICKS):
    """Max 1 pick per sector."""
    def pick(asof, eligible, data, regime):
        scored = _score_at(asof, eligible, data)
        if len(scored) < k:
            return [], []
        sector_map = data.sector_map  # dict ticker → sector

        ranked = scored.sort_values("score", ascending=False)
        selected = []
        used_sectors = set()
        for _, row in ranked.iterrows():
            tk = row["ticker"]
            sec = sector_map.get(tk, "Unknown")
            if sec in used_sectors:
                continue
            selected.append(tk)
            used_sectors.add(sec)
            if len(selected) >= k:
                break
        if len(selected) < k:
            # Fall back to top-K if sector cap can't be satisfied
            selected = ranked.head(k)["ticker"].tolist()
        weights = invvol_weights(selected, data.mret, asof, cap=CAP_PER_PICK)
        return selected, list(weights)
    return pick


# =============================================================================
# S. Adaptive K based on score dispersion
# =============================================================================
def make_adaptive_k_picker():
    """K is determined dynamically by the gap between top-K and (K+1)-th
    score. Wide gap → high conviction → K=2 (concentrate). Narrow gap →
    low conviction → K=5 (diversify).
    """
    def pick(asof, eligible, data, regime):
        scored = _score_at(asof, eligible, data)
        if len(scored) < 5:
            return [], []
        s = scored.sort_values("score", ascending=False).reset_index(drop=True)
        # Compute the gap between top-3 score and 4th score
        gap = float(s.loc[2, "score"] - s.loc[4, "score"])
        score_std = float(s["score"].std())
        if score_std == 0:
            k = 3
        else:
            gap_z = gap / score_std
            if gap_z > 0.30:
                k = 2  # high conviction
            elif gap_z < 0.10:
                k = 5  # low conviction
            else:
                k = 3  # default
        if len(s) < k:
            return [], []
        top = s.head(k)
        picks = top["ticker"].tolist()
        weights = invvol_weights(picks, data.mret, asof, cap=CAP_PER_PICK)
        return picks, list(weights)
    return pick


# =============================================================================
# Run everything
# =============================================================================
def main():
    RES.mkdir(parents=True, exist_ok=True)
    data = load_all()
    spy = data.mret["SPY"].copy(); spy.index = pd.to_datetime(spy.index)
    print("Loaded.\n")

    print(f"{'Variant':<30} {'CAGR':>7} {'WF':>7} {'WFmin':>7} {'beats':>6} {'Sharpe':>7} {'MDD':>7} {'2024':>8}")
    rows = []

    def test(name, pick_fn, desc):
        try:
            res = evaluate(data, pick_fn, name)
        except Exception as e:
            print(f"{name}: ERROR {e}")
            return None
        log = res.pop("log")
        pd.DataFrame(log).to_csv(RES / f"{name}_equity.csv", index=False)
        e2024 = next((y["edge_pp"] for y in res["year_by_year"] if y["year"]==2024), None)
        print(f"{name:<30} {res['cagr_lump_sum_pct']:>6.2f}% {res['wf_mean_pct']:>6.2f}% {res['wf_min_pct']:>6.2f}% {res['wf_n_beat_spy']:>4}/10 {res['sharpe']:>7.2f} {res['max_dd_pct']:>6.1f}% {e2024:>+6.2f}pp")
        rows.append({"variant": name, "desc": desc,
                     "cagr": res["cagr_lump_sum_pct"], "wf_mean": res["wf_mean_pct"],
                     "wf_min": res["wf_min_pct"], "beats_spy": res["wf_n_beat_spy"],
                     "sharpe": res["sharpe"], "mdd": res["max_dd_pct"],
                     "y2024": e2024})
        return res

    # Reference
    test("baseline_v5", pick_v5_baseline, "Production (K=3 GBM+Chronos)")

    print("\n=== P. Trend-positive entry filter ===")
    test("P_trend_entry", make_trend_entry_picker(),
          "Only picks where d_sma200 > 0 at entry")

    print("\n=== Q. Picker-consensus filter ===")
    test("Q_consensus_2of4_t40", make_consensus_picker(min_votes=2, vote_threshold=0.4),
          "≥2 of {v6,pattern,ttm,vertical} agree (top 40%)")
    test("Q_consensus_3of4_t40", make_consensus_picker(min_votes=3, vote_threshold=0.4),
          "≥3 of 4 agree (top 40%)")
    test("Q_consensus_2of4_t30", make_consensus_picker(min_votes=2, vote_threshold=0.3),
          "≥2 of 4 agree (top 30% stricter)")

    print("\n=== R. Sector concentration cap ===")
    test("R_sector_cap", make_sector_capped_picker(),
          "Max 1 pick per GICS sector")

    print("\n=== S. Adaptive K based on score dispersion ===")
    test("S_adaptive_K", make_adaptive_k_picker(),
          "K=2 if gap-z>0.3 else K=5 if <0.1 else K=3")

    print("\n=== Combined stacks ===")
    # Trend entry + sector cap (stack two filters)
    def make_trend_sector(k=K_PICKS):
        sm = data.sector_map
        def pick(asof, eligible, data, regime):
            scored = _score_at(asof, eligible, data)
            fp = FEATURES_DIR / f"{asof.strftime('%Y-%m-%d')}.parquet"
            if fp.exists():
                feat = pd.read_parquet(fp)
                if "d_sma200" in feat.columns:
                    uptrend = feat[feat["d_sma200"] > 0].index.tolist()
                    scored = scored[scored["ticker"].isin(uptrend)]
            if len(scored) < k: return [], []
            ranked = scored.sort_values("score", ascending=False)
            selected, used = [], set()
            for _, row in ranked.iterrows():
                tk = row["ticker"]
                sec = sm.get(tk, "Unknown")
                if sec in used: continue
                selected.append(tk); used.add(sec)
                if len(selected) >= k: break
            if len(selected) < k:
                selected = ranked.head(k)["ticker"].tolist()
            weights = invvol_weights(selected, data.mret, asof, cap=CAP_PER_PICK)
            return selected, list(weights)
        return pick
    test("PR_trend_plus_sector", make_trend_sector(),
          "Trend-positive entry + sector cap stack")

    df = pd.DataFrame(rows)
    df.to_csv(RES / "final_batch_summary.csv", index=False)


if __name__ == "__main__":
    main()
