"""Comprehensive v4 sweep:
  - v3 baseline (ml_3plus6 K=3 EW tight h=6 cap1.0) with various stop-loss / take-profit / score-threshold
  - v4 scorers (v4_only, v4_6m, v4_3m, stack_v2_v4, stack_v2_v4_quality)
  - K in {1, 2, 3, 5}
  - holds in {3, 6, 9}
  - weighting in {ew, invvol}
  - regime gate in {tight, v4}
  - score threshold in {0, 0.05, 0.10, 0.20}
  - stop-loss in {0, 0.30, 0.40}
  - take-profit in {0, 0.50, 0.75}
"""
from __future__ import annotations

import time
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from simulator_v4 import (
    Variant, simulate_variant_v4, evaluate, build_panel_with_score,
    load_spy_features, build_spy_aligned, _load_daily_prices, PIT, V2,
)

OUT = PIT / "v4_sweep_results.csv"


def main(stage="all"):
    t0 = time.time()
    print(f"[v4 sweep] stage={stage}")
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_features = load_spy_features()
    daily_prices = _load_daily_prices()

    panel_cache: dict[str, pd.DataFrame] = {}
    spy_aligned = None

    rows = []

    # ---------------------------- Stage A: v3 + stops/take-profits/threshold ----------------------------
    if stage in ("all", "stages"):
        configs_a = []
        # baseline
        configs_a.append(dict(name_suffix="baseline", k_normal=3, k_recovery=3, k_bull=3,
                              weighting="ew", regime_gate="tight", hold_months=6,
                              stop_loss_pct=0.0, take_profit_pct=0.0, score_threshold_pct=0.0))
        # stops
        for sl in (0.20, 0.25, 0.30, 0.35, 0.40):
            configs_a.append(dict(name_suffix=f"sl{sl:.2f}", k_normal=3, k_recovery=3, k_bull=3,
                                  weighting="ew", regime_gate="tight", hold_months=6,
                                  stop_loss_pct=sl, take_profit_pct=0.0, score_threshold_pct=0.0))
        # take-profit only
        for tp in (0.40, 0.50, 0.75, 1.00):
            configs_a.append(dict(name_suffix=f"tp{tp:.2f}", k_normal=3, k_recovery=3, k_bull=3,
                                  weighting="ew", regime_gate="tight", hold_months=6,
                                  stop_loss_pct=0.0, take_profit_pct=tp, score_threshold_pct=0.0))
        # combo
        for sl, tp in product((0.25, 0.30, 0.35), (0.50, 0.75, 1.00)):
            configs_a.append(dict(name_suffix=f"sl{sl}_tp{tp}", k_normal=3, k_recovery=3, k_bull=3,
                                  weighting="ew", regime_gate="tight", hold_months=6,
                                  stop_loss_pct=sl, take_profit_pct=tp, score_threshold_pct=0.0))
        # k variants
        for k in (1, 2, 5):
            configs_a.append(dict(name_suffix=f"k{k}", k_normal=k, k_recovery=k, k_bull=k,
                                  weighting="ew", regime_gate="tight", hold_months=6,
                                  stop_loss_pct=0.0, take_profit_pct=0.0, score_threshold_pct=0.0))
        # invvol
        configs_a.append(dict(name_suffix="invvol", k_normal=3, k_recovery=3, k_bull=3,
                              weighting="invvol", regime_gate="tight", hold_months=6,
                              stop_loss_pct=0.0, take_profit_pct=0.0, score_threshold_pct=0.0))

        scorer_a = "ml_3plus6"
        if scorer_a not in panel_cache:
            panel_cache[scorer_a] = build_panel_with_score(scorer_a)
        p = panel_cache[scorer_a]
        if spy_aligned is None:
            spy_aligned = build_spy_aligned(p)

        print(f"\n=== Stage A: v3 + stops/threshold ({len(configs_a)} configs) ===")
        for c in configs_a:
            v = Variant(name=f"{scorer_a}|{c['name_suffix']}", scorer=scorer_a,
                        cap_per_pick=1.0, **{k: c[k] for k in c if k != "name_suffix"})
            try:
                eq = simulate_variant_v4(p, monthly_returns, spy_features, v, daily_prices=daily_prices)
                m = evaluate(eq, spy_aligned, v.name)
                m.update(c)
                m["scorer"] = scorer_a
                rows.append(m)
                print(f"  {v.name:50s}  CAGR={m['cagr_full']*100:6.2f}%  WF_mean={m['wf_mean_cagr']*100:6.2f}%  "
                      f"WF_min={m['wf_min_cagr']*100:6.2f}%  beats={m['wf_n_beats']}/{m['wf_n_splits']}",
                      flush=True)
            except Exception as e:
                print(f"  ! {v.name}: {e}")

    # ---------------------------- Stage B: v4 scorers ----------------------------
    if stage in ("all", "v4"):
        v4_path = PIT / "ml_preds_v4.parquet"
        if not v4_path.exists():
            print("\n  v4 ML preds not available — skipping stage B")
            return rows
        scorers_b = ["v4_only", "v4_6m", "v4_3m", "stack_v2_v4", "stack_v2_v4_quality"]
        configs_b = []
        # core configs
        for k in (2, 3):
            for h in (3, 6, 9):
                for sl in (0.0, 0.30):
                    for tp in (0.0, 0.75):
                        configs_b.append(dict(k_normal=k, k_recovery=k, k_bull=k,
                                              weighting="ew", regime_gate="tight", hold_months=h,
                                              stop_loss_pct=sl, take_profit_pct=tp, score_threshold_pct=0.0))
        # invvol variants for top stack scorer
        for h in (6,):
            configs_b.append(dict(k_normal=3, k_recovery=3, k_bull=3,
                                  weighting="invvol", regime_gate="tight", hold_months=h,
                                  stop_loss_pct=0.0, take_profit_pct=0.0, score_threshold_pct=0.0))

        print(f"\n=== Stage B: v4 scorers ({len(scorers_b)} × {len(configs_b)} = {len(scorers_b)*len(configs_b)} configs) ===")
        for scorer in scorers_b:
            if scorer not in panel_cache:
                panel_cache[scorer] = build_panel_with_score(scorer)
            p = panel_cache[scorer]
            if spy_aligned is None:
                spy_aligned = build_spy_aligned(p)

            for c in configs_b:
                tag = f"k{c['k_normal']}|h{c['hold_months']}|{c['weighting']}|{c['regime_gate']}|sl{c['stop_loss_pct']}|tp{c['take_profit_pct']}"
                v = Variant(name=f"{scorer}|{tag}", scorer=scorer, cap_per_pick=1.0, **c)
                try:
                    eq = simulate_variant_v4(p, monthly_returns, spy_features, v, daily_prices=daily_prices)
                    m = evaluate(eq, spy_aligned, v.name)
                    m.update(c); m["scorer"] = scorer
                    rows.append(m)
                    print(f"  {v.name:60s}  CAGR={m['cagr_full']*100:6.2f}%  WF_mean={m['wf_mean_cagr']*100:6.2f}%  "
                          f"WF_min={m['wf_min_cagr']*100:6.2f}%  beats={m['wf_n_beats']}/{m['wf_n_splits']}",
                          flush=True)
                except Exception as e:
                    print(f"  ! {v.name}: {e}")

    df = pd.DataFrame(rows)
    df = df.sort_values("wf_mean_cagr", ascending=False)
    df.to_csv(OUT, index=False)
    print(f"\n{(time.time()-t0)/60:.1f} min total")
    print(f"Saved {len(df)} rows to {OUT}")
    print("\n=== TOP 10 by WF mean CAGR ===")
    cols = ["name", "cagr_full", "wf_mean_cagr", "wf_min_cagr", "wf_n_beats",
            "wf_n_pos", "sharpe", "max_dd", "n_cash"]
    print(df[cols].head(10).to_string(index=False))
    return rows


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    main(stage)
