"""Stage A2: deeper exploration of the take-profit + stop-loss winning families
on the existing v3 ml_3plus6 scorer.

We confirmed in stage A that tp=+50% alone gives WF mean 64.56% (vs 42.80% baseline).
This script explores:
  - tp values: {0.30, 0.40, 0.50, 0.60, 0.75, 1.00}
  - hold periods: {3, 6, 9, 12}
  - K values: {2, 3, 5}
  - stop-loss + take-profit combos
  - score-threshold filter
  - K=1 (most concentrated)
"""
from __future__ import annotations

import time
import sys
from itertools import product
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from simulator_v4 import (
    Variant, simulate_variant_v4, evaluate, build_panel_with_score,
    load_spy_features, build_spy_aligned, _load_daily_prices, PIT, V2,
)


def main():
    t0 = time.time()
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_features = load_spy_features()
    daily_prices = _load_daily_prices()
    panel = build_panel_with_score("ml_3plus6")
    spy_aligned = build_spy_aligned(panel)

    rows = []
    configs = []

    # -- TP-only sweep with various K, hold
    for k in (1, 2, 3, 5):
        for h in (3, 6, 9, 12):
            for tp in (0.30, 0.40, 0.50, 0.60, 0.75, 1.00):
                configs.append(dict(name_suffix=f"k{k}_h{h}_tp{tp}",
                                    k_normal=k, k_recovery=k, k_bull=k,
                                    weighting="ew", regime_gate="tight",
                                    hold_months=h,
                                    stop_loss_pct=0.0, take_profit_pct=tp,
                                    score_threshold_pct=0.0))
    # -- TP + SL combo with k=3, h=6
    for sl, tp in product((0.20, 0.25, 0.30, 0.35), (0.40, 0.50, 0.60, 0.75)):
        configs.append(dict(name_suffix=f"k3_h6_sl{sl}_tp{tp}",
                            k_normal=3, k_recovery=3, k_bull=3,
                            weighting="ew", regime_gate="tight",
                            hold_months=6,
                            stop_loss_pct=sl, take_profit_pct=tp,
                            score_threshold_pct=0.0))
    # -- v4 regime gate on top of TP=0.5
    for k in (2, 3):
        for h in (3, 6, 9):
            configs.append(dict(name_suffix=f"k{k}_h{h}_tp0.5_gate_v4",
                                k_normal=k, k_recovery=k, k_bull=k,
                                weighting="ew", regime_gate="v4",
                                hold_months=h,
                                stop_loss_pct=0.0, take_profit_pct=0.5,
                                score_threshold_pct=0.0))
    # -- invvol + tp
    for h in (3, 6):
        for tp in (0.50, 0.75):
            configs.append(dict(name_suffix=f"k3_h{h}_tp{tp}_invvol",
                                k_normal=3, k_recovery=3, k_bull=3,
                                weighting="invvol", regime_gate="tight",
                                hold_months=h,
                                stop_loss_pct=0.0, take_profit_pct=tp,
                                score_threshold_pct=0.0))
    # -- score threshold
    for thr in (0.05, 0.10, 0.20):
        for tp in (0.50, 0.75):
            configs.append(dict(name_suffix=f"k3_h6_tp{tp}_thr{thr}",
                                k_normal=3, k_recovery=3, k_bull=3,
                                weighting="ew", regime_gate="tight",
                                hold_months=6,
                                stop_loss_pct=0.0, take_profit_pct=tp,
                                score_threshold_pct=thr))

    print(f"=== Stage A2: {len(configs)} configs ===", flush=True)
    for c in configs:
        v = Variant(name=f"ml_3plus6|{c['name_suffix']}", scorer="ml_3plus6",
                    cap_per_pick=1.0,
                    **{k: c[k] for k in c if k != "name_suffix"})
        try:
            eq = simulate_variant_v4(panel, monthly_returns, spy_features, v, daily_prices=daily_prices)
            m = evaluate(eq, spy_aligned, v.name)
            m.update(c); m["scorer"] = "ml_3plus6"
            rows.append(m)
            print(f"  {v.name:55s}  CAGR={m['cagr_full']*100:6.2f}%  WF_mean={m['wf_mean_cagr']*100:6.2f}%  "
                  f"WF_min={m['wf_min_cagr']*100:6.2f}%  beats={m['wf_n_beats']}/{m['wf_n_splits']}  "
                  f"Sh={m['sharpe']:.2f}  MDD={m['max_dd']*100:6.1f}%  cash={m['n_cash']}",
                  flush=True)
        except Exception as e:
            print(f"  ! {v.name}: {e}", flush=True)

    df = pd.DataFrame(rows)
    df = df.sort_values("wf_mean_cagr", ascending=False)
    df.to_csv(PIT / "v4_stage_a2_results.csv", index=False)
    print(f"\n{(time.time()-t0)/60:.1f} min total")
    print(f"Saved {len(df)} rows.")
    print("\n=== TOP 15 by WF mean CAGR ===")
    cols = ["name", "cagr_full", "wf_mean_cagr", "wf_min_cagr", "wf_n_beats", "wf_n_pos",
            "sharpe", "max_dd", "n_cash"]
    print(df[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
