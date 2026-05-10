"""Advanced v4 sweep: test ideas that might give a real lift.

  1. Regime-conditional K (different K per regime)
  2. Multi-horizon ensemble with regime-specific weighting
  3. Score-margin filter (only invest if top-K vs bottom of top-K is wide)
  4. Different cost assumptions (lower cost in real world for larger positions)
"""
from __future__ import annotations
import time, sys
from pathlib import Path
import numpy as np
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
    configs = [
        # Baseline
        dict(name="baseline_v3", k_normal=3, k_recovery=3, k_bull=3,
             weighting="ew", regime_gate="tight", hold_months=6,
             stop_loss_pct=0.0, take_profit_pct=0.0),
        # Regime-conditional K
        dict(name="K_2_3_4", k_normal=3, k_recovery=2, k_bull=4,
             weighting="ew", regime_gate="tight", hold_months=6,
             stop_loss_pct=0.0, take_profit_pct=0.0),
        dict(name="K_3_2_3", k_normal=3, k_recovery=2, k_bull=3,
             weighting="ew", regime_gate="tight", hold_months=6,
             stop_loss_pct=0.0, take_profit_pct=0.0),
        dict(name="K_3_3_5", k_normal=3, k_recovery=3, k_bull=5,
             weighting="ew", regime_gate="tight", hold_months=6,
             stop_loss_pct=0.0, take_profit_pct=0.0),
        dict(name="K_4_3_5", k_normal=4, k_recovery=3, k_bull=5,
             weighting="ew", regime_gate="tight", hold_months=6,
             stop_loss_pct=0.0, take_profit_pct=0.0),
        dict(name="K_3_2_5", k_normal=3, k_recovery=2, k_bull=5,
             weighting="ew", regime_gate="tight", hold_months=6,
             stop_loss_pct=0.0, take_profit_pct=0.0),
        # K=2 with hold variants
        dict(name="K2_h3", k_normal=2, k_recovery=2, k_bull=2,
             weighting="ew", regime_gate="tight", hold_months=3,
             stop_loss_pct=0.0, take_profit_pct=0.0),
        dict(name="K2_h9", k_normal=2, k_recovery=2, k_bull=2,
             weighting="ew", regime_gate="tight", hold_months=9,
             stop_loss_pct=0.0, take_profit_pct=0.0),
        # Cap per pick (no concentration > 50%)
        dict(name="cap50", k_normal=3, k_recovery=3, k_bull=3,
             weighting="invvol", regime_gate="tight", hold_months=6,
             stop_loss_pct=0.0, take_profit_pct=0.0, cap_per_pick=0.5),
        dict(name="cap40_invvol", k_normal=3, k_recovery=3, k_bull=3,
             weighting="invvol", regime_gate="tight", hold_months=6,
             stop_loss_pct=0.0, take_profit_pct=0.0, cap_per_pick=0.4),
        # h_3 hold (faster turnover)
        dict(name="K3_h3", k_normal=3, k_recovery=3, k_bull=3,
             weighting="ew", regime_gate="tight", hold_months=3,
             stop_loss_pct=0.0, take_profit_pct=0.0),
        # Lower cost assumption (1bp instead of 10bp)
        dict(name="cost1bp", k_normal=3, k_recovery=3, k_bull=3,
             weighting="ew", regime_gate="tight", hold_months=6,
             stop_loss_pct=0.0, take_profit_pct=0.0, cost_bps=1.0),
    ]

    print(f"=== Advanced sweep: {len(configs)} configs ===", flush=True)
    for c in configs:
        cc = {k: v for k, v in c.items() if k != "name"}
        cc.setdefault("score_threshold_pct", 0.0)
        cc.setdefault("cap_per_pick", 1.0)
        cc.setdefault("cost_bps", 10.0)
        v = Variant(name=f"ml_3plus6|{c['name']}", scorer="ml_3plus6", **cc)
        try:
            eq = simulate_variant_v4(panel, monthly_returns, spy_features, v, daily_prices=daily_prices)
            m = evaluate(eq, spy_aligned, v.name)
            m.update(c); m["scorer"] = "ml_3plus6"
            rows.append(m)
            print(f"  {v.name:50s}  CAGR={m['cagr_full']*100:6.2f}%  WF_mean={m['wf_mean_cagr']*100:6.2f}%  "
                  f"WF_min={m['wf_min_cagr']*100:6.2f}%  beats={m['wf_n_beats']}/{m['wf_n_splits']}  "
                  f"Sh={m['sharpe']:.2f}  MDD={m['max_dd']*100:6.1f}%",
                  flush=True)
        except Exception as e:
            print(f"  ! {v.name}: {e}", flush=True)

    df = pd.DataFrame(rows).sort_values("wf_mean_cagr", ascending=False)
    df.to_csv(PIT / "v4_advanced_sweep_results.csv", index=False)
    print(f"\n{(time.time()-t0)/60:.1f} min total")
    print(f"\n=== TOP by WF mean CAGR ===")
    cols = ["name", "cagr_full", "wf_mean_cagr", "wf_min_cagr", "wf_n_beats",
            "sharpe", "max_dd", "n_cash"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
