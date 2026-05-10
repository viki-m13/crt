"""Honest v4 sweep using corrected TP semantics (cash for remainder of hold).

Tests:
  - Baseline (no TP, no SL)
  - TP only at honest semantics
  - SL only
  - TP + SL combos
  - Different K, hold, weighting
"""
from __future__ import annotations
import time, sys
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

    # Baseline (no TP/SL, no stops): exact v3
    configs.append(dict(name="baseline_v3", k_normal=3, k_recovery=3, k_bull=3,
                        weighting="ew", regime_gate="tight", hold_months=6,
                        stop_loss_pct=0.0, take_profit_pct=0.0))

    # TP only (honest)
    for tp in (0.30, 0.50, 0.75, 1.00, 1.50):
        configs.append(dict(name=f"tp{tp}", k_normal=3, k_recovery=3, k_bull=3,
                            weighting="ew", regime_gate="tight", hold_months=6,
                            stop_loss_pct=0.0, take_profit_pct=tp))
    # SL only
    for sl in (0.15, 0.20, 0.25, 0.30, 0.40):
        configs.append(dict(name=f"sl{sl}", k_normal=3, k_recovery=3, k_bull=3,
                            weighting="ew", regime_gate="tight", hold_months=6,
                            stop_loss_pct=sl, take_profit_pct=0.0))
    # SL + TP combos
    from itertools import product
    for sl, tp in product((0.20, 0.25, 0.30), (0.50, 0.75, 1.00)):
        configs.append(dict(name=f"sl{sl}_tp{tp}", k_normal=3, k_recovery=3, k_bull=3,
                            weighting="ew", regime_gate="tight", hold_months=6,
                            stop_loss_pct=sl, take_profit_pct=tp))
    # Different K, no stops
    for k in (1, 2, 5):
        configs.append(dict(name=f"k{k}_h6", k_normal=k, k_recovery=k, k_bull=k,
                            weighting="ew", regime_gate="tight", hold_months=6,
                            stop_loss_pct=0.0, take_profit_pct=0.0))
    # Different hold periods
    for h in (3, 9, 12):
        configs.append(dict(name=f"k3_h{h}", k_normal=3, k_recovery=3, k_bull=3,
                            weighting="ew", regime_gate="tight", hold_months=h,
                            stop_loss_pct=0.0, take_profit_pct=0.0))
    # Convex / inv-vol weighting
    for w in ("invvol", "conv", "softmax"):
        configs.append(dict(name=f"k3_h6_{w}", k_normal=3, k_recovery=3, k_bull=3,
                            weighting=w, regime_gate="tight", hold_months=6,
                            stop_loss_pct=0.0, take_profit_pct=0.0))
    # v4 regime gate
    configs.append(dict(name=f"k3_h6_gate_v4", k_normal=3, k_recovery=3, k_bull=3,
                        weighting="ew", regime_gate="v4", hold_months=6,
                        stop_loss_pct=0.0, take_profit_pct=0.0))

    # Score threshold
    for thr in (0.05, 0.10, 0.20, 0.30):
        configs.append(dict(name=f"thr{thr}", k_normal=3, k_recovery=3, k_bull=3,
                            weighting="ew", regime_gate="tight", hold_months=6,
                            stop_loss_pct=0.0, take_profit_pct=0.0,
                            score_threshold_pct=thr))

    print(f"=== Honest v4 sweep: {len(configs)} configs ===", flush=True)
    for c in configs:
        cc = {k: v for k, v in c.items() if k != "name"}
        # Defaults for missing fields
        cc.setdefault("score_threshold_pct", 0.0)
        v = Variant(name=f"ml_3plus6|{c['name']}", scorer="ml_3plus6",
                    cap_per_pick=1.0, **cc)
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

    df = pd.DataFrame(rows)
    df = df.sort_values("wf_mean_cagr", ascending=False)
    df.to_csv(PIT / "v4_honest_sweep_results.csv", index=False)
    print(f"\n{(time.time()-t0)/60:.1f} min total")
    print("\n=== TOP 15 by WF mean CAGR ===")
    cols = ["name", "cagr_full", "wf_mean_cagr", "wf_min_cagr", "wf_n_beats", "wf_n_pos",
            "sharpe", "max_dd", "n_cash"]
    print(df[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
