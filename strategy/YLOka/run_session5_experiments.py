"""Session 5 — regime-specialist GBM experiments."""
from __future__ import annotations

import sys, time
sys.path.insert(0, "/home/user/crt")

import pandas as pd

from strategy.YLOka.harness import (
    StratConfig, run_and_log,
    load_panel_specialist, load_monthly_returns, load_spy_features,
)


def main():
    print(">> loading specialist panel ...")
    panel = load_panel_specialist()
    mr = load_monthly_returns()
    spy = load_spy_features()
    print(f"   panel {panel.shape}")
    print(f"   columns: {sorted(panel.columns)[:10]}...")

    spec_cols = [c for c in panel.columns if "_bull" in c or "_normal" in c or "_recovery" in c]
    print(f"   specialist columns ({len(spec_cols)}): {spec_cols}")
    if "regime" in panel.columns:
        print(f"   regime distribution: {panel.drop_duplicates('asof')['regime'].value_counts().to_dict()}")

    cfgs = [
        StratConfig(name="exp_100_baseline_check"),
        # Pure router: use only the regime-matching specialist's prediction
        StratConfig(name="exp_101_specialist_router", score_fn_name="specialist_router"),
        # Blends with baseline
        StratConfig(name="exp_102_specialist_blend_03", score_fn_name="specialist_blend_03"),
        StratConfig(name="exp_103_specialist_blend_05", score_fn_name="specialist_blend_05"),
        StratConfig(name="exp_104_specialist_blend_07", score_fn_name="specialist_blend_07"),
        # Rank average
        StratConfig(name="exp_105_specialist_rank_avg", score_fn_name="specialist_rank_avg"),
        # Specialist + cash yield
        StratConfig(name="exp_106_specialist_blend_05_cy", score_fn_name="specialist_blend_05",
                    cash_yield_apr=0.03),
        # Specialist + K=5 (more diversification)
        StratConfig(name="exp_107_specialist_blend_05_K5", score_fn_name="specialist_blend_05", K=5),
    ]

    print("\n=== Session 5 specialist sweep (research window) ===")
    out = []
    for cfg in cfgs:
        t0 = time.time()
        try:
            met = run_and_log(cfg, panel, mr, spy, window="research")
        except Exception as e:
            print(f"  [{cfg.name}] FAILED: {e}")
            import traceback; traceback.print_exc()
            continue
        out.append({"name": cfg.name, **met})
        print(f"  [{cfg.name:35s}] CAGR={met['cagr']*100:6.2f}%  Sharpe={met['sharpe']:.3f}  "
              f"MaxDD={met['max_dd']*100:7.2f}%  cash={met['cash_months']}  ({time.time()-t0:.1f}s)")
    print()
    df = pd.DataFrame(out).sort_values("cagr", ascending=False)
    print(df[["name", "cagr", "sharpe", "sortino", "max_dd", "cash_months"]].to_string(index=False))
    return df


if __name__ == "__main__":
    main()
