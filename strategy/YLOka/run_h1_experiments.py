"""H1 experiments: multi-target ensemble using the new pred_12m + pred_12m_cls
heads trained in train_12m_head.py.

All runs restricted to RESEARCH window (2003-09 -> 2024-04).
Holdout (2024-05+) NOT touched.
"""
from __future__ import annotations

import sys, time
sys.path.insert(0, "/home/user/crt")

import pandas as pd

from strategy.YLOka.harness import (
    StratConfig, run_and_log,
    load_panel, load_panel_ensemble,
    load_monthly_returns, load_spy_features,
)


def main():
    print(">> loading panel + ensemble panel + mr + spy ...")
    base_panel = load_panel()
    ens_panel = load_panel_ensemble()
    mr = load_monthly_returns()
    spy = load_spy_features()
    print(f"   base panel {base_panel.shape}, ens panel {ens_panel.shape}")
    print(f"   pred_12m coverage on ens panel: {ens_panel['pred_12m'].notna().mean():.2%}")

    cfgs = [
        # baseline reference (from previous session)
        ("exp_19_baseline_repro",       StratConfig(name="exp_19_baseline_repro"), base_panel),

        # H1 — multi-target ensembles
        ("exp_20_ens_3_6_12_ew",        StratConfig(name="exp_20_ens_3_6_12_ew",
                                                     score_fn_name="ens_3_6_12"), ens_panel),
        ("exp_21_ens_3_6_12_cls",       StratConfig(name="exp_21_ens_3_6_12_cls",
                                                     score_fn_name="ens_3_6_12_cls"), ens_panel),
        ("exp_22_ens_3_6_12_invvol",    StratConfig(name="exp_22_ens_3_6_12_invvol",
                                                     score_fn_name="ens_3_6_12_invvol"), ens_panel),
        ("exp_23_ens_36_12wt",          StratConfig(name="exp_23_ens_36_12wt",
                                                     score_fn_name="ens_36_12wt"), ens_panel),

        # H1 + cash yield (free 0.07 pp)
        ("exp_24_ens_3_6_12_cy",        StratConfig(name="exp_24_ens_3_6_12_cy",
                                                     score_fn_name="ens_3_6_12",
                                                     cash_yield_apr=0.03), ens_panel),

        # H1 + K=5 (more diversification on top of stronger signal)
        ("exp_25_ens_3_6_12_K5",        StratConfig(name="exp_25_ens_3_6_12_K5",
                                                     score_fn_name="ens_3_6_12", K=5), ens_panel),

        # H1 + K=2 (concentrate on top of ensemble)
        ("exp_26_ens_3_6_12_K2",        StratConfig(name="exp_26_ens_3_6_12_K2",
                                                     score_fn_name="ens_3_6_12", K=2), ens_panel),
    ]

    out_rows = []
    for name, cfg, panel in cfgs:
        t0 = time.time()
        try:
            met = run_and_log(cfg, panel, mr, spy, window="research")
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            import traceback; traceback.print_exc()
            continue
        dt = time.time() - t0
        out_rows.append({"name": name, **met})
        print(f"  [{name}] CAGR={met['cagr']*100:6.2f}%  Sharpe={met['sharpe']:.3f}  "
              f"MaxDD={met['max_dd']*100:7.2f}%  cash={met['cash_months']}  ({dt:.1f}s)")

    df = pd.DataFrame(out_rows)
    print("\n=== H1 ENSEMBLE SUMMARY ===")
    cols = ["name", "n_months", "cagr", "sharpe", "sortino", "max_dd", "cash_months"]
    print(df[cols].to_string(index=False))
    return df


if __name__ == "__main__":
    main()
