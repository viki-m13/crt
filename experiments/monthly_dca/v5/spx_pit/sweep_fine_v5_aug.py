"""Phase 7b: focused 2-D sweep around K=2 to confirm the new optimum.

The Phase 7 broad sweep flagged K=2 as the winner on augmented PIT
(WF mean 49.39% vs deployed K=3's 32.68%). Run a finer K×chr_q×hold×cap
grid centred on K=2 to verify it's a robust peak, not a one-cell artifact.
"""
from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_v5_aug import (  # noqa: E402
    load_spy_features, run_one,
    AUG, PIT,
)


def main():
    t0 = time.time()
    print("Loading augmented data ...")
    panel = pd.read_parquet(AUG / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    ml = pd.read_parquet(AUG / "ml_preds.parquet")[["asof", "ticker", "pred_3m", "pred_6m"]]
    ml["asof"] = pd.to_datetime(ml["asof"])
    ml["ml_score"] = (ml["pred_3m"] + ml["pred_6m"]) / 2
    chr_ = pd.read_parquet(AUG / "ml_preds_chronos.parquet")[["asof", "ticker", "chronos_p70_3m"]]
    chr_["asof"] = pd.to_datetime(chr_["asof"])
    spy = load_spy_features()
    mr = pd.read_parquet(AUG / "monthly_returns_clean.parquet").fillna(0.0)
    mp = pd.read_parquet(AUG / "monthly_prices_clean.parquet")
    if not isinstance(mr.index, pd.DatetimeIndex):
        mr.index = pd.to_datetime(mr.index)
        mp.index = pd.to_datetime(mp.index)
    members = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    members["asof"] = pd.to_datetime(members["asof"])
    members_g = members.groupby("asof")["ticker"].apply(set).to_dict()

    panel_by_asof = {a: g for a, g in panel.groupby("asof")}
    ml_by_asof = {a: g for a, g in ml.groupby("asof")}
    chr_by_asof = {a: g for a, g in chr_.groupby("asof")}
    months = sorted(set(panel["asof"]).intersection(set(spy.index)))
    months = [pd.Timestamp(m) for m in months]

    data = dict(panel=panel, ml=ml, chr_=chr_, spy=spy, mr=mr, mp=mp, members_g=members_g,
                panel_by_asof=panel_by_asof, ml_by_asof=ml_by_asof, chr_by_asof=chr_by_asof,
                months=months)

    # Focused 2D grid around K=2
    K_grid = [1, 2, 3, 4]
    Q_grid = [0.20, 0.30, 0.40, 0.45, 0.50, 0.60]
    H_grid = [3, 6, 9, 12]
    CAP_grid = [0.34, 0.40, 0.50]

    configs = []
    for k, q, h, c in product(K_grid, Q_grid, H_grid, CAP_grid):
        configs.append(dict(k=k, chr_q=q, hold=h, cap=c))
    print(f"Total configs: {len(configs)}")

    results = []
    for i, cfg in enumerate(configs):
        r = run_one(cfg, data)
        results.append(r)
        if (i + 1) % 20 == 0 or i == len(configs) - 1:
            elapsed = time.time() - t0
            best = max(results, key=lambda x: x["wf_mean_cagr"] or 0)
            print(f"  [{i+1}/{len(configs)}]  elapsed={elapsed:.0f}s  "
                  f"best so far: k={best['k']} q={best['chr_q']:.2f} h={best['hold']} "
                  f"cap={best['cap']:.2f}  wf_mean={best['wf_mean_cagr']*100:.1f}%")

    df = pd.DataFrame(results)
    df.to_csv(AUG / "v5_param_sweep_fine.csv", index=False)
    print(f"\nSaved -> {AUG / 'v5_param_sweep_fine.csv'}")

    # Top 10 by WF mean CAGR
    df_top = df.sort_values("wf_mean_cagr", ascending=False).head(10)
    print(f"\nTop 10 by WF mean CAGR:")
    cols = ["k", "chr_q", "hold", "cap", "cagr_full", "wf_mean_cagr",
            "wf_min_cagr", "sharpe", "max_dd", "wf_n_beats_spy"]
    print(df_top[cols].to_string(index=False))

    winner = df.sort_values("wf_mean_cagr", ascending=False).iloc[0].to_dict()
    (AUG / "v5_param_sweep_fine_winner.json").write_text(json.dumps(winner, indent=2, default=str))

    # Also: top 10 by Sharpe (alternative criterion)
    df_top_sh = df.sort_values("sharpe", ascending=False).head(10)
    print(f"\nTop 10 by Sharpe:")
    print(df_top_sh[cols].to_string(index=False))


if __name__ == "__main__":
    main()
