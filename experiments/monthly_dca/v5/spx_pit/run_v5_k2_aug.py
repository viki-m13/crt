"""Phase 7c: K=2 v5 — both lump-sum and crash-aware-staggered — on augmented PIT.

The parameter sweep flagged K=2 (instead of deployed K=3) as a clear
improvement on the PIT-corrected panel:
  - Full CAGR  49.21 % (vs deployed K=3's 32.92 %)
  - WF mean    49.39 % (vs 32.68 %)
  - Sharpe     1.04   (vs 0.92)
  - Max DD     -52.5 %  (matched K=3)
  - WF beats SPY 10/10 (vs 8/10)

Run K=2 in both the lump-sum and crash-aware-staggered modes for the
deployment-ready comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Monkey-patch K_PICKS in both modules
import run_v5_winner_aug as lump_mod
import run_v5_staggered_aug as stag_mod
import run_v5_staggered_crash_aware_aug as ca_mod

lump_mod.K_PICKS = 2
stag_mod.K_PICKS = 2
ca_mod.K_PICKS = 2

from run_v5_winner_aug import main as lump_main  # noqa: E402
from run_v5_staggered_aug import (  # noqa: E402
    AUG, PIT,
    classify_regime_tight, load_spy_features,
    pick_v5, _tranche_value,
    compute_returns_from_equity, yearly_table, walkforward_table,
    HOLD_MONTHS, COST_BPS,
)
from run_v5_staggered_crash_aware_aug import run_staggered_crash_aware  # noqa: E402


def main():
    print("=" * 64)
    print("Phase 7c: K=2 v5 — lump-sum and crash-aware-staggered")
    print("=" * 64)

    # Load shared data
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

    print("\n[1] Running K=2 crash-aware-staggered ...")
    eq_ca, tr_ca = run_staggered_crash_aware(panel, ml, chr_, mr, mp, members_g, spy)
    eq_ca.to_csv(AUG / "v5_k2_staggered_ca_equity.csv", index=False)
    if len(tr_ca):
        tr_ca.to_csv(AUG / "v5_k2_staggered_ca_tranches.csv", index=False)
    yr_ca = yearly_table(eq_ca, mr)
    yr_ca.to_csv(AUG / "v5_k2_staggered_ca_yearly.csv", index=False)
    wf_ca = walkforward_table(eq_ca, mr)
    wf_ca.to_csv(AUG / "v5_k2_staggered_ca_walkforward.csv", index=False)
    rets_ca = compute_returns_from_equity(eq_ca, mr)
    n_months = len(rets_ca)
    cagr_ca = (1 + rets_ca).prod() ** (12.0 / n_months) - 1
    spy_full = (1 + mr["SPY"].loc[rets_ca.index[0]:rets_ca.index[-1]].dropna()).prod() ** (12.0 / n_months) - 1
    sharpe_ca = (rets_ca.mean() / max(rets_ca.std(), 1e-9)) * np.sqrt(12)
    ec = (1 + rets_ca).cumprod()
    peak = ec.cummax()
    mdd_ca = float(((ec - peak) / peak).min())
    summary_ca = {
        "variant_name": "v5_K2_staggered_ca",
        "panel": "augmented_PIT",
        "n_months": int(n_months),
        "cagr_full": float(cagr_ca),
        "spy_cagr_full": float(spy_full),
        "edge_full_pp": float((cagr_ca - spy_full) * 100),
        "sharpe": float(sharpe_ca),
        "max_dd": float(mdd_ca),
        "n_tranches_closed": int(len(tr_ca)),
        "wf_mean_cagr": float(wf_ca["cagr"].mean()) if len(wf_ca) else None,
        "wf_median_cagr": float(wf_ca["cagr"].median()) if len(wf_ca) else None,
        "wf_min_cagr": float(wf_ca["cagr"].min()) if len(wf_ca) else None,
        "wf_mean_edge_pp": float(wf_ca["edge_pp"].mean()) if len(wf_ca) else None,
        "wf_n_positive": int((wf_ca["cagr"] > 0).sum()) if len(wf_ca) else 0,
        "wf_n_beats_spy": int((wf_ca["cagr"] > wf_ca["spy_cagr"]).sum()) if len(wf_ca) else 0,
        "wf_n_splits": int(len(wf_ca)),
    }
    (AUG / "v5_k2_staggered_ca_summary.json").write_text(json.dumps(summary_ca, indent=2))
    print(f"  crash-aware K=2: cagr={cagr_ca*100:.2f}%  sharpe={sharpe_ca:.2f}  "
          f"maxDD={mdd_ca*100:.2f}%  WF_mean={summary_ca['wf_mean_cagr']*100:.2f}%  "
          f"beats={summary_ca['wf_n_beats_spy']}/{summary_ca['wf_n_splits']}")

    print("\n[2] Running K=2 lump-sum inline (uses sweep_v5_aug.run_one) ...")
    from sweep_v5_aug import run_one as sweep_run_one
    panel_by_asof = {a: g for a, g in panel.groupby("asof")}
    ml_by_asof = {a: g for a, g in ml.groupby("asof")}
    chr_by_asof = {a: g for a, g in chr_.groupby("asof")}
    months = sorted(set(panel["asof"]).intersection(set(spy.index)))
    months = [pd.Timestamp(m) for m in months]
    data = dict(panel=panel, ml=ml, chr_=chr_, spy=spy, mr=mr, mp=mp, members_g=members_g,
                panel_by_asof=panel_by_asof, ml_by_asof=ml_by_asof, chr_by_asof=chr_by_asof,
                months=months)
    lump_k2_summary = sweep_run_one({"k": 2, "chr_q": 0.45, "hold": 6, "cap": 0.40,
                                     "scorer": "ml_3plus6"}, data)
    (AUG / "v5_k2_lump_summary.json").write_text(json.dumps(lump_k2_summary, indent=2))
    print(f"  lump-sum K=2: cagr={lump_k2_summary['cagr_full']*100:.2f}%  "
          f"sharpe={lump_k2_summary['sharpe']:.2f}  "
          f"maxDD={lump_k2_summary['max_dd']*100:.2f}%  "
          f"WF_mean={lump_k2_summary['wf_mean_cagr']*100:.2f}%  "
          f"beats={lump_k2_summary['wf_n_beats_spy']}/{lump_k2_summary['wf_n_splits']}")

    print(f"\n[3] Comparison (K=2 vs deployed K=3 vs all-prior on augmented PIT):")
    cmp = {
        "deployed_k3_lump": json.loads((AUG / "v5_winner_summary.json").read_text()),
        "deployed_k3_stag_ca": json.loads((AUG / "v5_staggered_ca_summary.json").read_text()),
        "new_k2_lump": lump_k2_summary,
        "new_k2_stag_ca": summary_ca,
    }
    print(f"{'Config':<26}{'CAGR':>10}{'WF_mean':>10}{'Sharpe':>8}{'MaxDD':>8}{'beats':>8}")
    for name, s in cmp.items():
        wf_mean = s.get("wf_mean_cagr") or 0
        beats = f"{s.get('wf_n_beats_spy', 0)}/{s.get('wf_n_splits', 10)}"
        print(f"{name:<26}{s.get('cagr_full', 0)*100:>9.2f}% {wf_mean*100:>9.2f}% "
              f"{s.get('sharpe', 0):>7.2f} {s.get('max_dd', 0)*100:>7.2f}% {beats:>8}")

    (AUG / "v5_k2_vs_deployed.json").write_text(json.dumps(cmp, indent=2))
    print(f"\n[saved] {AUG / 'v5_k2_vs_deployed.json'}")


if __name__ == "__main__":
    main()
