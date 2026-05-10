"""Extended validation of Option C — Chronos filter.

  * Quantile-sensitivity check (q in 0.2..0.6)
  * Per-split decomposition + 2024-05→2025 holdout
  * Walk-forward threshold-selection test (choose q on first 5 splits, test on last 5)
  * Simple leakage shuffle test (random Chronos rank should not produce edge)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
V6DIR = ROOT / "experiments" / "monthly_dca" / "v6"
sys.path.insert(0, str(V6DIR))

from lib_engine import (  # noqa: E402
    V2, PIT, V6Config, build_spy_aligned, evaluate, load_score_panel,
    load_spy_features, simulate, WF_SPLITS, cagr_monthly, sharpe_monthly, maxdd_monthly,
)

OUT = ROOT / "experiments" / "monthly_dca" / "v5" / "cache"
OUT.mkdir(parents=True, exist_ok=True)


def build_filtered_panel(panel: pd.DataFrame, chr_df: pd.DataFrame, q: float) -> pd.DataFrame:
    m = panel.merge(chr_df[["asof", "ticker", "chronos_p70_3m"]], on=["asof", "ticker"], how="left")
    m["chr_p70_rk"] = m.groupby("asof")["chronos_p70_3m"].rank(pct=True)
    return m[m["chr_p70_rk"].fillna(0.0) >= q][["asof", "ticker", "score", "vol_1y"]].copy()


def run_cfg(panel, monthly_returns, spy_feats, cfg):
    eq = simulate(cfg, panel, monthly_returns, spy_feats)
    spy_aln = build_spy_aligned(eq, monthly_returns)
    return evaluate(eq, spy_aln, cfg.name), eq, spy_aln


def main():
    print("[load]")
    panel = load_score_panel("ml_3plus6", "sp500_pit")
    chr_df = pd.read_parquet(PIT / "ml_preds_chronos.parquet")
    chr_df["asof"] = pd.to_datetime(chr_df["asof"])
    panel["asof"] = pd.to_datetime(panel["asof"])
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_feats = load_spy_features()

    cfg_base = V6Config(name="C", scorer="ml_3plus6", regime_gate="tight",
                        k_normal=3, k_recovery=3, k_bull=3, weighting="ew",
                        hold_months=6, cost_bps=10.0)

    # --- 1. Quantile sensitivity sweep ---
    print("\n=== 1. Quantile sensitivity ===")
    qs = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]
    sens_rows = []
    for q in qs:
        if q == 0.0:
            filt = panel
        else:
            filt = build_filtered_panel(panel, chr_df, q)
        cfg = V6Config(**{**cfg_base.__dict__, "name": f"C_q{q}"})
        m, eq, _ = run_cfg(filt, monthly_returns, spy_feats, cfg)
        sens_rows.append({"q": q, **m})
        print(f"  q={q}: CAGR={m['cagr_full']*100:.2f}% Sh={m['sharpe']:.3f} "
              f"WFmean={m['wf_mean_cagr']*100:.2f}% beats={m['wf_n_beats_spy']}/{m['wf_n_splits']}")

    pd.DataFrame(sens_rows).to_csv(OUT / "C_quantile_sensitivity.csv", index=False)

    # --- 2. Per-split with chosen q=0.4 ---
    print("\n=== 2. Per-split with q=0.4 ===")
    filt_04 = build_filtered_panel(panel, chr_df, 0.4)
    cfg = V6Config(**{**cfg_base.__dict__, "name": "C_q0.4"})
    m_C, eq_C, spy_C = run_cfg(filt_04, monthly_returns, spy_feats, cfg)
    cfg_v3 = V6Config(**{**cfg_base.__dict__, "name": "v3"})
    m_v3, eq_v3, spy_v3 = run_cfg(panel, monthly_returns, spy_feats, cfg_v3)

    per_split = []
    for split, lo, hi in WF_SPLITS:
        lo_t, hi_t = pd.Timestamp(lo), pd.Timestamp(hi)
        for label, eq in [("v3", eq_v3), ("C", eq_C)]:
            e = eq[(eq["date"] >= lo_t) & (eq["date"] <= hi_t)]
            if len(e) == 0: continue
            sa = spy_C[(spy_C["date"] >= lo_t) & (spy_C["date"] <= hi_t)]
            r = e["ret_m"].astype(float); sr = sa["spy_ret_m"].astype(float)
            per_split.append({"strategy": label, "split": split,
                              "cagr": cagr_monthly(r), "sharpe": sharpe_monthly(r),
                              "max_dd": maxdd_monthly(r),
                              "edge_pp": (cagr_monthly(r) - cagr_monthly(sr)) * 100})
    df_ps = pd.DataFrame(per_split)
    df_ps.to_csv(OUT / "C_per_split.csv", index=False)
    print(df_ps.pivot_table(index="split", columns="strategy", values="cagr").round(4).to_string())

    # --- 3. 2024-05→2025-12 holdout ---
    print("\n=== 3. Holdout 2024-05 → 2025-12 ===")
    for label, eq in [("v3", eq_v3), ("C", eq_C)]:
        e = eq[(eq["date"] >= "2024-05-01") & (eq["date"] <= "2025-12-31")].reset_index(drop=True)
        spy_aln = build_spy_aligned(e, monthly_returns)
        m = evaluate(e, spy_aln, label)
        print(f"  [{label}] CAGR={m['cagr_full']*100:6.2f}% SPY={m['spy_cagr_full']*100:6.2f}% "
              f"edge={m['edge_full_pp']:5.2f}pp MDD={m['max_dd']*100:6.2f}%")

    # --- 4. Walk-forward threshold selection ---
    # Choose q on first 5 splits (A1-A3, R1, R2), test on last 5 (R3-STRICT)
    print("\n=== 4. Out-of-sample q selection ===")
    train_splits = ["A1", "A2", "A3", "R1_GFC", "R2"]
    test_splits = ["R3", "R4", "R5_COVID", "R6_AI", "STRICT"]
    qcand = [0.2, 0.3, 0.4, 0.5, 0.6]
    train_perf = {}
    for q in qcand:
        if q == 0.0:
            filt = panel
        else:
            filt = build_filtered_panel(panel, chr_df, q)
        cfg = V6Config(**{**cfg_base.__dict__, "name": f"oos_q{q}"})
        m, eq, spy_aln = run_cfg(filt, monthly_returns, spy_feats, cfg)
        train_cagrs = []
        for split, lo, hi in WF_SPLITS:
            if split not in train_splits: continue
            lo_t, hi_t = pd.Timestamp(lo), pd.Timestamp(hi)
            e = eq[(eq["date"] >= lo_t) & (eq["date"] <= hi_t)]
            r = e["ret_m"].astype(float)
            train_cagrs.append(cagr_monthly(r))
        train_perf[q] = np.mean(train_cagrs)
        print(f"  q={q}: train_mean_cagr={train_perf[q]*100:.2f}%")
    best_q = max(train_perf, key=train_perf.get)
    print(f"  → BEST q on TRAIN: {best_q}")
    # Test on holdout splits
    filt_best = build_filtered_panel(panel, chr_df, best_q) if best_q > 0 else panel
    cfg_t = V6Config(**{**cfg_base.__dict__, "name": f"oos_test_q{best_q}"})
    m_t, eq_t, spy_t = run_cfg(filt_best, monthly_returns, spy_feats, cfg_t)
    test_cagrs = []
    for split, lo, hi in WF_SPLITS:
        if split not in test_splits: continue
        lo_t, hi_t = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq_t[(eq_t["date"] >= lo_t) & (eq_t["date"] <= hi_t)]
        r = e["ret_m"].astype(float)
        test_cagrs.append(cagr_monthly(r))
    print(f"  TEST splits ({test_splits}) mean_cagr at best q={best_q}: {np.mean(test_cagrs)*100:.2f}%")
    # Compare to v3 on test splits
    v3_test = []
    for split, lo, hi in WF_SPLITS:
        if split not in test_splits: continue
        lo_t, hi_t = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq_v3[(eq_v3["date"] >= lo_t) & (eq_v3["date"] <= hi_t)]
        r = e["ret_m"].astype(float)
        v3_test.append(cagr_monthly(r))
    print(f"  TEST splits v3: {np.mean(v3_test)*100:.2f}%")
    print(f"  TEST splits lift (C - v3): {(np.mean(test_cagrs) - np.mean(v3_test))*100:.2f}pp")

    # --- 5. Leakage shuffle test ---
    # If we shuffle Chronos ranks within each asof, the filter becomes random
    # — expect performance to revert to ~v3 level.
    print("\n=== 5. Shuffle leakage test (5 seeds) ===")
    rng = np.random.default_rng(42)
    shuffle_cagrs = []
    for seed in range(5):
        m_shuf = panel.merge(chr_df[["asof", "ticker", "chronos_p70_3m"]],
                             on=["asof", "ticker"], how="left").copy()
        # Shuffle chronos_p70_3m within each asof
        for d, idx in m_shuf.groupby("asof").groups.items():
            shuffled = rng.permutation(m_shuf.loc[idx, "chronos_p70_3m"].values)
            m_shuf.loc[idx, "chronos_p70_3m"] = shuffled
        m_shuf["chr_p70_rk"] = m_shuf.groupby("asof")["chronos_p70_3m"].rank(pct=True)
        filt = m_shuf[m_shuf["chr_p70_rk"].fillna(0.0) >= 0.4][["asof", "ticker", "score", "vol_1y"]]
        cfg = V6Config(**{**cfg_base.__dict__, "name": f"C_shuffle_s{seed}"})
        m, _, _ = run_cfg(filt, monthly_returns, spy_feats, cfg)
        shuffle_cagrs.append(m["cagr_full"])
        print(f"  seed={seed}: CAGR={m['cagr_full']*100:.2f}% WFmean={m['wf_mean_cagr']*100:.2f}%")
    print(f"  Shuffle MEAN CAGR: {np.mean(shuffle_cagrs)*100:.2f}% "
          f"(v3 baseline: {m_v3['cagr_full']*100:.2f}%, C real q=0.4: {m_C['cagr_full']*100:.2f}%)")
    print(f"  → If chronos provides real info, real >> shuffle ≈ v3")


if __name__ == "__main__":
    main()
