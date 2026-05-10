"""Tier 3 — stacked variants on top of `22_fallback_tlt` (current leader)
plus a staggered ensemble harness.

Output:
  experiments/monthly_dca/v8/results/tier3_stacks.csv
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys
from dataclasses import replace as _replace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "monthly_dca" / "v6"))

from lib_engine import (  # noqa: E402
    V2, V6Config, build_spy_aligned, evaluate,
    load_score_panel, load_spy_features, simulate,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def base_cfg(**kwargs) -> V6Config:
    """exp_01 + TLT-fallback template."""
    cfg = V6Config(
        name="x", scorer="ml_3plus6plus1", universe="sp500_pit",
        regime_gate="safer",
        k_normal=1, k_recovery=1, k_bull=1,
        weighting="invvol", hold_months=1, cost_bps=10.0,
        crash_fallback="tlt", fallback_ticker="TLT",
    )
    return _replace(cfg, **kwargs) if kwargs else cfg


def variants():
    yield ("00_tlt_only", base_cfg(name="00"))
    # TLT + add-ons
    yield ("01_tlt_dderisk15", base_cfg(name="01", drawdown_de_risk=0.15))
    yield ("02_tlt_dderisk20", base_cfg(name="02", drawdown_de_risk=0.20))
    yield ("03_tlt_smartreentry", base_cfg(name="03", smart_reentry=True))
    yield ("04_tlt_minmom30", base_cfg(name="04", min_pick_mom=0.30))
    yield ("05_tlt_minmom50", base_cfg(name="05", min_pick_mom=0.50))
    yield ("06_tlt_voltgt30", base_cfg(name="06", vol_target_yr=0.30))
    yield ("07_tlt_ts25", base_cfg(name="07", trailing_stop=0.25))
    yield ("08_tlt_volp10", base_cfg(name="08", vol_penalty=0.10))
    # Hold horizon
    yield ("10_tlt_h2", base_cfg(name="10", hold_months=2))
    yield ("11_tlt_h3", base_cfg(name="11", hold_months=3))
    # Different scorer
    yield ("12_tlt_h6_3plus6", base_cfg(
        name="12", hold_months=6, scorer="ml_3plus6"))
    # Different regime gate
    yield ("13_tlt_combo", base_cfg(name="13", regime_gate="combo"))
    yield ("14_tlt_strict_dd", base_cfg(name="14", regime_gate="strict_dd"))
    yield ("15_tlt_tight", base_cfg(name="15", regime_gate="tight"))
    # k=2 with conviction weight + TLT
    yield ("16_tlt_k2_conv", base_cfg(
        name="16", k_normal=2, k_recovery=2, k_bull=2, weighting="conv"))
    # k=2 invvol + TLT
    yield ("17_tlt_k2_invvol", base_cfg(
        name="17", k_normal=2, k_recovery=2, k_bull=2))
    yield ("18_tlt_k3_invvol", base_cfg(
        name="18", k_normal=3, k_recovery=3, k_bull=3))
    # Triple-stack: TLT + dderisk + smart re-entry
    yield ("20_tlt_dd15_sre", base_cfg(
        name="20", drawdown_de_risk=0.15, smart_reentry=True))
    yield ("21_tlt_dd20_minmom30", base_cfg(
        name="21", drawdown_de_risk=0.20, min_pick_mom=0.30))
    # Volp + min-pick mom + TLT
    yield ("22_tlt_volp5_minmom30", base_cfg(
        name="22", vol_penalty=0.05, min_pick_mom=0.30))
    # SPY-DD continuous scaling + TLT
    yield ("23_tlt_spdd15", base_cfg(
        name="23", spy_dd_scale=0.15, monthly_exposure=True))
    yield ("24_tlt_spdd20", base_cfg(
        name="24", spy_dd_scale=0.20, monthly_exposure=True))
    # Cash yield (additive to TLT-fallback when TLT not allocated — though TLT covers crashes)
    yield ("25_tlt_cy3", base_cfg(name="25", cash_yield_yr=0.03))


def simulate_staggered(cfg: V6Config, panel: pd.DataFrame,
                       monthly_returns: pd.DataFrame, spy_feats: pd.DataFrame,
                       n_legs: int, hold_months_per_leg: int) -> pd.DataFrame:
    """Run `n_legs` parallel sub-baskets, each rebalancing every
    `hold_months_per_leg` months but staggered.

    Combines monthly returns by averaging legs equal-weight.
    Each leg uses cfg as its rule (typically k=1).
    """
    leg_eqs = []
    for offset in range(n_legs):
        cfg_leg = _replace(cfg,
                           name=f"{cfg.name}_leg{offset}",
                           hold_months=hold_months_per_leg)
        # Drop the first `offset` asofs so this leg's rebalance schedule
        # is offset by `offset` months relative to leg 0.
        # We do this by constructing a panel filtered to start `offset`
        # months later — same simulator, shifted entry.
        if offset > 0:
            asofs = sorted(panel["asof"].unique())
            start = asofs[offset]
            sub = panel[panel["asof"] >= start].copy()
            spy_sub = spy_feats[spy_feats.index >= start].copy()
        else:
            sub = panel.copy()
            spy_sub = spy_feats.copy()
        eq_leg = simulate(cfg_leg, sub, monthly_returns, spy_sub)
        leg_eqs.append(eq_leg)

    # Align all legs on date and equal-weight returns
    df_rets = []
    for eq in leg_eqs:
        df_rets.append(eq.set_index("date")["ret_m"].rename("r"))
    rets = pd.concat(df_rets, axis=1)
    rets.columns = [f"leg{i}" for i in range(n_legs)]
    avg_ret = rets.mean(axis=1, skipna=True).fillna(0.0)
    eq_curve = (1.0 + avg_ret).cumprod()
    out = pd.DataFrame({
        "date": eq_curve.index,
        "equity": eq_curve.values,
        "ret_m": avg_ret.values,
        "regime": "stagger",
        "n_picks": n_legs,
        "gross": 1.0,
    })
    return out


def main():
    print("[load]")
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_feats = load_spy_features()
    panel = load_score_panel("ml_3plus6plus1", "sp500_pit", attach_pullback=True)
    panel_3plus6 = load_score_panel("ml_3plus6", "sp500_pit", attach_pullback=True)

    rows = []
    cfgs = list(variants())
    print(f"[run] {len(cfgs)} stacked variants + 6 staggered ensembles")
    t0 = time.time()
    for i, (label, cfg) in enumerate(cfgs):
        cfg2 = _replace(cfg, name=label)
        p = panel_3plus6 if "3plus6" == cfg.scorer else panel
        eq = simulate(cfg2, p, monthly_returns, spy_feats)
        spy_aln = build_spy_aligned(eq, monthly_returns)
        m = evaluate(eq, spy_aln, label)
        rows.append(m)

    # Staggered ensembles — these are *averaging* multiple legs, so cost
    # scales as well; we use the same cfg.cost_bps per leg.
    base_for_stagger = base_cfg()
    for n_legs, hold in [(2, 2), (3, 3), (2, 3), (3, 6), (4, 4), (6, 6)]:
        label = f"stagger_n{n_legs}_h{hold}"
        eq_stag = simulate_staggered(
            base_for_stagger, panel, monthly_returns, spy_feats,
            n_legs=n_legs, hold_months_per_leg=hold)
        spy_aln = build_spy_aligned(eq_stag, monthly_returns)
        m = evaluate(eq_stag, spy_aln, label)
        rows.append(m)

    df = pd.DataFrame(rows).sort_values("wf_mean_cagr", ascending=False)
    df.to_csv(OUT / "tier3_stacks.csv", index=False)

    cols = ["name", "wf_mean_cagr", "wf_min_cagr", "wf_mean_sharpe",
            "max_dd", "wf_n_pos", "wf_n_beats_spy", "cagr_full"]
    print()
    print("All variants by WF mean CAGR:")
    print(df[cols].to_string(index=False))
    print()
    print(f"[done] in {time.time()-t0:.0f}s")

    floors = (
        (df["wf_min_cagr"] >= 0.0)
        & (df["wf_mean_sharpe"] >= 1.0)
        & (df["max_dd"] >= -0.50)
        & (df["wf_n_beats_spy"] >= 8)
    )
    df_pass = df[floors].sort_values("wf_mean_cagr", ascending=False)
    print(f"\n{len(df_pass)} pass floors. Top 10:")
    print(df_pass.head(10)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
