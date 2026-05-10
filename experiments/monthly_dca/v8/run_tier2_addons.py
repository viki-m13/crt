"""Tier 2 — additive risk-controls on top of the exp_01 winner, plus
staggered ensemble and regime-conditional k. Still uses existing GBM preds.

Output:
  experiments/monthly_dca/v8/results/tier2_addons.csv
  experiments/monthly_dca/v8/results/tier2_addons_top.csv
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys
from dataclasses import replace as _replace

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
    """exp_01 winner template."""
    cfg = V6Config(
        name="x", scorer="ml_3plus6plus1", universe="sp500_pit",
        regime_gate="safer",
        k_normal=1, k_recovery=1, k_bull=1,
        weighting="invvol", hold_months=1, cost_bps=10.0,
    )
    return _replace(cfg, **kwargs) if kwargs else cfg


def variants():
    """Generator of named (name, V6Config) variants."""
    # 0. Baseline (exp_01 winner replayed for reference)
    yield ("00_exp01_winner", base_cfg(name="00_exp01_winner"))

    # ---- Single-knob explorations ----

    # 1. Quality-blend (multiply score by trend_health rank)
    for q in [0.10, 0.25, 0.50]:
        yield (f"10_quality{int(q*100)}", base_cfg(name=f"q{q}", quality_blend=q))

    # 2. Vol penalty (penalise high-vol picks at score level)
    for v in [0.05, 0.10, 0.20]:
        yield (f"11_volp{int(v*100)}", base_cfg(name=f"vp{v}", vol_penalty=v))

    # 3. Half-cash on warning (only fires for 'safer' regime which has
    #    a 'warning' branch — well, 'safer' returns 'warning' on dsma<0)
    yield ("12_halfcash", base_cfg(name="hc", half_cash_warning=True))

    # 4. Cash yield credit (3% T-bills during 4 cash months)
    yield ("13_cashyield3", base_cfg(name="cy3", cash_yield_yr=0.03))

    # 5. Drawdown de-risk (halve gross when running dd <= -X)
    for dd in [0.10, 0.15, 0.20]:
        yield (f"14_dderisk{int(dd*100)}", base_cfg(name=f"dd{dd}", drawdown_de_risk=dd))

    # 6. Trailing stop on portfolio
    for ts in [0.10, 0.15, 0.20, 0.25, 0.30]:
        yield (f"15_ts{int(ts*100)}", base_cfg(name=f"ts{ts}", trailing_stop=ts))

    # 7. Vol target on basket
    for vt in [0.20, 0.25, 0.30]:
        yield (f"16_voltgt{int(vt*100)}", base_cfg(name=f"vt{vt}", vol_target_yr=vt))

    # 8. SPY drawdown continuous scaling
    for s in [0.10, 0.15, 0.20]:
        yield (f"17_spdd{int(s*100)}", base_cfg(name=f"spdd{s}", spy_dd_scale=s, monthly_exposure=True))

    # 9. Min pick momentum filter (drop falling-knife picks)
    for mp in [0.20, 0.30, 0.40, 0.50]:
        yield (f"18_minmom{int(mp*100)}", base_cfg(name=f"mp{mp}", min_pick_mom=mp))

    # 10. Pullback filter
    for pb in [0.30, 0.40, 0.50]:
        yield (f"19_pbfilt{int(pb*100)}", base_cfg(name=f"pb{pb}", pullback_filter=pb))

    # 11. Smart re-entry
    yield ("20_smartreentry", base_cfg(name="sre", smart_reentry=True))

    # 12. Sticky cash 1 / 2 months
    for s in [1, 2]:
        yield (f"21_sticky{s}", base_cfg(name=f"sk{s}", cash_sticky=s))

    # 13. Crash fallback to TLT (long bonds)
    yield ("22_fallback_tlt", base_cfg(
        name="ftlt", crash_fallback="tlt", fallback_ticker="TLT"))

    # 14. Crash fallback to SPY (still in market through gate-down)
    yield ("23_fallback_spy", base_cfg(
        name="fspy", crash_fallback="spy", fallback_ticker="SPY"))

    # 15. Conviction weighting at k=2 and k=3
    for k in [2, 3]:
        yield (f"24_conv_k{k}", base_cfg(
            name=f"conv_k{k}", k_normal=k, k_recovery=k, k_bull=k,
            weighting="conv"))

    # 16. Softmax weighting at k=2 and k=3
    for k in [2, 3]:
        yield (f"25_softmax_k{k}", base_cfg(
            name=f"sm_k{k}", k_normal=k, k_recovery=k, k_bull=k,
            weighting="softmax"))

    # 17. Regime-conditional k: bull=1, normal=2, recovery=3
    yield ("26_kbull1_norm2_rec3", base_cfg(
        name="kbnr", k_bull=1, k_normal=2, k_recovery=3, weighting="invvol"))
    # And bull=1, normal=1, recovery=2
    yield ("27_kbull1_norm1_rec2", base_cfg(
        name="k112", k_bull=1, k_normal=1, k_recovery=2, weighting="invvol"))

    # 18. Combine winners — best stack
    yield ("30_stack_qual_minmom", base_cfg(
        name="stk1", quality_blend=0.25, min_pick_mom=0.30))
    yield ("31_stack_qual_volp_minmom", base_cfg(
        name="stk2", quality_blend=0.25, vol_penalty=0.10, min_pick_mom=0.30))
    yield ("32_stack_qual_minmom_smartre", base_cfg(
        name="stk3", quality_blend=0.25, min_pick_mom=0.30, smart_reentry=True))
    yield ("33_stack_qual_minmom_dderisk", base_cfg(
        name="stk4", quality_blend=0.25, min_pick_mom=0.30, drawdown_de_risk=0.15))
    yield ("34_stack_qual_minmom_ts", base_cfg(
        name="stk5", quality_blend=0.25, min_pick_mom=0.30, trailing_stop=0.20))
    yield ("35_stack_qual_minmom_voltgt", base_cfg(
        name="stk6", quality_blend=0.25, min_pick_mom=0.30, vol_target_yr=0.30))


def main():
    print("[load] panels and SPY features")
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_feats = load_spy_features()
    panel = load_score_panel("ml_3plus6plus1", "sp500_pit", attach_pullback=True)
    print(f"  panel rows={len(panel)}")

    print("[run] tier 2 add-on sweep")
    t0 = time.time()
    rows = []
    cfgs = list(variants())
    for i, (label, cfg) in enumerate(cfgs):
        cfg2 = _replace(cfg, name=label)
        eq = simulate(cfg2, panel, monthly_returns, spy_feats)
        spy_aln = build_spy_aligned(eq, monthly_returns)
        m = evaluate(eq, spy_aln, label)
        rows.append(m)
        if (i + 1) % 10 == 0 or (i + 1) == len(cfgs):
            print(f"  {i+1}/{len(cfgs)} ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tier2_addons.csv", index=False)

    floors = (
        (df["wf_min_cagr"] >= 0.0)
        & (df["wf_mean_sharpe"] >= 1.0)
        & (df["max_dd"] >= -0.50)
        & (df["wf_n_beats_spy"] >= 8)
    )
    df_pass = df[floors].sort_values("wf_mean_cagr", ascending=False)
    df_top = df.sort_values("wf_mean_cagr", ascending=False)

    df_pass.to_csv(OUT / "tier2_addons_passing.csv", index=False)
    df_top.head(40).to_csv(OUT / "tier2_addons_top.csv", index=False)

    cols = ["name", "wf_mean_cagr", "wf_min_cagr", "wf_mean_sharpe",
            "max_dd", "wf_n_pos", "wf_n_beats_spy", "cagr_full"]
    print()
    print("Top 15 unconstrained:")
    print(df_top.head(15)[cols].to_string(index=False))
    print()
    print(f"Passing floors ({len(df_pass)}):")
    if len(df_pass):
        print(df_pass.head(15)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
