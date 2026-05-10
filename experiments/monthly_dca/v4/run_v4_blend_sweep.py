"""Honest v4 blend sweep: explore creative ML+factor blends and ensembles
that DON'T rely on the buggy intra-month TP."""
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


def panel_with_custom_score(scorer_fn_name: str) -> pd.DataFrame:
    """Build panel with a custom blended score."""
    panel = pd.read_parquet(PIT / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])

    # Attach v2 ml predictions
    ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred", "pred_1m", "pred_3m", "pred_6m"]]
    ml["asof"] = pd.to_datetime(ml["asof"])
    panel = panel.merge(ml, on=["asof", "ticker"], how="left")

    # Convert ml preds to cross-sectional ranks within the PIT universe each month
    for c in ("pred", "pred_1m", "pred_3m", "pred_6m"):
        panel[f"{c}_rk"] = panel.groupby("asof")[c].rank(pct=True)

    if scorer_fn_name == "ml_3plus6_baseline":
        # Equivalent to v3
        panel["score"] = (panel["pred_3m"] + panel["pred_6m"]) / 2
    elif scorer_fn_name == "ml_136_blend":
        # Blend 1m/3m/6m equally
        panel["score"] = (panel["pred_1m"] + panel["pred_3m"] + panel["pred_6m"]) / 3
    elif scorer_fn_name == "ml_36_qmom":
        # ml_3plus6 + quality + idio momentum
        ml_score = (panel["pred_3m_rk"] + panel["pred_6m_rk"]) / 2
        q = (panel["sharpe_5y_xs"] + panel["trend_health_5y_xs"]
             + panel["quality_score_5y_xs"]) / 3
        idio = panel["idio_mom_12_1_xs"]
        panel["score"] = 0.6 * ml_score + 0.2 * q + 0.2 * (idio + 1) / 2  # rescale idio to [0,1]
    elif scorer_fn_name == "ml_36_idio_winner":
        # ml_3plus6 + idio mom + 12m momentum (pick winners)
        ml_score = (panel["pred_3m_rk"] + panel["pred_6m_rk"]) / 2
        panel["score"] = 0.5 * ml_score + 0.25 * panel["idio_mom_12_1_xs"] + 0.25 * panel["mom_12_1_xs"]
    elif scorer_fn_name == "ml_36_strict_winner":
        # ml_3plus6 hardened: require trend_health > 0
        ml_score = (panel["pred_3m_rk"] + panel["pred_6m_rk"]) / 2
        # penalize stocks below 200dma
        below200_pen = (panel["d_sma200"] < 0).astype(float) * 0.3
        panel["score"] = ml_score - below200_pen
    elif scorer_fn_name == "ml_36_breakout":
        # ml_3plus6 with breakout strength tilt
        ml_score = (panel["pred_3m_rk"] + panel["pred_6m_rk"]) / 2
        panel["score"] = 0.7 * ml_score + 0.3 * panel["breakout_strength_60_xs"]
    elif scorer_fn_name == "ml_36_low_dd":
        # ml_3plus6 with low max_dd_5y bias (more durable winners)
        ml_score = (panel["pred_3m_rk"] + panel["pred_6m_rk"]) / 2
        panel["score"] = 0.7 * ml_score + 0.3 * (1 + panel["max_dd_5y_xs"]) / 2
    elif scorer_fn_name == "ml_36_momconv":
        # ml_3plus6 hardened with momentum/quality stack
        ml_score = (panel["pred_3m_rk"] + panel["pred_6m_rk"]) / 2
        mom = panel["mom_12_1_xs"]
        sharpe = panel["sharpe_5y_xs"]
        trend = panel["trend_health_5y_xs"]
        panel["score"] = 0.50 * ml_score + 0.20 * mom + 0.15 * sharpe + 0.15 * trend
    elif scorer_fn_name == "ml_36_v2_3plus6_avg":
        # average of v2 pred (1m/3m/6m blend) and 3plus6
        panel["score"] = (panel["pred"] + (panel["pred_3m"] + panel["pred_6m"]) / 2) / 2
    elif scorer_fn_name == "ml_36_dispersion_aware":
        # ml_3plus6 / vol_1y (Sharpe-like)
        ml_score = (panel["pred_3m"] + panel["pred_6m"]) / 2
        vol = panel["vol_1y"].clip(lower=0.05)
        panel["score"] = ml_score / vol  # tilt away from very high-vol picks
    elif scorer_fn_name == "ml_36_top_decile_filter":
        # only use ml_3plus6 score, but cross-sectionally filter to top decile
        ml_score = (panel["pred_3m_rk"] + panel["pred_6m_rk"]) / 2
        panel["score"] = ml_score
    else:
        raise ValueError(scorer_fn_name)
    return panel


def main():
    t0 = time.time()
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_features = load_spy_features()
    daily_prices = _load_daily_prices()

    rows = []
    scorers = ["ml_3plus6_baseline", "ml_136_blend", "ml_36_qmom",
               "ml_36_idio_winner", "ml_36_strict_winner", "ml_36_breakout",
               "ml_36_low_dd", "ml_36_momconv", "ml_36_v2_3plus6_avg",
               "ml_36_dispersion_aware"]

    print(f"=== Blend sweep: {len(scorers)} scorers x [k3,h6] [k2,h6] [k3,h12] [k3,h6 invvol] ===", flush=True)
    for scorer in scorers:
        panel = panel_with_custom_score(scorer)
        spy_aligned = build_spy_aligned(panel)
        for cfg in [
            dict(name=f"{scorer}|k3_h6_ew", k_normal=3, k_recovery=3, k_bull=3,
                 weighting="ew", regime_gate="tight", hold_months=6,
                 stop_loss_pct=0.0, take_profit_pct=0.0),
            dict(name=f"{scorer}|k3_h12_ew", k_normal=3, k_recovery=3, k_bull=3,
                 weighting="ew", regime_gate="tight", hold_months=12,
                 stop_loss_pct=0.0, take_profit_pct=0.0),
            dict(name=f"{scorer}|k2_h6_ew", k_normal=2, k_recovery=2, k_bull=2,
                 weighting="ew", regime_gate="tight", hold_months=6,
                 stop_loss_pct=0.0, take_profit_pct=0.0),
            dict(name=f"{scorer}|k5_h6_ew", k_normal=5, k_recovery=5, k_bull=5,
                 weighting="ew", regime_gate="tight", hold_months=6,
                 stop_loss_pct=0.0, take_profit_pct=0.0),
            dict(name=f"{scorer}|k3_h6_invvol", k_normal=3, k_recovery=3, k_bull=3,
                 weighting="invvol", regime_gate="tight", hold_months=6,
                 stop_loss_pct=0.0, take_profit_pct=0.0),
        ]:
            cc = {k: v for k, v in cfg.items() if k != "name"}
            cc.setdefault("score_threshold_pct", 0.0)
            v = Variant(name=cfg["name"], scorer=scorer, cap_per_pick=1.0, **cc)
            try:
                eq = simulate_variant_v4(panel, monthly_returns, spy_features, v, daily_prices=daily_prices)
                m = evaluate(eq, spy_aligned, v.name)
                m["scorer"] = scorer
                m.update(cfg)
                rows.append(m)
                print(f"  {v.name:60s}  CAGR={m['cagr_full']*100:6.2f}%  WF_mean={m['wf_mean_cagr']*100:6.2f}%  "
                      f"WF_min={m['wf_min_cagr']*100:6.2f}%  beats={m['wf_n_beats']}/{m['wf_n_splits']}  "
                      f"Sh={m['sharpe']:.2f}  MDD={m['max_dd']*100:6.1f}%",
                      flush=True)
            except Exception as e:
                print(f"  ! {v.name}: {e}", flush=True)

    df = pd.DataFrame(rows)
    df = df.sort_values("wf_mean_cagr", ascending=False)
    df.to_csv(PIT / "v4_blend_sweep_results.csv", index=False)
    print(f"\n{(time.time()-t0)/60:.1f} min total")
    print(f"\n=== TOP 15 by WF mean CAGR ===")
    cols = ["name", "cagr_full", "wf_mean_cagr", "wf_min_cagr", "wf_n_beats", "wf_n_pos",
            "sharpe", "max_dd", "n_cash"]
    print(df[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
