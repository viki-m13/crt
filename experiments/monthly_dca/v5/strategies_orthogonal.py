"""Orthogonal strategy library for the v5 ensemble.

Each strategy returns a per-(asof, ticker) score within the PIT S&P 500 panel.
Strategies are designed to be diversified: different feature emphases, different
regime sensitivities.  At deployment, we pick top-K from each strategy and
unite/vote/blend.

Strategies:
  S1. ml_3plus6 — the v3 baseline (multi-horizon ML rank ensemble).
  S2. pure_momentum — 12m absolute momentum, vol-adjusted.
  S3. quality_pullback — long-term winners on 15-50% pullback w/ recovery rate.
  S4. breakout_winner — tight consolidation + breakout strength + multibagger ratio.
  S5. low_vol_quality — high Sharpe + trend health + low max drawdown.
  S6. multibagger_lottery — multibagger ratio + fip + acceleration_2y.
  S7. idio_winner — idiosyncratic momentum (residualised against SPY).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"


def load_panel():
    panel = pd.read_parquet(PIT / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    return panel


def attach_ml(panel):
    ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred", "pred_1m", "pred_3m", "pred_6m"]]
    ml["asof"] = pd.to_datetime(ml["asof"])
    return panel.merge(ml, on=["asof", "ticker"], how="left")


def s1_ml_3plus6(p: pd.DataFrame) -> pd.Series:
    return (p["pred_3m"] + p["pred_6m"]) / 2


def s2_pure_momentum(p: pd.DataFrame) -> pd.Series:
    # 12m mom adjusted by 1y vol; quality filter via trend_health_5y
    score = (p["mom_12_1_xs"] + p["mom_per_unit_vol_12_xs"]) / 2
    # Filter: must be above 200dma
    above200 = (p["d_sma200"] > 0).astype(float)
    return score * above200


def s3_quality_pullback(p: pd.DataFrame) -> pd.Series:
    # Long-term winner on 15-50% pullback (pullback_1y in [-0.50, -0.15])
    score = p["trend_health_5y_xs"] + p["sharpe_5y_xs"] + p["recovery_rate"]
    pullback_ok = ((p["pullback_1y"] >= -0.50) & (p["pullback_1y"] <= -0.15)).astype(float)
    decel = (p["accel"] > 0).astype(float)
    return score * pullback_ok * decel


def s4_breakout_winner(p: pd.DataFrame) -> pd.Series:
    score = (p["tight_consolidation_60_xs"] + p["breakout_strength_60_xs"]
             + p["multibagger_ratio_24m_xs"]) / 3
    above200 = (p["d_sma200"] > 0).astype(float)
    return score * above200


def s5_low_vol_quality(p: pd.DataFrame) -> pd.Series:
    # Tilt toward consistent compounders
    score = (p["sharpe_5y_xs"] + p["trend_health_5y_xs"] - p["max_dd_5y_xs"]) / 3
    # avoid penny stocks: log_price > some threshold via xs rank
    return score * (p["log_price_xs"] > -0.5).astype(float)


def s6_multibagger_lottery(p: pd.DataFrame) -> pd.Series:
    score = (p["multibagger_ratio_24m_xs"]
             + p["fip_score_xs"]
             + p["acceleration_2y_xs"]) / 3
    return score


def s7_idio_winner(p: pd.DataFrame) -> pd.Series:
    score = (p["idio_mom_12_1_xs"] + p["mom_consistency_12m_xs"]) / 2
    above200 = (p["d_sma200"] > 0).astype(float)
    return score * above200


STRATS = {
    "S1_ml_3plus6": s1_ml_3plus6,
    "S2_pure_momentum": s2_pure_momentum,
    "S3_quality_pullback": s3_quality_pullback,
    "S4_breakout_winner": s4_breakout_winner,
    "S5_low_vol_quality": s5_low_vol_quality,
    "S6_multibagger_lottery": s6_multibagger_lottery,
    "S7_idio_winner": s7_idio_winner,
}
