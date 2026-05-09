"""Ensemble strategies: combine multiple signals from many strategies.

Lessons from sweep_alpha results:
- Top strategies: quality_pullback, explosive_winners, rank_intersect, consensus_top_decile
- All converge around 17-18% CAGR on 2002-2024 window
- Different strategies work in different regimes
- High win-rate strategies (rank_intersect, consensus_top_decile) have very high consistency

These ensembles try to extract MORE alpha by combining signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_engine import Strategy
from experiments.monthly_dca.strategies_fast import (
    _safe, _z, quality_pullback, explosive_winners, pullback_in_winner,
    blended_pullback_momentum, dual_momentum,
)
from experiments.monthly_dca.strategies_alpha import (
    nova_star, persistent_winner, smooth_trend_compounder, multibagger_engine,
    consensus_top_decile, alpha_intersect, rank_intersect,
)


# ---------------------------------------------------------------------------
def grand_ensemble(df: pd.DataFrame) -> pd.Series:
    """Average percentile rank across our 8 best strategies.

    Picks stocks that are highly ranked across MANY independent signals.
    """
    sigs = [
        quality_pullback(df),
        explosive_winners(df),
        pullback_in_winner(df),
        rank_intersect(df),
        consensus_top_decile(df),
        nova_star(df),
        dual_momentum(df),
        persistent_winner(df),
    ]
    ranks = [s.rank(pct=True, na_option="keep") for s in sigs]
    df_r = pd.concat(ranks, axis=1)
    n_pass = df_r.notna().sum(axis=1)
    avg = df_r.mean(axis=1, skipna=True)
    high_count = (df_r > 0.85).sum(axis=1)
    score = avg + 0.05 * high_count
    return score.where(n_pass >= 3)


def diamond_ensemble(df: pd.DataFrame) -> pd.Series:
    """Pick stocks that pass the gate of MULTIPLE strategies.

    Quality bar: must be in top 3 of at least 3 different strategies' picks.
    """
    sigs = {
        "qp": quality_pullback(df),
        "ew": explosive_winners(df),
        "pin": pullback_in_winner(df),
        "ri": rank_intersect(df),
        "ctd": consensus_top_decile(df),
        "ns": nova_star(df),
    }
    # For each strategy, identify top-N (e.g. top-10) stocks
    top_n = 10
    in_top = {}
    for name, s in sigs.items():
        s_clean = s.dropna()
        if s_clean.empty:
            in_top[name] = pd.Series(False, index=df.index)
            continue
        thresh = s_clean.nlargest(top_n).iloc[-1]
        in_top[name] = s.fillna(-np.inf) >= thresh
    df_top = pd.concat(in_top.values(), axis=1, keys=in_top.keys())
    n_overlap = df_top.sum(axis=1)
    score = n_overlap.astype(float)
    return score.where(n_overlap >= 2)


def strategy_rotation(df: pd.DataFrame) -> pd.Series:
    """Pick from 'best regime strategy' based on SPY conditions.

    Bull: explosive_winners (momentum)
    Recovery: pullback_in_winner (deep value rebound)
    Sideways/normal: quality_pullback (compounders on dip)
    """
    if "SPY" not in df.index:
        return quality_pullback(df)
    spy_dsma = float(df.loc["SPY", "d_sma200"]) if "d_sma200" in df.columns else 0.0
    spy_rsi = float(df.loc["SPY", "rsi_14"]) if "rsi_14" in df.columns else 50.0
    spy_mom = float(df.loc["SPY", "mom_12_1"]) if "mom_12_1" in df.columns else 0.0

    # Bear: hold cash (return all-NaN)
    if spy_dsma < -0.10 and spy_rsi < 35:
        return pd.Series(np.nan, index=df.index)
    # Recovery: SPY just reclaimed 200dma
    if -0.05 < spy_dsma < 0.03:
        return pullback_in_winner(df)
    # Strong bull (SPY mom > 15%)
    if spy_mom > 0.15:
        return explosive_winners(df)
    # Default: quality_pullback
    return quality_pullback(df)


def best_of_top4(df: pd.DataFrame) -> pd.Series:
    """For each ticker, take MAX of percentile rank across 4 best strategies.

    A stock with high score in ANY of {quality_pullback, explosive_winners,
    pullback_in_winner, consensus_top_decile} qualifies.
    """
    sigs = [
        quality_pullback(df).rank(pct=True, na_option="keep"),
        explosive_winners(df).rank(pct=True, na_option="keep"),
        pullback_in_winner(df).rank(pct=True, na_option="keep"),
        consensus_top_decile(df).rank(pct=True, na_option="keep"),
    ]
    df_r = pd.concat(sigs, axis=1)
    return df_r.max(axis=1)


def best_of_top4_intersect(df: pd.DataFrame) -> pd.Series:
    """Stocks that score high in MULTIPLE of the top-4 strategies."""
    sigs = [
        quality_pullback(df).rank(pct=True, na_option="keep"),
        explosive_winners(df).rank(pct=True, na_option="keep"),
        pullback_in_winner(df).rank(pct=True, na_option="keep"),
        consensus_top_decile(df).rank(pct=True, na_option="keep"),
    ]
    df_r = pd.concat(sigs, axis=1)
    n_pass = df_r.notna().sum(axis=1)
    avg = df_r.mean(axis=1, skipna=True)
    in_top10 = (df_r > 0.90).sum(axis=1)
    score = avg + 0.10 * in_top10
    return score.where(n_pass >= 2)


def quality_pullback_amped(df: pd.DataFrame) -> pd.Series:
    """quality_pullback with new alpha features added.

    Boost factors:
      - rs_12m_spy (relative strength)
      - trend_r2_12m (smooth trend)
      - mom_consistency_12m (consistent up months)
    """
    base = quality_pullback(df)
    rs12 = _safe(df, "rs_12m_spy")
    r2 = _safe(df, "trend_r2_12m")
    cons = _safe(df, "mom_consistency_12m", 0.5)
    sharpe = _safe(df, "sharpe_12m")
    boost = (
        0.4 * _z(rs12).clip(-1, 4)
        + 0.3 * _z(r2).clip(-1, 4)
        + 0.3 * _z(cons).clip(-1, 4)
        + 0.3 * _z(sharpe).clip(-1, 4)
    )
    return base + boost


def explosive_winners_amped(df: pd.DataFrame) -> pd.Series:
    """explosive_winners with alpha features."""
    base = explosive_winners(df)
    rs12 = _safe(df, "rs_12m_spy")
    r2 = _safe(df, "trend_r2_12m")
    tail = _safe(df, "tail_ratio_24m", 1.0)
    mom_acc = _safe(df, "mom_accel", 0)
    boost = (
        0.4 * _z(rs12).clip(-1, 4)
        + 0.3 * _z(r2).clip(-1, 4)
        + 0.3 * _z(tail).clip(-1, 4)
        + 0.3 * _z(mom_acc).clip(-1, 4)
    )
    return base + boost


def pullback_in_winner_amped(df: pd.DataFrame) -> pd.Series:
    """pullback_in_winner with alpha features."""
    base = pullback_in_winner(df)
    rs3 = _safe(df, "rs_3m_spy")
    rs12 = _safe(df, "rs_12m_spy")
    r2 = _safe(df, "trend_r2_12m")
    accel = _safe(df, "accel")
    cons = _safe(df, "mom_consistency_12m", 0.5)
    boost = (
        0.4 * _z(rs3).clip(-1, 4)
        + 0.3 * _z(rs12).clip(-1, 4)
        + 0.3 * _z(r2).clip(-1, 4)
        + 0.4 * _z(accel).clip(-1, 4)
        + 0.3 * _z(cons).clip(-1, 4)
    )
    return base + boost


def strategy_rotation_v2(df: pd.DataFrame) -> pd.Series:
    """Improved regime rotation:
    - Bear (SPY < -8% from 200dma OR RSI < 32): SKIP MONTH (no buy)
    - Recovery (SPY just reclaimed 200dma): pullback_in_winner_amped (deep value rebound)
    - Strong bull (SPY mom > 15%): explosive_winners_amped (momentum)
    - Default: grand_ensemble (consensus across signals)
    """
    if "SPY" not in df.index:
        return grand_ensemble(df)
    spy_dsma = float(df.loc["SPY", "d_sma200"]) if "d_sma200" in df.columns else 0.0
    spy_rsi = float(df.loc["SPY", "rsi_14"]) if "rsi_14" in df.columns else 50.0
    spy_mom = float(df.loc["SPY", "mom_12_1"]) if "mom_12_1" in df.columns else 0.0

    # Hard cash regime: bear market
    if spy_dsma < -0.08 or spy_rsi < 32:
        return pd.Series(np.nan, index=df.index)
    # Recovery: SPY just reclaimed 200dma (-5% to +3%)
    if -0.05 < spy_dsma < 0.03:
        return pullback_in_winner_amped(df)
    # Strong bull
    if spy_mom > 0.15:
        return explosive_winners_amped(df)
    # Default: grand_ensemble (uses 8-strategy consensus)
    return grand_ensemble(df)


def all_ensemble_strategies(top_k: int = 5) -> list[Strategy]:
    return [
        Strategy("grand_ensemble", grand_ensemble, top_k=top_k),
        Strategy("diamond_ensemble", diamond_ensemble, top_k=top_k),
        Strategy("strategy_rotation", strategy_rotation, top_k=top_k),
        Strategy("strategy_rotation_v2", strategy_rotation_v2, top_k=top_k),
        Strategy("best_of_top4", best_of_top4, top_k=top_k),
        Strategy("best_of_top4_intersect", best_of_top4_intersect, top_k=top_k),
        Strategy("quality_pullback_amped", quality_pullback_amped, top_k=top_k),
        Strategy("explosive_winners_amped", explosive_winners_amped, top_k=top_k),
        Strategy("pullback_in_winner_amped", pullback_in_winner_amped, top_k=top_k),
    ]
