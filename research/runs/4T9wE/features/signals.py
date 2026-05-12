"""
Score functions for stock selection.
All functions: DataFrame (indexed by ticker) -> Series of scores.
Higher score = more preferred. NaN = excluded from selection.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _g(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return df[col].astype(float)


def _rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, na_option="keep")


def _z(s: pd.Series) -> pd.Series:
    mu, sigma = s.mean(), s.std()
    if sigma < 1e-10:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma


# ---------------------------------------------------------------------------
# RUNG 1: Pure 12-1 momentum (Jegadeesh-Titman)
# ---------------------------------------------------------------------------
def momentum_12_1(df: pd.DataFrame) -> pd.Series:
    return _g(df, "mom_12_1")


# ---------------------------------------------------------------------------
# RUNG 2: Momentum + low-vol filter (remove top-33% vol stocks)
# ---------------------------------------------------------------------------
def momentum_lowvol(df: pd.DataFrame) -> pd.Series:
    mom = _g(df, "mom_12_1")
    vol = _g(df, "vol_12m")
    vol_cutoff = vol.quantile(0.67)
    result = mom.copy()
    mask_highvol = vol > vol_cutoff
    result[mask_highvol] = np.nan
    return result


# ---------------------------------------------------------------------------
# RUNG 3: Momentum + quality + low-vol
# ---------------------------------------------------------------------------
def momentum_quality_lowvol(df: pd.DataFrame) -> pd.Series:
    mom = _g(df, "mom_12_1")
    vol = _g(df, "vol_12m")
    trend = _g(df, "trend_health_5y", 0.5)
    sharpe5y = _g(df, "sharpe_5y", 0.0)
    mdd5y = _g(df, "max_dd_5y", -0.5)

    quality = _rank(trend) * 0.4 + _rank(sharpe5y) * 0.4 + _rank(-mdd5y.abs()) * 0.2
    vol_cutoff = vol.quantile(0.67)
    quality_cutoff = quality.quantile(0.33)

    result = _rank(mom) * 0.5 + quality * 0.5
    result[(vol > vol_cutoff) | (quality < quality_cutoff)] = np.nan
    return result


# ---------------------------------------------------------------------------
# RUNG 4 / 5: Composite v1 (z-score composite of multiple signals)
# ---------------------------------------------------------------------------
def composite_v1(df: pd.DataFrame) -> pd.Series:
    mom12 = _z(_rank(_g(df, "mom_12_1")))
    mom6 = _z(_rank(_g(df, "mom_6_1")))
    mom3 = _z(_rank(_g(df, "mom_3")))
    sharpe12 = _z(_rank(_g(df, "sharpe_12m")))
    trend = _z(_rank(_g(df, "trend_health_5y", 0.5)))
    frac50 = _z(_rank(_g(df, "frac_above_50dma_1y", 0.5)))
    cons = _z(_rank(_g(df, "mom_consistency_12m", 0.5)))
    inv_vol = _z(_rank(-_g(df, "vol_12m", 0.3)))

    return (
        mom12 * 1.5 + mom6 * 1.0 + mom3 * 0.5
        + sharpe12 * 1.2 + trend * 0.8 + frac50 * 0.5
        + cons * 0.8 + inv_vol * 1.2
    )


# ---------------------------------------------------------------------------
# Smooth compounder: selects stocks with consistent, low-vol uptrends
# Target: Sharpe improvement through stock-level risk-adjusted selection
# ---------------------------------------------------------------------------
def smooth_compounder(df: pd.DataFrame) -> pd.Series:
    mom12 = _g(df, "mom_12_1")
    mom6 = _g(df, "mom_6_1")
    sharpe12 = _g(df, "sharpe_12m")
    sharpe5y = _g(df, "sharpe_5y", 0.0)
    trend = _g(df, "trend_health_5y", 0.5)
    frac50 = _g(df, "frac_above_50dma_1y", 0.5)
    cons = _g(df, "mom_consistency_12m", 0.5)
    r2 = _g(df, "trend_r2_12m", 0.5)
    vol12 = _g(df, "vol_12m", 0.25)
    vol3m = _g(df, "vol_3m", 0.25)

    score = (
        _z(_rank(sharpe12)) * 2.0
        + _z(_rank(sharpe5y)) * 1.5
        + _z(_rank(mom12)) * 1.0
        + _z(_rank(mom6)) * 0.5
        + _z(_rank(trend)) * 1.0
        + _z(_rank(frac50)) * 0.8
        + _z(_rank(cons)) * 1.0
        + _z(_rank(r2)) * 0.8
        + _z(_rank(-vol12)) * 1.2
        + _z(_rank(-vol3m)) * 0.8
    )
    # Hard filter: must have positive 12m momentum
    score[mom12 <= 0] = np.nan
    return score


# ---------------------------------------------------------------------------
# ML-based score: uses existing GBM pred_3m from PIT panel
# Combined with Sharpe filter for risk-adjusted selection
# ---------------------------------------------------------------------------
def ml_score_v3(df: pd.DataFrame) -> pd.Series:
    """
    Use the existing GBM predictions from the PIT panel as the primary score.
    Falls back to composite_v1 if not available.
    Requires set_date_context() to be called before scoring.
    """
    global _CURRENT_DATE
    if _CURRENT_DATE is None:
        return composite_v1(df)

    from backtest.engine import get_pit_scores_at
    pit = get_pit_scores_at(_CURRENT_DATE)
    if pit.empty:
        return composite_v1(df)

    # Use pred (3+6m combined) as primary score
    common_tickers = df.index.intersection(pit.index)
    if len(common_tickers) < 10:
        return composite_v1(df)

    scores = pd.Series(np.nan, index=df.index)
    for col in ["pred", "pred_3m", "pred_6m"]:
        if col in pit.columns:
            scores.loc[common_tickers] = pit.loc[common_tickers, col].values
            break

    return scores


def ml_plus_sharpe(df: pd.DataFrame) -> pd.Series:
    """
    Combine ML prediction with Sharpe-based quality filter.
    Only invest in stocks where ML agrees AND Sharpe12m > 0.
    """
    ml = ml_score_v3(df)
    sharpe12 = _g(df, "sharpe_12m")
    # Filter: require positive 12m Sharpe
    ml[sharpe12 <= 0] = np.nan
    return ml


def ml_plus_lowvol(df: pd.DataFrame) -> pd.Series:
    """ML score + low-vol filter (exclude top-33% vol)."""
    ml = ml_score_v3(df)
    vol = _g(df, "vol_12m")
    vol_cutoff = vol.quantile(0.67)
    ml[vol > vol_cutoff] = np.nan
    return ml


def ml_plus_smooth(df: pd.DataFrame) -> pd.Series:
    """ML score combined with smooth compounder composite."""
    global _CURRENT_DATE
    if _CURRENT_DATE is None:
        return smooth_compounder(df)

    from backtest.engine import get_pit_scores_at
    pit = get_pit_scores_at(_CURRENT_DATE)
    if pit.empty:
        return smooth_compounder(df)

    common_tickers = df.index.intersection(pit.index)
    if len(common_tickers) < 10:
        return smooth_compounder(df)

    # ML score (normalized)
    ml_raw = pd.Series(np.nan, index=df.index)
    for col in ["pred", "pred_3m"]:
        if col in pit.columns:
            ml_raw.loc[common_tickers] = pit.loc[common_tickers, col].values
            break
    ml_z = _z(ml_raw)

    # Smooth compounder features
    sharpe12 = _z(_rank(_g(df, "sharpe_12m")))
    trend = _z(_rank(_g(df, "trend_health_5y", 0.5)))
    cons = _z(_rank(_g(df, "mom_consistency_12m", 0.5)))
    inv_vol = _z(_rank(-_g(df, "vol_12m", 0.25)))
    r2 = _z(_rank(_g(df, "trend_r2_12m", 0.5)))
    mom12 = _g(df, "mom_12_1")

    score = (
        ml_z * 2.0
        + sharpe12 * 1.5
        + trend * 1.0
        + cons * 1.0
        + inv_vol * 1.2
        + r2 * 0.8
    )
    score[mom12 <= 0] = np.nan
    return score


# Date context for ML functions (set by the backtest engine)
_CURRENT_DATE: pd.Timestamp | None = None


def set_date_context(date: pd.Timestamp | None) -> None:
    global _CURRENT_DATE
    _CURRENT_DATE = date
