"""
Feature Engineering for TMD-ARC Strategy
==========================================
Novel features that form the basis of the patentable strategy:

1. Multi-Timeframe Momentum Dispersion Index (MTMDI)
   - Measures disagreement between momentum at different horizons
   - High dispersion = regime transition = predictable resolution

2. Cross-Asset Cascade Score (CACS)
   - Detects information propagation lag from sector leaders to followers
   - Exploits the delay in price discovery across related assets

3. Volatility Regime State (VRS)
   - Hidden Markov Model-based regime detection
   - Adapts strategy behavior to current market state

4. Momentum Persistence Ratio (MPR)
   - Novel measure of whether momentum is accelerating or decelerating
   - Uses the curvature of the cumulative return function

IMPORTANT: All features are computed using ONLY past data at each point.
No lookahead bias - every feature uses a strict lookback window.
"""

import numpy as np
import pandas as pd
from scipy import stats


# === MOMENTUM TIMEFRAMES (in trading days) ===
MOMENTUM_WINDOWS = [5, 10, 21, 63, 126, 252]


def compute_returns(close, windows=None):
    """Compute log returns over multiple lookback windows."""
    if windows is None:
        windows = MOMENTUM_WINDOWS
    rets = {}
    for w in windows:
        rets[f"ret_{w}d"] = np.log(close / close.shift(w))
    return pd.DataFrame(rets, index=close.index)


def mtmdi(close, windows=None):
    """
    Multi-Timeframe Momentum Dispersion Index (MTMDI)
    ==================================================
    NOVEL FEATURE: Measures the cross-timeframe disagreement in momentum signals.

    When short-term momentum strongly disagrees with long-term momentum,
    a predictable resolution follows. This is fundamentally different from
    simple mean reversion because it measures the SHAPE of the momentum
    curve across timeframes, not just levels.

    MTMDI = std(z-scored momentum across timeframes)

    A high MTMDI means timeframes disagree → regime transition likely.
    The DIRECTION of resolution depends on which timeframes dominate.

    Returns DataFrame with:
    - mtmdi: the dispersion index (0 = agreement, high = disagreement)
    - mtmdi_direction: positive = short-term leads up, negative = short-term leads down
    - mtmdi_zscore: rolling z-score of MTMDI for regime-relative comparison
    """
    if windows is None:
        windows = MOMENTUM_WINDOWS

    rets = compute_returns(close, windows)

    # Z-score each momentum within its own rolling history (252-day window)
    # This normalizes for the fact that short-term momentum is naturally
    # more volatile than long-term momentum
    z_scored = pd.DataFrame(index=close.index)
    for col in rets.columns:
        rolling_mean = rets[col].rolling(252, min_periods=126).mean()
        rolling_std = rets[col].rolling(252, min_periods=126).std()
        z_scored[col] = (rets[col] - rolling_mean) / rolling_std.clip(lower=1e-8)

    # MTMDI = standard deviation across z-scored timeframes at each point
    dispersion = z_scored.std(axis=1)

    # Direction: positive if short-term z-scores > long-term z-scores
    # Uses the slope of z-scores across timeframes
    short_term_z = z_scored.iloc[:, :len(windows)//2].mean(axis=1)  # Fast timeframes
    long_term_z = z_scored.iloc[:, len(windows)//2:].mean(axis=1)   # Slow timeframes
    direction = short_term_z - long_term_z

    # Rolling z-score of MTMDI itself (is current dispersion unusual?)
    mtmdi_mean = dispersion.rolling(252, min_periods=126).mean()
    mtmdi_std = dispersion.rolling(252, min_periods=126).std()
    mtmdi_z = (dispersion - mtmdi_mean) / mtmdi_std.clip(lower=1e-8)

    return pd.DataFrame({
        "mtmdi": dispersion,
        "mtmdi_direction": direction,
        "mtmdi_zscore": mtmdi_z,
    }, index=close.index)


def cross_asset_cascade_score(stock_close, leader_close, lookback=21):
    """
    Cross-Asset Cascade Score (CACS)
    ==================================
    NOVEL FEATURE: Measures how much a stock is lagging its sector/market leader.

    Information propagates from leaders to followers with a delay.
    This score quantifies that delay and predicts catch-up moves.

    Method:
    1. Compute rolling correlation between stock and leader at lag 0
    2. Compute rolling correlation at lags 1-5 days
    3. If lagged correlations > contemporaneous → stock is a follower
    4. The "cascade gap" = leader's recent move × (1 - stock's recent move / leader's recent move)

    Returns:
    - cacs: cascade score (positive = stock lagging behind leader's move)
    - cacs_lag: optimal lag in days where correlation peaks
    - cacs_beta: rolling beta to the leader
    """
    stock_ret = stock_close.pct_change()
    leader_ret = leader_close.pct_change()

    # Align indices
    common = stock_ret.index.intersection(leader_ret.index)
    stock_ret = stock_ret.reindex(common)
    leader_ret = leader_ret.reindex(common)

    # Rolling beta to leader
    cov = stock_ret.rolling(lookback, min_periods=lookback//2).cov(leader_ret)
    var = leader_ret.rolling(lookback, min_periods=lookback//2).var()
    beta = cov / var.clip(lower=1e-10)

    # Find optimal lag: at which lag does correlation peak?
    max_lag = 5
    lag_corrs = {}
    for lag in range(0, max_lag + 1):
        lagged = leader_ret.shift(lag)
        lag_corrs[lag] = stock_ret.rolling(
            lookback, min_periods=lookback//2
        ).corr(lagged)

    lag_df = pd.DataFrame(lag_corrs)
    # Handle rows that are all NaN (early warmup period)
    # Use apply to safely handle all-NaN rows
    def safe_idxmax(row):
        if row.isna().all():
            return 0
        return row.idxmax()
    optimal_lag = lag_df.apply(safe_idxmax, axis=1)

    # Cascade gap: how much has the stock NOT yet responded to leader's move?
    leader_move = leader_close.pct_change(lookback).reindex(common)
    stock_move = stock_close.pct_change(lookback).reindex(common)
    expected_move = leader_move * beta
    cascade_gap = expected_move - stock_move

    return pd.DataFrame({
        "cacs": cascade_gap,
        "cacs_lag": optimal_lag,
        "cacs_beta": beta,
    }, index=common)


def momentum_persistence_ratio(close, fast=5, slow=63):
    """
    Momentum Persistence Ratio (MPR)
    ==================================
    NOVEL FEATURE: Measures the curvature of cumulative returns.

    Instead of just measuring momentum (first derivative of price),
    this measures whether momentum is accelerating or decelerating
    (second derivative).

    MPR > 1: momentum is accelerating (trend strengthening)
    MPR < 1: momentum is decelerating (trend weakening)
    MPR ≈ 1: steady state

    Method:
    - Compute the ratio of recent momentum to older momentum
    - Use geometric segments to capture the curvature

    Returns:
    - mpr: the persistence ratio
    - mpr_zscore: z-scored MPR for cross-sectional comparison
    """
    ret_fast = close.pct_change(fast)
    ret_slow = close.pct_change(slow)

    # Momentum of the most recent 'fast' period vs the average over 'slow'
    avg_daily_slow = ret_slow / slow
    avg_daily_fast = ret_fast / fast

    # Ratio (with protection against division by tiny numbers)
    mpr = avg_daily_fast / avg_daily_slow.clip(lower=1e-8)
    # Clamp to reasonable range
    mpr = mpr.clip(-10, 10)

    # Z-score
    mpr_mean = mpr.rolling(252, min_periods=63).mean()
    mpr_std = mpr.rolling(252, min_periods=63).std()
    mpr_z = (mpr - mpr_mean) / mpr_std.clip(lower=1e-8)

    return pd.DataFrame({
        "mpr": mpr,
        "mpr_zscore": mpr_z,
    }, index=close.index)


def volatility_features(close, volume=None):
    """
    Volatility and volume-based features for regime detection.
    """
    log_ret = np.log(close / close.shift(1))

    # Realized volatility at multiple horizons
    vol_5 = log_ret.rolling(5).std() * np.sqrt(252)
    vol_21 = log_ret.rolling(21).std() * np.sqrt(252)
    vol_63 = log_ret.rolling(63).std() * np.sqrt(252)

    # Volatility ratio (term structure of vol)
    vol_ratio_short = vol_5 / vol_21.clip(lower=1e-8)
    vol_ratio_long = vol_21 / vol_63.clip(lower=1e-8)

    features = {
        "vol_5d": vol_5,
        "vol_21d": vol_21,
        "vol_63d": vol_63,
        "vol_ratio_5_21": vol_ratio_short,
        "vol_ratio_21_63": vol_ratio_long,
    }

    # Volume features
    if volume is not None:
        vol_ma20 = volume.rolling(20).mean()
        vol_relative = volume / vol_ma20.clip(lower=1)
        features["volume_relative"] = vol_relative

        # Volume trend (are volumes expanding or contracting?)
        vol_ma5 = volume.rolling(5).mean()
        features["volume_trend"] = vol_ma5 / vol_ma20.clip(lower=1)

    return pd.DataFrame(features, index=close.index)


def drawdown_features(close):
    """
    Drawdown-based features (complementary to existing CRT approach,
    but used differently in TMD-ARC context).
    """
    # Rolling max (252-day)
    rolling_max = close.rolling(252, min_periods=21).max()
    drawdown = (close - rolling_max) / rolling_max

    # Rate of drawdown change (is the drawdown deepening or recovering?)
    dd_change_5d = drawdown - drawdown.shift(5)
    dd_change_21d = drawdown - drawdown.shift(21)

    # Distance from 52-week low
    rolling_min = close.rolling(252, min_periods=21).min()
    position_in_range = (close - rolling_min) / (rolling_max - rolling_min).clip(lower=1e-8)

    return pd.DataFrame({
        "drawdown_252d": drawdown,
        "dd_change_5d": dd_change_5d,
        "dd_change_21d": dd_change_21d,
        "position_in_52w_range": position_in_range,
    }, index=close.index)


def compute_all_features(stock_close, stock_volume=None, market_close=None):
    """
    Compute the complete feature set for a single stock.

    Parameters:
    - stock_close: pd.Series of adjusted close prices
    - stock_volume: pd.Series of volume (optional)
    - market_close: pd.Series of market benchmark close (e.g., SPY)

    Returns: pd.DataFrame with all features, NaN-trimmed.
    """
    features = []

    # 1. MTMDI (core novel feature)
    features.append(mtmdi(stock_close))

    # 2. Momentum returns at various horizons
    features.append(compute_returns(stock_close))

    # 3. Momentum Persistence Ratio
    features.append(momentum_persistence_ratio(stock_close))

    # 4. Volatility features
    features.append(volatility_features(stock_close, stock_volume))

    # 5. Drawdown features
    features.append(drawdown_features(stock_close))

    # 6. Cross-Asset Cascade Score (if market benchmark available)
    if market_close is not None:
        features.append(cross_asset_cascade_score(stock_close, market_close))

    # Combine all features
    result = pd.concat(features, axis=1)

    # Drop rows where core features are missing
    # (initial warmup period where rolling windows aren't filled)
    result = result.dropna(subset=["mtmdi", "vol_21d"])

    return result


def compute_forward_returns(close, horizons=None):
    """
    Compute forward returns for signal evaluation.
    IMPORTANT: These are targets, NOT features. Never used as inputs.
    """
    if horizons is None:
        horizons = [5, 10, 21, 63, 126, 252]

    fwd = {}
    for h in horizons:
        fwd[f"fwd_ret_{h}d"] = close.shift(-h) / close - 1

    return pd.DataFrame(fwd, index=close.index)
