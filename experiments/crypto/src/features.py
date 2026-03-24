"""
Feature Engineering for Crypto TMD-ARC Strategy
================================================
Adapted from the stock TMD-ARC features with crypto-specific modifications:

1. Multi-Timeframe Momentum Dispersion Index (MTMDI)
   - Crypto uses calendar days (24/7 market) → adjusted windows
   - Higher baseline volatility → different z-score normalization

2. Cross-Asset Cascade Score (CACS)
   - BTC as market leader (instead of SPY)
   - Crypto cascade effects are faster (minutes to hours vs days)
   - Adjusted lookback windows for faster information propagation

3. Volatility Regime State (VRS)
   - Crypto vol regimes are more extreme
   - Adjusted thresholds for high/low vol classification

4. Momentum Persistence Ratio (MPR)
   - Same core concept, adjusted for crypto timeframes

IMPORTANT: All features use ONLY past data. No lookahead bias.
"""

import numpy as np
import pandas as pd
from scipy import stats


# Crypto momentum windows (calendar days, 24/7 market)
# Roughly equivalent to stock windows: 5→7, 10→14, 21→30, 63→90, 126→180, 252→365
MOMENTUM_WINDOWS = [7, 14, 30, 90, 180, 365]

# Annualization factor for crypto (365 days/year)
CRYPTO_ANN_FACTOR = 365


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
    Adapted for crypto: uses 365-day rolling normalization
    (crypto trades 24/7, so 365 calendar days ≈ 252 stock trading days).
    """
    if windows is None:
        windows = MOMENTUM_WINDOWS

    rets = compute_returns(close, windows)

    # Z-score each momentum within its own rolling history
    z_scored = pd.DataFrame(index=close.index)
    for col in rets.columns:
        rolling_mean = rets[col].rolling(365, min_periods=180).mean()
        rolling_std = rets[col].rolling(365, min_periods=180).std()
        z_scored[col] = (rets[col] - rolling_mean) / rolling_std.clip(lower=1e-8)

    # MTMDI = standard deviation across z-scored timeframes
    dispersion = z_scored.std(axis=1)

    # Direction: short-term z-scores vs long-term z-scores
    n_fast = len(windows) // 2
    short_term_z = z_scored.iloc[:, :n_fast].mean(axis=1)
    long_term_z = z_scored.iloc[:, n_fast:].mean(axis=1)
    direction = short_term_z - long_term_z

    # Rolling z-score of MTMDI itself
    mtmdi_mean = dispersion.rolling(365, min_periods=180).mean()
    mtmdi_std = dispersion.rolling(365, min_periods=180).std()
    mtmdi_z = (dispersion - mtmdi_mean) / mtmdi_std.clip(lower=1e-8)

    return pd.DataFrame({
        "mtmdi": dispersion,
        "mtmdi_direction": direction,
        "mtmdi_zscore": mtmdi_z,
    }, index=close.index)


def cross_asset_cascade_score(coin_close, btc_close, lookback=14):
    """
    Cross-Asset Cascade Score (CACS) for Crypto
    =============================================
    BTC leads the crypto market. This measures how much an altcoin
    is lagging BTC's move.

    Crypto-specific: shorter lookback (14 days vs 21 for stocks)
    because information propagates faster in crypto markets.
    """
    coin_ret = coin_close.pct_change()
    btc_ret = btc_close.pct_change()

    common = coin_ret.index.intersection(btc_ret.index)
    coin_ret = coin_ret.reindex(common)
    btc_ret = btc_ret.reindex(common)

    # Rolling beta to BTC
    cov = coin_ret.rolling(lookback, min_periods=lookback // 2).cov(btc_ret)
    var = btc_ret.rolling(lookback, min_periods=lookback // 2).var()
    beta = cov / var.clip(lower=1e-10)

    # Cascade gap: expected move vs actual
    btc_move = btc_close.pct_change(lookback).reindex(common)
    coin_move = coin_close.pct_change(lookback).reindex(common)
    expected_move = btc_move * beta
    cascade_gap = expected_move - coin_move

    return pd.DataFrame({
        "cacs": cascade_gap,
        "cacs_beta": beta,
    }, index=common)


def momentum_persistence_ratio(close, fast=7, slow=90):
    """
    Momentum Persistence Ratio (MPR) for Crypto
    =============================================
    Adjusted windows: fast=7 days, slow=90 days (crypto calendar days).
    """
    ret_fast = close.pct_change(fast)
    ret_slow = close.pct_change(slow)

    avg_daily_slow = ret_slow / slow
    avg_daily_fast = ret_fast / fast

    mpr = avg_daily_fast / avg_daily_slow.clip(lower=1e-8)
    mpr = mpr.clip(-10, 10)

    mpr_mean = mpr.rolling(365, min_periods=90).mean()
    mpr_std = mpr.rolling(365, min_periods=90).std()
    mpr_z = (mpr - mpr_mean) / mpr_std.clip(lower=1e-8)

    return pd.DataFrame({
        "mpr": mpr,
        "mpr_zscore": mpr_z,
    }, index=close.index)


def volatility_features(close, volume=None):
    """
    Volatility features adapted for crypto.
    Uses crypto annualization (sqrt(365)).
    """
    log_ret = np.log(close / close.shift(1))

    vol_7 = log_ret.rolling(7).std() * np.sqrt(CRYPTO_ANN_FACTOR)
    vol_30 = log_ret.rolling(30).std() * np.sqrt(CRYPTO_ANN_FACTOR)
    vol_90 = log_ret.rolling(90).std() * np.sqrt(CRYPTO_ANN_FACTOR)

    vol_ratio_short = vol_7 / vol_30.clip(lower=1e-8)
    vol_ratio_long = vol_30 / vol_90.clip(lower=1e-8)

    features = {
        "vol_7d": vol_7,
        "vol_30d": vol_30,
        "vol_90d": vol_90,
        "vol_ratio_7_30": vol_ratio_short,
        "vol_ratio_30_90": vol_ratio_long,
    }

    if volume is not None:
        vol_ma20 = volume.rolling(20).mean()
        vol_relative = volume / vol_ma20.clip(lower=1)
        features["volume_relative"] = vol_relative
        vol_ma7 = volume.rolling(7).mean()
        features["volume_trend"] = vol_ma7 / vol_ma20.clip(lower=1)

    return pd.DataFrame(features, index=close.index)


def drawdown_features(close):
    """Drawdown features for crypto (365-day rolling window)."""
    rolling_max = close.rolling(365, min_periods=30).max()
    drawdown = (close - rolling_max) / rolling_max

    dd_change_7d = drawdown - drawdown.shift(7)
    dd_change_30d = drawdown - drawdown.shift(30)

    rolling_min = close.rolling(365, min_periods=30).min()
    position_in_range = (close - rolling_min) / (rolling_max - rolling_min).clip(lower=1e-8)

    return pd.DataFrame({
        "drawdown_365d": drawdown,
        "dd_change_7d": dd_change_7d,
        "dd_change_30d": dd_change_30d,
        "position_in_range": position_in_range,
    }, index=close.index)


def compute_all_features(coin_close, coin_volume=None, btc_close=None):
    """
    Compute the complete feature set for a single cryptocurrency.
    """
    features = []

    # 1. MTMDI
    features.append(mtmdi(coin_close))

    # 2. Momentum returns
    features.append(compute_returns(coin_close))

    # 3. MPR
    features.append(momentum_persistence_ratio(coin_close))

    # 4. Volatility features
    features.append(volatility_features(coin_close, coin_volume))

    # 5. Drawdown features
    features.append(drawdown_features(coin_close))

    # 6. CACS (if BTC benchmark available and this isn't BTC itself)
    if btc_close is not None:
        features.append(cross_asset_cascade_score(coin_close, btc_close))

    result = pd.concat(features, axis=1)
    result = result.dropna(subset=["mtmdi", "vol_30d"])

    return result


def compute_forward_returns(close, horizons=None):
    """Forward returns for evaluation only. NEVER used as inputs."""
    if horizons is None:
        horizons = [7, 14, 30, 90, 180, 365]

    fwd = {}
    for h in horizons:
        fwd[f"fwd_ret_{h}d"] = close.shift(-h) / close - 1

    return pd.DataFrame(fwd, index=close.index)
