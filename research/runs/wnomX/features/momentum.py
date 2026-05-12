"""
Point-in-time momentum and quality features.
All features computed using only price data available at or before asof date.
Leakage guard: this module is a pure function of prices[:asof].
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mom_12_1(prices: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """12-1 month momentum (skip last month to avoid reversal).
    Uses prices at asof (month-end) looking back 12 months, skip last 1 month.
    """
    hist = prices.loc[:asof]
    if len(hist) < 2:
        return pd.Series(dtype=float)
    p_now = hist.iloc[-1]          # price at asof (skip = price 21 days ago)
    p_skip = hist.iloc[-22] if len(hist) >= 22 else hist.iloc[0]
    p_12m = hist.iloc[-253] if len(hist) >= 253 else hist.iloc[0]
    mom = (p_skip / p_12m) - 1
    return mom.replace([np.inf, -np.inf], np.nan)


def mom_6_1(prices: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    hist = prices.loc[:asof]
    if len(hist) < 2:
        return pd.Series(dtype=float)
    p_skip = hist.iloc[-22] if len(hist) >= 22 else hist.iloc[0]
    p_6m = hist.iloc[-127] if len(hist) >= 127 else hist.iloc[0]
    return ((p_skip / p_6m) - 1).replace([np.inf, -np.inf], np.nan)


def vol_12m(prices: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    """Annualised 12-month return volatility."""
    hist = prices.loc[:asof]
    rets = hist.pct_change().dropna()
    if len(rets) < 20:
        return pd.Series(dtype=float)
    window = rets.iloc[-252:]
    return (window.std() * np.sqrt(252)).replace([np.inf, -np.inf], np.nan)


def trend_health(prices: pd.DataFrame, asof: pd.Timestamp, lookback_days: int = 756) -> pd.Series:
    """Fraction of days in lookback where price > 200-day SMA.
    Proxy for 'quality' in absence of fundamentals.
    """
    hist = prices.loc[:asof]
    if len(hist) < 200:
        return pd.Series(dtype=float)
    window = hist.iloc[-lookback_days:]
    result = {}
    for col in window.columns:
        s = window[col].dropna()
        if len(s) < 200:
            result[col] = np.nan
            continue
        ma200 = s.rolling(200).mean()
        above = (s > ma200).dropna()
        result[col] = above.mean() if len(above) > 0 else np.nan
    return pd.Series(result)


def cross_sectional_rank(series: pd.Series) -> pd.Series:
    """Rank-normalize to [0,1]."""
    return series.rank(pct=True)


def build_features_at_asof(
    prices: pd.DataFrame,
    asof: pd.Timestamp,
    universe: list,
    compute_trend_health: bool = True,
) -> pd.DataFrame:
    """
    Build a feature DataFrame for the given universe at a single asof date.
    All features are strictly PIT (no look-ahead).
    """
    prices_pit = prices.loc[:asof, universe].copy()

    feats = pd.DataFrame(index=universe)
    feats.index.name = "ticker"

    m12 = mom_12_1(prices_pit, asof).reindex(universe)
    m6 = mom_6_1(prices_pit, asof).reindex(universe)
    v12 = vol_12m(prices_pit, asof).reindex(universe)

    feats["mom_12_1"] = m12.values
    feats["mom_6_1"] = m6.values
    feats["vol_12m"] = v12.values

    # Composite score: rank of 12-1 mom minus rank of vol (low vol is better)
    feats["mom_rank"] = cross_sectional_rank(feats["mom_12_1"])
    feats["vol_rank"] = cross_sectional_rank(feats["vol_12m"])
    feats["mom_lo_vol"] = feats["mom_rank"] - 0.5 * feats["vol_rank"]

    if compute_trend_health:
        th = trend_health(prices_pit, asof).reindex(universe)
        feats["trend_health"] = th.values
        feats["quality_score"] = feats["mom_rank"] * feats["trend_health"].rank(pct=True)

    feats["asof"] = asof
    feats = feats.reset_index()
    return feats
