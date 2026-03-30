#!/usr/bin/env python3
"""
30+ novel signal generators for sector ETF strategy research.
Each function returns a weights DataFrame (date x ticker).
All signals use ONLY past data - no lookahead.
"""
import numpy as np
import pandas as pd
from .engine import SECTOR_ETFS, BENCHMARK


def _rank_normalize(df):
    """Cross-sectional rank normalize each row to [0, 1]."""
    return df.rank(axis=1, pct=True)


def _top_n_equal_weight(scores, n=3, gate=None):
    """
    From a scores DataFrame, pick top N sectors each day.
    gate: Series of bool, True = invest, False = cash.
    Returns weights DataFrame.
    """
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for date in scores.index:
        row = scores.loc[date].dropna()
        if len(row) == 0:
            continue
        if gate is not None and date in gate.index and not gate.loc[date]:
            continue
        top = row.nlargest(n).index
        weights.loc[date, top] = 1.0 / n
    return weights


def _momentum(close_df, lookback):
    """Simple momentum: return over lookback period."""
    return close_df.pct_change(lookback)


# ============================================================
# SIGNAL 1: Classic Momentum (baseline)
# Top 3 sectors by 63-day momentum
# ============================================================
def signal_momentum_63d(close_df, **kw):
    mom = _momentum(close_df[SECTOR_ETFS], 63)
    return _top_n_equal_weight(mom, n=3)


# ============================================================
# SIGNAL 2: SMA-Gated Momentum
# Top 3 sectors when SPY > SMA50, else cash
# ============================================================
def signal_sma_gated_mom(close_df, sma_period=50, **kw):
    spy = close_df[BENCHMARK]
    sma = spy.rolling(sma_period).mean()
    gate = spy > sma
    mom = _momentum(close_df[SECTOR_ETFS], 63)
    return _top_n_equal_weight(mom, n=3, gate=gate)


# ============================================================
# SIGNAL 3: Volatility-Adjusted Momentum
# Rank by momentum / volatility (Sharpe-like)
# ============================================================
def signal_vol_adj_momentum(close_df, **kw):
    rets = close_df[SECTOR_ETFS].pct_change()
    mom = rets.rolling(63).mean() * 252
    vol = rets.rolling(63).std() * np.sqrt(252)
    score = mom / vol.clip(lower=0.01)
    return _top_n_equal_weight(score, n=3)


# ============================================================
# SIGNAL 4: Momentum + Trend Confirmation
# Momentum > 0 AND price > SMA50
# ============================================================
def signal_mom_trend_confirm(close_df, **kw):
    mom = _momentum(close_df[SECTOR_ETFS], 63)
    sma = close_df[SECTOR_ETFS].rolling(50).mean()
    above = close_df[SECTOR_ETFS] > sma
    score = mom.where(above, -999)
    return _top_n_equal_weight(score, n=3)


# ============================================================
# SIGNAL 5: Dual Momentum (absolute + relative)
# Only sectors with positive absolute momentum, ranked by relative
# ============================================================
def signal_dual_momentum(close_df, **kw):
    mom = _momentum(close_df[SECTOR_ETFS], 63)
    spy_mom = _momentum(close_df[[BENCHMARK]], 63)[BENCHMARK]
    # Absolute: sector must be positive
    # Relative: must beat SPY
    score = mom.copy()
    for col in score.columns:
        score.loc[mom[col] < 0, col] = -999
        score.loc[mom[col] < spy_mom, col] = -999
    gate = spy_mom > 0
    return _top_n_equal_weight(score, n=3, gate=gate)


# ============================================================
# SIGNAL 6: Mean Reversion (short-term reversal)
# Bottom 3 sectors by 5-day return (contrarian)
# ============================================================
def signal_mean_reversion_5d(close_df, **kw):
    mom = _momentum(close_df[SECTOR_ETFS], 5)
    score = -mom  # reverse: worst recent = best score
    return _top_n_equal_weight(score, n=3)


# ============================================================
# SIGNAL 7: Momentum + Mean Reversion Combo
# Long-term momentum + short-term reversal
# ============================================================
def signal_mom_reversion_combo(close_df, **kw):
    long_mom = _rank_normalize(_momentum(close_df[SECTOR_ETFS], 63))
    short_rev = _rank_normalize(-_momentum(close_df[SECTOR_ETFS], 5))
    score = 0.6 * long_mom + 0.4 * short_rev
    return _top_n_equal_weight(score, n=3)


# ============================================================
# SIGNAL 8: Sector Dispersion Timing
# When cross-sector dispersion is HIGH, go to top momentum
# When LOW, equal weight all (mean reversion regime)
# ============================================================
def signal_dispersion_timing(close_df, **kw):
    rets_21d = close_df[SECTOR_ETFS].pct_change(21)
    dispersion = rets_21d.std(axis=1)
    disp_z = (dispersion - dispersion.rolling(252).mean()) / dispersion.rolling(252).std().clip(lower=1e-8)
    mom = _momentum(close_df[SECTOR_ETFS], 63)
    weights = pd.DataFrame(0.0, index=close_df.index, columns=SECTOR_ETFS)
    for date in close_df.index:
        if date not in disp_z.index or pd.isna(disp_z.loc[date]):
            continue
        row = mom.loc[date].dropna()
        if len(row) == 0:
            continue
        if disp_z.loc[date] > 0.5:
            # High dispersion: concentrated in top 2
            top = row.nlargest(2).index
            weights.loc[date, top] = 0.5
        else:
            # Low dispersion: spread across top 5
            top = row.nlargest(5).index
            weights.loc[date, top] = 0.2
    return weights


# ============================================================
# SIGNAL 9: Volatility Regime Gate + Momentum
# Only invest when VIX proxy (SPY realized vol) is below threshold
# ============================================================
def signal_vol_regime_gate(close_df, **kw):
    spy_ret = close_df[BENCHMARK].pct_change()
    vol_21 = spy_ret.rolling(21).std() * np.sqrt(252)
    vol_ma = vol_21.rolling(126).mean()
    low_vol = vol_21 < vol_ma  # vol below its own average
    mom = _momentum(close_df[SECTOR_ETFS], 63)
    return _top_n_equal_weight(mom, n=3, gate=low_vol)


# ============================================================
# SIGNAL 10: Sector Acceleration
# Second derivative of momentum (momentum of momentum)
# ============================================================
def signal_sector_acceleration(close_df, **kw):
    mom_21 = _momentum(close_df[SECTOR_ETFS], 21)
    mom_21_prev = mom_21.shift(21)
    accel = mom_21 - mom_21_prev
    return _top_n_equal_weight(accel, n=3)


# ============================================================
# SIGNAL 11: Relative Strength Index Timing
# Buy sectors with RSI crossing above 50 from below
# ============================================================
def signal_rsi_timing(close_df, period=14, **kw):
    weights = pd.DataFrame(0.0, index=close_df.index, columns=SECTOR_ETFS)
    for etf in SECTOR_ETFS:
        if etf not in close_df.columns:
            continue
        delta = close_df[etf].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.clip(lower=1e-8)
        rsi = 100 - (100 / (1 + rs))
        # Buy when RSI crosses above 50
        cross_up = (rsi > 50) & (rsi.shift(1) <= 50)
        # Hold while RSI > 40
        hold = rsi > 40
        in_trade = False
        for date in close_df.index:
            if date not in rsi.index:
                continue
            if pd.isna(rsi.loc[date]):
                continue
            if cross_up.loc[date] and not in_trade:
                in_trade = True
            elif in_trade and not hold.loc[date]:
                in_trade = False
            if in_trade:
                weights.loc[date, etf] = 1.0
    # Normalize rows to sum <= 1
    row_sums = weights.sum(axis=1).clip(lower=1)
    weights = weights.div(row_sums, axis=0)
    return weights


# ============================================================
# SIGNAL 12: Correlation Breakdown
# When sector correlations drop, go concentrated; when high, go cash
# ============================================================
def signal_correlation_regime(close_df, **kw):
    rets = close_df[SECTOR_ETFS].pct_change().dropna()
    # Rolling pairwise average correlation
    avg_corr = rets.rolling(63).corr().groupby(level=0).mean().mean(axis=1)
    avg_corr = avg_corr[~avg_corr.index.duplicated(keep='first')]
    corr_ma = avg_corr.rolling(126).mean()
    low_corr = avg_corr < corr_ma  # decorrelation = opportunity
    mom = _momentum(close_df[SECTOR_ETFS], 63)
    return _top_n_equal_weight(mom, n=3, gate=low_corr)


# ============================================================
# SIGNAL 13: Sector Rotation Velocity
# Rate of change of sector leadership
# When leadership is STABLE, ride the winner
# ============================================================
def signal_rotation_velocity(close_df, **kw):
    mom = _momentum(close_df[SECTOR_ETFS], 63)
    ranks = mom.rank(axis=1)
    rank_change = ranks.diff(21).abs().mean(axis=1)
    rank_ma = rank_change.rolling(63).mean()
    stable = rank_change < rank_ma  # stable leadership
    return _top_n_equal_weight(mom, n=2, gate=stable)


# ============================================================
# SIGNAL 14: Breadth-Weighted Momentum
# Weight by how many sectors have positive momentum
# ============================================================
def signal_breadth_weighted(close_df, **kw):
    mom = _momentum(close_df[SECTOR_ETFS], 63)
    breadth = (mom > 0).sum(axis=1)
    breadth_pct = breadth / len(SECTOR_ETFS)
    gate = breadth_pct > 0.5  # majority positive
    return _top_n_equal_weight(mom, n=3, gate=gate)


# ============================================================
# SIGNAL 15: Vol Compression Entry
# Enter when sector's vol compresses (squeeze), expect breakout
# ============================================================
def signal_vol_compression(close_df, **kw):
    rets = close_df[SECTOR_ETFS].pct_change()
    vol_5 = rets.rolling(5).std()
    vol_63 = rets.rolling(63).std()
    compression = vol_5 / vol_63.clip(lower=1e-8)
    # Low compression = squeeze
    mom = _momentum(close_df[SECTOR_ETFS], 63)
    score = _rank_normalize(mom) + _rank_normalize(-compression) * 0.5
    return _top_n_equal_weight(score, n=3)


# ============================================================
# SIGNAL 16: Multi-Timeframe Consensus
# Momentum must be positive on 21d, 63d, and 126d
# ============================================================
def signal_multi_tf_consensus(close_df, **kw):
    m21 = _momentum(close_df[SECTOR_ETFS], 21) > 0
    m63 = _momentum(close_df[SECTOR_ETFS], 63) > 0
    m126 = _momentum(close_df[SECTOR_ETFS], 126) > 0
    consensus = m21.astype(int) + m63.astype(int) + m126.astype(int)
    mom = _momentum(close_df[SECTOR_ETFS], 63)
    score = mom.where(consensus >= 2, -999)
    return _top_n_equal_weight(score, n=3)


# ============================================================
# SIGNAL 17: Drawdown Recovery
# Buy sectors recovering from drawdowns (crossed above -10% from worse)
# ============================================================
def signal_drawdown_recovery(close_df, **kw):
    weights = pd.DataFrame(0.0, index=close_df.index, columns=SECTOR_ETFS)
    for etf in SECTOR_ETFS:
        if etf not in close_df.columns:
            continue
        peak = close_df[etf].rolling(252, min_periods=63).max()
        dd = (close_df[etf] - peak) / peak
        # Recovery: was below -10%, now crossed above -5%
        was_bad = dd.shift(1) < -0.05
        now_better = dd > -0.05
        entry = was_bad & now_better
        # Hold until new high or drawdown > 15%
        in_trade = False
        for date in close_df.index:
            if date not in dd.index or pd.isna(dd.loc[date]):
                continue
            if entry.loc[date] and not in_trade:
                in_trade = True
            elif in_trade and (dd.loc[date] > -0.01 or dd.loc[date] < -0.15):
                in_trade = False
            if in_trade:
                weights.loc[date, etf] = 1.0
    row_sums = weights.sum(axis=1).clip(lower=1)
    weights = weights.div(row_sums, axis=0)
    return weights


# ============================================================
# SIGNAL 18: Information Ratio Ranking
# Rank by alpha vs SPY (regression residual momentum)
# ============================================================
def signal_info_ratio(close_df, **kw):
    spy_ret = close_df[BENCHMARK].pct_change()
    scores = pd.DataFrame(index=close_df.index, columns=SECTOR_ETFS)
    for etf in SECTOR_ETFS:
        if etf not in close_df.columns:
            continue
        etf_ret = close_df[etf].pct_change()
        # Rolling alpha = excess return beyond beta * spy
        cov = etf_ret.rolling(63).cov(spy_ret)
        var = spy_ret.rolling(63).var().clip(lower=1e-10)
        beta = cov / var
        alpha_daily = etf_ret - beta * spy_ret
        scores[etf] = alpha_daily.rolling(63).mean() / alpha_daily.rolling(63).std().clip(lower=1e-8)
    scores = scores.astype(float)
    return _top_n_equal_weight(scores, n=3)


# ============================================================
# SIGNAL 19: Sector Pair Spread Mean Reversion
# When a sector diverges too far from SPY, buy it back
# ============================================================
def signal_spread_reversion(close_df, **kw):
    scores = pd.DataFrame(index=close_df.index, columns=SECTOR_ETFS)
    spy_norm = close_df[BENCHMARK] / close_df[BENCHMARK].rolling(252).mean()
    for etf in SECTOR_ETFS:
        if etf not in close_df.columns:
            continue
        etf_norm = close_df[etf] / close_df[etf].rolling(252).mean()
        spread = etf_norm - spy_norm
        spread_z = (spread - spread.rolling(126).mean()) / spread.rolling(126).std().clip(lower=1e-8)
        scores[etf] = -spread_z  # buy undervalued
    scores = scores.astype(float)
    return _top_n_equal_weight(scores, n=3)


# ============================================================
# SIGNAL 20: Regime-Adaptive Ensemble
# Bull: momentum; Bear: defensive sectors; Transition: cash
# ============================================================
def signal_regime_adaptive(close_df, **kw):
    spy = close_df[BENCHMARK]
    sma50 = spy.rolling(50).mean()
    sma200 = spy.rolling(200).mean()
    mom = _momentum(close_df[SECTOR_ETFS], 63)

    defensive = ["XLU", "XLP", "XLV"]
    weights = pd.DataFrame(0.0, index=close_df.index, columns=SECTOR_ETFS)

    for date in close_df.index:
        if date not in sma50.index or pd.isna(sma50.loc[date]) or pd.isna(sma200.loc[date]):
            continue
        s = spy.loc[date]
        if s > sma50.loc[date] and sma50.loc[date] > sma200.loc[date]:
            # Bull: top 3 momentum
            row = mom.loc[date].dropna()
            if len(row) > 0:
                top = row.nlargest(3).index
                weights.loc[date, top] = 1.0 / 3
        elif s < sma200.loc[date]:
            # Bear: defensive
            avail = [d for d in defensive if d in close_df.columns]
            if avail:
                weights.loc[date, avail] = 1.0 / len(avail)
        # Else: transition = cash
    return weights


# ============================================================
# SIGNAL 21: Momentum Persistence Filter
# Only buy sectors where momentum has been positive 4 of last 5 months
# ============================================================
def signal_momentum_persistence(close_df, **kw):
    monthly_mom = _momentum(close_df[SECTOR_ETFS], 21)
    pos_count = (monthly_mom > 0).rolling(105).sum()  # ~5 months of daily
    # Approximate: 4/5 months positive = 80+ days positive in 105
    persistent = pos_count > 80
    mom = _momentum(close_df[SECTOR_ETFS], 63)
    score = mom.where(persistent, -999)
    return _top_n_equal_weight(score, n=3)


# ============================================================
# SIGNAL 22: Risk Parity Weighting
# Equal risk contribution from top sectors
# ============================================================
def signal_risk_parity(close_df, **kw):
    rets = close_df[SECTOR_ETFS].pct_change()
    vol = rets.rolling(63).std()
    inv_vol = 1.0 / vol.clip(lower=1e-8)
    mom = _momentum(close_df[SECTOR_ETFS], 63)

    weights = pd.DataFrame(0.0, index=close_df.index, columns=SECTOR_ETFS)
    for date in close_df.index:
        if date not in mom.index:
            continue
        row = mom.loc[date].dropna()
        if len(row) < 3:
            continue
        top = row.nlargest(3).index
        iv = inv_vol.loc[date, top]
        if iv.sum() > 0:
            w = iv / iv.sum()
            weights.loc[date, top] = w
    return weights


# ============================================================
# SIGNAL 23: MACD Sector Timing
# Enter when MACD crosses signal line
# ============================================================
def signal_macd_timing(close_df, **kw):
    weights = pd.DataFrame(0.0, index=close_df.index, columns=SECTOR_ETFS)
    for etf in SECTOR_ETFS:
        if etf not in close_df.columns:
            continue
        ema12 = close_df[etf].ewm(span=12).mean()
        ema26 = close_df[etf].ewm(span=26).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()
        bullish = macd > signal_line
        weights.loc[bullish, etf] = 1.0
    row_sums = weights.sum(axis=1).clip(lower=1)
    weights = weights.div(row_sums, axis=0)
    return weights


# ============================================================
# SIGNAL 24: Sector Flow Proxy (Volume-Price Divergence)
# High volume + positive price = accumulation
# ============================================================
def signal_volume_price(close_df, open_df=None, data=None, **kw):
    if data is None:
        return pd.DataFrame(0.0, index=close_df.index, columns=SECTOR_ETFS)
    scores = pd.DataFrame(index=close_df.index, columns=SECTOR_ETFS)
    for etf in SECTOR_ETFS:
        if etf not in data or "Volume" not in data[etf].columns:
            continue
        vol = data[etf]["Volume"]
        vol_ma = vol.rolling(20).mean().clip(lower=1)
        vol_ratio = vol / vol_ma
        ret = close_df[etf].pct_change()
        # Accumulation score: positive return * high volume
        scores[etf] = (ret * vol_ratio).rolling(21).mean()
    scores = scores.astype(float)
    return _top_n_equal_weight(scores, n=3)


# ============================================================
# SIGNAL 25: Cross-Sector Lead-Lag
# Financials lead industrials, tech leads comm services, etc.
# ============================================================
def signal_lead_lag(close_df, **kw):
    leaders = {"XLF": "XLI", "XLK": "XLC", "XLE": "XLB", "XLY": "XLP"}
    rets = close_df[SECTOR_ETFS].pct_change()
    weights = pd.DataFrame(0.0, index=close_df.index, columns=SECTOR_ETFS)
    for leader, follower in leaders.items():
        if leader not in rets.columns or follower not in rets.columns:
            continue
        # If leader had good 5-day return, buy follower
        leader_mom = rets[leader].rolling(5).sum()
        entry = leader_mom > leader_mom.rolling(63).quantile(0.8)
        weights.loc[entry, follower] = 0.25
    row_sums = weights.sum(axis=1).clip(lower=1)
    weights = weights.div(row_sums, axis=0)
    return weights


# ============================================================
# SIGNAL 26: Trend Following with Adaptive Lookback
# Use the lookback period with highest autocorrelation
# ============================================================
def signal_adaptive_lookback(close_df, **kw):
    lookbacks = [21, 42, 63, 126]
    moms = {lb: _momentum(close_df[SECTOR_ETFS], lb) for lb in lookbacks}

    weights = pd.DataFrame(0.0, index=close_df.index, columns=SECTOR_ETFS)
    for date in close_df.index[252:]:
        # Pick lookback with highest avg return for top picks
        best_lb = 63
        best_score = -999
        for lb in lookbacks:
            if date not in moms[lb].index:
                continue
            row = moms[lb].loc[date].dropna()
            if len(row) >= 3:
                score = row.nlargest(3).mean()
                if score > best_score:
                    best_score = score
                    best_lb = lb
        row = moms[best_lb].loc[date].dropna()
        if len(row) >= 3:
            top = row.nlargest(3).index
            weights.loc[date, top] = 1.0 / 3
    return weights


# ============================================================
# SIGNAL 27: Sector Momentum + SPY Hedge
# 70% top sectors, 30% inverse when SPY trends down
# ============================================================
def signal_hedged_momentum(close_df, **kw):
    spy = close_df[BENCHMARK]
    sma = spy.rolling(50).mean()
    mom = _momentum(close_df[SECTOR_ETFS], 63)
    weights = pd.DataFrame(0.0, index=close_df.index, columns=SECTOR_ETFS)

    for date in close_df.index:
        if date not in sma.index or pd.isna(sma.loc[date]):
            continue
        row = mom.loc[date].dropna()
        if len(row) < 3:
            continue
        if spy.loc[date] > sma.loc[date]:
            # Bull: full allocation to top 3
            top = row.nlargest(3).index
            weights.loc[date, top] = 1.0 / 3
        else:
            # Bear: reduced allocation + defensive
            top = row.nlargest(2).index
            weights.loc[date, top] = 0.15
            for d in ["XLU", "XLP"]:
                if d in close_df.columns:
                    weights.loc[date, d] = 0.15
    return weights


# ============================================================
# SIGNAL 28: Entropy-Based Regime Detection
# Market entropy (return distribution evenness) as a gate
# ============================================================
def signal_entropy_gate(close_df, **kw):
    rets = close_df[SECTOR_ETFS].pct_change(5)
    # Discretize returns into bins and compute entropy
    def row_entropy(row):
        r = row.dropna()
        if len(r) < 3:
            return np.nan
        # Use rank-based proxy
        ranks = r.rank() / len(r)
        # Entropy of ranks (higher = more dispersed)
        p = np.histogram(ranks, bins=4, density=True)[0]
        p = p[p > 0]
        p = p / p.sum()
        return -np.sum(p * np.log2(p))

    entropy = rets.apply(row_entropy, axis=1)
    entropy_ma = entropy.rolling(63).mean()
    # High entropy = diverse opportunities = invest
    gate = entropy > entropy_ma
    mom = _momentum(close_df[SECTOR_ETFS], 63)
    return _top_n_equal_weight(mom, n=3, gate=gate)


# ============================================================
# SIGNAL 29: Hurst Exponent Filter
# Only trade sectors showing trending behavior (H > 0.5)
# ============================================================
def signal_hurst_filter(close_df, **kw):
    mom = _momentum(close_df[SECTOR_ETFS], 63)
    # Simplified Hurst: ratio of range to std over different windows
    scores = pd.DataFrame(index=close_df.index, columns=SECTOR_ETFS)
    for etf in SECTOR_ETFS:
        if etf not in close_df.columns:
            continue
        log_ret = np.log(close_df[etf] / close_df[etf].shift(1))
        # R/S statistic as proxy
        r = log_ret.rolling(63).max() - log_ret.rolling(63).min()
        s = log_ret.rolling(63).std().clip(lower=1e-8)
        rs = r / s
        # Higher R/S = more trending
        scores[etf] = rs
    scores = scores.astype(float)
    trending = scores > scores.rolling(126).median()
    score = mom.where(trending, -999)
    return _top_n_equal_weight(score, n=3)


# ============================================================
# SIGNAL 30: Sector Momentum with Volatility Scaling
# Scale position size inversely with market volatility
# ============================================================
def signal_vol_scaled_momentum(close_df, **kw):
    spy_ret = close_df[BENCHMARK].pct_change()
    vol_21 = spy_ret.rolling(21).std() * np.sqrt(252)
    target_vol = 0.15
    vol_scale = (target_vol / vol_21.clip(lower=0.05)).clip(upper=1.5)

    mom = _momentum(close_df[SECTOR_ETFS], 63)
    base_weights = _top_n_equal_weight(mom, n=3)
    # Scale by vol
    for col in base_weights.columns:
        base_weights[col] = base_weights[col] * vol_scale
    # Cap at 1.0 total
    row_sums = base_weights.sum(axis=1).clip(lower=1e-8)
    excess = row_sums > 1.0
    base_weights[excess] = base_weights[excess].div(row_sums[excess], axis=0)
    return base_weights


# ============================================================
# SIGNAL 31: Relative Momentum Change (Momentum Derivative)
# Buy sectors where momentum is IMPROVING fastest
# ============================================================
def signal_momentum_change(close_df, **kw):
    mom = _momentum(close_df[SECTOR_ETFS], 63)
    mom_change = mom - mom.shift(21)
    return _top_n_equal_weight(mom_change, n=3)


# ============================================================
# SIGNAL 32: Sector Divergence Score
# When ONE sector breaks away from the pack, follow it
# ============================================================
def signal_breakaway(close_df, **kw):
    mom = _momentum(close_df[SECTOR_ETFS], 21)
    cross_mean = mom.mean(axis=1)
    cross_std = mom.std(axis=1).clip(lower=1e-8)
    z_scores = mom.sub(cross_mean, axis=0).div(cross_std, axis=0)
    # Only buy strong breakaways (z > 1.5)
    score = z_scores.where(z_scores > 1.0, -999)
    return _top_n_equal_weight(score, n=2)


# ============================================================
# SIGNAL 33: Multi-Signal Composite (NOVEL)
# Combine: SMA gate + vol-adj momentum + breadth + vol scaling
# This is the PROPRIETARY composite
# ============================================================
def signal_composite_v1(close_df, **kw):
    spy = close_df[BENCHMARK]
    sma = spy.rolling(50).mean()
    gate = spy > sma

    # Component 1: Vol-adjusted momentum
    rets = close_df[SECTOR_ETFS].pct_change()
    mom = rets.rolling(63).mean() * 252
    vol = rets.rolling(63).std() * np.sqrt(252)
    sharpe_score = _rank_normalize(mom / vol.clip(lower=0.01))

    # Component 2: Momentum persistence
    pos_21 = (_momentum(close_df[SECTOR_ETFS], 21) > 0).rolling(63).mean()
    persist_score = _rank_normalize(pos_21)

    # Component 3: Relative strength vs SPY
    spy_mom = _momentum(close_df[[BENCHMARK]], 63)[BENCHMARK]
    rel_str = _momentum(close_df[SECTOR_ETFS], 63).sub(spy_mom, axis=0)
    rel_score = _rank_normalize(rel_str)

    # Composite
    composite = 0.4 * sharpe_score + 0.3 * persist_score + 0.3 * rel_score

    # Vol scaling
    spy_ret = close_df[BENCHMARK].pct_change()
    vol_21 = spy_ret.rolling(21).std() * np.sqrt(252)
    vol_scale = (0.15 / vol_21.clip(lower=0.05)).clip(upper=1.0)

    weights = _top_n_equal_weight(composite, n=3, gate=gate)
    for col in weights.columns:
        weights[col] = weights[col] * vol_scale
    row_sums = weights.sum(axis=1)
    excess = row_sums > 1.0
    if excess.any():
        weights.loc[excess] = weights.loc[excess].div(row_sums[excess], axis=0)
    return weights


# Master list of all signals
ALL_SIGNALS = {
    "01_momentum_63d": signal_momentum_63d,
    "02_sma_gated_mom": signal_sma_gated_mom,
    "03_vol_adj_momentum": signal_vol_adj_momentum,
    "04_mom_trend_confirm": signal_mom_trend_confirm,
    "05_dual_momentum": signal_dual_momentum,
    "06_mean_reversion_5d": signal_mean_reversion_5d,
    "07_mom_reversion_combo": signal_mom_reversion_combo,
    "08_dispersion_timing": signal_dispersion_timing,
    "09_vol_regime_gate": signal_vol_regime_gate,
    "10_sector_acceleration": signal_sector_acceleration,
    "11_rsi_timing": signal_rsi_timing,
    "12_correlation_regime": signal_correlation_regime,
    "13_rotation_velocity": signal_rotation_velocity,
    "14_breadth_weighted": signal_breadth_weighted,
    "15_vol_compression": signal_vol_compression,
    "16_multi_tf_consensus": signal_multi_tf_consensus,
    "17_drawdown_recovery": signal_drawdown_recovery,
    "18_info_ratio": signal_info_ratio,
    "19_spread_reversion": signal_spread_reversion,
    "20_regime_adaptive": signal_regime_adaptive,
    "21_momentum_persistence": signal_momentum_persistence,
    "22_risk_parity": signal_risk_parity,
    "23_macd_timing": signal_macd_timing,
    "24_volume_price": signal_volume_price,
    "25_lead_lag": signal_lead_lag,
    "26_adaptive_lookback": signal_adaptive_lookback,
    "27_hedged_momentum": signal_hedged_momentum,
    "28_entropy_gate": signal_entropy_gate,
    "29_hurst_filter": signal_hurst_filter,
    "30_vol_scaled_momentum": signal_vol_scaled_momentum,
    "31_momentum_change": signal_momentum_change,
    "32_breakaway": signal_breakaway,
    "33_composite_v1": signal_composite_v1,
}
