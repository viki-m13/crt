#!/usr/bin/env python3
"""
Strategy Validation Script
Core Question: What stocks should I buy today that will most likely be profitable in 1, 3, 5 years?

Tests 4 strategies using historical data to see which best predicts future positive returns.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# Small test universe for quick validation
TEST_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ", "PG", "XOM", "WMT", "KO"]
BENCHMARK = "SPY"

print("=" * 80)
print("STRATEGY VALIDATION - Testing which approach best predicts future gains")
print("=" * 80)

# Download data
print("\n[1] Downloading 10 years of data...")
tickers = TEST_STOCKS + [BENCHMARK]
data = yf.download(tickers, period="10y", interval="1d", auto_adjust=True, progress=False)
close = data["Close"].dropna()
high = data["High"].reindex(close.index).ffill()
low = data["Low"].reindex(close.index).ffill()
volume = data["Volume"].reindex(close.index).ffill()

print(f"    Data from {close.index[0].date()} to {close.index[-1].date()}")
print(f"    {len(close)} trading days")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_sma(prices, period):
    return prices.rolling(period).mean()

def calc_volatility(prices, period=60):
    returns = np.log(prices / prices.shift(1))
    return returns.rolling(period).std() * np.sqrt(252) * 100

# =============================================================================
# MODEL 1: ALPHA MODEL (Momentum & 52-Week High)
# =============================================================================
def model1_score(ticker, close_df, high_df, volume_df, date_idx):
    """Score based on proximity to 52-week high + momentum."""
    try:
        px = close_df[ticker].loc[:date_idx].dropna()
        hi = high_df[ticker].loc[:date_idx].dropna()
        vol = volume_df[ticker].loc[:date_idx].dropna()
        if len(px) < 252:
            return np.nan

        # 52-week high proximity (25% weight)
        high_52w = hi.rolling(252).max().iloc[-1]
        current = px.iloc[-1]
        dist = (current - high_52w) / high_52w
        if dist >= -0.02: h52_score = 100
        elif dist >= -0.05: h52_score = 90
        elif dist >= -0.10: h52_score = 75
        elif dist >= -0.15: h52_score = 60
        elif dist >= -0.20: h52_score = 45
        else: h52_score = 30

        # Momentum (20% weight)
        mom_3m = (current / px.iloc[-63] - 1) * 100 if len(px) >= 63 else 0
        mom_score = min(100, max(0, 50 + mom_3m * 2))

        # Trend - price vs SMA200 (20% weight)
        sma200 = px.rolling(200).mean().iloc[-1] if len(px) >= 200 else px.mean()
        trend_score = 80 if current > sma200 else 40

        # Volatility (15% weight) - lower is better
        vol_ann = calc_volatility(px, 60).iloc[-1]
        vol_score = max(0, 100 - vol_ann) if not np.isnan(vol_ann) else 50

        # Volume trend (10% weight)
        avg_vol = vol.rolling(20).mean().iloc[-1]
        recent_vol = vol.iloc[-5:].mean()
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
        volume_score = min(100, 50 + (vol_ratio - 1) * 50)

        # RSI (10% weight)
        rsi = calc_rsi(px).iloc[-1]
        if 40 <= rsi <= 60: rsi_score = 70
        elif 30 <= rsi <= 70: rsi_score = 60
        else: rsi_score = 40

        score = (h52_score * 0.25 + mom_score * 0.20 + trend_score * 0.20 +
                 vol_score * 0.15 + volume_score * 0.10 + rsi_score * 0.10)
        return score
    except:
        return np.nan

# =============================================================================
# MODEL 2: PREDICTION MODEL (Long-Term Probability)
# =============================================================================
def model2_score(ticker, close_df, high_df, low_df, date_idx):
    """Probability-based model for long-term returns."""
    try:
        px = close_df[ticker].loc[:date_idx].dropna()
        hi = high_df[ticker].loc[:date_idx].dropna()
        lo = low_df[ticker].loc[:date_idx].dropna()
        if len(px) < 252:
            return np.nan

        current = px.iloc[-1]

        # Value position (inverted - low = good value)
        high_52w = hi.rolling(252).max().iloc[-1]
        low_52w = lo.rolling(252).min().iloc[-1]
        range_52w = high_52w - low_52w
        if range_52w > 0:
            value_pos = (current - low_52w) / range_52w
            value_score = (1 - value_pos) * 100  # Inverted: low position = high score
        else:
            value_score = 50

        # Mean reversion potential
        ret_12m = (current / px.iloc[-252] - 1) if len(px) >= 252 else 0
        if ret_12m < -0.20:
            mr_score = 80  # Fallen a lot = rebound potential
        elif ret_12m < 0:
            mr_score = 60
        elif ret_12m < 0.30:
            mr_score = 50
        else:
            mr_score = 30  # Up a lot = less upside

        # Trend quality (SMA slope)
        sma200 = px.rolling(200).mean()
        if len(sma200.dropna()) >= 63:
            slope = (sma200.iloc[-1] / sma200.iloc[-63] - 1) * 100
            trend_score = min(100, max(0, 50 + slope * 5))
        else:
            trend_score = 50

        # Volatility rank (lower = safer)
        vol = calc_volatility(px, 60).iloc[-1]
        if vol < 20: vol_score = 90
        elif vol < 30: vol_score = 70
        elif vol < 40: vol_score = 50
        else: vol_score = 30

        # Above SMA200 bonus
        sma200_val = sma200.iloc[-1] if len(sma200.dropna()) > 0 else current
        above_sma = 15 if current > sma200_val else -10

        score = value_score * 0.25 + mr_score * 0.25 + trend_score * 0.25 + vol_score * 0.25 + above_sma
        return min(100, max(0, score))
    except:
        return np.nan

# =============================================================================
# MODEL 3: STOCK MODEL (Multi-Factor with Regime)
# =============================================================================
def model3_score(ticker, close_df, spy_close, date_idx):
    """Regime-aware multi-factor model."""
    try:
        px = close_df[ticker].loc[:date_idx].dropna()
        spy = spy_close.loc[:date_idx].dropna()
        if len(px) < 252 or len(spy) < 252:
            return np.nan

        current = px.iloc[-1]

        # Detect market regime
        spy_ret_3m = (spy.iloc[-1] / spy.iloc[-63] - 1) if len(spy) >= 63 else 0
        spy_vol = calc_volatility(spy, 60).iloc[-1]

        if spy_ret_3m > 0.05 and spy_vol < 20:
            regime = "BULL"
        elif spy_ret_3m < -0.05 or spy_vol > 25:
            regime = "BEAR"
        else:
            regime = "NEUTRAL"

        # Momentum (12-1 month)
        if len(px) >= 252:
            mom_12_1 = (px.iloc[-21] / px.iloc[-252] - 1) * 100
        else:
            mom_12_1 = 0
        mom_score = min(100, max(0, 50 + mom_12_1))

        # Quality (win rate of monthly returns)
        monthly = px.resample('M').last().pct_change().dropna()
        if len(monthly) >= 12:
            win_rate = (monthly > 0).mean() * 100
        else:
            win_rate = 50

        # Relative strength vs SPY
        stock_ret = (current / px.iloc[-63] - 1) if len(px) >= 63 else 0
        spy_ret = (spy.iloc[-1] / spy.iloc[-63] - 1) if len(spy) >= 63 else 0
        rel_strength = (stock_ret - spy_ret) * 100
        rs_score = min(100, max(0, 50 + rel_strength * 2))

        # Trend score
        sma50 = px.rolling(50).mean().iloc[-1] if len(px) >= 50 else current
        sma200 = px.rolling(200).mean().iloc[-1] if len(px) >= 200 else current
        trend_score = 50
        if current > sma50: trend_score += 15
        if current > sma200: trend_score += 20
        if sma50 > sma200: trend_score += 15

        # Apply regime weights
        if regime == "BULL":
            score = mom_score * 0.35 + rs_score * 0.25 + trend_score * 0.25 + win_rate * 0.15
        elif regime == "BEAR":
            score = win_rate * 0.35 + mom_score * 0.15 + rs_score * 0.25 + trend_score * 0.25
        else:
            score = mom_score * 0.25 + win_rate * 0.25 + rs_score * 0.25 + trend_score * 0.25

        return min(100, max(0, score))
    except:
        return np.nan

# =============================================================================
# MODEL 4: VALUE MODEL (Historical Similarity - CRT Style)
# =============================================================================
def model4_score(ticker, close_df, high_df, low_df, date_idx):
    """Historical similarity matching - finds similar past situations."""
    try:
        px = close_df[ticker].loc[:date_idx].dropna()
        hi = high_df[ticker].loc[:date_idx].dropna()
        lo = low_df[ticker].loc[:date_idx].dropna()

        if len(px) < 756:  # Need 3+ years
            return np.nan

        current = px.iloc[-1]

        # Current conditions
        high_52w = hi.rolling(252).max().iloc[-1]
        low_52w = lo.rolling(252).min().iloc[-1]
        range_52w = high_52w - low_52w

        if range_52w > 0:
            curr_value_pos = (current - low_52w) / range_52w * 100
        else:
            curr_value_pos = 50

        curr_drawdown = (current / high_52w - 1) * 100

        sma200 = px.rolling(200).mean()
        curr_sma_dist = (current / sma200.iloc[-1] - 1) * 100 if len(sma200.dropna()) > 0 else 0

        curr_rsi = calc_rsi(px).iloc[-1]

        # Find similar historical instances and their outcomes
        similar_outcomes_1y = []
        similar_outcomes_3y = []
        similar_outcomes_5y = []

        # Sample every 63 days (quarterly) going back, need 1Y forward data
        sample_indices = list(range(252 + 63, len(px) - 252, 63))  # Leave room for 1Y forward

        for idx in sample_indices[-50:]:  # Last 50 samples for speed
            hist_px = px.iloc[idx]
            hist_high_52w = hi.iloc[max(0,idx-252):idx].max()
            hist_low_52w = lo.iloc[max(0,idx-252):idx].min()
            hist_range = hist_high_52w - hist_low_52w

            if hist_range > 0:
                hist_value_pos = (hist_px - hist_low_52w) / hist_range * 100
            else:
                continue

            hist_drawdown = (hist_px / hist_high_52w - 1) * 100
            hist_sma200 = px.iloc[max(0,idx-200):idx].mean()
            hist_sma_dist = (hist_px / hist_sma200 - 1) * 100 if hist_sma200 > 0 else 0

            # Calculate similarity
            similarity = 100 - (
                abs(curr_value_pos - hist_value_pos) * 0.4 +
                abs(curr_drawdown - hist_drawdown) * 0.3 +
                abs(curr_sma_dist - hist_sma_dist) * 0.3
            )

            if similarity > 60:  # Similar enough
                # 1Y forward return
                if idx + 252 < len(px):
                    ret_1y = (px.iloc[idx + 252] / hist_px - 1)
                    similar_outcomes_1y.append(ret_1y)

                # 3Y forward return
                if idx + 756 < len(px):
                    ret_3y = (px.iloc[idx + 756] / hist_px - 1)
                    similar_outcomes_3y.append(ret_3y)

                # 5Y forward return
                if idx + 1260 < len(px):
                    ret_5y = (px.iloc[idx + 1260] / hist_px - 1)
                    similar_outcomes_5y.append(ret_5y)

        # Calculate probabilities from similar instances
        if len(similar_outcomes_1y) >= 5:
            prob_1y = np.mean([r > 0 for r in similar_outcomes_1y]) * 100
            med_1y = np.median(similar_outcomes_1y) * 100
        else:
            prob_1y, med_1y = 50, 0

        if len(similar_outcomes_3y) >= 5:
            prob_3y = np.mean([r > 0 for r in similar_outcomes_3y]) * 100
            med_3y = np.median(similar_outcomes_3y) * 100
        else:
            prob_3y, med_3y = 60, 10

        if len(similar_outcomes_5y) >= 5:
            prob_5y = np.mean([r > 0 for r in similar_outcomes_5y]) * 100
            med_5y = np.median(similar_outcomes_5y) * 100
        else:
            prob_5y, med_5y = 70, 20

        # Value score (lower position = better value)
        value_score = max(0, 100 - curr_value_pos)

        # Combine: probability weighted by time horizon
        prob_score = prob_1y * 0.2 + prob_3y * 0.35 + prob_5y * 0.45

        score = prob_score * 0.6 + value_score * 0.4
        return min(100, max(0, score))
    except:
        return np.nan

# =============================================================================
# BACKTEST: Walk-forward validation
# =============================================================================
print("\n[2] Running walk-forward backtest...")
print("    Testing each model's ability to predict 1Y, 3Y, 5Y positive returns")

# Test dates: every quarter from 2016 to 2021 (so we have forward returns to validate)
test_dates = pd.date_range('2016-01-01', '2021-01-01', freq='Q')
test_dates = [d for d in test_dates if d in close.index or close.index[close.index <= d].size > 0]

results = {
    'Model1_Alpha': {'predictions': [], 'actual_1y': [], 'actual_3y': [], 'actual_5y': []},
    'Model2_Prediction': {'predictions': [], 'actual_1y': [], 'actual_3y': [], 'actual_5y': []},
    'Model3_Stock': {'predictions': [], 'actual_1y': [], 'actual_3y': [], 'actual_5y': []},
    'Model4_Value': {'predictions': [], 'actual_1y': [], 'actual_3y': [], 'actual_5y': []},
}

spy_close = close[BENCHMARK]

for test_date in test_dates:
    # Find closest trading day
    valid_dates = close.index[close.index <= test_date]
    if len(valid_dates) == 0:
        continue
    date_idx = valid_dates[-1]

    for ticker in TEST_STOCKS:
        # Get scores from each model
        s1 = model1_score(ticker, close, high, volume, date_idx)
        s2 = model2_score(ticker, close, high, low, date_idx)
        s3 = model3_score(ticker, close, spy_close, date_idx)
        s4 = model4_score(ticker, close, high, low, date_idx)

        # Get actual future returns
        future_idx_1y = close.index[close.index >= date_idx + pd.Timedelta(days=365)]
        future_idx_3y = close.index[close.index >= date_idx + pd.Timedelta(days=365*3)]
        future_idx_5y = close.index[close.index >= date_idx + pd.Timedelta(days=365*5)]

        px_now = close[ticker].loc[date_idx]

        ret_1y = (close[ticker].loc[future_idx_1y[0]] / px_now - 1) if len(future_idx_1y) > 0 else np.nan
        ret_3y = (close[ticker].loc[future_idx_3y[0]] / px_now - 1) if len(future_idx_3y) > 0 else np.nan
        ret_5y = (close[ticker].loc[future_idx_5y[0]] / px_now - 1) if len(future_idx_5y) > 0 else np.nan

        # Store results
        for model_name, score in [('Model1_Alpha', s1), ('Model2_Prediction', s2),
                                   ('Model3_Stock', s3), ('Model4_Value', s4)]:
            if not np.isnan(score):
                results[model_name]['predictions'].append(score)
                results[model_name]['actual_1y'].append(ret_1y)
                results[model_name]['actual_3y'].append(ret_3y)
                results[model_name]['actual_5y'].append(ret_5y)

# =============================================================================
# ANALYZE RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("RESULTS: Which model best predicts positive future returns?")
print("=" * 80)

def analyze_model(name, data):
    predictions = np.array(data['predictions'])
    actual_1y = np.array(data['actual_1y'])
    actual_3y = np.array(data['actual_3y'])
    actual_5y = np.array(data['actual_5y'])

    # Split into high-score (top 30%) vs low-score (bottom 30%)
    high_thresh = np.percentile(predictions, 70)
    low_thresh = np.percentile(predictions, 30)

    high_mask = predictions >= high_thresh
    low_mask = predictions <= low_thresh

    print(f"\n{name}:")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  High-score threshold (top 30%): {high_thresh:.1f}")

    # For high-score stocks, what % were actually positive?
    for horizon, actual in [('1Y', actual_1y), ('3Y', actual_3y), ('5Y', actual_5y)]:
        valid_high = actual[high_mask & ~np.isnan(actual)]
        valid_low = actual[low_mask & ~np.isnan(actual)]
        valid_all = actual[~np.isnan(actual)]

        if len(valid_high) > 0 and len(valid_low) > 0:
            win_high = np.mean(valid_high > 0) * 100
            win_low = np.mean(valid_low > 0) * 100
            win_all = np.mean(valid_all > 0) * 100
            med_high = np.median(valid_high) * 100
            med_low = np.median(valid_low) * 100

            edge = win_high - win_low
            print(f"  {horizon}: High-score win rate: {win_high:.1f}% | Low-score: {win_low:.1f}% | Edge: {edge:+.1f}pp | Median return (high): {med_high:+.1f}%")

    return {
        'high_win_1y': np.mean(actual_1y[high_mask & ~np.isnan(actual_1y)] > 0) * 100 if np.sum(high_mask & ~np.isnan(actual_1y)) > 0 else 0,
        'high_win_3y': np.mean(actual_3y[high_mask & ~np.isnan(actual_3y)] > 0) * 100 if np.sum(high_mask & ~np.isnan(actual_3y)) > 0 else 0,
        'high_win_5y': np.mean(actual_5y[high_mask & ~np.isnan(actual_5y)] > 0) * 100 if np.sum(high_mask & ~np.isnan(actual_5y)) > 0 else 0,
    }

model_stats = {}
for name, data in results.items():
    if len(data['predictions']) > 0:
        model_stats[name] = analyze_model(name, data)

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: Win Rates for High-Score Stocks (Top 30%)")
print("=" * 80)
print(f"{'Model':<20} {'1Y Win%':<12} {'3Y Win%':<12} {'5Y Win%':<12} {'Avg':<12}")
print("-" * 68)

best_model = None
best_avg = 0

for name, stats in model_stats.items():
    avg = (stats['high_win_1y'] + stats['high_win_3y'] + stats['high_win_5y']) / 3
    print(f"{name:<20} {stats['high_win_1y']:<12.1f} {stats['high_win_3y']:<12.1f} {stats['high_win_5y']:<12.1f} {avg:<12.1f}")
    if avg > best_avg:
        best_avg = avg
        best_model = name

print("-" * 68)
print(f"\nBest performing model: {best_model} (avg win rate: {best_avg:.1f}%)")

# Compare to SPY baseline
spy_rets = []
for test_date in test_dates:
    valid_dates = close.index[close.index <= test_date]
    if len(valid_dates) == 0:
        continue
    date_idx = valid_dates[-1]

    future_1y = close.index[close.index >= date_idx + pd.Timedelta(days=365)]
    if len(future_1y) > 0:
        ret = close[BENCHMARK].loc[future_1y[0]] / close[BENCHMARK].loc[date_idx] - 1
        spy_rets.append(ret)

spy_win_rate = np.mean([r > 0 for r in spy_rets]) * 100 if spy_rets else 0
spy_median_ret = np.median(spy_rets) * 100 if spy_rets else 0

print(f"\nSPY Baseline (buy and hold):")
print(f"  1Y positive rate: {spy_win_rate:.1f}%")
print(f"  Median 1Y return: {spy_median_ret:+.1f}%")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
