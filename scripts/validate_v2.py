#!/usr/bin/env python3
"""
Strategy Validation V2 - Focus on ALPHA (beating SPY) not just positive returns
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

TEST_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ", "PG", "XOM", "WMT", "KO"]
BENCHMARK = "SPY"

print("=" * 80)
print("VALIDATION V2: Testing for ALPHA (returns above SPY)")
print("=" * 80)

# Download data
tickers = TEST_STOCKS + [BENCHMARK]
data = yf.download(tickers, period="10y", interval="1d", auto_adjust=True, progress=False)
close = data["Close"].dropna()
high = data["High"].reindex(close.index).ffill()
low = data["Low"].reindex(close.index).ffill()
volume = data["Volume"].reindex(close.index).ffill()

print(f"Data: {close.index[0].date()} to {close.index[-1].date()}")

# Helper functions
def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_volatility(prices, period=60):
    returns = np.log(prices / prices.shift(1))
    return returns.rolling(period).std() * np.sqrt(252) * 100

# =============================================================================
# SIMPLIFIED MODELS - Focus on what matters for long-term
# =============================================================================

def model_momentum(ticker, close_df, date_idx):
    """Pure momentum: recent winners keep winning."""
    px = close_df[ticker].loc[:date_idx].dropna()
    if len(px) < 252: return np.nan

    # 12-1 month momentum (skip last month)
    if len(px) >= 252:
        mom = (px.iloc[-21] / px.iloc[-252] - 1) * 100
        return min(100, max(0, 50 + mom))
    return 50

def model_value(ticker, close_df, high_df, low_df, date_idx):
    """Value: stocks down from highs tend to rebound."""
    px = close_df[ticker].loc[:date_idx].dropna()
    hi = high_df[ticker].loc[:date_idx].dropna()
    lo = low_df[ticker].loc[:date_idx].dropna()
    if len(px) < 252: return np.nan

    high_52w = hi.rolling(252).max().iloc[-1]
    low_52w = lo.rolling(252).min().iloc[-1]
    current = px.iloc[-1]

    # Position in range (inverted: low = high score)
    if high_52w > low_52w:
        pos = (current - low_52w) / (high_52w - low_52w)
        return (1 - pos) * 100
    return 50

def model_quality(ticker, close_df, date_idx):
    """Quality: consistent performers with low volatility."""
    px = close_df[ticker].loc[:date_idx].dropna()
    if len(px) < 252: return np.nan

    # Monthly win rate
    monthly = px.resample('M').last().pct_change().dropna()
    win_rate = (monthly > 0).mean() * 100 if len(monthly) >= 12 else 50

    # Low volatility
    vol = calc_volatility(px, 60).iloc[-1]
    vol_score = max(0, 100 - vol * 2) if not np.isnan(vol) else 50

    return win_rate * 0.6 + vol_score * 0.4

def model_trend(ticker, close_df, date_idx):
    """Trend: stocks in uptrends continue."""
    px = close_df[ticker].loc[:date_idx].dropna()
    if len(px) < 200: return np.nan

    current = px.iloc[-1]
    sma50 = px.rolling(50).mean().iloc[-1]
    sma200 = px.rolling(200).mean().iloc[-1]

    score = 50
    if current > sma50: score += 15
    if current > sma200: score += 20
    if sma50 > sma200: score += 15
    return min(100, score)

def model_historical_similarity(ticker, close_df, high_df, low_df, date_idx):
    """Historical: find similar past situations, use their outcomes."""
    px = close_df[ticker].loc[:date_idx].dropna()
    hi = high_df[ticker].loc[:date_idx].dropna()
    lo = low_df[ticker].loc[:date_idx].dropna()
    if len(px) < 756: return np.nan

    current = px.iloc[-1]
    high_52w = hi.rolling(252).max().iloc[-1]
    low_52w = lo.rolling(252).min().iloc[-1]

    if high_52w <= low_52w: return 50

    curr_pos = (current - low_52w) / (high_52w - low_52w) * 100
    curr_dd = (current / high_52w - 1) * 100

    outcomes = []
    for idx in range(300, len(px) - 252, 21):  # Sample every month
        hist_px = px.iloc[idx]
        hist_high = hi.iloc[max(0,idx-252):idx].max()
        hist_low = lo.iloc[max(0,idx-252):idx].min()

        if hist_high <= hist_low: continue

        hist_pos = (hist_px - hist_low) / (hist_high - hist_low) * 100
        hist_dd = (hist_px / hist_high - 1) * 100

        similarity = 100 - (abs(curr_pos - hist_pos) * 0.5 + abs(curr_dd - hist_dd) * 0.5)

        if similarity > 70:
            fwd_ret = (px.iloc[idx + 252] / hist_px - 1)
            outcomes.append(fwd_ret)

    if len(outcomes) >= 5:
        return np.mean([o > 0 for o in outcomes]) * 100
    return 50

# =============================================================================
# BACKTEST
# =============================================================================
print("\nRunning backtest (testing for alpha over SPY)...")

test_dates = pd.date_range('2016-01-01', '2021-06-01', freq='Q')
spy_close = close[BENCHMARK]

models = {
    'Momentum': model_momentum,
    'Value': model_value,
    'Quality': model_quality,
    'Trend': model_trend,
    'Historical': model_historical_similarity,
}

results = {name: {'scores': [], 'alpha_1y': [], 'alpha_3y': []} for name in models}

for test_date in test_dates:
    valid_dates = close.index[close.index <= test_date]
    if len(valid_dates) == 0: continue
    date_idx = valid_dates[-1]

    # SPY returns
    spy_now = spy_close.loc[date_idx]
    future_1y = close.index[close.index >= date_idx + pd.Timedelta(days=365)]
    future_3y = close.index[close.index >= date_idx + pd.Timedelta(days=365*3)]

    spy_ret_1y = (spy_close.loc[future_1y[0]] / spy_now - 1) if len(future_1y) > 0 else np.nan
    spy_ret_3y = (spy_close.loc[future_3y[0]] / spy_now - 1) if len(future_3y) > 0 else np.nan

    for ticker in TEST_STOCKS:
        px_now = close[ticker].loc[date_idx]
        stock_ret_1y = (close[ticker].loc[future_1y[0]] / px_now - 1) if len(future_1y) > 0 else np.nan
        stock_ret_3y = (close[ticker].loc[future_3y[0]] / px_now - 1) if len(future_3y) > 0 else np.nan

        # Alpha = stock return - SPY return
        alpha_1y = stock_ret_1y - spy_ret_1y if not np.isnan(stock_ret_1y) and not np.isnan(spy_ret_1y) else np.nan
        alpha_3y = stock_ret_3y - spy_ret_3y if not np.isnan(stock_ret_3y) and not np.isnan(spy_ret_3y) else np.nan

        for name, func in models.items():
            if name == 'Value' or name == 'Historical':
                score = func(ticker, close, high, low, date_idx)
            else:
                score = func(ticker, close, date_idx)

            if not np.isnan(score):
                results[name]['scores'].append(score)
                results[name]['alpha_1y'].append(alpha_1y)
                results[name]['alpha_3y'].append(alpha_3y)

# =============================================================================
# ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("RESULTS: Which model best predicts ALPHA (beating SPY)?")
print("=" * 80)

summary = []

for name, data in results.items():
    scores = np.array(data['scores'])
    alpha_1y = np.array(data['alpha_1y'])
    alpha_3y = np.array(data['alpha_3y'])

    high_thresh = np.percentile(scores, 70)
    high_mask = scores >= high_thresh
    low_mask = scores <= np.percentile(scores, 30)

    # Key metric: Do high-score picks beat SPY?
    high_alpha_1y = alpha_1y[high_mask & ~np.isnan(alpha_1y)]
    low_alpha_1y = alpha_1y[low_mask & ~np.isnan(alpha_1y)]
    high_alpha_3y = alpha_3y[high_mask & ~np.isnan(alpha_3y)]

    beat_spy_1y_high = np.mean(high_alpha_1y > 0) * 100 if len(high_alpha_1y) > 0 else 0
    beat_spy_1y_low = np.mean(low_alpha_1y > 0) * 100 if len(low_alpha_1y) > 0 else 0
    beat_spy_3y_high = np.mean(high_alpha_3y > 0) * 100 if len(high_alpha_3y) > 0 else 0

    avg_alpha_1y = np.mean(high_alpha_1y) * 100 if len(high_alpha_1y) > 0 else 0
    avg_alpha_3y = np.mean(high_alpha_3y) * 100 if len(high_alpha_3y) > 0 else 0

    edge = beat_spy_1y_high - beat_spy_1y_low

    print(f"\n{name}:")
    print(f"  High-score beats SPY (1Y): {beat_spy_1y_high:.1f}% | Low-score: {beat_spy_1y_low:.1f}% | Edge: {edge:+.1f}pp")
    print(f"  High-score beats SPY (3Y): {beat_spy_3y_high:.1f}%")
    print(f"  Avg alpha (1Y): {avg_alpha_1y:+.1f}% | Avg alpha (3Y): {avg_alpha_3y:+.1f}%")

    summary.append({
        'name': name,
        'beat_spy_1y': beat_spy_1y_high,
        'beat_spy_3y': beat_spy_3y_high,
        'edge_1y': edge,
        'avg_alpha_1y': avg_alpha_1y,
        'avg_alpha_3y': avg_alpha_3y,
    })

# =============================================================================
# ENSEMBLE TEST
# =============================================================================
print("\n" + "=" * 80)
print("ENSEMBLE TEST: Combining models")
print("=" * 80)

# Combine all scores for each prediction
ensemble_results = {'scores': [], 'alpha_1y': [], 'alpha_3y': []}

for test_date in test_dates:
    valid_dates = close.index[close.index <= test_date]
    if len(valid_dates) == 0: continue
    date_idx = valid_dates[-1]

    spy_now = spy_close.loc[date_idx]
    future_1y = close.index[close.index >= date_idx + pd.Timedelta(days=365)]
    future_3y = close.index[close.index >= date_idx + pd.Timedelta(days=365*3)]

    spy_ret_1y = (spy_close.loc[future_1y[0]] / spy_now - 1) if len(future_1y) > 0 else np.nan
    spy_ret_3y = (spy_close.loc[future_3y[0]] / spy_now - 1) if len(future_3y) > 0 else np.nan

    for ticker in TEST_STOCKS:
        px_now = close[ticker].loc[date_idx]
        stock_ret_1y = (close[ticker].loc[future_1y[0]] / px_now - 1) if len(future_1y) > 0 else np.nan
        stock_ret_3y = (close[ticker].loc[future_3y[0]] / px_now - 1) if len(future_3y) > 0 else np.nan

        alpha_1y = stock_ret_1y - spy_ret_1y if not np.isnan(stock_ret_1y) and not np.isnan(spy_ret_1y) else np.nan
        alpha_3y = stock_ret_3y - spy_ret_3y if not np.isnan(stock_ret_3y) and not np.isnan(spy_ret_3y) else np.nan

        # Calculate all model scores
        s_mom = model_momentum(ticker, close, date_idx)
        s_val = model_value(ticker, close, high, low, date_idx)
        s_qual = model_quality(ticker, close, date_idx)
        s_trend = model_trend(ticker, close, date_idx)
        s_hist = model_historical_similarity(ticker, close, high, low, date_idx)

        all_scores = [s for s in [s_mom, s_val, s_qual, s_trend, s_hist] if not np.isnan(s)]

        if len(all_scores) >= 3:
            # Weighted ensemble: Historical (best at predicting) + Quality + Trend
            weights = {'hist': 0.35, 'qual': 0.25, 'trend': 0.20, 'val': 0.10, 'mom': 0.10}
            ensemble_score = 0
            total_w = 0
            if not np.isnan(s_hist): ensemble_score += s_hist * weights['hist']; total_w += weights['hist']
            if not np.isnan(s_qual): ensemble_score += s_qual * weights['qual']; total_w += weights['qual']
            if not np.isnan(s_trend): ensemble_score += s_trend * weights['trend']; total_w += weights['trend']
            if not np.isnan(s_val): ensemble_score += s_val * weights['val']; total_w += weights['val']
            if not np.isnan(s_mom): ensemble_score += s_mom * weights['mom']; total_w += weights['mom']

            if total_w > 0:
                ensemble_score = ensemble_score / total_w
                ensemble_results['scores'].append(ensemble_score)
                ensemble_results['alpha_1y'].append(alpha_1y)
                ensemble_results['alpha_3y'].append(alpha_3y)

scores = np.array(ensemble_results['scores'])
alpha_1y = np.array(ensemble_results['alpha_1y'])
alpha_3y = np.array(ensemble_results['alpha_3y'])

high_thresh = np.percentile(scores, 70)
high_mask = scores >= high_thresh
low_mask = scores <= np.percentile(scores, 30)

high_alpha_1y = alpha_1y[high_mask & ~np.isnan(alpha_1y)]
low_alpha_1y = alpha_1y[low_mask & ~np.isnan(alpha_1y)]
high_alpha_3y = alpha_3y[high_mask & ~np.isnan(alpha_3y)]

beat_spy_1y_high = np.mean(high_alpha_1y > 0) * 100 if len(high_alpha_1y) > 0 else 0
beat_spy_1y_low = np.mean(low_alpha_1y > 0) * 100 if len(low_alpha_1y) > 0 else 0
beat_spy_3y_high = np.mean(high_alpha_3y > 0) * 100 if len(high_alpha_3y) > 0 else 0
edge = beat_spy_1y_high - beat_spy_1y_low
avg_alpha_1y = np.mean(high_alpha_1y) * 100 if len(high_alpha_1y) > 0 else 0
avg_alpha_3y = np.mean(high_alpha_3y) * 100 if len(high_alpha_3y) > 0 else 0

print(f"\nWeighted Ensemble (Hist 35% + Quality 25% + Trend 20% + Value 10% + Momentum 10%):")
print(f"  High-score beats SPY (1Y): {beat_spy_1y_high:.1f}% | Low-score: {beat_spy_1y_low:.1f}% | Edge: {edge:+.1f}pp")
print(f"  High-score beats SPY (3Y): {beat_spy_3y_high:.1f}%")
print(f"  Avg alpha (1Y): {avg_alpha_1y:+.1f}% | Avg alpha (3Y): {avg_alpha_3y:+.1f}%")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"{'Model':<20} {'Beat SPY 1Y':<15} {'Beat SPY 3Y':<15} {'Edge 1Y':<12} {'Avg Alpha 1Y':<15}")
print("-" * 77)
for s in sorted(summary, key=lambda x: x['avg_alpha_1y'], reverse=True):
    print(f"{s['name']:<20} {s['beat_spy_1y']:<15.1f} {s['beat_spy_3y']:<15.1f} {s['edge_1y']:<+12.1f} {s['avg_alpha_1y']:<+15.1f}")
print(f"{'ENSEMBLE':<20} {beat_spy_1y_high:<15.1f} {beat_spy_3y_high:<15.1f} {edge:<+12.1f} {avg_alpha_1y:<+15.1f}")
print("-" * 77)

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
