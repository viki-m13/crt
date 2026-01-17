#!/usr/bin/env python3
"""
CRT Recovery Predictor - Evidence-Based Stock Selection

Core Question: What stocks should I buy today that will most likely be profitable in 1, 3, 5 years?

Approach (based on backtest validation):
1. MOMENTUM is the strongest alpha predictor (stocks going up keep going up)
2. QUALITY filters out risky names
3. Historical data provides probability estimates
4. Compare everything to SPY baseline

Output: For each stock, show:
- Probability of positive return at 1Y, 3Y, 5Y
- Expected return (median from similar past situations)
- Probability of beating SPY
- Confidence level based on sample size
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
BENCHMARK = "SPY"
MIN_HISTORY_DAYS = 756  # 3 years
OUT_DIR = os.path.join("docs", "data")
CHUNK_SIZE = 80

# iShares IVV holdings URL (S&P 500)
ISHARES_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund"
)

def fetch_sp500_tickers():
    """Fetch S&P 500 tickers from iShares IVV."""
    import requests
    from io import StringIO

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    }
    try:
        resp = requests.get(ISHARES_HOLDINGS_URL, headers=headers, timeout=45)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}")

        raw_text = resp.content.decode("utf-8", errors="ignore")
        lines = raw_text.splitlines()

        header_idx = None
        for i, line in enumerate(lines[:700]):
            if line.strip().startswith("Ticker,") or line.strip().startswith('"Ticker",'):
                header_idx = i
                break

        if header_idx is None:
            raise RuntimeError("Could not find Ticker header")

        trimmed = "\n".join(lines[header_idx:])
        df = pd.read_csv(StringIO(trimmed))
        df.columns = [c.strip() for c in df.columns]

        if "Asset Class" in df.columns:
            df = df[df["Asset Class"].astype(str).str.contains("Equity", case=False, na=False)]

        tickers = (
            df["Ticker"].astype(str)
            .str.strip()
            .str.replace(" ", "", regex=False)
            .str.replace(".", "-", regex=False)
            .str.upper()
        )
        keep = tickers[
            tickers.ne("") &
            tickers.ne("NAN") &
            (~tickers.str.contains("CASH", case=False, na=False))
        ].dropna().unique().tolist()

        print(f"    Fetched {len(keep)} S&P 500 tickers from iShares")
        return sorted(keep)
    except Exception as e:
        print(f"    Warning: Could not fetch holdings ({e}), using fallback list")
        # Fallback to top 100
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "JNJ",
            "V", "UNH", "HD", "PG", "MA", "DIS", "NFLX", "CRM", "COST", "WMT",
            "AVGO", "LLY", "PEP", "KO", "MRK", "TMO", "ABBV", "CVX", "XOM", "ORCL",
            "BAC", "MCD", "CSCO", "ACN", "ABT", "DHR", "NKE", "AMD", "QCOM", "TXN",
            "NEE", "HON", "IBM", "LOW", "UPS", "RTX", "SPGI", "GS", "CAT", "BA",
            "INTC", "SBUX", "PFE", "T", "VZ", "PM", "GE", "LMT", "BLK", "AXP",
            "MDT", "NOW", "ISRG", "SYK", "MMM", "ADI", "GILD", "BKNG", "MDLZ", "ADP",
            "CME", "REGN", "VRTX", "ZTS", "ETN", "PLD", "CI", "SCHW", "CB", "SO",
            "DUK", "MO", "BDX", "CL", "EOG", "SLB", "FCX", "PSX", "WM", "APD",
            "EMR", "ITW", "NSC", "PCAR", "AON", "MCK", "CTAS", "TGT", "ORLY", "ROP",
        ]

# =============================================================================
# HELPERS
# =============================================================================
def calc_volatility(prices, period=60):
    returns = np.log(prices / prices.shift(1))
    return returns.rolling(period).std() * np.sqrt(252) * 100

def calc_momentum(prices, months):
    days = months * 21
    if len(prices) < days:
        return np.nan
    return (prices.iloc[-1] / prices.iloc[-days] - 1) * 100

def calc_drawdown(prices, lookback=252):
    if len(prices) < lookback:
        return np.nan
    high = prices.rolling(lookback).max().iloc[-1]
    return (prices.iloc[-1] / high - 1) * 100

# =============================================================================
# CORE SCORING MODEL
# =============================================================================
def calculate_stock_metrics(ticker, close, high, low, spy_close):
    """Calculate all metrics for a single stock."""

    px = close[ticker].dropna()
    hi = high[ticker].dropna() if ticker in high.columns else px
    lo = low[ticker].dropna() if ticker in low.columns else px
    spy = spy_close.dropna()

    if len(px) < MIN_HISTORY_DAYS:
        return None

    # Align indices
    common_idx = px.index.intersection(spy.index)
    if len(common_idx) < MIN_HISTORY_DAYS:
        return None

    px = px.reindex(common_idx)
    spy = spy.reindex(common_idx)

    current_price = px.iloc[-1]

    # ----- MOMENTUM METRICS (strongest alpha predictor) -----
    mom_1m = calc_momentum(px, 1)
    mom_3m = calc_momentum(px, 3)
    mom_6m = calc_momentum(px, 6)
    mom_12m = calc_momentum(px, 12)

    # 12-1 momentum (academic standard: skip last month)
    if len(px) >= 252:
        mom_12_1 = (px.iloc[-21] / px.iloc[-252] - 1) * 100
    else:
        mom_12_1 = 0

    # ----- TREND METRICS -----
    sma_50 = px.rolling(50).mean().iloc[-1] if len(px) >= 50 else current_price
    sma_200 = px.rolling(200).mean().iloc[-1] if len(px) >= 200 else current_price

    above_sma50 = current_price > sma_50
    above_sma200 = current_price > sma_200
    sma50_above_200 = sma_50 > sma_200  # Golden cross

    # ----- QUALITY METRICS -----
    volatility = calc_volatility(px, 60).iloc[-1]

    # Monthly win rate
    monthly_rets = px.resample('M').last().pct_change().dropna()
    monthly_win_rate = (monthly_rets > 0).mean() * 100 if len(monthly_rets) >= 12 else 50

    # ----- VALUE METRICS -----
    drawdown = calc_drawdown(px, 252)

    high_52w = hi.rolling(252).max().iloc[-1] if len(hi) >= 252 else hi.max()
    low_52w = lo.rolling(252).min().iloc[-1] if len(lo) >= 252 else lo.min()

    if high_52w > low_52w:
        position_in_range = (current_price - low_52w) / (high_52w - low_52w) * 100
    else:
        position_in_range = 50

    # ----- RELATIVE STRENGTH VS SPY -----
    stock_ret_3m = calc_momentum(px, 3) / 100 if not np.isnan(calc_momentum(px, 3)) else 0
    spy_ret_3m = calc_momentum(spy, 3) / 100 if not np.isnan(calc_momentum(spy, 3)) else 0
    rel_strength = (stock_ret_3m - spy_ret_3m) * 100

    # ----- HISTORICAL PROBABILITY CALCULATION -----
    # Find similar past situations and see what happened
    outcomes_1y = []
    outcomes_3y = []
    outcomes_5y = []
    alpha_1y = []  # vs SPY

    # Current conditions to match
    curr_mom_12_1 = mom_12_1
    curr_above_200 = above_sma200
    curr_drawdown = drawdown

    # Sample historical points
    for idx in range(300, len(px) - 252, 21):  # Every month, leave room for 1Y forward
        hist_px = px.iloc[idx]

        # Calculate historical metrics at that point
        hist_mom = (px.iloc[idx-21] / px.iloc[idx-252] - 1) * 100 if idx >= 252 else 0
        hist_sma200 = px.iloc[max(0,idx-200):idx].mean()
        hist_above_200 = hist_px > hist_sma200
        hist_high = px.iloc[max(0,idx-252):idx].max()
        hist_dd = (hist_px / hist_high - 1) * 100

        # Similarity score
        mom_sim = max(0, 100 - abs(curr_mom_12_1 - hist_mom) * 2)
        trend_sim = 100 if curr_above_200 == hist_above_200 else 50
        dd_sim = max(0, 100 - abs(curr_drawdown - hist_dd) * 2)

        similarity = (mom_sim * 0.5 + trend_sim * 0.3 + dd_sim * 0.2)

        if similarity > 60:  # Similar enough
            # 1Y forward
            if idx + 252 < len(px):
                ret_1y = (px.iloc[idx + 252] / hist_px - 1)
                outcomes_1y.append(ret_1y)

                # Alpha vs SPY
                spy_ret = (spy.iloc[idx + 252] / spy.iloc[idx] - 1)
                alpha_1y.append(ret_1y - spy_ret)

            # 3Y forward
            if idx + 756 < len(px):
                ret_3y = (px.iloc[idx + 756] / hist_px - 1)
                outcomes_3y.append(ret_3y)

            # 5Y forward
            if idx + 1260 < len(px):
                ret_5y = (px.iloc[idx + 1260] / hist_px - 1)
                outcomes_5y.append(ret_5y)

    # Calculate probabilities
    if len(outcomes_1y) >= 5:
        prob_positive_1y = np.mean([r > 0 for r in outcomes_1y]) * 100
        prob_beat_spy_1y = np.mean([a > 0 for a in alpha_1y]) * 100
        median_return_1y = np.median(outcomes_1y) * 100
        p10_return_1y = np.percentile(outcomes_1y, 10) * 100
        p90_return_1y = np.percentile(outcomes_1y, 90) * 100
        sample_size_1y = len(outcomes_1y)
    else:
        prob_positive_1y = 65  # Default base rate
        prob_beat_spy_1y = 50
        median_return_1y = 10
        p10_return_1y = -20
        p90_return_1y = 40
        sample_size_1y = 0

    if len(outcomes_3y) >= 5:
        prob_positive_3y = np.mean([r > 0 for r in outcomes_3y]) * 100
        median_return_3y = np.median(outcomes_3y) * 100
        sample_size_3y = len(outcomes_3y)
    else:
        prob_positive_3y = 75
        median_return_3y = 35
        sample_size_3y = 0

    if len(outcomes_5y) >= 5:
        prob_positive_5y = np.mean([r > 0 for r in outcomes_5y]) * 100
        median_return_5y = np.median(outcomes_5y) * 100
        sample_size_5y = len(outcomes_5y)
    else:
        prob_positive_5y = 85
        median_return_5y = 65
        sample_size_5y = 0

    # ----- COMPOSITE SCORE -----
    # Weighted by what actually predicts alpha (from our backtest)
    momentum_score = min(100, max(0, 50 + mom_12_1))
    trend_score = 50 + (15 if above_sma50 else 0) + (20 if above_sma200 else 0) + (15 if sma50_above_200 else 0)
    quality_score = monthly_win_rate * 0.6 + max(0, 100 - volatility * 2) * 0.4

    # Final score (momentum-weighted based on backtest results)
    composite_score = (
        momentum_score * 0.40 +  # Strongest predictor
        trend_score * 0.25 +
        quality_score * 0.20 +
        prob_beat_spy_1y * 0.15
    )
    composite_score = min(100, max(0, composite_score))

    # ----- CONFIDENCE -----
    min_samples = min(sample_size_1y, sample_size_3y)
    confidence = min(100, max(0, min_samples * 2))  # 50 samples = 100% confidence

    # ----- SIGNAL -----
    if composite_score >= 70 and prob_positive_3y >= 75 and prob_beat_spy_1y >= 55:
        signal = "STRONG_BUY"
    elif composite_score >= 60 and prob_positive_3y >= 70:
        signal = "BUY"
    elif composite_score >= 45 and prob_positive_1y >= 60:
        signal = "HOLD"
    else:
        signal = "AVOID"

    return {
        "ticker": ticker,
        "price": round(current_price, 2),
        "signal": signal,
        "composite_score": round(composite_score, 1),
        "confidence": round(confidence, 1),

        # Probabilities (THE KEY OUTPUT)
        "prob_positive_1y": round(prob_positive_1y, 1),
        "prob_positive_3y": round(prob_positive_3y, 1),
        "prob_positive_5y": round(prob_positive_5y, 1),
        "prob_beat_spy_1y": round(prob_beat_spy_1y, 1),

        # Expected returns
        "median_return_1y": round(median_return_1y, 1),
        "median_return_3y": round(median_return_3y, 1),
        "median_return_5y": round(median_return_5y, 1),
        "downside_risk_1y": round(p10_return_1y, 1),  # 10th percentile
        "upside_potential_1y": round(p90_return_1y, 1),  # 90th percentile

        # Sample sizes
        "sample_size_1y": sample_size_1y,
        "sample_size_3y": sample_size_3y,
        "sample_size_5y": sample_size_5y,

        # Key metrics
        "momentum_12m": round(mom_12m, 1) if not np.isnan(mom_12m) else None,
        "momentum_3m": round(mom_3m, 1) if not np.isnan(mom_3m) else None,
        "rel_strength_vs_spy": round(rel_strength, 1),
        "volatility": round(volatility, 1) if not np.isnan(volatility) else None,
        "drawdown_from_high": round(drawdown, 1) if not np.isnan(drawdown) else None,
        "position_in_52w_range": round(position_in_range, 1),
        "above_sma200": bool(above_sma200),
        "monthly_win_rate": round(monthly_win_rate, 1),
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("CRT RECOVERY PREDICTOR - Evidence-Based Stock Selection")
    print("=" * 80)

    # Fetch S&P 500 universe
    print("\n[1] Fetching stock universe...")
    stock_universe = fetch_sp500_tickers()

    # Download data in chunks and extract Close/High/Low
    print("\n[2] Downloading data...")
    tickers = list(set(stock_universe + [BENCHMARK]))

    close_dfs = []
    high_dfs = []
    low_dfs = []

    for i in range(0, len(tickers), CHUNK_SIZE):
        batch = tickers[i:i + CHUNK_SIZE]
        print(f"    Downloading batch {i//CHUNK_SIZE + 1}/{(len(tickers)-1)//CHUNK_SIZE + 1}...")
        data = yf.download(batch, period="10y", interval="1d", auto_adjust=True, progress=False)

        if data.empty:
            continue

        # Extract Close, High, Low for this batch
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.get_level_values(0):
                close_dfs.append(data["Close"])
            if "High" in data.columns.get_level_values(0):
                high_dfs.append(data["High"])
            if "Low" in data.columns.get_level_values(0):
                low_dfs.append(data["Low"])
        else:
            # Single ticker
            ticker = batch[0]
            close_dfs.append(pd.DataFrame({ticker: data["Close"]}))
            high_dfs.append(pd.DataFrame({ticker: data["High"]}))
            low_dfs.append(pd.DataFrame({ticker: data["Low"]}))

    # Combine all dataframes
    close = pd.concat(close_dfs, axis=1) if close_dfs else pd.DataFrame()
    high = pd.concat(high_dfs, axis=1) if high_dfs else pd.DataFrame()
    low = pd.concat(low_dfs, axis=1) if low_dfs else pd.DataFrame()

    # Clean up
    close = close.loc[:, ~close.columns.duplicated()].dropna(how='all')
    high = high.loc[:, ~high.columns.duplicated()].reindex(close.index).ffill()
    low = low.loc[:, ~low.columns.duplicated()].reindex(close.index).ffill()

    if BENCHMARK not in close.columns:
        raise RuntimeError(f"Missing {BENCHMARK} data")

    spy_close = close[BENCHMARK]

    print(f"    Data from {close.index[0].date()} to {close.index[-1].date()}")
    print(f"    {len(close.columns)} tickers loaded")

    # Analyze each stock
    print("\n[3] Analyzing stocks...")
    results = []
    processed = 0

    for ticker in stock_universe:
        if ticker not in close.columns:
            continue

        metrics = calculate_stock_metrics(ticker, close, high, low, spy_close)
        if metrics:
            results.append(metrics)
            processed += 1
            if processed % 50 == 0:
                print(f"    Processed {processed} stocks...")

    # Sort by composite score
    results.sort(key=lambda x: x['composite_score'], reverse=True)

    print(f"    Completed: {len(results)} stocks analyzed")

    # Summary - show top 30
    print("\n" + "=" * 80)
    print("TOP 30 STOCKS BY COMPOSITE SCORE")
    print("=" * 80)
    print(f"\n{'Rank':<5} {'Ticker':<8} {'Score':<8} {'Signal':<12} {'P(+1Y)':<10} {'P(+3Y)':<10} {'P(+5Y)':<10} {'Beat SPY':<10}")
    print("-" * 85)

    for i, r in enumerate(results[:30], 1):
        print(f"{i:<5} {r['ticker']:<8} {r['composite_score']:<8.0f} {r['signal']:<12} "
              f"{r['prob_positive_1y']:<10.0f} {r['prob_positive_3y']:<10.0f} "
              f"{r['prob_positive_5y']:<10.0f} {r['prob_beat_spy_1y']:<10.0f}")

    # Top picks
    strong_buys = [r for r in results if r['signal'] == 'STRONG_BUY']
    buys = [r for r in results if r['signal'] == 'BUY']

    print("\n" + "=" * 80)
    print("TOP PICKS")
    print("=" * 80)

    if strong_buys:
        print("\nSTRONG BUY signals:")
        for r in strong_buys:
            print(f"  {r['ticker']}: {r['prob_positive_1y']:.0f}% chance positive in 1Y, "
                  f"{r['prob_beat_spy_1y']:.0f}% chance to beat SPY, "
                  f"median return {r['median_return_1y']:+.0f}%")

    if buys:
        print("\nBUY signals:")
        for r in buys[:5]:
            print(f"  {r['ticker']}: {r['prob_positive_1y']:.0f}% chance positive in 1Y, "
                  f"median return {r['median_return_1y']:+.0f}%")

    # Save results
    os.makedirs(OUT_DIR, exist_ok=True)

    output = {
        "as_of": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "model": "CRT Recovery Predictor v1",
        "methodology": "Momentum-weighted scoring with historical probability estimation",
        "benchmark": BENCHMARK,
        "items": results,
    }

    with open(os.path.join(OUT_DIR, "predictions.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Results saved to {OUT_DIR}/predictions.json")

    return results


if __name__ == "__main__":
    main()
