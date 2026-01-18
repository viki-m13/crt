#!/usr/bin/env python3
"""
CRT Recovery Predictor v3.0 - Find Undervalued Stocks Likely to Outperform

Core Question: What stocks should I buy TODAY that are undervalued and
most likely to outperform SPY in 1, 3, and 5 years?

Philosophy:
- Find QUALITY companies trading at a DISCOUNT (down from highs)
- With EARLY RECOVERY signals (not catching falling knives)
- Backed by STATISTICAL EVIDENCE (minimum sample sizes)
- Simple output with clear "Why Buy Now" thesis
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

MIN_HISTORY_YEARS = 5          # Minimum years of price history required
MIN_SAMPLE_SIZE = 15           # Minimum historical matches for valid probability
MIN_DISCOUNT_PCT = 10          # Minimum % below 52-week high to be "undervalued"
MAX_DISCOUNT_PCT = 50          # Maximum % below high (avoid distressed)

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "data"

# =============================================================================
# DATA FETCHING
# =============================================================================

def get_sp500_tickers():
    """Fetch S&P 500 constituents from iShares IVV holdings."""
    url = "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund"
    try:
        df = pd.read_csv(url, skiprows=9)
        tickers = df['Ticker'].dropna().tolist()
        tickers = [t for t in tickers if isinstance(t, str) and t.isalpha() and len(t) <= 5]
        return tickers[:503]
    except Exception as e:
        print(f"    Warning: Could not fetch S&P 500 list: {e}")
        return []

def download_price_data(tickers, use_max_period=True):
    """Download historical price data for all tickers."""
    all_close = {}
    all_high = {}
    all_low = {}

    batch_size = 80
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        print(f"    Downloading batch {batch_num}/{total_batches}...")

        try:
            data = yf.download(
                batch,
                period="max",  # Get ALL available history
                progress=False,
                group_by='ticker',
                auto_adjust=True,
                threads=True
            )

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        close = data['Close']
                        high = data['High']
                        low = data['Low']
                    else:
                        close = data[ticker]['Close'] if ticker in data.columns.get_level_values(0) else None
                        high = data[ticker]['High'] if ticker in data.columns.get_level_values(0) else None
                        low = data[ticker]['Low'] if ticker in data.columns.get_level_values(0) else None

                    if close is not None and len(close.dropna()) > 252:
                        all_close[ticker] = close.dropna()
                        all_high[ticker] = high.dropna()
                        all_low[ticker] = low.dropna()
                except:
                    pass
        except Exception as e:
            print(f"    Warning: Batch download error: {e}")

    if not all_close:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    close_df = pd.DataFrame(all_close)
    high_df = pd.DataFrame(all_high)
    low_df = pd.DataFrame(all_low)

    return close_df, high_df, low_df

# =============================================================================
# ANALYSIS - Find Undervalued Opportunities
# =============================================================================

def calculate_opportunity_metrics(ticker, close, high, low, spy_close):
    """
    Calculate metrics focused on finding undervalued stocks with recovery potential.

    Key insight: We want stocks that are:
    1. DOWN from highs (undervalued/discounted)
    2. QUALITY (historically profitable, not garbage)
    3. RECOVERING (early momentum, not falling knife)
    4. HISTORICALLY SUCCESSFUL (similar setups led to outperformance)
    """

    try:
        if len(close) < 252 * MIN_HISTORY_YEARS:
            return None  # Not enough history

        current_price = close.iloc[-1]

        # 1. DISCOUNT METRICS - How undervalued is this stock?
        high_52w = close.iloc[-252:].max()
        low_52w = close.iloc[-252:].min()
        discount_from_high = (high_52w - current_price) / high_52w * 100
        position_in_range = (current_price - low_52w) / (high_52w - low_52w) * 100 if high_52w != low_52w else 50

        # Skip if not discounted enough or too distressed
        if discount_from_high < MIN_DISCOUNT_PCT:
            return None  # Not undervalued - trading near highs
        if discount_from_high > MAX_DISCOUNT_PCT:
            return None  # Too distressed - might be broken

        # 2. QUALITY METRICS - Is this a quality company?
        returns = close.pct_change().dropna()

        # Sharpe ratio (risk-adjusted returns)
        annual_return = (close.iloc[-1] / close.iloc[-252] - 1) if len(close) >= 252 else 0
        annual_vol = returns.iloc[-252:].std() * np.sqrt(252) if len(returns) >= 252 else 0.3
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # Long-term trend (5-year return)
        years_of_data = len(close) / 252
        if years_of_data >= 5:
            five_year_return = (close.iloc[-1] / close.iloc[-252*5] - 1) * 100
        else:
            five_year_return = (close.iloc[-1] / close.iloc[0] - 1) * 100

        # Monthly win rate (consistency)
        monthly = close.resample('M').last().pct_change().dropna()
        monthly_win_rate = (monthly > 0).mean() * 100 if len(monthly) > 12 else 50

        # Skip low quality stocks
        if five_year_return < -20:
            return None  # Long-term loser
        if monthly_win_rate < 40:
            return None  # Too inconsistent

        # 3. RECOVERY SIGNALS - Is it starting to bounce?
        sma20 = close.iloc[-20:].mean()
        sma50 = close.iloc[-50:].mean() if len(close) >= 50 else sma20
        sma200 = close.iloc[-200:].mean() if len(close) >= 200 else sma50

        above_sma20 = current_price > sma20
        above_sma50 = current_price > sma50
        price_vs_sma50 = (current_price / sma50 - 1) * 100

        # Recent momentum (last 1-3 months)
        mom_1m = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) >= 21 else 0
        mom_3m = (close.iloc[-1] / close.iloc[-63] - 1) * 100 if len(close) >= 63 else 0

        # Recovery signal: stock is down from highs BUT showing recent strength
        early_recovery = above_sma20 and mom_1m > 0

        # 4. HISTORICAL PROBABILITY - What happened in similar situations?
        # Find times when stock had similar discount + early recovery pattern

        spy_returns_1y = {}
        spy_returns_3y = {}

        for i in range(252, len(spy_close) - 252):
            date = spy_close.index[i]
            spy_returns_1y[date] = (spy_close.iloc[i + 252] / spy_close.iloc[i] - 1) * 100 if i + 252 < len(spy_close) else None
            spy_returns_3y[date] = (spy_close.iloc[i + 756] / spy_close.iloc[i] - 1) * 100 if i + 756 < len(spy_close) else None

        matches_1y = []
        matches_3y = []
        matches_5y = []
        beat_spy_1y = []
        beat_spy_3y = []

        for i in range(252, len(close) - 252):
            try:
                hist_price = close.iloc[i]
                hist_high_52w = close.iloc[i-252:i].max()
                hist_low_52w = close.iloc[i-252:i].min()
                hist_discount = (hist_high_52w - hist_price) / hist_high_52w * 100

                hist_sma20 = close.iloc[i-20:i].mean()
                hist_above_sma20 = hist_price > hist_sma20
                hist_mom_1m = (hist_price / close.iloc[i-21] - 1) * 100 if i >= 21 else 0
                hist_early_recovery = hist_above_sma20 and hist_mom_1m > 0

                # Match criteria: similar discount level AND similar recovery signal
                if abs(hist_discount - discount_from_high) < 10 and hist_early_recovery == early_recovery:
                    date = close.index[i]

                    # 1-year outcome
                    if i + 252 < len(close):
                        ret_1y = (close.iloc[i + 252] / hist_price - 1) * 100
                        matches_1y.append(ret_1y)

                        if date in spy_returns_1y and spy_returns_1y[date] is not None:
                            beat_spy_1y.append(ret_1y > spy_returns_1y[date])

                    # 3-year outcome
                    if i + 756 < len(close):
                        ret_3y = (close.iloc[i + 756] / hist_price - 1) * 100
                        matches_3y.append(ret_3y)

                        if date in spy_returns_3y and spy_returns_3y[date] is not None:
                            beat_spy_3y.append(ret_3y > spy_returns_3y[date])

                    # 5-year outcome
                    if i + 1260 < len(close):
                        ret_5y = (close.iloc[i + 1260] / hist_price - 1) * 100
                        matches_5y.append(ret_5y)
            except:
                continue

        # Require minimum sample size for valid probabilities
        if len(matches_1y) < MIN_SAMPLE_SIZE:
            return None  # Not enough historical data for this pattern

        # Calculate probabilities
        prob_positive_1y = np.mean([r > 0 for r in matches_1y]) * 100 if matches_1y else 50
        prob_positive_3y = np.mean([r > 0 for r in matches_3y]) * 100 if len(matches_3y) >= 10 else None
        prob_positive_5y = np.mean([r > 0 for r in matches_5y]) * 100 if len(matches_5y) >= 10 else None

        prob_beat_spy_1y = np.mean(beat_spy_1y) * 100 if len(beat_spy_1y) >= MIN_SAMPLE_SIZE else None
        prob_beat_spy_3y = np.mean(beat_spy_3y) * 100 if len(beat_spy_3y) >= 10 else None

        median_return_1y = np.median(matches_1y) if matches_1y else 0
        median_return_3y = np.median(matches_3y) if matches_3y else None
        median_return_5y = np.median(matches_5y) if matches_5y else None

        downside_1y = np.percentile(matches_1y, 10) if len(matches_1y) >= 10 else None
        upside_1y = np.percentile(matches_1y, 90) if len(matches_1y) >= 10 else None

        # 5. OPPORTUNITY SCORE - Combine all factors
        # Higher score = better opportunity (undervalued + quality + recovering + good odds)

        score = 0

        # Discount bonus (more discount = more opportunity, up to a point)
        score += min(discount_from_high, 35)  # Max 35 points for discount

        # Quality bonus
        if five_year_return > 50:
            score += 20
        elif five_year_return > 20:
            score += 15
        elif five_year_return > 0:
            score += 10

        # Recovery signal bonus
        if early_recovery:
            score += 15
        if above_sma50:
            score += 5

        # Historical probability bonus
        if prob_beat_spy_1y and prob_beat_spy_1y > 60:
            score += 15
        elif prob_beat_spy_1y and prob_beat_spy_1y > 50:
            score += 10

        if prob_positive_1y > 70:
            score += 10
        elif prob_positive_1y > 60:
            score += 5

        # Cap at 100
        score = min(score, 100)

        # Determine signal
        if score >= 70 and prob_beat_spy_1y and prob_beat_spy_1y >= 55 and early_recovery:
            signal = "STRONG_BUY"
        elif score >= 55 and prob_positive_1y >= 60:
            signal = "BUY"
        elif score >= 40:
            signal = "WATCH"
        else:
            signal = "PASS"

        # Generate "Why Buy Now" thesis
        thesis = generate_thesis(ticker, discount_from_high, early_recovery,
                                  prob_beat_spy_1y, median_return_1y, five_year_return)

        return {
            "ticker": ticker,
            "price": round(current_price, 2),
            "signal": signal,
            "opportunity_score": round(score, 1),

            # Valuation
            "discount_from_high": round(discount_from_high, 1),
            "position_in_52w_range": round(position_in_range, 1),
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),

            # Quality
            "five_year_return": round(five_year_return, 1),
            "monthly_win_rate": round(monthly_win_rate, 1),
            "sharpe": round(sharpe, 2),

            # Recovery
            "early_recovery": bool(early_recovery),
            "above_sma20": bool(above_sma20),
            "above_sma50": bool(above_sma50),
            "momentum_1m": round(mom_1m, 1),
            "momentum_3m": round(mom_3m, 1),

            # Probabilities (the key output)
            "prob_positive_1y": round(prob_positive_1y, 0),
            "prob_positive_3y": round(prob_positive_3y, 0) if prob_positive_3y else None,
            "prob_positive_5y": round(prob_positive_5y, 0) if prob_positive_5y else None,
            "prob_beat_spy_1y": round(prob_beat_spy_1y, 0) if prob_beat_spy_1y else None,
            "prob_beat_spy_3y": round(prob_beat_spy_3y, 0) if prob_beat_spy_3y else None,

            # Expected returns
            "median_return_1y": round(median_return_1y, 1),
            "median_return_3y": round(median_return_3y, 1) if median_return_3y else None,
            "median_return_5y": round(median_return_5y, 1) if median_return_5y else None,
            "downside_1y": round(downside_1y, 1) if downside_1y else None,
            "upside_1y": round(upside_1y, 1) if upside_1y else None,

            # Sample sizes (for credibility)
            "sample_size_1y": len(matches_1y),
            "sample_size_3y": len(matches_3y),
            "sample_size_5y": len(matches_5y),

            # Thesis
            "thesis": thesis,

            # Price series (last 5 years for chart)
            "series": {
                "dates": [d.strftime("%Y-%m-%d") for d in close.index[-1260:]],
                "prices": [round(p, 2) for p in close.values[-1260:]]
            }
        }

    except Exception as e:
        return None


def generate_thesis(ticker, discount, early_recovery, prob_beat_spy, median_return, five_year_return):
    """Generate a simple 'Why Buy Now' thesis."""

    parts = []

    # Discount
    parts.append(f"{ticker} is trading {discount:.0f}% below its 52-week high")

    # Quality
    if five_year_return > 50:
        parts.append(f"with strong long-term performance (+{five_year_return:.0f}% over 5 years)")
    elif five_year_return > 0:
        parts.append(f"with solid long-term track record (+{five_year_return:.0f}% over 5 years)")

    # Recovery
    if early_recovery:
        parts.append("and showing early signs of recovery")

    # Probability
    if prob_beat_spy and prob_beat_spy > 55:
        parts.append(f"Historically, similar setups beat SPY {prob_beat_spy:.0f}% of the time with median return of +{median_return:.0f}%")
    elif median_return > 0:
        parts.append(f"Historically, similar setups returned +{median_return:.0f}% median over 1 year")

    return ". ".join(parts) + "."


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("CRT RECOVERY PREDICTOR v3.0 - Finding Undervalued Opportunities")
    print("=" * 80)

    # Check if market is open (skip on weekends unless forced)
    today = datetime.now()
    if today.weekday() >= 5 and not os.environ.get("FORCE_RUN"):
        print("\n[SKIP] Weekend - use FORCE_RUN=1 to override")
        return

    # 1. Get tickers
    print("\n[1] Building stock universe...")
    tickers = get_sp500_tickers()
    tickers.extend(["SPY", "QQQ", "DIA", "IWM"])  # Add ETFs for reference
    tickers = list(set(tickers))
    print(f"    Total universe: {len(tickers)} tickers")

    # 2. Download data
    print("\n[2] Downloading price data (max history)...")
    close_df, high_df, low_df = download_price_data(tickers)  # Fetch max available history

    if close_df.empty:
        print("[ERROR] No data downloaded")
        return

    print(f"    Data from {close_df.index[0].date()} to {close_df.index[-1].date()}")
    print(f"    {len(close_df.columns)} tickers with sufficient data")

    # Get SPY for benchmark comparison
    spy_close = close_df["SPY"] if "SPY" in close_df.columns else None
    if spy_close is None:
        print("[ERROR] SPY data required for benchmark comparison")
        return

    # 3. Analyze each stock
    print("\n[3] Analyzing stocks for undervalued opportunities...")
    results = []

    for i, ticker in enumerate(close_df.columns):
        if ticker in ["SPY", "QQQ", "DIA", "IWM"]:
            continue  # Skip ETFs

        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1} stocks...")

        close = close_df[ticker].dropna()
        high = high_df[ticker].dropna() if ticker in high_df.columns else close
        low = low_df[ticker].dropna() if ticker in low_df.columns else close

        metrics = calculate_opportunity_metrics(ticker, close, high, low, spy_close)

        if metrics:
            results.append(metrics)

    print(f"    Found {len(results)} undervalued opportunities")

    # Sort by opportunity score
    results.sort(key=lambda x: x["opportunity_score"], reverse=True)

    # 4. Summary
    print("\n" + "=" * 80)
    print("TOP 20 UNDERVALUED OPPORTUNITIES")
    print("=" * 80)
    print(f"\n{'Rank':<6}{'Ticker':<8}{'Score':<8}{'Signal':<12}{'Discount':<10}{'P(Beat SPY)':<12}{'Recovery':<10}")
    print("-" * 75)

    for i, item in enumerate(results[:20], 1):
        recovery = "Yes" if item["early_recovery"] else "No"
        beat_spy = f"{item['prob_beat_spy_1y']:.0f}%" if item["prob_beat_spy_1y"] else "N/A"
        print(f"{i:<6}{item['ticker']:<8}{item['opportunity_score']:<8.0f}{item['signal']:<12}"
              f"{item['discount_from_high']:.0f}%{'':<5}{beat_spy:<12}{recovery:<10}")

    # Count signals
    strong_buy = len([r for r in results if r["signal"] == "STRONG_BUY"])
    buy = len([r for r in results if r["signal"] == "BUY"])
    watch = len([r for r in results if r["signal"] == "WATCH"])

    print("\n" + "=" * 80)
    print(f"SUMMARY: {strong_buy} STRONG_BUY | {buy} BUY | {watch} WATCH")
    print("=" * 80)

    if strong_buy > 0:
        print("\nTop STRONG_BUY opportunities:")
        for item in [r for r in results if r["signal"] == "STRONG_BUY"][:5]:
            print(f"  {item['ticker']}: {item['thesis'][:100]}...")

    # 5. Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "as_of": datetime.now().astimezone().isoformat(),
        "model": {
            "name": "CRT Recovery Predictor",
            "version": "v3.0",
            "methodology": "Find undervalued quality stocks with early recovery signals",
            "min_history_years": MIN_HISTORY_YEARS,
            "min_sample_size": MIN_SAMPLE_SIZE,
            "discount_range": f"{MIN_DISCOUNT_PCT}%-{MAX_DISCOUNT_PCT}%"
        },
        "summary": {
            "total_analyzed": len(results),
            "strong_buy": strong_buy,
            "buy": buy,
            "watch": watch
        },
        "items": results
    }

    # Save main file
    with open(OUTPUT_DIR / "full.json", "w") as f:
        json.dump(output, f)

    # Save individual ticker files
    ticker_dir = OUTPUT_DIR / "tickers"
    ticker_dir.mkdir(exist_ok=True)

    for item in results:
        with open(ticker_dir / f"{item['ticker']}.json", "w") as f:
            json.dump(item, f)

    # Save timestamp
    with open(OUTPUT_DIR / "last_run.txt", "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d"))

    print(f"\n[OK] Results saved to {OUTPUT_DIR / 'full.json'}")


if __name__ == "__main__":
    main()
