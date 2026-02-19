# Implementation Plan: Quality-Filtered Rebound Scanner

## Goal
Simple, accurate tool: "What stock should I buy today that is most likely to be higher in 1, 3, 5 years?"

## Changes to daily_scan.py

### 1. Quality Gate (price-based, no fundamentals)

**A. Long-term trend health (new function: `trend_quality`)**
- Compute % of trading days in last 5Y where stock was above its 200-day SMA
- Score 0-100. Stocks with >60% of days above 200 SMA = healthy long-term trend
- This keeps AMZN/TSLA (mostly above 200 SMA even with wild swings)
- Flags GE/INTC-type multi-year decliners (spent most of 5Y below 200 SMA)
- Used as a multiplier on the final probability, NOT a hard filter

**B. Recovery history (new function: `recovery_track_record`)**
- Look at all prior drawdowns of >=20% for this stock in its history
- For each, measure: did it recover to prior high within 3 years?
- Recovery rate = (# recovered) / (# drawdowns)
- A stock that has recovered from 4/5 prior drawdowns = strong track record
- A stock that has recovered from 1/5 = weak (even if fundamentally great, it hasn't historically bounced)
- Used to adjust confidence in the rebound thesis

**C. Selling deceleration (new function: `selling_momentum`)**
- Compare recent 5-day return vs 20-day return
- If 5d > 20d (less negative recently), selling is decelerating = positive signal
- If 5d < 20d (more negative recently), still in freefall = negative signal
- Also check: is RSI(14) > 30 and turning up from below 30?
- Used as a gate: don't recommend stocks in active freefall

### 2. Survivorship Bias Fix

**In `forward_returns()` and analog scoring:**
- When forward price is NaN (stock delisted), treat as -100% return
- This means: if a stock went bankrupt 6 months after a "washed out" signal, that signal's 1Y return = -100%
- Dramatically more honest probabilities
- Add a "data_coverage" field: what % of analog forward returns have actual data vs imputed -100%

### 3. Simplified Output

**New item fields (replacing current complex scoring):**
- `prob_1y`: Probability of positive return in 1 year (0-100%)
- `prob_3y`: Probability of positive return in 3 years (0-100%)
- `prob_5y`: Probability of positive return in 5 years (0-100%)
- `quality`: Quality score 0-100 (from trend health + recovery history + selling deceleration)
- `conviction`: Combined quality × probability score - the "should I buy" number
- `median_1y`, `median_3y`, `median_5y`: Typical (median) return at each horizon
- `downside_1y`: 10th percentile 1Y return (worst realistic case)
- `n_analogs`: How many similar historical days this is based on

**Keep existing fields for detail view:**
- edge_score, washout_today, confidence, stability (for the detail cards)

### 4. Starting Universe (small, testable)
- Keep ALWAYS_PLOT as the initial universe (17 tickers)
- Add a few quality test cases: some solid recoverers + some structural decliners
- This lets us validate before expanding

## Changes to app.js / index.html

### 1. Hero Section Redesign
- "Top Convictions" at top: only stocks where quality >= 70 AND prob_1y >= 70%
- Each card shows: ticker, 1Y/3Y/5Y probability, quality badge, median expected return

### 2. All Stocks Table
- Default sort: by 1Y probability (descending)
- Columns: Ticker | Quality | 1Y Prob | 3Y Prob | 5Y Prob | Typical 1Y | Downside | Analogs
- User can click column headers to re-sort
- User can click a row to see the detail card

### 3. Detail Cards (when clicked)
- Price chart with score overlay (keep existing)
- Outcomes box (keep existing)
- Evidence section (keep existing)
- NEW: Recovery track record ("This stock has recovered from X/Y prior similar drawdowns")
- NEW: Quality breakdown (trend health, recovery history, selling momentum)

### 4. Remove/Simplify
- Remove "Final Score" as the hero metric (replace with probability)
- Remove "Edge Score" from the main table (keep in detail only)
- Remove "Washout" from main table (keep in detail only)
- Sort buttons: "1Y Prob" (default), "3Y Prob", "5Y Prob", "Quality"
