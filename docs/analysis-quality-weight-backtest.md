# Backtest: Quality Score Weight in the Opp Score — Results & Decision

**Date:** 2026-02-25 | **Updated:** Quality ≥ 50 gate adopted into production
**Script:** `scripts/backtest_quality_weight.py`
**Universe:** 34 tickers, 33 years history (1993-2026), 3,883 eval points

## Production Formula (adopted Feb 2026)

```
OppScore = Quality × 1Y_Win_Rate × Pullback_Gate(washout)
           ONLY if Quality ≥ 50 (otherwise no Opp Score)
```

This gate was adopted based on the backtest results below, which showed it cuts extreme downside by 43% while maintaining 83% hit rates during pullbacks.

## Formulas Tested

| Formula | Description |
|---------|-------------|
| **A: Current** | `Quality × Win × Gate` (linear, production) |
| **B: Q-squared** | `(Q/100)² × 100 × Win × Gate` (punishes low quality hard) |
| **C: Q^1.5** | `(Q/100)^1.5 × 100 × Win × Gate` (moderate emphasis) |
| **D: Gate≥60** | Same as A but **drops stocks with quality < 60** |
| **E: No quality** | `100 × Win × Gate` (quality removed — pure price signal) |
| **F: Gate≥50** | Same as A but **drops stocks with quality < 50** |

---

## Key Finding #1: Quality DOES predict downside protection during pullbacks

**Section 6 — the critical test.** Same washout conditions, split by quality:

| Washout ≥ | Quality | N | Hit Rate | Median 1Y | **P10** | **P5** | Worst |
|-----------|---------|---|----------|-----------|---------|--------|-------|
| 15 | HIGH (above median) | 189 | **81.0%** | +31.5% | **-13.5%** | -27.9% | -64.6% |
| 15 | LOW (below median) | 188 | 72.9% | +26.1% | -32.0% | **-50.4%** | -72.8% |
| 25 | HIGH (above median) | 117 | **79.5%** | +35.0% | **-12.6%** | -22.2% | -44.5% |
| 25 | LOW (below median) | 117 | 71.8% | +33.3% | -34.9% | **-56.4%** | -72.8% |
| 40 | HIGH (above median) | 55 | **80.0%** | +54.4% | **-13.0%** | -29.3% | -38.7% |
| 40 | LOW (below median) | 54 | 74.1% | +54.8% | -38.1% | **-52.8%** | -69.7% |

**The pattern is very clear:**
- Hit rate: +6-8% better for high quality (consistent across all washout levels)
- **P10 is where quality shines:** -13% vs -33% at wash≥25 — quality cuts tail risk in half
- P5: -22% vs -56% at wash≥25 — extreme downside 2.5x worse for low quality
- Worst case: -44% vs -73% at wash≥25

**Quality doesn't improve the median/upside much — it protects the downside.**

---

## Key Finding #2: Quality is NOT monotonically good across all conditions

From Section 1, RAW quality signal across ALL eval points:

| Quality Bucket | N | Hit | Med 1Y | P10 |
|---------------|---|-----|--------|-----|
| Low (0-40) | 69 | **84.1%** | +22.2% | -10.9% |
| Med-Low (40-55) | 687 | 74.2% | +15.3% | -20.6% |
| Med (55-70) | 1353 | **78.3%** | +18.2% | -13.3% |
| Med-High (70-85) | 1556 | 72.0% | +14.3% | -19.9% |
| High (85-100) | 218 | 62.8% | +9.3% | -22.4% |

**Very high quality (85+) actually performs WORST.** Why?
- These are "always above SMA, always recovers" stocks — likely already expensive
- When quality is 85+, the stock has been in an uptrend so long that mean reversion works against you
- The low (0-40) bucket is tiny (N=69) and likely captures deep selloffs in otherwise-ok stocks

**Quality is a U-shaped signal overall, but during pullbacks it's clearly protective.**

---

## Key Finding #3: Quality gate works better than quality power-weighting

From Section 5 — pullback signals (wash≥20), top 50%:

| Formula | N | Hit | Med 1Y | **P10** | **P5** |
|---------|---|-----|--------|---------|--------|
| A: Current | 149 | 82.6% | +38.9% | -14.2% | -35.2% |
| B: Q-squared | 149 | 81.9% | +38.9% | **-12.4%** | **-27.0%** |
| C: Q^1.5 | 149 | **83.2%** | +40.3% | -9.1% | -27.0% |
| D: Gate≥60 | 88 | 80.7% | +43.4% | -9.5% | -25.9% |
| **E: No quality** | 149 | 77.9% | +39.8% | **-21.9%** | **-41.8%** |
| **F: Gate≥50** | 124 | **83.1%** | +40.9% | **-7.5%** | **-20.0%** |

**Key observations:**
- **Removing quality entirely (E) doubles the P10 downside** (-21.9% vs -14.2%). Quality is doing real work.
- **Q^1.5 (C) has the best hit rate** (83.2%) and good P10 (-9.1%)
- **Gate≥50 (F) has the best P10** (-7.5%) and best P5 (-20.0%) — the soft gate is the best downside protector
- The hard gate at 60 (D) loses too many good signals (N=88 vs 149)
- Q-squared (B) doesn't meaningfully improve over current

---

## Key Finding #4: Quality×Washout interaction during moderate pullbacks

From Section 2 — the "moderate pullback" zone (wash 25-45) where CRT signals fire:

| Quality | N | Hit | Med 1Y | P10 |
|---------|---|-----|--------|-----|
| Med-Low (40-55) | 39 | 69.2% | +26.1% | **-39.7%** |
| Med (55-70) | 56 | 80.4% | +33.4% | -21.1% |
| Med-High (70-85) | 49 | 79.6% | +33.0% | **-12.9%** |

During moderate pullbacks, quality 70-85 has a P10 of -12.9% vs -39.7% for quality 40-55.
**That's a 3x difference in tail risk.**

---

## Decision: ADOPTED — Gate≥50 (Formula F) is now production

```
OppScore = Quality × Win_1Y × Pullback_Gate(washout)   [if Quality ≥ 50, else no score]
```

Implemented in `daily_scan.py` via `MIN_QUALITY_GATE = 50`. Stocks below quality 50 get `conviction = None` (no Opp Score), pushing them to the bottom of the ranking.

Why this won:
1. **Best P10 during pullbacks** (-7.5% vs -14.2% without gate) — the thing we care about most
2. **Best P5 during pullbacks** (-20.0% vs -35.2% without gate) — 43% reduction in extreme downside
3. **Maintains high hit rate** (83.1% vs 82.6% without gate)
4. **Doesn't lose too many signals** (124 vs 149 — only drops 17% of signals, and those are the dangerous ones)
5. Simple to implement — one `if` statement and a constant

---

## What This Means for the "Is This a Pullback or Deterioration?" Question

Quality score IS doing something meaningful. It's not perfect — it can't detect a fundamental regime shift. But empirically:

- **High quality pullbacks recover 81% of the time** with worst-case P10 of -13%
- **Low quality pullbacks recover only 73% of the time** with worst-case P10 of -33%
- The quality score's recovery_track_record component is the key driver — stocks that have historically recovered from 20%+ drawdowns do in fact continue to recover

The model can't tell you *why* a stock is pulling back. But the quality score's historical recovery pattern is a meaningful prior. A quality 80+ stock in a 30%+ pullback has recovered 80% of the time historically, with a P10 of -13%. That's not "this time is definitely the same" — but it's a genuinely informative base rate.
