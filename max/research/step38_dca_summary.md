# Step 38: DCA Scaling by Market Drawdown — Summary

**Date:** 2026-04-21
**Hypothesis:** Scaling the monthly DCA amount up during SPY drawdowns
(value-averaging style) should improve 20Y outcomes vs flat DCA.

## Headline finding (step 38b, 20Y full period)

| Variant | CAGR | Sharpe | Invested | Final |
|---|---|---|---|---|
| flat (incumbent) | +17.41% | 1.34 | $241k | $5.92M |
| 2x at SPY -15% | +17.83% | 1.36 | $262k | $6.90M |
| 3x at SPY -15% | +18.62% | 1.39 | $323k | $9.75M |
| **5x at SPY -15%** | **+19.44%** | **1.42** | $405k | $14.0M |
| 5x at SPY -10% | +19.09% | 1.43 | $477k | $15.6M |

On 20Y full-period, dd-scaling looks like a clear win: +1-2pp CAGR
and higher Sharpe. ROIC per extra dollar invested: 4000-5000% over 20Y.

## BUT — rolling 10Y and calendar year (step 38c/38d)

**The 20Y win is almost entirely a GFC artifact.**

Rolling 10Y CAGR (step 38c):

| Window | Flat | 3x @ -15% | Δ |
|---|---|---|---|
| 2006-04 → 2016-04 | +459.29% | +490.09% | **+30.80pp** ← GFC window |
| 2008-04 → 2018-04 | +37.55% | +44.49% | +6.94pp |
| 2010-04 → 2020-04 | +30.49% | +27.55% | **-2.94pp** |
| 2012-04 → 2022-04 | +29.15% | +26.19% | **-2.96pp** |
| 2014-04 → 2024-05 | +31.68% | +30.22% | **-1.46pp** |

**Median rolling-10Y CAGR is WORSE for dd-scaled** (+30.22% vs +31.68%).
Mean is higher only because of the GFC window.

Calendar-year win rate since 2008 (step 38d):

| Variant | Wins vs flat |
|---|---|
| 3x at -20% | 5/18 |
| 3x at -20% + above 50DMA (rebound confirm) | 6/18 |
| 3x at -20% + above 20D high (strong confirm) | 6/18 |
| 3x when SPY < 0.9×200DMA (persistent bear) | 3/18 |

**Most variants lose in 2/3 of years.** The big gains are concentrated
in 2008-2009 (GFC: 3x-at-20% scored +36.72% / +207.14% vs +8.96% /
+111.90% for flat), with small wins in 2010, 2021, 2022, 2023.

## Interpretation

- The -15% threshold triggers on corrections like 2011, 2015, 2018,
  2020 where the drawdown was short-lived. Scaling up capital there
  just dilutes the bull market with below-avg entries.
- -20% triggers less often but also catches "falling knife" moments.
- Rebound-confirmation filters (above 50DMA, above 20D high) help
  marginally but don't change the structural verdict: dd-scaling loses
  outside of protracted downturns (2008-10, 2022).

## Decision

**DO NOT promote dd-scaling to default.** It's not a robust alpha
improvement — it's a crisis hedge that pays off only in 2008-style
extended crashes.

Offer as an **optional crisis overlay** for investors with:
1. Extra discretionary capital (variant uses 12-70% more $$ over 20Y)
2. Willingness to underperform in bull markets
3. Belief that another GFC-like event is plausible

Recommended crisis overlay: **2x at SPY -20% + rebound confirm (above
50DMA)**. Minimal capital commitment (~10% more over 20Y), modest
drag in bull markets (-1pp CAGR), meaningful crisis alpha (+20-30pp
during GFC-like periods).

## What this validates about CAP5

**Static flat-DCA CAP5 remains the best robust strategy.** Every
sophisticated overlay attempted so far (walk-forward, factor combos,
sector caps, score thresholds, dd-scaling) either ties CAP5 or trades
CAGR for MaxDD. CAP5's edge is mean-reversion at the stock level, not
market timing at the portfolio level.
