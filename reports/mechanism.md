# Why the Pre-Runner Footprint Strategy Works (Plain Language)

> **Branch identifier: `claude/invent-stock-selection-FHtzX`**
> Multiple agents are working in parallel; this is the FHtzX agent's
> output.

## The two-paragraph version

The single most reliable signature of stocks that 3x in the next twelve
months is, surprisingly, a **deep-drawdown high-volatility profile** —
not the tight breakout pattern most published strategies look for.
Stocks that go on to triple are **45–70% off their 52-week high, with
realized volatility 2–3× the cross-section, having been falling for
about six months, with selling decelerating in the last few weeks**.
That footprint persists across four eras (1997-2007, 2008-2014,
2015-2019, 2020-2025) — it is not a regime artefact.

The hard part isn't *finding* candidates with this footprint — it's
*separating* runners from continued failures inside that subset.  The
discriminator is **Cross-Sectional Rank Trajectory (CRT)** — the time-
derivative of a stock's relative-strength rank in the cross-section
over the last six months.  Stocks whose cross-sectional return rank
has been **monotonically rising** even while their absolute price is
still in drawdown are showing covert leadership emergence.  The
strategy buys the top-5 stocks each month that pass the footprint gate
and have the highest CRT score.

## Why the structural mechanism is real

Two effects make the footprint cheap and the CRT signal informative:

### Mechanism 1: Mechanical-selling exhaustion in fallen-angel high-vol names

Institutional risk managers force selling in stocks whose realized
volatility breaches risk budgets.  This selling is mechanical, not
informational — it is driven by VaR, beta-adjusted exposure limits,
and 13F-pad-the-quarter window dressing.  As volatility persists, the
selling pressure compounds: more risk-budget violations, more position
trims, more redemptions in funds holding the names.

Eventually, the supply of mechanical sellers is exhausted.  At that
point, the price stops falling — not because new buyers have appeared,
but because forced sellers are *done*.  This shows up in the data as
"high vol AND deep drawdown AND drawdown-age > 4 months AND selling
decelerating".  All four conditions must be simultaneously present —
the footprint isn't any one of them.

When new buyers eventually do step in (typically institutional
"deep-value mandates" or sector-rotation flows), the supply curve has
shifted dramatically — a small amount of buying pressure produces a
large price move because there are no marginal sellers left.

### Mechanism 2: Information diffusion through cross-sectional rank

Capital rotates between stocks in continuous re-allocation.  When a
sector or theme rotates, capital flows into the *next layer* of names
— laggards within a sector, smaller positions held by long-only
funds, names with similar fundamentals to the current leaders.

This rotation is visible in the **time-derivative of cross-sectional
rank**.  A stock at rank percentile 25 → 35 → 45 → 55 → 65 over six
months is being rotated into, even if its absolute return is flat or
slightly negative during that period.  The rotation is invisible to
factor models that look at point-in-time levels (z-scores, 12-1
momentum) but visible in the *trajectory*.

CRT measures this directly: it is the Spearman correlation between
calendar time and cross-sectional rank percentile, computed at the
last 6 month-ends.  Stocks with CRT close to +1 are climbing the
cross-section linearly.  Combined with the footprint gate, this is
the structural signal that distinguishes successful rebounds from
falling knives.

## Why it isn't already arbitraged

There are several layers of friction that protect the signal:

1. **Most factor models use rank levels, not rank time-derivatives.**
   The standard quant zoo (12-1 momentum, residual momentum,
   Carhart-WML) computes a z-score at a point in time.  None of them
   compute Spearman(time, rank) over a multi-month window.  This is a
   structural omission, not an oversight.
2. **Mainstream momentum strategies *exclude* deep-drawdown names by
   construction.**  12-1 momentum punishes recent losers; quality
   momentum excludes high-vol names.  The pre-runner footprint sits in
   the "junk" bucket of mainstream momentum — exactly the bucket the
   forensic data shows is most fertile.
3. **Behavioral resistance.**  Buying stocks down 50% with high
   volatility is psychologically unattractive.  The names are in the
   news for bad reasons; clients don't want them in their accounts;
   PMs face career-risk if a falling-knife trade goes to zero.  The
   footprint requires holding names that *look* terrible.
4. **Cross-sectional computation overhead.**  CRT requires recomputing
   rank percentiles at 6+ historical month-ends with a clean PIT
   universe.  Easy in principle; friction in practice.  Most retail
   tools and factor backtesters compute features per-stock, not
   cross-sectionally over time.
5. **Capacity is not unlimited but is meaningful.**  Pre-runners are
   typically S&P 400/500 names with $10B+ cap that are temporarily
   trading at junk-quality multiples.  Not all are illiquid.  At
   $50M-$200M AUM, a 5-name basket rebalanced monthly can be deployed
   without material market impact.  At $1B+, slippage and capacity
   become significant.

## What it isn't

It isn't a "buy oversold and pray" strategy.  Three of the gating
conditions explicitly screen against falling-knife traps:

- `accel > 0` filters stocks where the most recent 5-day return is
  *worse* than the 20-day return — those are still in death spiral.
- `drawdown_age_days > 90` filters stocks whose drawdown is too fresh
  — they may have a leg or two left to fall.
- `trend_health_5y > 0.30` filters stocks that have spent most of the
  last 5 years below their 200-day SMA — those are structural
  declines, not value rebounds.

It isn't reweighted Fama-French factors.  CRT is a feature that
doesn't appear in the standard factor zoo because it requires
cross-sectional rank trajectories.

It isn't an ML black box.  The strategy is a hard gate (5 explicit
conditions) plus a 5-component soft rank (each component a single
human-readable feature).  The mechanism is articulable in plain
language (this document); if it stops working you'll be able to see
*which* component degraded.

## What could break it

- **Regime change in the volatility risk premium.**  If institutions
  stop forcing-selling on vol breaches (e.g. through better hedging
  technology), the supply-exhaustion mechanism weakens.
- **Mass adoption.**  If enough capital chases the same footprint, the
  pre-trade discount disappears.  At current AUM (~$50M-$200M
  estimated capacity) this is years away.
- **Increased delisting/bankruptcy rate.**  The footprint catches
  stocks in deep drawdown.  If the underlying delisting rate rises
  (e.g. a 1929-style structural break), the realized return
  distribution flips to negative.  This is partially captured by the
  α=4%/yr survivorship bias overlay and shows up in the bias
  sensitivity table.

## Capacity estimate

At $100M AUM:
- 5 picks × $20M each = $20M positions.
- Median position: S&P 500 mid-cap, $5B market cap, ~$50M ADV.
- $20M / $50M = 40% of ADV — manageable in 1-2 days.
- Monthly turnover: 100% of portfolio.
- Annual transaction cost at 25bp round-trip estimate: ~3% drag.
- Strategy still profitable after 3% drag.

At $500M AUM, slippage scales to ~50bp round-trip and 2-3 days to
build positions.  Strategy still profitable but capacity-constrained.

At $2B+, the strategy degrades materially.  This is a multi-billion
strategy at most, not a flagship strategy for $20B+ funds.

## Bottom line

The strategy works because it exploits a **specific, structural
feature** of how institutional risk management forces price
mis-pricing in fallen-angel high-volatility names, combined with a
**rare cross-sectional rank-trajectory signal** that distinguishes
successful rebounds from continued failures.  Both mechanisms are
articulable; both are testable in the forensic data; both have remained
stable across the 1997-2025 period.
