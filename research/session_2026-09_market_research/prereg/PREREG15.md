# PRE-REGISTRATION v15 — WEEKLY CROSS-SECTIONAL REVERSAL AT REAL ALPACA COSTS
Frozen 2026-09-05 before any v15 number is computed. Trial 42.

## Why this, after 41 failures (the reasoning, stated first)
Every one of the 41 died on cost. The binding ratio is (gross edge)/(round trip).
I have been testing 1-min to 1-day horizons where I trade often and the toll is
paid often. I measured today that the real Alpaca toll is ~5.4 bp round trip
(ZERO commission + 2.72 bp median Corwin-Schultz spread), not the 8-10 bp I
assumed. At a WEEKLY horizon that toll is paid once per week instead of ~26x
per day. That is a ~100x improvement in the cost arithmetic and it is the one
axis I have not moved.
This is deliberately NOT exotic. After 41 inventive failures the honest highest-
probability test is whether a documented effect survives at realistic cost on a
survivorship-free universe. Novelty is subordinated to probability of profit here,
and I am saying so rather than dressing it up.

## Data (survivorship-free, PIT)
equity_daily_broad open/close/dvol, 800 symbols, 2016-01-04..2026-08-11.
sp500_pit_members.csv -> PIT membership (includes delisted tickers).
Universe on week w = PIT members, valid prices, trailing-20d ADV > $20M.
TRAIN 2016-2021 | VALID 2022-2023 | TEST 2024-01-01.. LOCKED, not read.

## Signals (fixed now, no additions)
 R1 weekly reversal: rank on last week's close-to-close return, long losers /
    short winners, hold 1 week, rebalance weekly (Lehmann 1990, Jegadeesh 1990).
 R2 same, but SKIP the most recent day (avoids the bid-ask-bounce contamination
    that this session has repeatedly shown dominates shared-price tests).
 R3 12-1 month momentum, monthly rebalance (Jegadeesh-Titman), as the
    slow-horizon comparison.
 R4 NOVEL INCREMENT: condition R2 on sigma (entropy production). PREREG9 showed
    sigma is a stable 65x-ranging SYMBOL property (per-symbol t +2.8..+58.9) that
    measures information arrival. Hypothesis: reversal should be STRONGER in
    low-sigma names (moves are liquidity, they unwind) and WEAKER in high-sigma
    names (moves are information, they stick). sigma computed from daily returns
    over a trailing 60d window, causal.

## Costs (measured, not assumed)
Zero commission. Charge a FULL round-trip spread per name per rebalance, using
each name's Corwin-Schultz spread estimated from its own daily high/low.
Report gross, cost, and net separately. No net claim without the cost line.

## Pre-registered tests
 W1 R1/R2 net > 0, day-clustered t > 3 on VALID, same sign on TRAIN.
 W2 turnover and per-name cost reported explicitly.
 W3 SHUFFLE NULL: permute the signal across names within each week, 200 draws;
    real |t| must exceed the 99th percentile.
 W4 R4 must ADD to R2 -- residualize sigma on size/vol/spread first, or it is a
    liquidity proxy in disguise (the control that has killed 3 ideas this session).
 W5 effect present in both halves of each split.

## Kill
Net <= 0 after real costs -> dead, and if R1/R2 fail then the weekly horizon is
closed too and I will say the cost wall holds at every horizon I can reach.

## RESULT (2026-09-05): FAILED — and it failed differently, which matters
 R1 weekly reversal : TRAIN gross +19.7bp (t+1.71) -> VALID -17.7bp (t-0.97). SIGN FLIP.
 R2 skip-last-day   : TRAIN  +9.7bp (t+0.88)       -> VALID -10.2bp (t-0.54). SIGN FLIP.
 cost = 2.8 bp against a 20 bp gross edge -> COST WAS NOT BINDING HERE.
 The signal itself did not replicate. R3/R4 not run (R1/R2 gate failed).

## POSITIVE CONTROL (test OF THE TEST): 12-1 momentum, monthly
 TRAIN +3.1 bp/mo (t+0.05) | VALID -29.1 (t-0.33) | pooled -6.3 bp/mo t-0.12, n=82 months.
 INCONCLUSIVE, not a pass and not proof the harness is broken: momentum genuinely
 was flat-to-negative in S&P large caps 2017-2023. But n=82 months has almost no power.

## THE STRUCTURAL FINDING OF THIS SESSION (quantified, not asserted)
 t = Sharpe * sqrt(years). My bar is t>3. Years needed to CONFIRM:
   Sharpe 2.0 -> 2.2y | 1.5 -> 4.0y | 1.2 -> 6.2y | 1.0 -> 9.0y | 0.6 -> 25y | 0.5 -> 36y
 I hold 6y TRAIN + 2y VALID. At weekly/monthly rebalance I can ONLY ever confirm
 Sharpe >= ~1.2. A REAL Sharpe-0.6 strategy is INVISIBLE in my sample -- it would
 look identical to the 42 nulls I have reported.
 At intraday frequency I have thousands of observations and ample power, but there
 the edge/cost ratio is 0.1-0.2 and nothing survives.
 THE TWO CONSTRAINTS ARE OPPOSED:
   short horizon -> power YES, economics NO
   long  horizon -> economics YES, power NO
 Consequence: the ONLY thing I can both detect and afford is a LARGE effect
 (Sharpe > 1.2). Small-but-real edges are permanently undecidable on this data.

## TRIAL 42. TOTAL: 42 hypotheses, 0 tradable.
