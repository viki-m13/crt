# PRE-REGISTRATION v16 — VOLATILITY CARRY (no options, Alpaca ETFs)
Frozen 2026-09-05 before any v16 number is computed. Trial 45.

## Why this passes my own detectability bar when 44 others did not
Power analysis (PREREG15): with 8y I can only confirm Sharpe >= ~1.2; with the
36y VIX history and ~15y of ETP history I can confirm ~0.8. Short-vol carry is
the canonical large-effect, non-prediction, structural-premium trade -- the same
SHAPE as this account's only live-validated winner (Polymarket weather fade:
sell insurance, fat margin). It is not a forecast. No options involved.

## THE INTEGRITY ISSUE, STATED BEFORE ANY RESULT
Short-vol backtests are the most survivorship-poisoned in finance.
 - XIV (the -1x VIX ETN) LOST 96% ON 2018-02-05 and was liquidated. It is GONE
   from every dataset. Any dataset showing a clean short-vol equity curve is
   showing the SURVIVOR.
 - SVXY still exists ONLY because it re-levered from -1.0x to -0.5x after that
   day. Its post-2018 series is a DIFFERENT INSTRUMENT from its pre-2018 series.
 - Therefore: a headline Sharpe here is NOT evidence. The tail is the whole
   question, and I will report worst-day and max-drawdown BEFORE any Sharpe.
Any result presented without those two numbers is invalid by this registration.

## Data
daily_multiasset: IDX_VIX (1990-2026, 9219d), IDX_VIX3M (5032d), VXX (2146d),
SVXY (3733d), VIXY (3922d), SPY (8439d). Fields o/h/l/c/adj, date col 'd'.
TRAIN through 2019-12-31 | VALID 2020-01-01..2023-12-31 |
TEST 2024-01-01.. LOCKED, not read.

## Tests, in this order (tail first, deliberately)
 T1 TAIL: worst 1-day return, worst 5-day, max drawdown of each candidate
    position, over the FULL available history. Reported before anything else.
 T2 UNCONDITIONAL: short VXX / long SVXY buy-and-hold. Sharpe, and the tail.
 T3 CONDITIONED on term structure: hold the short-vol position only when
    VIX3M > VIX (contango); flat otherwise. Signal uses PRIOR close only.
 T4 Does the condition actually reduce the tail, or just the return? Report the
    tail WITH and WITHOUT the filter. A filter that keeps Sharpe but keeps the
    -90% day is worthless.
 T5 COSTS: ETF round-trip spread + short borrow. Report turnover explicitly.
 T6 SVXY REGIME SPLIT: measure pre-2018-02-05 and post separately. They are
    different instruments and must not be pooled into one curve.

## Kill
If the tail-conditioned strategy still carries a >50% drawdown, it is not
deployable at any Sharpe and I will say so rather than quote the Sharpe.
If T3 does not beat T2 on tail-adjusted terms, the timing adds nothing.
