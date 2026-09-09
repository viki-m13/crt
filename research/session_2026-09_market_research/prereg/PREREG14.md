# PRE-REGISTRATION v14 — THE ASE INVERSION: BREAKOUT CONTINUATION
Frozen 2026-09-05 before any v14 number is computed. Trial 41.

## Where this comes from (a measurement, not a guess)
PREREG12 measured Adverse-Selection Efficiency across venues:
  binance-perp 1.005 | builder-dex 1.175 | US EQUITIES 1.213
ASE > 1 means adverse selection EXCEEDS the maker's offset. I filed that as
"market making is unattractive here". Its MIRROR IMAGE is untested: if a maker
posting at +delta loses 1.21*delta, the TAKER lifting them gains 0.21*delta
beyond the offset. ASE>1 IS a direct measurement of breakout CONTINUATION.
Median maker gross P&L in US equities was -4.35 bp per filled trade; the
counterparty's gross is the mirror of that.

## Why the cost arithmetic differs from my previous 40 attempts
Alpaca charges ZERO commission on US equities. The only cost is the spread.
Measured Corwin-Schultz median on these names: 2.72 bp. Prior tests assumed an
8-10 bp all-in toll. That assumption, not the signal, may have been the binding
constraint in several earlier kills.

## Construction (causal, PIT)
20 US equities/ETFs, 1-min bars resampled to 15-min, 2018..2023.
At bar t: if high_t > c_{t-1}*(1+delta) -> LONG breakout, entry at that level.
          if low_t  < c_{t-1}*(1-delta) -> SHORT breakout, entry at that level.
delta in {10, 25, 50} bp (fixed now). Hold k in {1,2,4} 15-min bars, exit at close.
ENTRY SLIPPAGE: a taker crosses the spread. Charge a FULL half-spread on entry
and a full half-spread on exit, using each symbol's measured Corwin-Schultz
spread. No commission (Alpaca).
TRAIN 2018-01-01..2021-12-31 | VALID 2022-01-01..2023-12-31
TEST 2024-01-01..2026-07-31 LOCKED, not read in this batch.

## Pre-registered tests
 B1 net P&L per trade > 0 after the spread charge, day-clustered t > 3 on VALID,
    same sign on TRAIN.
 B2 CONSISTENCY WITH THE MEASUREMENT: realized continuation must be ~0.2*delta,
    matching ASE=1.21. If the P&L does not scale with delta as ASE predicts,
    then B1 (if positive) is coming from something else and the mechanism claim
    is wrong -- report that.
 B3 SHUFFLE NULL: randomize the entry timestamps within each day (same count,
    same symbol), 200 draws. Real |t| must exceed the 99th percentile. This
    catches "any intraday entry makes money" artifacts.
 B4 the effect must appear in >= 12 of 20 symbols on VALID.
 B5 CONTROL vs a same-cost baseline: random-entry trades with identical holding
    period and identical spread charge. Breakout must beat it, clustered t > 3.

## Kill
B1 <= 0 net, or B3/B5 fails -> dead, report and move on.
No re-tuning of delta/hold after seeing results; the grid is fixed now.

## RESULT (2026-09-05): FAILED, and the MECHANISM CLAIM WAS WRONG
B1 net P&L: -3.24 to -6.21 bp in ALL 18 cells, TRAIN and VALID agreeing.
B5 control: breakout entries are WORSE than RANDOM entries (diff -0.03 to -1.88 bp,
TRAIN t to -4.57). Not merely unprofitable -- actively worse than noise entries.
WHY MY INVERSION WAS WRONG: ASE>1 says the MAKER loses relative to their offset,
but the maker's "gross" already embeds the +delta price improvement. The taker on
the other side does NOT inherit that as a gain -- they pay spread on both legs.
The two sides' frictions do not mirror. My reasoning error, stated plainly.
TRIAL 41. TOTAL: 41 hypotheses, 0 tradable.
