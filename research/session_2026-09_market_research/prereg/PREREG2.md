# PRE-REGISTRATION v2 — Frequency-Resolved Lead-Lag ("phase flip")
Frozen BEFORE results. Carries a stated debt: 2 prior trials on TRAIN (v1 H3, v1 mechanism).
TEST 2024-01-01..2026-07-31 remains LOCKED and untouched.

## What v1 taught (why this design differs)
v1 died because: whole-spectrum statistic, single asset, single timescale, and an
estimator whose sd equalled the entire observed range (white-noise null matched to
4 decimals). v2 inverts all four: BAND-limited, TWO assets, PHASE not power, and
Welch-averaged over segments x days so estimator variance falls like 1/N.

## Mechanism
Information reaches related assets at timescale-dependent speed:
  FAST band  - idiosyncratic news hits the SINGLE NAME first -> constituent LEADS ETF
  SLOW band  - basket flow (creation/redemption, rebalancing, macro) hits the
               BASKET first -> ETF LEADS constituent
If the sign of the lead FLIPS across bands, a single-lag regression averages the two
into ~0. That is why standard lead-lag tests on liquid US pairs report nothing.

## Hypotheses
H1 (existence)  Band-resolved lead-lag beta is significantly NON-ZERO in at least one
   band for ETF-constituent pairs, while the ALL-BAND (unfiltered) beta is ~0.
H2 (the flip)   sign(beta_fast) != sign(beta_slow) for ETF-constituent pairs.
H3 (specificity) The flip is WEAKER or absent for pairs with no basket relationship
   (e.g. GLD/TLT), i.e. it is not a generic artifact of bandpass filtering.

## Bands (fixed now, in minutes of period)
FAST  2-6      MID  6-20      SLOW  20-90

## Pairs (fixed now)
BASKET pairs   : SMH-NVDA, SMH-AMD, QQQ-AAPL, QQQ-MSFT, QQQ-NVDA, XLF-... (n/a), SPY-AAPL
CONTROL pairs  : GLD-TLT, GLD-HYG, TLT-XLE   (no basket membership)
PAIR-CONTROL   : GLD-SLV (related but NOT basket/constituent)

## Measurement (step 1, non-causal filter ALLOWED - establishes the phenomenon only)
Per session, bandpass both 1-min log-return series by zeroing FFT coefficients
outside the band. Regress follower_band[t+1] on leader_band[t], pooled, clustered by day.

## Trading test (step 2, STRICTLY CAUSAL - only this counts for P&L)
Bandpass replaced by causal EMA differences: band = EMA(n1) - EMA(n2), one-sided.
Signal at bar t uses only bars <= t. Trade next bar. Charge measured spread.

## Splits
TRAIN 2018-2021 (all decisions)  VALID 2022-2023 (sign must replicate)  TEST locked.

## Kill criteria (pre-committed)
- H2 flip does not appear in TRAIN                       -> DEAD
- Flip sign does not replicate in VALID                  -> DEAD
- H3 fails (controls show the same flip)                 -> ARTIFACT of filtering, DEAD
- Causal (step 2) beta is not the same sign as step 1    -> not implementable, DEAD
- Net of costs negative                                  -> report as measurement only
