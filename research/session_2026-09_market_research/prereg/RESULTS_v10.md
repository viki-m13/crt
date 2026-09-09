# PREREG10 — DISSIPATION PER UNIT ORDER FLOW.  KILLED.
2,500,930 events, 8 Binance perps, 1,096 days, 2023-01-01..2025-12-31.
TEST 2026-01-01..2026-08 LOCKED, not read.

## The idea
Kyle's lambda = price impact per unit flow (standard). Proposed: ENTROPY
PRODUCTION per unit flow, D = sigma_local/|F|, as a permanent-vs-transient
discriminator. Informed flow -> permanent -> irreversible -> high sigma.
Liquidity flow -> transient -> reversible -> low sigma. Fade only low-D.

## Result: dead on every pre-registered gate
 H3 low-D minus high-D spread: TRAIN k15 +0.47bp t+3.24 -> VALID +0.07bp t+0.32
 H2 residualized on (lambda, vol, trade size, volume), R^2 0.279:
    k15 t+0.66, k30 t+0.27, k60 t-0.65   -> adds nothing beyond known controls
 H5 per-symbol: 4 of 8 positive, none significant (best LINK t+1.45)

## The bigger error, stated plainly
H1 net payoff was ~-9.6 to -10.4 bp in EVERY D quintile. Because the
UNCONDITIONAL 5-min reversal payoff is ~zero gross: TRAIN +0.14/+0.05/-0.00 bp,
VALID +0.33/+0.15/+0.29 bp at k=15/30/60, against a 10 bp cost.
There was no base-rate edge for a conditioner to sort. I built a discriminator
for a trade that does not exist. CHECK THE BASE RATE BEFORE BUILDING A
CONDITIONER FOR IT.

## What IS real (and still not tradable)
Taker-flow fade: heavy taker BUYING predicts NEGATIVE forward returns.
  buy-quintile minus sell-quintile forward return
  TRAIN k15 -1.33 bp t-7.33 | k30 -1.60 t-6.23 | k60 -2.19 t-6.45
  VALID k15 -1.31 bp t-3.96 | k30 -1.32 t-2.98 | k60 -2.33 t-3.63
BOUNCE CONTROL (target started at t+1 instead of t) -- it SURVIVES:
  TRAIN -1.33 -> -1.33 (t-7.38);  VALID -1.31 -> -0.79 (t-2.47), k30 -1.32 -> -1.09 (t-2.46)
So: a genuine, replicating, non-artifact flow-reversal effect of ~0.8-1.6 bp.
Against a 10 bp taker round trip that is edge/cost 0.08-0.16. Not tradable as a taker.

## Also refuted this tick
The sigma venue map: crypto perps dissipate LESS than equities (median 0.00116
vs 0.00303; FX 0.00054). The "find a high-dissipation venue" thesis is dead, and
it was already invalid since PREREG9's T3' showed sigma ranks assets backwards.
I ran a test whose logic I had already falsified. Discarded.
Correct screen instead: 5-min move / round-trip cost = crypto 3.07, equity 2.04,
fx 1.81. Crypto headroom is 1.5x -- real but modest.

## RUNNING TOTAL: 37 hypotheses. 0 tradable.
