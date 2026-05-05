# Stillpoint — Compression-Conditioned Microbuffer Engine

A **proprietary novel** credit-spread engine that publishes **tight strikes**
(short strike within ~5% of spot) on **short expirations** (5-21 trading
days, i.e. < 21 DTE) with **walk-forward out-of-sample win rate ≥ 95%**.

This is a *third* engine on the `/spreads/` page, kept entirely separate
from the existing CreditFloor (100%-OOS) and Option C (event-triggered)
strategies. It does not modify either of them.

## The novel idea

Most credit-spread engines try to set a buffer never breached over a horizon.
That works only when buffers are wide (10-25%). With a *tight* strike at
short DTE, the empirical 100%-of-history threshold balloons. Stillpoint
takes a different route:

1. **Stillness regime gate (proprietary)** — restrict to days where the
   stock is in a compression-volatility regime: 20d annualized vol < 40%,
   5d/20d vol ratio < 1.05 (calm or compressing), 20d range < 15%, price
   within 4% of its 20d SMA, RSI(14) in [25, 75], and |5d return| < 8%.
   Conditional on this, the next-h-day path distribution is far thinner
   than the unconditional one — its 97th percentile is materially smaller.
2. **Conformal q-quantile strike** — set the buffer to the 97th-percentile
   path-buffer in training plus 0.5% safety. By the order-statistic
   interpretation that targets ~97% in-sample coverage. We then *honestly
   verify* this on walk-forward test years and reject anything whose
   pooled OOS win rate falls below 95%, or whose worst fold drops below
   85%.
3. **Pinpoint** — when both put and call sides clear all gates
   simultaneously, the combined trade is an iron-condor pin: spot
   expected to stay in a narrow band through expiry, each side carrying
   its own ≥95% historical win rate.

## Files

- `sp_common.py` — Stillpoint regime features, masks, buffer helpers.
  Importing the data loaders and the path-buffer math from
  `../credit_spread/common.py` via `importlib` (avoids module-name
  collision with the sibling package).
- `research.py` — walk-forward conformal-q research engine. Writes
  `results/stillpoint_signals.json`.
- `scan.py` — convenience driver: runs research and copies the signals
  JSON into `spreads/docs/data/`.
- `sweep.py`, `diagnose.py` — experiment scaffolding used to tune the
  proprietary thresholds.
- `results/` — generated output.

## Algorithm — TL;DR

For each `(ticker, side, horizon h ∈ {5,7,10,14,21})`:

1. Compute Stillpoint mask using only `close[0..t]`.
2. For each fold year `Y ∈ {2020..2026}`:
   - Train mask: `(t < Jan 1 Y) AND (t+h < Jan 1 Y)` AND `Stillpoint(t)` — purged.
   - Test mask: `(Jan 1 Y ≤ t < Jan 1 Y+1)` AND `Stillpoint(t)`.
   - `b̂ = quantile_0.97(b*(t,h) on train) + 0.5%`
   - Win iff `b*(t,h) ≤ b̂`.
3. Eligible iff:
   - pooled OOS win rate ≥ 0.95
   - every fold ≥ 0.85
   - ≥ 4 distinct fold years tested, ≥ 40 pooled tests
   - final live `b̂` ≤ 5%
4. Deployable today iff today's features satisfy the Stillpoint mask.

Strike arithmetic:
- Put: `K = spot × (1 - b̂)` (just below spot)
- Call: `K = spot × (1 + b̂)` (just above spot)

## Running

```bash
cd strategies/stillpoint
python3 research.py            # full universe; writes results/stillpoint_signals.json
python3 scan.py                # research + publish to spreads/docs/data/
SP_LIMIT=N python3 research.py # smoke-test with first N tickers
```

## What this does *not* guarantee

A 95%+ historical OOS win rate is empirical, not axiomatic. The regime
gate is the entire defense — if a stock is no longer in stillness today,
we do not publish. Future paths can break the conformal bound. Use at
your own risk. Not financial advice.
