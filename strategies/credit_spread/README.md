# CreditFloor — Walk-Forward Conformal Credit-Spread Engine

A deliberately selective engine that publishes `(ticker, side, expiry h,
short strike K)` triples whose walk-forward out-of-sample win rate is
exactly 100%. Runs in **both directions**:

- **Put side** (floor below spot): stock's close never fell below `K` on
  any of the `h` trading days after entry. Sell put credit spreads.
- **Call side** (ceiling above spot): stock's close never rose above
  `K` on any of the `h` trading days after entry. Sell call credit
  spreads.

The historical guarantee is on the short strike — it was never breached
on any OOS test across every walk-forward fold.

## Files

- `common.py`        — data loading, causal features, forward targets, walk-forward masks.
- `research.py`      — the walk-forward conformal strike estimator and universe evaluation. Writes `results/research_full.json` and `results/signals.json`.
- `scan.py`          — convenience driver that runs `research.py` and publishes `signals.json` into `spreads/docs/data/` for the webapp.
- `results/`         — generated outputs (gitignored in production; checked in here so the static webapp has something to load).

## Algorithm — TL;DR

For each ticker `T`, each side `∈ {put, call}`, each horizon `h ∈ {21, 42, 63, 126}`:

1. Compute the path buffer:
   - Put:  `b*(t, h) = 1 - min(close[t+1..t+h]) / close[t]` (how far below entry)
   - Call: `b*(t, h) = max(close[t+1..t+h]) / close[t] - 1` (how far above entry)
2. For each walk-forward fold year `Y ∈ {2020..2026}`:
   - Train mask: `t < Jan 1 Y` AND `t+h < Jan 1 Y` (purged gap — no overlap with test).
   - Test mask: `Jan 1 Y ≤ t < Jan 1 Y+1`.
   - Conformal buffer: `b̂ = max(b* over train) + 1%` (flat safety margin).
   - Win iff `b*(t, h) ≤ b̂` on the test sample.
3. Eligible iff every fold is 100% with ≥1 win, pooled test count ≥ 50,
   and final full-history buffer ≤ 25%.
4. Run two variants per ticker per side: plain, and regime-gated.
   - Put regime:  `close ≥ SMA200` AND `dd252 ≤ 20%` (uptrend)
   - Call regime: `close ≤ SMA200` AND `up252 ≤ 20%` (bearish, no post-bottom rally)
   The regime variant is only deployable today if today's features
   also satisfy the gate. Per ticker we keep whichever passes with the
   tighter buffer.
5. Aggregate eligible (ticker, side, variant, horizon, fold) OOS
   predictions across the entire universe — any loss anywhere, on
   either side, fails the entire method.

Strike arithmetic:
- Put:  `K = spot * (1 - b̂)`    (below spot)
- Call: `K = spot * (1 + b̂)`    (above spot)

## Leakage controls

- Features at `t` use only `close[0..t]`.
- Targets use only `(t, t+h]`, never mixed into features.
- Purged training: any sample whose forward window crosses into the test
  year is excluded from training.
- Fixed safety margin (1%) and buffer cap (25%), never tuned on test data.
- Variant choice (plain vs regime) is made on buffer size, not on win rate
  (both must already be 100%).
- 252-day warmup dropped per ticker before any feature is consumed.

## Running

```bash
cd strategies/credit_spread
python3 research.py          # ~60s for ~964 tickers; writes results/
python3 scan.py              # same + publishes into spreads/docs/data/
```

Env knobs:
- `CS_LIMIT=N` — process only the first `N` tickers (smoke test).

## Results (last run)

See `results/signals.json`. Two-sided, pooled across the whole universe:
- 964 tickers scanned
- **Put side: 35 eligible, 33,099 OOS tests, 0 losses**
- **Call side: 10 eligible, 6,158 OOS tests, 0 losses**
- Combined: 45 signals, 39,257 OOS tests, **100.000%** pooled win rate

## Webapp

The `/spreads/` route serves `spreads/docs/` and consumes
`spreads/docs/data/signals.json` (shape:
`{summary, put_signals[], call_signals[]}`). The page has two
collapsible sections — put-credit floors and call-credit ceilings —
**both collapsed by default**. Each card shows a ticker with a
multi-horizon ladder; clicking "Show fold-by-fold walk-forward
breakdown" reveals the per-year test results.

## What this does *not* guarantee

A 100% historical OOS win rate is empirical, not axiomatic. Future
drawdowns can exceed any observed in training/validation. Use at your
own risk. Not financial advice.
