# CreditFloor v3 ("Sigma-Clear") — Replay-Validated Credit-Spread Engine

Publishes vertical credit spreads `(ticker, side, expiry, short strike)`
daily, in both directions (put-credit floors below spot, call-credit
ceilings above), across durations from 1 week to 1 year.

**The honest claim** (see [`VALIDATION.md`](VALIDATION.md) for the full
protocol): the exact published rule set, replayed point-in-time day by
day on an 18-year full-history panel, produced **zero losses across the
2008–2018 design window** (GFC included) and, on untouched 2019–2026
validation data, a **99.4% win rate (9 losses / 1,508 independent
trades) with positive P&L net of all losses under conservative fills**.
The residual ~0.6% — systemic crashes and single-name M&A/earnings
shocks — is irreducible at tradeable premium and is disclosed
everywhere. This engine does **not** claim 100%; `VALIDATION.md` §4
documents why any engine that does is grading its own homework.

## How it works

Two stages, both fail-closed:

1. **Conformal certification** (selection): per (ticker, side, horizon,
   variant), walk-forward calendar-year folds 2006–2026 with purged
   gaps. The certified buffer is the worst h-day move ever seen in
   training plus 1%; a combo is certified only if every fold's test
   year stayed inside it, with ≥50 pooled OOS tests and a certified
   buffer ≤25%. Plain and regime-gated variants, both sides.
2. **Sigma-Clear publication** (what you can actually trade):
   - published buffer `b = 2.5·σ60_daily·√h + 1%`,
   - `b ≥ 0.8 ×` the worst h-day move in the ticker's **entire listed
     history** (decades, not 2015+),
   - `b ≤ 25%` (h ≤ 21) / `45%` (h ≥ 42),
   - expiry snapped **down** to the latest standard weekly/monthly
     expiry the certified window covers,
   - conservative net fill ≥ $0.05/share (BS + put smile at 1.3×
     realized IV, tenor haircut below mid, bid-ask floor, commissions),
     with a stress estimate at bare realized vol published per rung,
   - listed options chain required, ≥10 years of history, series ≤5
     sessions stale.
3. **Reality + liquidity gate** (`reality.py`, both tiers): the rung is
   snapped onto the ACTUAL live chain — real listed expiration in the
   certified window, real strikes in the safe direction, priced at the
   **natural credit** (sell bid / buy ask). It is published only if it
   also clears the liquidity discipline: underlying average dollar
   volume ≥ **$100M/day** (`adv.json`, refreshed weekly), open interest
   ≥ **25** per leg, and short-leg bid/ask spread ≤ **40% of width**
   (phantom-bid protection). Env-tunable via `CS_MIN_ADV_USD`,
   `CS_MIN_OI`, `CS_MAX_SHORT_SPREAD_FRAC`.

## Files

| File | Role |
|---|---|
| `common.py` | data loading, causal features, forward buffers, folds, NYSE expiry math (incl. snap-down `covered_options_expiry`) |
| `research.py` | certification + v3 publication layer; writes `results/signals.json` |
| `pricing.py` | conservative fill model (smile, tenor haircut, commissions) |
| `fetch_full_history.py` | full-history price panel (`cache_full/`, period=max, one consistent adjustment basis) |
| `fetch_optionable.py` | listed-options map (`results/optionable.json`) |
| `replay.py` | **the validation instrument**: point-in-time daily replay of the live protocol |
| `replay_analysis.py` | replay row loader + vectorized pricing for gate research |
| `validate_v3.py` | reproduces the validation tables; asserts the design-window invariant |
| `build_breadth.py` | causal universe-breadth series (diagnostics) |
| `backfill_prices.py` | legacy site-panel backfiller (now with split-seam integrity check) |
| `scan.py` | daily driver: refresh data → research → publish → live log |
| `live_log.py` | append-only, survivorship-bias-free live scoreboard |
| `VALIDATION.md` | the full validation report — read this first |

## Tier 2 — "Vol-Alpha" (`tier2.py`, engine `t2-volalpha-gbm`)

A second published tier for maximum profit at high accuracy: put
verticals at 0.6·σ60·√14 below spot, 2.5%-of-spot width, ~2-week listed
expiry, hold to expiry, selected by a gradient-boosted model over the
13-feature causal library (fit only on 2008–2018, committed artifact
`results/tier2_model.joblib`, frozen deep-confidence threshold).
Untouched 2019–2026 validation: **98.2% win rate, 24.3% net ROR/trade
(19.7% under zero-vol-premium stress), ~7 trades/week, worst trade
−$516**. Runs fail-soft after the Tier 1 scan inside `scan.py`;
reality-verified; live-log entries carry the `t2` engine tag with
`:t2`-suffixed ids. See VALIDATION.md §12–§16 (including why exits
were rejected: a learned exit policy converges to never exiting).

## Live scoreboard

`live_log.py` keeps an append-only record of every rung ever published
(now tagged with the engine version) and resolves each at its expiry
close. Losses stay forever — including the legacy engine's 36. That log,
not any backtest, is the strategy's unbiased forward measure.

## Running

```bash
cd strategies/credit_spread
python3 scan.py                  # full daily pipeline (~7 min)
CS_SKIP_FETCH=1 python3 scan.py  # data already fresh
CS_LIMIT=60 CS_DATA_DIR=$PWD/cache_full python3 research.py  # smoke
python3 validate_v3.py           # reproduce validation tables
```

## What this does *not* guarantee

A 99.4% replay-validated win rate is an estimate from one (carefully
de-biased) history, on today's surviving universe, with modeled fills.
Future drawdowns can exceed historical ones; M&A and earnings shocks
are not predictable from price history. Defined-risk verticals cap
each loss at width − credit. Use at your own risk. Not financial
advice.
