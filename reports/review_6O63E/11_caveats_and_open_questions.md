# 11 — Caveats and open questions

Things I'd want to validate further before committing live capital, in
rough priority order.

## Caveats acknowledged

### 1. Chronos temporal leakage (medium concern)

Chronos-bolt-tiny was released by Amazon in early 2024. The training
corpus (a curated mix of public time-series across many domains) may have
some overlap with the 2003-2023 financial time-series we're backtesting
on. The model is unlikely to have *memorised* specific stock paths
because:

- The training corpus is described as cross-domain (weather, transportation,
  energy, etc.) and synthetic — not financial-specific
- The model has only 9M parameters (small enough that memorisation of
  individual long sequences is unlikely)
- The published Chronos paper benchmarks against zero-shot performance on
  held-out series

Empirical evidence against leakage:
- The shuffle-test (real chronos at q=0.4: 44.81% CAGR; random shuffled
  chronos: 27.34%) confirms the signal is real cross-section info, not
  just any filter. See section 5 of original review.
- The WF-q-selection picks q=0.4 on training windows ending as early as
  2010-12 (only 7 years of training history before A1). If leakage were
  driving the result, the selector's choice would shift with training
  horizon. It doesn't.

**Mitigation**: re-run the same test using an older Chronos snapshot
(if available) trained on a corpus pre-dating 2024. If the q=0.4 lift
persists, leakage is essentially ruled out.

### 2. Universe survivorship bias on tech_broad

The 212-name tech_broad universe is a 2025-vintage list (QQQ + IYW + IGM
holdings as of ~early 2025) applied to historical data 2003-2025. Names
that:
- Existed in 2003 but were delisted before 2025: NOT in our universe.
  Examples: Sun Microsystems, Yahoo, Palm, Garmin (acquired). The
  backtest never tries to pick these.
- Listed after 2003 but before 2025: only in scope after they listed.
  Modern names like META (2012), TSLA (2010), GOOGL (2004) are in scope
  from their IPO. This is correct (no future-leakage on existence).

The forward bias is: we're testing the strategy on a universe that
*succeeded enough to still be listed in 2025*. The strategy would have
picked some delisted names in production that we're not modelling.

**Mitigation**: build a PIT version of tech_broad using historical IYW /
QQQ membership lists. This would add ~5-10% additional tickers (the
historical delisters) but isn't trivial to source.

### 3. The bull-regime concentration in B is real, just not robust

Section 5 shows the WF kb-selector picks kb=1 in 8 of 10 splits, not
kb=2 the agent claimed. But the per-split data (section 9) shows kb=2
IS the right choice on R5_COVID (+11.75pp lift) and R6_AI (+1.15pp).

The mechanism may have some merit *in those specific regimes*; the
sweep-overfit issue is that the agent picked kb=2 from a 45-cell sweep
on the same data they were claiming a +3.28pp lift on.

**Mitigation if you want to use kb=2**: only apply it in specific
regimes where it can be justified ex-ante (e.g., after a major bear
market when the recovery is concentrated in mega-cap names). Treat as
a tactical overlay, not a strategic default.

### 4. A's WF-honest CAGR cost is real

Section 5: A's "always-on invvol" loses CAGR in 7 of 10 splits when the
WF selector chooses ew on training data. The full-period 22-year Sharpe
and MaxDD improvements are real, but they're an aggregation effect.

A user who deploys A expecting "Sharpe improvement at no CAGR cost
quarter-by-quarter" will be disappointed in some quarters.

**Mitigation**: communicate honestly that A is a *risk-management* play,
not an alpha play. The CAGR improvement comes from C; the MaxDD/Sharpe
improvement comes from A.

### 5. 2024-05 holdout is just 20 months

The +17pp edge on tech_broad for A+C is impressive but it's a 20-month
sample. Returns 20 months from now could easily be 0 or negative for the
same strategy — there's no guarantee the recent edge persists.

The more reliable estimate is the 22-year WF mean (51.7%) and full-period
Sharpe (1.43), not the recent 20-month numbers.

## Open questions I'd want answered before real money

### Q1. Can we get historical PIT IYW / QQQ membership?

iShares and Invesco publish historical fund composition data. With this
we could build a properly PIT tech_broad universe and verify the
survivorship bias is small.

Effort: ~1-2 weeks of data engineering. Not blocking but worth it.

### Q2. What is Chronos's behaviour during structural breaks?

The backtest includes 2008 GFC and 2020 COVID as test windows. The
chronos filter's behaviour during these regimes: does it correctly
de-confidence on most stocks (filter applies broadly) or does it have
specific failure modes? I haven't analysed Chronos prediction
distributions during stress regimes.

Effort: a 1-day analysis on the existing parquet files.

### Q3. How much capacity does the tech_broad strategy have?

3 picks × monthly rebalance × 6m hold on a 212-name universe = roughly
half the picks are mega-caps (NVDA, MSFT, AAPL, GOOGL, META) and half are
mid-caps. At $10M AUM, this is trivially executable. At $1B AUM, the
mid-cap picks become slippage-sensitive.

Effort: simple liquidity analysis on average daily volume per pick.

### Q4. Does the strategy survive transaction-cost shocks?

The backtest assumes 10bp round-trip per pick. At 50bp (small-cap
realised), the lift would shrink ~3-5pp. At 100bp, possibly 5-8pp.

Effort: 1-day cost-sensitivity sweep using the existing engine.

### Q5. Does an A+C ensemble across (tech_broad, sp500_pit, iyw_tech)
beat any single-universe deployment?

Not tested. A 33/33/33 blend across the three universes would diversify
the universe-choice risk. Worth exploring.

Effort: 1-day analysis.

## What I'm explicitly NOT worried about

- v3 model leakage: thoroughly audited by multiple agents (especially
  `G0nfM` v6 parity test). Cached predictions are exactly the deployed
  v3.
- Engine bugs: v6 engine reproduces v3 to 16 decimal places. Same engine
  used for A, B, C — no risk of differential bugs.
- Look-ahead in regime gate: features computed strictly from data with
  `asof < T`. Verified.
- Universe-selection bias on sp500_pit: PIT membership properly tracked
  back to 2003-01.

## Summary

The validation is comprehensive enough to ship. The main residual risks
are (1) Chronos temporal leakage (mitigable with an older model
snapshot test) and (2) tech_broad survivorship bias (mitigable with a PIT
membership build). Neither is a blocker — both are 1-2 week mitigations
that can run in parallel with shadow deployment.
