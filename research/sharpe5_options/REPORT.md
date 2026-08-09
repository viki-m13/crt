# Can an options strategy reach Sharpe 5, validated honestly?

**Answer: no — not in this market, and the shortfall is structural rather than
a failure of search.** This report gives the honest best strategy the data
supports, a quantitative ceiling on what *any* strategy built from these
signals could achieve, and a precise account of which accounting shortcuts
manufacture a Sharpe-5 headline.

Everything below is computed from **real end-of-day option quotes** — bid, ask,
implied vol and greeks — for **1,254 observation dates spanning 2019-02-09 to
2026-08-06 across 128 symbols**, pulled from the DoltHub
`post-no-preference/options` database with zero fetch failures. The window
contains the COVID crash, the 2022 bear market, the August 2024 vol spike and
the 2025 tariff shock.

The methodology was **pre-registered before any strategy was run**
(`RESEARCH_LOG.md`, first commit), and every trial — including the failures and
two bugs I found in my own code — is logged there.

---

## 1. The rules I held myself to

| Rule | Why it matters |
|---|---|
| **Worst-side fills always** — sell at bid, buy at ask | Mid fills are the single largest source of fake options alpha |
| **No exit-price invention** — legs with bid=0 cannot be sold | Otherwise you "close" positions nobody would buy |
| **Marked M/W/F on real quotes**, shorts marked at ask | Entry→expiry-only P&L hides the volatility that Sharpe divides by |
| **Settlement at intrinsic** vs corrected spot, at the observation nearest expiry | 92% of expirations land exactly on an observation date |
| **Returns on committed margin**, cash earns T-bills | Return-on-premium inflates Sharpe several-fold |
| **Dev set 2019–2024; 2025–2026 held out** | Touched once, by the final candidate only |
| **Every trial counted** toward the Deflated Sharpe Ratio | ~164 configurations were tried; DSR prices that in |

## 2. What I tested, and what happened

Seven strategy families, ~164 logged configurations. Screening Sharpe on the
full corrected panel:

| Family | Best config | Sharpe |
|---|---|---:|
| **Index put credit spreads** | SPY 4%/9%, unconditional | **+0.53** |
| Short-dated (8–14 DTE) put spreads | SPY 3%/7% | +0.57 |
| Term-inversion strangles (earnings crush) | 25-delta, idio inv >0.09 | +0.30 |
| Short strangles | 25-delta, unconditional | +0.03 |
| Delta-hedged short straddles | SPY, unconditional | −0.16 |
| Iron condors | unconditional | −0.15 |
| **Dispersion** (short index vol / long components) | vega-matched, 36 names | **−0.43** |
| Reverse dispersion | | −0.61 |
| Intra-chain smile arbitrage | — | **no tradable candidates at all** |

Three families died for reasons worth stating plainly:

- **Smile arbitrage doesn't exist here.** Fitting vega-weighted quadratic
  smiles across 580 dates, residual σ was 2.2 vol points and the 95th
  percentile about 1 point. Every quote flagged "rich" was a 3–40¢ far-OTM
  option with negligible vega — estimated edge ~$0.08 per spread against an $8
  cost hurdle. **Zero tradable candidates.** Liquid EOD chains are
  smile-efficient within costs.
- **Every short-hold variant loses.** Round-trip spread exceeds the theta
  captured. Weekend theta screens at −2.2 with a 2.5% hit rate. Anything
  viable must be held to expiry, paying the half-spread once.
- **Dispersion is dead both directions.** Implied correlation (computed
  point-in-time from the chains, median 0.25) shows the correlation premium is
  really there — but capturing it means crossing the spread on 37 straddles,
  which costs more than the premium is worth.

## 3. Why Sharpe 5 was never reachable

This is the core result. Grinold's fundamental law says `IR ≈ IC × √BR`, so
measure both terms on the actual data:

| Quantity | Measured |
|---|---:|
| IC of the best signal (credit_yield → straddle return) | **+0.0371** (t = **+5.54**, 1237 dates) |
| Tradable liquid names | 38 |
| Mean pairwise return correlation | +0.185 |
| **Effective independent names** (eigenvalue participation ratio) | **13.7** — 36% of nominal |
| Independent bets per year (13.7 × 12 monthly rounds) | 164 |
| **Implied IR ceiling, gross of all costs** | **0.48** |

**To reach Sharpe 5 you would need 18,130 independent bets per year — 111× the
breadth this market offers — or an IC of 0.391, which is 11× what I measured.**

The signals are real. `credit_yield` (t=+8.4), term structure (t=−7.9),
momentum (t=+5.6) and IV−RV (t=+4.0) all rank rich options with high
statistical significance, and the ICs got *stronger* as data was added. The
problem is not signal quality. It is that 38 short-vol positions behave like
~14 independent ones, and no amount of signal engineering fixes a breadth
deficit of two orders of magnitude.

This is why the answer is "no" rather than "not yet."

## 4. The out-of-sample test: the best candidate lost money

The holdout (2025-01 → 2026-08) was touched exactly once, by the final
ensemble, with inverse-volatility weights frozen on the development window.

| Period | Sharpe | CAGR | Max DD |
|---|---:|---:|---:|
| Development 2019–2024 | **+0.57** | +6.6% | −15% |
| **Holdout 2025–2026 (single test)** | **−1.12** | **−28.1%** | **−43%** |
| Full period | −0.24 | −1.9% | −43% |

95% bootstrap CI on the holdout: (−2.05, 0.25). **The strategy that looked
mildly profitable in development lost money out of sample.**

One sleeve did not merely underperform — it went **insolvent**. The
term-inversion strangle sleeve (`B_str25`) survived the entire development
window *including the COVID crash*, then took equity through zero on
**2025-05-12**. I deliberately left it in the holdout book: the exclusion rule
is applied on development data only, because removing a sleeve after seeing it
fail out of sample would hide exactly what the holdout exists to reveal.
(Two other sleeves went bust *within* development and were excluded on that
basis — a legitimate in-sample decision.)

This is the most important number in the report. Not "Sharpe 5 was out of
reach" but "the best honest candidate did not survive contact with unseen
data."

## 5. How a Sharpe-5 options backtest actually gets made

Every figure below comes from the **same strategy on the same real quotes with
the same worst-side fills**. Only the accounting convention changes.

| Convention | Sharpe |
|---|---:|
| **Honest: worst-side fills, weekly marks, margin base, full period** | **+0.79** |
| Report 2021 only | **+15.94** |
| Report 2019 only | **+17.02** |
| 2021 only + per-trade annualization (×√149) | **+6.51** |
| 2023 only + per-trade annualization (×√151) | **+7.48** |
| 2021 + per-trade + drop worst 2% of trades | **+15.86** |

And the individual shortcuts, isolated on identical trades:

| Shortcut | Sharpe | Inflation |
|---|---:|---:|
| Honest baseline | +0.53 | — |
| Per-trade Sharpe × √(164 trades/yr) | +1.76 | **3.3×** |
| Drop the worst 1% of weeks (4 weeks removed) | +1.35 | **2.5×** |
| No intermediate marks (entry-bucketed) | +0.55 | 1.04× |
| Return on premium instead of margin | +0.45 | 0.85× |

The two dominant illusions are **treating 164 overlapping monthly trades as
164 independent bets** and **removing a handful of bad weeks**. Combine either
with subperiod selection and Sharpe 5 appears without a single fabricated
price.

Short premium selling has exactly this shape: it earns a small steady carry
and repays it in rare violent episodes. A high Sharpe over any window that
excludes the episode is the *expected* result, not evidence of edge — which is
why the pre-registered protocol scored the whole window including both crashes.

## 6. The honest best strategy

**SPY put credit spread, 4% short strike / 9% long wing, front expiry
(15–50 DTE), held to expiry, entered at worst-side quotes.**

| Metric | Value |
|---|---:|
| Sharpe, development (weekly, excess of T-bills) | **0.57** |
| 95% bootstrap CI (stationary block) | (−0.18, 1.99) |
| CAGR | 11.9% |
| Max drawdown | **−38%** |
| Hit rate | 87% |
| **Deflated Sharpe (170 trials)** | **≈ 0.00** |

The confidence interval includes zero, the Deflated Sharpe is indistinguishable
from zero once the trial count is priced in, and the ensemble built around it
was negative out of sample. **This is not a validated edge.** It is the best
thing I found, reported with the uncertainty it actually carries.

## 7. Two bugs I found in my own work

Both were caught by internal audits and both are logged. They matter because
each one, uncaught, would have changed the conclusion:

1. **The long-short sleeve that looked best was fake.** It credited the long
   leg with the negated short-leg return instead of pricing it at the real ask.
   Priced honestly, it went **+2.32 → −1.64**. Long ATM straddle legs bleed
   ~5%/month at worst-side fills plus hedge costs. This was the single most
   promising result in the whole project and it did not survive.

2. **My "spot" was actually the forward.** Put-call parity's `C−P+K` recovers
   `F = S·e^{(r−q)T}`, so spot drifted with whichever expiry happened to be
   nearest — **+15 bps at 2024 rates**, +1 bp under 2021 ZIRP. That drift falls
   between entry and settlement and *flattered put spreads*, my best family.
   Fixed by fitting `ln F = ln S + cT` across the expiration curve and taking
   the intercept, recovering spot and carry from the quotes alone. Corrected
   SPY on 2024-06-03: 528.33 against a true close of ~527.8; the naive forward
   said 528.98. **The entire panel was re-run on corrected spots** — the
   uncorrected log is kept at `cache/rebuild_uncorrected_spot.log`.

## 8. What would actually be required

Sharpe 5 in options is not impossible in general — it exists in market making
and latency-sensitive strategies. Reaching it requires breadth or edge this
data cannot supply:

- **Intraday/tick data** — thousands of independent bets per year instead of 164
- **Passive fills at or inside the spread** — a market maker earns the
  half-spread I pay; on the numbers above that alone flips the sign
- **Instruments with structural carry not in this database** — VIX futures term
  structure, variance swaps
- **Genuinely uncorrelated sleeves** — the binding constraint is 0.185 mean
  correlation, not signal strength

## 9. Reproducing this

```bash
cd research/sharpe5_options
python3 fetch/fetch_chains.py        # tier 1 (~60 symbols); FETCH_TIER=2 for tier 2
USE_TIER2=1 bash rebuild_all.sh      # features -> structures -> studies
USE_TIER2=1 python3 study5_ceiling.py    # the breadth/ceiling result
USE_TIER2=1 python3 run_finals.py dev    # event-loop sleeve runs
```

The data cache is regenerable and deliberately untracked. `RESEARCH_LOG.md`
holds the full chronological ledger, including every failed trial.

---

**Bottom line.** I could not honestly build a Sharpe-5 options strategy. Three
findings, in order of importance:

1. **The best candidate failed out of sample.** +0.57 in development, **−1.12
   in the holdout**, with one sleeve going insolvent in May 2025 after
   surviving COVID. The honest answer is not "I reached 2 instead of 5" — it is
   "the thing that looked like an edge was not one."
2. **Sharpe 5 was never reachable here, and that is measurable.** IC of 0.037
   against 13.7 effective independent names caps IR at **~0.48 gross of
   costs**. Reaching 5 needs 111× this market's breadth or 11× the measured
   signal. This is a structural property of EOD equity options, not a search
   failure.
3. **A Sharpe-5 headline is trivially manufacturable from these very same
   trades** — 15.94 by reporting 2021 alone, 7.48 by annualizing overlapping
   trades as independent — with no fabricated prices anywhere. Which is
   precisely why the protocol was pre-registered before the first backtest ran.

If a strategy claiming Sharpe 5 in options crosses your desk, the fastest
diagnostics are: what window, marks or entry-only P&L, capital or premium in
the denominator, and how many trials produced it.
