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

## 4. How a Sharpe-5 options backtest actually gets made

The best sleeve, run through the full event-loop engine, by year:

> 2019: **12.18** · 2020: 0.36 · 2021: **15.90** · 2022: 0.00 · 2023: 7.74 ·
> 2024: 8.87 — **full period: 0.63, max drawdown −36%**

Every calm year prints a Sharpe far above 5. Report 2021 alone and you have a
"Sharpe 15.9 options strategy" — from real quotes, real fills, no fabrication.
The full-period figure is 0.63. **Subperiod selection alone is sufficient to
manufacture the headline**, which is why the pre-registered protocol scored the
whole window including both crashes.

Short premium selling has exactly this shape: it earns a small, steady carry
and pays it back in rare, violent episodes. High Sharpe over any window that
excludes the episode is the *expected* result, not evidence of edge.

## 5. The honest best strategy

**SPY put credit spread, 4% short strike / 9% long wing, front expiry
(15–50 DTE), held to expiry, entered at worst-side quotes.**

| Metric | Value |
|---|---:|
| Sharpe (weekly, excess of T-bills) | **0.57** |
| 95% bootstrap CI (stationary block) | **(−0.18, 1.99)** |
| CAGR | 11.9% |
| Max drawdown | **−38%** |
| Annualized vol | 18.9% |
| Hit rate | 87% |
| **Deflated Sharpe (164 trials)** | **≈ 0.00** |

The confidence interval includes zero and the Deflated Sharpe is
indistinguishable from zero once the trial count is priced in. **I would not
describe this as a validated edge.** It is the best thing I found, reported
with the uncertainty it actually carries.

## 6. Two bugs I found in my own work

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

## 7. What would actually be required

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

## 8. Reproducing this

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

**Bottom line.** I could not honestly build a Sharpe-5 options strategy, and I
can now show with measured numbers why no strategy built from these signals and
this breadth could reach it — the ceiling is about **0.5**, and the best honest
result I produced sits right at that ceiling with a confidence interval that
includes zero. The Sharpe-5 version of this project exists only if you report
2021 by itself, price long legs at mid, or divide by premium instead of
capital. Each of those was tested, and each is documented above.
