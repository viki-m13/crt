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
| IC of the best **clean** signal (vol-of-vol → loss term) | **0.0330** (t = **−5.30**) |
| Tradable liquid names | 38 |
| Mean pairwise return correlation | +0.185 |
| **Effective independent names** (eigenvalue participation ratio) | **11.7–13.7** — ~36% of nominal |
| Independent bets per year (× 12 monthly rounds) | ~141–164 |
| **Implied IR ceiling, gross of all costs** | **0.39** |

**To reach Sharpe 5 you would need an IC of 0.421 — 13× the best clean signal —
or roughly 100× the breadth this market offers.**

The signals are real: vol-of-vol (t=−5.30) and 25-delta skew (t=+3.81) predict
losses with genuine significance. The problem is not signal quality. It is that
38 short-vol positions behave like ~12 independent ones, and no amount of
signal engineering fixes a breadth deficit of two orders of magnitude.

> **Note.** My first pass reported this ceiling as 0.48 using `credit_yield`
> (IC 0.037, t=+5.54). That number was contaminated by an accounting identity —
> see §7b. The corrected ceiling on genuinely predictable content is **0.39**.
> The conclusion strengthens; the arithmetic changed.

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

## 7. Second research pass: attacking the ceiling's own terms

The ceiling formula IR = IC × √BR names its own levers. The first pass used
one signal, one holding period, one bet direction. This pass attacked each.

### 7a. Box spreads — the one family with unbounded Sharpe potential

A box (long call K1 / short call K2 / short put K1 / long put K2) pays exactly
K2−K1 at expiry regardless of path. It is pure financing, so a mispriced box is
true arbitrage, and arbitrage has no Sharpe ceiling. If Sharpe 5 lives anywhere
in options, it lives here. **114,649 boxes scanned at worst-side fills:**

| Implied lending rate from boxes | Value |
|---|---:|
| Median | **−99.9%** |
| 99th percentile | −4.8% |
| T-bill reference | +3.2% |
| Boxes lending above T-bill + 2% | **0.023%** of sample |
| Hard arbitrage (cost ≤ 0 for positive payoff) | **10 of 114,649 (0.0087%)** |

Crossing four bid-ask spreads costs vastly more than any financing dislocation
in EOD quotes. The distribution is not marginal — it is *catastrophically*
negative, with the median box losing essentially its entire cost.

And the handful of apparent free money is a **data artifact, not an
opportunity**: all 10 cases concentrate in a single name (F), with a median box
width of $2.00 — stale quotes on a low-priced stock. This doubles as a warning
about the dataset: any strategy that appears to find edge in sub-$5 structures
on cheap names is reading quote noise.

### 7b. My headline signal was an accounting identity

Study 5 reported credit_yield with IC +0.037 (t=+5.54) and called it real. It
is not. For a premium structure held to expiry:

> **ret = credit_yield − loss/margin**, and the credit is **known at entry**

With **76% of credit spreads expiring worthless**, `ret` *literally equals*
`credit_yield` most of the time. Regressing return on credit_yield partly
regresses a variable on itself.

| Measurement | Value |
|---|---:|
| corr(credit_yield, **return**) | **+0.53** ← what I reported |
| corr(credit_yield, **−loss**) | **−0.13** ← real predictive content |

The true relationship is *negative*: richer premium predicts **bigger** losses.
This resolves the paradox in the first pass — a t=+8.4 signal that never
produced a profitable sleeve. It was measuring an identity. **Any "signal" that
is a component of the payoff (premium, credit, IV level) will show a large
spurious IC in an options backtest.** I would not have caught this from the
Sharpe numbers alone; it took asking why a strong signal made no money.

### 7c. The true ceiling: 0.39

Recomputing on signals that are *not* components of the payoff:

| Clean signal | IC vs −loss | t |
|---|---:|---:|
| **Vol-of-vol** | **−0.0330** | **−5.30** |
| **25-delta skew** | **+0.0231** | **+3.81** |
| ivrv | −0.0078 | −1.32 |
| momentum | −0.0084 | −1.34 |
| IV rank | −0.0006 | −0.09 |

Best clean IC 0.033 against 11.7 effective names → **IR ceiling 0.39** (not the
0.48 I first reported). Reaching 5 needs IC 0.421 — **13× the best clean
signal**.

### 7d. What the clean signals actually buy

Vol-of-vol and skew are *genuinely* predictive of losses, so I used them as a
risk-avoidance filter rather than a return-picker:

| Filter (credit put spreads) | Mean return | Loss rate |
|---|---:|---:|
| Unfiltered | −0.0145 | 20.2% |
| Low vol-of-vol | −0.0094 | 18.4% |
| High skew | −0.0060 | 20.2% |
| **Low vol-of-vol + high skew** | **−0.0024** | **17.0%** |

The filter works — it cuts the loss rate by a fifth and removes 83% of the
deficit. **It still does not cross zero.** Also tested and dead: multi-signal
composites with walk-forward weights (the apparent +0.40 OOS IC was the same
tautology), the shortest available tenor to double turnover, stress overlays,
and selling cheap premium instead of rich (every credit-yield quintile is
negative for every structure).

## 8. Third pass: bringing the stock in

Every failure above paid the **option** spread — 100–500 bps per round trip
against a gross edge of similar size. But nothing forces a trade to be
expressed in options. Stock spreads on these names are 1–5 bps.

### 8a. Option-implied market timing of SPY — no rule beats holding

| Strategy | Sharpe | CAGR | Max DD |
|---|---:|---:|---:|
| **Buy & hold SPY** | **+0.85** | +14.6% | −34.2% |
| Vol-target on option-implied IV | +0.84 | +12.1% | **−24.5%** |
| Vol-target on trailing realized | +0.79 | +11.8% | −21.2% |
| Long only in contango | +0.67 | +6.4% | −15.2% |
| Long only when VRP > 0 | +0.40 | +4.7% | −31.2% |

Every gate that sits out of the market surrenders more return than risk. The
one durable result: **vol-targeting on implied vol matches buy-and-hold's
Sharpe while cutting drawdown by ten points**, and it is the most stable thing
in this entire project — dev 0.93, holdout 0.85, essentially no decay. That is
risk scaling, not alpha, and I present it as such.

### 8b. Stock + option combinations — every overlay is worse than the stock

Held to expiry, benchmarked against buy-and-hold of the same stock over
identical dates (46,037 observations per structure):

| Structure | Sharpe | vs stock | Hit rate |
|---|---:|---:|---:|
| **Stock only** | **+0.77** | — | 55.2% |
| Protective put 5% | +0.65 | −0.11 | 48.1% |
| Buy-write 5% OTM | +0.56 | −0.21 | 61.0% |
| **Cash-secured put write 5%** | +0.46 | −0.30 | **83.2%** |
| Buy-write ATM | +0.44 | −0.33 | 70.0% |
| Collar 5% | +0.25 | −0.51 | 53.7% |

**Read the hit-rate column against the Sharpe column — they run in opposite
directions.** The put write wins **83% of the time** (92.6% on SPY alone) and
delivers the second-worst risk-adjusted return in the table. Options reshape
the payoff into many small wins and rare large losses; they do not manufacture
edge. Any strategy sold on "wins 9 months out of 10" is describing its payoff
shape, not its profitability, and this table is the cleanest demonstration of
that distinction I produced.

## 9. Two bugs I found in my own work

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

## 10. What would actually be required

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

## 11. Reproducing this

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
2. **Sharpe 5 was never reachable here, and that is measurable.** The best
   *clean* IC of 0.033 against ~12 effective independent names caps IR at
   **~0.39 gross of costs**. Reaching 5 needs 13× the signal or ~100× the
   breadth. Confirmed from the opposite direction by the box scan: the one
   family with unbounded Sharpe potential prices at a **median −99.9% implied
   rate** once you cross four spreads.
3. **A Sharpe-5 headline is trivially manufacturable from these very same
   trades** — 15.94 by reporting 2021 alone, 7.48 by annualizing overlapping
   trades as independent — with no fabricated prices anywhere. Which is
   precisely why the protocol was pre-registered before the first backtest ran.
4. **The subtlest trap is an accounting identity posing as a signal.** My own
   headline IC survived a pre-registered protocol, a walk-forward, and a
   deflated-Sharpe penalty — and was still measuring `ret ≡ credit_yield` on
   the 76% of trades that expire worthless. What exposed it was not any
   validation test but asking why a t=+8.4 signal made no money.

If a strategy claiming Sharpe 5 in options crosses your desk, the fastest
diagnostics are: what window, marks or entry-only P&L, capital or premium in
the denominator, and how many trials produced it.
