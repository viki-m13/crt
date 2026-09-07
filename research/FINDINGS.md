# Two product ideas, tested against their baselines

Both ideas were tested with the pass criteria written down before the runs,
and with the controls chosen to make each idea fail if it was not real. One
survived convincingly. One did not.

---

## 1. Municipal bond execution quality — VALIDATED

**Claim under test:** retail-size muni investors are systematically
overcharged, by an amount large enough that a wealth manager would pay to
have their fills checked.

**Data:** MSRB EMMA regulatory trade tape (`viki-m13/bonds`), 3,085 bonds,
every dealer-reported customer print.

**Method:** same bond, same day, same side — which cancels bond quality,
coupon, maturity, credit and the day's rate move, leaving only trade size.
Retail ≤ $100k par vs institutional ≥ $1m, with a gap between the buckets.

| | Retail penalty (median) | Retail worse | t (clustered by bond) |
|---|---:|---:|---:|
| Buying | **+0.311 pts** ($311/$100k) | 63.3% | +34.4 (2,581 bonds) |
| Selling | **+0.176 pts** ($176/$100k) | 59.7% | +21.5 (2,123 bonds) |
| Buying, within 60 min | **+0.156 pts** ($156/$100k) | 57.5% | +26.6 (2,351 bonds) |

**Controls, and what each ruled out:**

- **Symmetry (the decisive one).** Retail is penalised buying *and* selling.
  Market drift cannot do that — drift that makes retail overpay on buys
  would make them overreceive on sells. Only a dealer spread hurts both
  directions. This is what makes the finding a markup rather than an
  artifact.
- **Intraday drift.** Timestamps do vary within a day (~3 distinct times per
  bond-day), so the comparison was re-run on pairs traded within 60 minutes.
  The buy penalty falls from 0.311 to 0.156 — about half the raw gap *was*
  timing — but what remains is still overwhelming. **0.156 is the honest
  number to quote.**
- **Correlated observations.** Significance is computed on per-bond means,
  not bond-days. The bond-day count is not the sample size.
- **Yield check.** Price could mislead; yield cannot. Retail gives up 5.7bp
  buying and 5.3bp selling, worse on 68% and 72% of bond-days.

84.8% of 2,631 bonds show retail worse on average. On a $2m ladder the
penalty is **$3,100–$6,200 one way**.

**Known limitations, stated rather than buried:** these are the most liquid
bonds on the tape, so this *understates* the typical retail experience; the
comparison requires both a retail and an institutional print on the same day;
and EMMA prints are anonymous, so this measures what retail-*sized* trades
paid, never what a named firm's clients paid.

**Bug caught:** the per-$100k figures were printed 100× too small ($3 rather
than $311) because one point of par is $1,000 on $100k, not $10. The $2m
totals used a separate, correct conversion and were unaffected — which is
exactly how a unit error survives a read-through.

---

## 2. Daily "high quality but cheap" stock list — NOT PROVEN

**Claim under test:** a daily list of high-quality, currently-cheap stocks
beats the market by enough to sell with a reportable track record.

**Why it needed retesting:** `docs/analysis-quality-weight-backtest.md`
reports an **81% one-year hit rate** and a production gate was adopted on
it — but it contains **no baseline at all**, and it used 34 tickers that all
still exist. Over a period when most stocks rose in most twelve-month
windows, a hit rate without a baseline is not evidence of anything.

**Method:** point-in-time S&P 500 membership (985 tickers *including* names
later acquired or bankrupted), 245 month-ends 2003–2025, twelve-month holds,
scored against three controls on the same months and horizon.

| | 1y hit rate | 1y median return |
|---|---:|---:|
| Picks (high quality ∧ cheap) | 66.7% | +11.98% |
| **Every S&P member that month** | **65.2%** | **+9.12%** |
| Matched random, same count | — | +9.07% |
| **Excess** | **+1.5 pts** | **+2.85%** |

**Verdict against the pre-stated criteria:**

| Criterion | Result |
|---|---|
| Beats base rate by >1%/yr | **YES** (+2.85%) |
| t > 2 on non-overlapping annual cohorts | **NO** (+1.91) |
| Holdout excess positive | **YES** (+2.71%) |

**=> NOT PROVEN. Do not sell this as an edge.**

**What is genuinely interesting anyway:**

- The effect is remarkably *stable*: dev +2.94%, holdout +2.71%.
- The combination beats either half — quality alone +1.56%, cheap alone
  −0.76%, both +2.85% — so there is a real interaction, not just a
  repackaged momentum or value tilt.
- But it has **decayed**: 2000s +3.81%, 2010s +4.14%, **2020s −1.15%**.
- And the hit rate is the killer for the product framing: **66.7% vs 65.2%**.
  Any "our picks win two-thirds of the time" claim is describing the stock
  market, not the strategy. The excess comes from winning *bigger*, not
  *more often* — which is far harder to sell and far slower to demonstrate.
- This tests a **twelve-month hold**. A *daily* list implies turnover that
  would consume a 2.85% edge outright.

**Bug caught, and it nearly produced a false verdict:** the first clean run
reported −36.9% excess against an equal-weight S&P baseline of +50%/yr. A
+50% annual base return is impossible, which prompted the check: TIE appears
in the panel at **$24,500** having started at $5.47 — a +447,797% "return"
from one corrupted price, enough on its own to move the mean of a 380-name
universe. Headline statistics moved to medians and implausible moves are now
dropped as data errors. The hit-rate comparison was immune throughout and
never moved: 66.6% vs 65.1%.

---

## 2b. The picks idea, re-examined — a real signal, but not the product

The "not proven" verdict above was about a **14-name portfolio**, and that
turned out to be the wrong thing to test. Two further experiments:

**Signal survey** (`signal_survey.py`) — rank-IC for 20+ signals at four
horizons, significance on non-overlapping periods. Six clear |t|>2 with a
consistent holdout sign. The two that matter:

| signal | horizon | IC | t | dev | holdout | 2020s |
|---|---|---:|---:|---:|---:|---:|
| `pct_above_sma` | 12m | +0.0747 | +2.76 | +0.0644 | +0.0915 | +0.0164 |
| `quality_x_cheap` | 12m | +0.0473 | **+3.00** | +0.0476 | +0.0467 | +0.0237 |

`quality_x_cheap` clears the bar at **every** horizon tested (1m, 3m, 6m,
12m), which is much harder to produce by chance than a single lucky window.

**Breadth test** (`test_breadth.py`) — same signal, varying only how many
names are held. This resolved the contradiction between IC t=+3.00 and the
portfolio's t=+1.91:

| names held | excess/yr | t (annual) | holdout | 2020s |
|---:|---:|---:|---:|---:|
| 10 | +2.99% | +2.17 | +2.97% | −0.30% |
| 50 | +1.85% | +2.38 | +2.00% | +0.46% |
| 100 | +1.69% | +2.90 | +1.54% | +0.51% |
| 150 | +1.53% | **+3.22** | +1.40% | +0.79% |

Excess return **falls** with breadth while its t-statistic **rises
monotonically** — the signature of a real but modest signal swamped by
idiosyncratic noise in a concentrated book. The original 14-name test was
not measuring the signal; it was measuring the noise around it.

**So the honest verdict flips, into something other than what was asked
for.** There is a real effect: roughly **+1.5% a year** over the equal-weight
universe, t≈3.2, positive in dev, in holdout, and in the 2020s. But it is a
**100–150 name, twelve-month-hold systematic tilt** — a strategy or an index,
not a daily stock-picking list. The concentrated ten-name version that would
actually make a "picks" product has the higher headline (+2.99%) and is the
weakest evidence (t=+2.17, and negative in the 2020s).

Not included: trading costs, taxes, or the tracking error of a portfolio this
far from the index. +1.5% before those is not +1.5% after them.

## Conclusion

The muni finding is stronger than the picks finding by orders of magnitude —
t=+26.6 after every control, versus t=+1.91 and decaying. It is also
quantified in dollars a specific client actually lost, which is what an
outbound message needs, and it rests on data access (EMMA's image-rendered
CUSIPs and blocked HTTP clients) that has already been solved here and would
cost a competitor real effort to reproduce.

The picks idea is not worthless, but it cannot honestly carry the claim that
would sell it.

---

# Muni product shape — market check and which of the two products to build

Two candidate products, same tape, same locked rules in `api/_munisignal.py`:
a **nightly list** of dislocated bonds, or a **pre-trade price check** where
the advisor pastes a CUSIP and the quoted price. Scripts:
`research/muni_edge/product_shape.py`, `price_check_control.py`,
`who_needs_it.py`. Window: 2025-07-15 → 2026-07-10, 3,085 bonds, 5.0m prints.

## Is the buyer segment real and growing?

Measured on the tape (`market_check.py`): retail-size share of muni trades
rose from **84.2%** (first three years) to **92.9%** (last three), +0.74
pts/year. Not a shrinking segment.

Published market structure, which changes who the customer is:

- Muni SMAs hold **~$1.3tn across ~180 managers**, ~30% of the $4tn market,
  up from 13–14% penetration in 2018 ([JPMorgan via Bloomberg][b],
  [IMTC][i]).
- **Odd-lot customer trading is no longer mostly retail.** MSRB research
  finds institutional dealers accounted for **44% of odd-lot customer trades
  in 2024**, and judges it likely that **≥50%** of odd-lot customer
  transactions were with institutional clients — SMA growth being the cause
  ([ION Analytics summarising MSRB][ion]).

This retires the earlier objection that dislocations happen in lots too
small for professional buyers. They happen in exactly the lots the fastest
growing professional segment trades.

[b]: https://www.bloomberg.com/news/articles/2026-02-09/jpmorgan-says-bespoke-muni-bond-accounts-grew-to-1-3-trillion
[i]: https://imtc.com/insights/sma-muni-bond-revolution/
[ion]: https://ionanalytics.com/insights/debtwire/muni-sma-growth-reshapes-bond-trading-patterns-and-curve-valuations/

## Product 1 — the nightly list: not enough inventory

| | |
|---|---|
| dislocations in 250 days | 1,627 across 625 distinct bonds |
| per day | median **1**, mean 6.5, max 195 |
| days with an empty list | **104/250 (42%)** |
| days with nothing inside the limit | 114/250 (46%) |
| discount when it fires | median 3.82 pts |

A subscription that has nothing to show on 42% of days is a churn machine.
The list is a feature, not a product.

## Product 2 — the price check: fires on almost every trade

Replaying all 385,610 retail-size customer buys in the window as if the
advisor had asked first: **58.4% "too rich"**, 33.0% within limit, 8.7% no
opinion. Median overpayment 1.13 pts.

### The bias control that matters

The limit is set off the *prior* mark, so in a rising market a fair buy
clears it mechanically and the check would look valuable while measuring
drift. Two controls:

- **Drift is nil.** Median day-over-day change in mid **+0.0000 pts**, mean
  +0.0226 — an order of magnitude too small to produce a +0.64 buy gap.
- **Symmetry.** Customers are penalised on **both** sides: buys **+0.640**
  pts median above the prior mark, sells **+0.190** below it. Drift can only
  push one side. Clustered one-vote-per-bond: buys +0.693 pts across 3,081
  bonds, **t=+74.3**; sells +0.345 across 3,068, **t=+47.6**.

The reference mid is itself often built from customer prints, which already
contain a markup — so the measured penalty is if anything understated.

### Who actually needs it (median pts vs the prior mark, customer buys)

| lot size | trades | median | past the 0.25 cap | t |
|---|---|---|---|---|
| under $25k | 150,351 | +0.490 | 60.3% | +61.4 |
| $25k–$100k | 196,764 | **+0.742** | 67.0% | +73.6 |
| $100k–$500k | 97,597 | +0.500 | 61.5% | +61.0 |
| $500k–$1m | 11,958 | +0.156 | 42.9% | +26.2 |
| $1m+ | 28,085 | **+0.000** | 29.1% | +10.9 |

Monotone above $100k and it goes to **exactly zero** for block buyers. A
$1m+ buyer already trades at the mark and has nothing to buy from us. Every
dollar of value sits in the odd-lot band — which is both the direct-buying
RIA and the SMA manager working an account-level ladder.

Per firing: median **$274** saved, mean $451, 90th percentile $1,112, on a
median $25,000 lot.

## Verdict

Build the **price check** as the product and ship the **list** inside it as
a daily feed. The check has an opinion on 91% of real quotes and objects to
58% of them; the list is empty two days in five. Sell to whoever trades odd
lots, not to whoever has the most AUM — above $1m of par the edge is zero
and the pitch is false.
