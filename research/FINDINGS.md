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

## Conclusion

The muni finding is stronger than the picks finding by orders of magnitude —
t=+26.6 after every control, versus t=+1.91 and decaying. It is also
quantified in dollars a specific client actually lost, which is what an
outbound message needs, and it rests on data access (EMMA's image-rendered
CUSIPs and blocked HTTP clients) that has already been solved here and would
cost a competitor real effort to reproduce.

The picks idea is not worthless, but it cannot honestly carry the claim that
would sell it.
