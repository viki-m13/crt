# The short side: predicting which stocks fall

**The short side is roughly half as predictable as the long side, and the part
that looks promising cannot be borrowed.** Inside a universe where a short
could actually be established, the names the model flags as most dangerous go
**up** by a median 2.05% over the next 63 sessions.

There was good reason to expect otherwise. Short-side anomalies are documented
as stronger than long-side ones, the standard explanation being that shorting
is costly and constrained so the mispricing is not arbitraged away. That
explanation also contains the warning: it predicts the edge lives exactly
where it cannot be captured. It does.

## 1. Symmetry — the same machinery aimed down

Orthogonalised channels, gate inverted so every channel must be maximally
dangerous. Whole universe, 63 sessions:

| coverage | P(down) delisting=win | P(down) delisting=loss | median return | P(up), long gate |
|---|---|---|---|---|
| 2.00% | 53.1% | 48.2% | −1.24% | 59.6% |
| 0.50% | 53.6% | 47.5% | −1.50% | 61.7% |
| 0.20% | 54.2% | **47.5%** | −2.02% | **62.4%** |

Base rate P(lower) is 44.3%, so the tightest short gate lifts it **+3.2
points**. The same machinery on the long side lifts P(higher) **+7.4 points**.
**The long side is about twice as good.**

Delisting is scored both ways because for a short it is genuinely ambiguous —
a bankruptcy is the best possible outcome and an acquisition is usually the
worst, and a close-only tape cannot tell them apart. The optimistic column
assumes every delisting is a winning short, which is certainly false.

## 2. The borrowability test, which decides it

Of the 24,772 signals at the tightest gate:

| dollar volume | share of signals | P(down) | median return |
|---|---|---|---|
| under $100k/day | **92.9%** | 47.1% | −1.59% |
| $100k–$1m/day | 7.0% | 53.2% | −6.12% |
| $1m–$10m/day | 0.1% | — | — |

| price | share | P(down) | median return |
|---|---|---|---|
| under $5 | **87.4%** | 46.8% | −1.76% |
| $5–$20 | 8.0% | 52.7% | −2.89% |
| over $20 | 4.6% | 52.8% | −6.03% |

93% of the signals are in names trading under $100k a day; 87% are under $5 a
share. Those are not borrowable at any sensible cost, and in size they are not
borrowable at all.

The borrowable slice looked *better* per signal (−6% median), which was the
opposite of what I expected — so I tested it properly rather than reporting
it. Ranking across the whole market makes the most dangerous names microcaps
by construction, so the honest test is to restrict the universe to borrowable
names **first** and rank inside it.

### Inside the borrowable universe the signal disappears

Price ≥ $5 and median 63-day dollar volume ≥ $1m. 5.76m of 12.39m observations
survive, 1,981 tickers:

| coverage | P(down) | median return of the name | edge vs shorting the average name |
|---|---|---|---|
| 1.00% | 44.2% | +1.93% | — |
| 0.20% | **44.6%** | **+2.05%** | **+0.34%** |

Base rate P(lower) in this universe is 41.8%, so the lift is +2.7 points and
the 95% lower bound is 35.1%. The flagged names **rise** a median 2.05%; a
short in them loses money, and the edge over shorting the average borrowable
name is **+0.34%** — nothing.

The detail underneath is the classic pattern: within the borrowable set the
"most dangerous" $5–$20 names had P(down) of **38.9%** and a median return of
**+6.40%**. The cheap, distressed, heavily-flagged, borrowable name is the one
that squeezes.

## 3. What does work: predicting corporate death

The one genuinely strong result in this line of work. Can the same channels
predict which names stop trading within a year?

| gate coverage | signals | P(delists within 252 sessions) | lift |
|---|---|---|---|
| base rate | — | 7.86% | 1.0× |
| 5.00% | 619,289 | 12.97% | 1.6× |
| 1.00% | 123,860 | 19.04% | 2.4× |
| 0.20% | 24,772 | **22.86%** | **2.9×** |

Nearly **three times** the base rate. The model is substantially better at
predicting *failure* than at predicting *price direction* — which is what the
distress literature would predict, and what a panel containing 4,390 dead
tickers is uniquely able to measure.

But the same constraint applies. Inside the borrowable universe the lift falls
from 2.9× to **1.4×** (7.70% → 10.49%). Companies that die are small and
illiquid long before they die, and that is most of what the signal is reading.

## Conclusion

Three separate findings, all pointing the same way:

1. The short side lifts the base rate about half as much as the long side.
2. 93% of short signals are unborrowable, and inside the borrowable universe
   the signal is gone — the flagged names go up.
3. Predicting delisting works well (2.9×) and is the most genuinely
   predictable thing measured in this repo, but it too collapses to 1.4× once
   restricted to names you could trade.

None of this is priced for borrow cost, which would only make it worse. Nor
for the squeeze risk that the $5–$20 bucket's +6.40% median is a picture of.

## A data defect found here, affecting every mean in this repo

The first run of this printed a base mean 63-day return of **+6,729%**.
`ZXZZT` is a NASDAQ order-routing **test symbol**, not a security; it prints
$0.0001 → $6,000 on a single bar. Twelve such symbols were in the panel, plus
540 series with a greater-than-10x single-day move — failed price adjustments.

Both are now excluded at the data layer (`research/uptrend/data.py`), and
every return figure is a median rather than a mean. **Direction-based results
are unaffected**, which was verified rather than assumed: the 252-session base
rate moved 57.0% → 57.8% and the 504-session rate 59.7% → 60.5%. Sign is
robust to magnitude outliers; means are not.

## Reproduce

```
python research/failure_first/shorts.py --horizon 63
python research/failure_first/shorts.py --horizon 63 --borrowable
```
