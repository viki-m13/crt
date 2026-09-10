# Predicting failure first: model every way the call breaks, buy when all are quiet

> **Correction (2026-09-10).** Every number below was first computed on a
> panel containing twelve exchange test symbols (ZXZZT prints $0.0001 →
> $6,000 on one bar) and 540 series with failed price adjustments. Those are
> now excluded at the data layer. Direction-based results survive but shift,
> and the shift is against this idea: on the clean panel the conjunction
> reaches **62.1%** (base 53.3%) but its margin over the best single channel
> falls from +2.8/+6.4 points to **+1.2/+1.8**. Part of what looked like an
> ensemble advantage was junk series. The clean table is at the end of this
> section; the original is kept so the correction is visible.

**Verdict: the mechanism is real and it is the best thing in this repo. It
reaches 62.1%, not 90%.** And the premise it rests on — that the failure
channels are independent risks — is measurably false until you force it to be
true.

## The idea

Rather than modelling P(rises), model each distinct way the call fails, and
issue a BUY only when every channel is unusually quiet. The appeal is real:
our earlier single classifier never emitted a probability above 0.70 at any
horizon, because one model averaging over all risks has nowhere to put extreme
confidence. A conjunction can be arbitrarily selective.

Seven channels, each a point-in-time risk score in [0,1] (`channels.py`):

| channel | the failure it models |
|---|---|
| `market` | the whole market falls and takes everything with it |
| `trend` | the stock is below or under a falling long average |
| `volatility` | dispersion alone makes a down outcome likely |
| `tail` | gap risk — a recent history of violent single-day moves |
| `stretch` | overextension above its own trend, in vol units |
| `liquidity` | thin dollar volume and a low price; the exit is the risk |
| `distress` | near the lows, sustained underperformance — the pre-delisting shape |

Every score is a cross-sectional rank **within a single date**, from that
date's past. `market` is one time series so it uses an expanding-window
percentile of its own history. Verified by test: quintupling every bar after
day 1000 changes no score before it (drift 0.00e+00).

## The premise is false as stated

Pairwise correlation of the raw risk scores, 1.5m observations:

| | trend | stretch | distress | tail |
|---|---|---|---|---|
| **trend** | 1.00 | **−0.91** | **+0.87** | +0.25 |
| **volatility** | +0.17 | −0.30 | +0.26 | **+0.82** |

**`trend` and `stretch` correlate −0.91.** A stock safe in an uptrend is by
construction overextended — they are two readings of the same number with
opposite signs, and you cannot be in the safe decile of both. `trend` and
`distress` are +0.87; `volatility` and `tail` are +0.82. Seven channels behave
like roughly three.

The consequence is measurable. Requiring all seven below their median covers
**0.110%** of observations, not the **0.781%** independence would give — a
7× shortfall, entirely from channels fighting each other.

### And the raw conjunction fails the idea's own prediction

Tightening the gate is supposed to raise precision. It does the opposite:

| coverage | conjunction | best single channel | base rate |
|---|---|---|---|
| 5.00% | 57.9% | 61.3% (`tail`) | 52.9% |
| 1.00% | 57.3% | 55.2% (`market`) | 52.9% |
| 0.20% | **55.1%** | 58.0% (`market`) | 52.9% |

Ahead of its best single channel at 3 of 5 levels — noise. And the ablation is
damning: **dropping `market` raises precision by +6.3 points**, dropping
`trend` by +3.7, `stretch` +2.4. Five of seven channels were actively harmful.

## Forcing independence rescues the mechanism

Residualise each channel against the other six and re-rank
(`orthogonalise()`). Mean off-diagonal correlation falls from +0.044 with
extremes of ±0.9 to −0.017, and coverage at q=0.50 rises from 0.110% to
0.507% — much closer to the 0.781% of true independence.

Now it behaves exactly as the idea predicts. Precision **rises** with
tightness, and the margin over the best single channel **grows**:

| coverage | orth. conjunction | best RAW single channel | delta | 95% lower bound |
|---|---|---|---|---|
| 5.00% | 58.0% | 61.3% (`tail`) | −3.3 | 48.1% |
| 2.00% | 59.3% | 55.3% (`market`) | +4.0 | 49.4% |
| 1.00% | 59.6% | 55.2% (`market`) | +4.4 | 49.7% |
| 0.50% | 60.0% | 53.6% (`market`) | +6.4 | 50.1% |
| 0.20% | **60.7%** | 58.0% (`market`) | +2.8 | 50.8% |

The control is the best **raw** channel, not the residualised one. Comparing
against residualised singles flattered the result by up to +11.5 points, which
is meaningless — the orthogonalisation cripples a channel's standalone power
(`tail` alone falls from 61.3% to 49.3%). Beating a control you damaged is not
evidence.

At the 30-session horizon the user asked about, the same shape holds at a
lower level: **58.8%** at 0.20% coverage against a **52.0%** base rate, ahead
of the best raw single channel at 4 of 5 levels (+2.6 to +4.9).

### On the cleaned panel

| coverage | orth. conjunction | best RAW single | delta | 95% lower bound |
|---|---|---|---|---|
| 5.00% | 57.9% | 62.2% (`tail`) | −4.3 | 48.0% |
| 2.00% | 59.3% | 58.0% (`tail`) | +1.2 | 49.4% |
| 1.00% | 60.3% | 58.8% (`market`) | +1.6 | 50.4% |
| 0.50% | 61.3% | 59.6% (`market`) | +1.8 | 51.4% |
| 0.20% | **62.1%** | 60.4% (`market`) | +1.7 | 52.2% |

Base rate 53.3%, so +8.8 points over base at the tightest gate — but only
**+1.7 over simply using the single best channel**. The seven-channel
construction buys less than two points over one number.

Scored on the second half of the panel only, the gain survives (+4.9 to +13.6
at tight coverage for 63 sessions; +5.8 to +6.7 for 30). The orthogonalisation is fitted on all history at once, so
these are an in-sample upper bound — a deliberate advantage handed to the idea
under test.

### What is actually doing the work

Ablation on the orthogonalised gate:

| dropped | precision change |
|---|---|
| `volatility` | **−5.8** |
| `tail` | **−6.1** |
| `distress` | −2.7 |
| `liquidity` | −0.0 |
| `stretch` | +0.0 |
| `trend` | +0.3 |
| `market` | +1.8 |

Three channels carry it, and all three are variants of *thin left tail*. The
elaborate seven-channel construction reduces to the low-volatility anomaly,
which has been documented for forty years. `market`, `trend`, `stretch` and
`liquidity` contribute nothing.

## Why 90% is not reachable here, and would not be provable if it were

At the tightest gate on the clean panel: **62.1%** on 24,772 signals against
a 53.3% base rate. 95% lower bound **52.2%**. Distance to target: **27.9
points**.

The harder limit is sample size. Those signals fall in **69 non-overlapping
horizon blocks**. A rule that was *genuinely* 90% accurate would, with 69
blocks, still only certify **82.5%** at 95% confidence. **90% is not provable
on this data no matter how good the rule is** — the evidence to support the
claim does not exist in 36 years of daily prices.

Worst calendar year at the tightest gate: **28.0%** (2026), on 546 signals.

## Where this leaves the idea

It is the best-performing thing built in this line of work: +8.8 points over
base rate, a real margin over any single channel, surviving out of sample, and
a mechanism that behaves the way the theory says it should. It is also
~61%, it is mostly the low-volatility anomaly wearing seven hats, and its bad
years are very bad.

Two things would change the picture, and neither is more modelling:

1. **Non-price channels.** All seven are derived from the same price series,
   which is why they collapse to three. Fundamentals, financing, ownership and
   short interest would be genuinely separate failure channels rather than
   re-projections of one.
2. **Accepting that 90% needs a different claim.** At any coverage worth
   trading, 36 years of daily data cannot certify 90% for a single name. It
   can certify it for a diversified basket at a long horizon — with the
   worst-year caveat in `research/adaptive_path/FINDINGS.md`.

## Reproduce

```
python tests/test_failure_first.py                              # 20 checks
python research/failure_first/validate.py --horizon 63           # raw
python research/failure_first/validate.py --horizon 63 --orthogonal
python research/failure_first/fair_control.py --horizon 63       # the real test
```
