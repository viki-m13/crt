# Does analog pattern matching predict what a stock does next?

**No.** Directional accuracy 50.5% out of sample, against a 55.7% baseline
you get by assuming the stock goes up. The edge over that baseline is
**−5.2 points, t = −3.19** clustered by date. Choosing the windows at random
instead of matching them scores the same, which is how we know the matching
itself contributes nothing.

This was built because a 99%-accurate version was requested. That target is
not reachable — a method that called three-month direction 99% of the time
would be worth more than the market it trades — so what was built instead is
the honest version: the tool, plus the measurement of what the tool is worth,
displayed together and inseparably.

## What was tested

The technique in `api/_analog.py`: index the last 120 trading days to 100,
sweep the ticker's history for the windows whose shape matches, and read what
followed each of them over the next 60 trading days. The forecast scored here
is the median forward return of the five closest matches.

Panel: `experiments/monthly_dca/cache/prices_extended.parquet` — 8,133 daily
bars × 1,833 tickers, 1995-01-03 → 2026-05-07. The test ran on the 400
tickers with the most history, 608,618 candidate windows.

## Controls

Each of these was included because without it the result would look better
than it is.

**No look-ahead.** At test date `t`, a candidate window ending at `s` is
admissible only if `s + 60 < t` — its own outcome had to already be knowable.
Matching windows whose futures overlap the forecast period is the standard
way this technique manufactures skill. Enforced in `_analog.find_analogs` and
checked in `tests/test_analog.py`, including the decisive test: tripling every
future bar must not change the answer.

**Non-overlapping outcomes.** Test dates are spaced 60 trading days apart, so
no two of the 3,968 forecasts share a forward window.

**Clustered significance.** Every stock moves with the market, so each *date*
is one observation, not each ticker-date. Without this the same 50.5% comes
with t ≈ +60 and reads as overwhelming evidence of a coin flip.

**Matched random control.** The identical pipeline with matching switched off
— windows drawn at random from the same admissible set.

**The right baseline.** Stocks rise more often than they fall, so "always say
up" is the bar to clear, not 50%.

**Deduplication.** Matches must be 60 trading days apart. Without it the top
five are five shifts of one window: one event counted five times, and an
agreement score that is pure double-counting.

**Same code, verified.** `validate.py` reimplements the search in numpy for
speed. `check_equivalence.py` requires it to return byte-identical analogs to
`api/_analog.py` on real ticker-dates — 12/12 matched. Otherwise the accuracy
figures would describe code no user runs.

**A known effect, to prove the test can find one.** 12-1 momentum on this same
panel: **IC +0.0188, t = +2.31** over 364 months. The harness detects a real
cross-sectional signal. Analog matching's IC is −0.0071, t = −0.43.

## Results

| | accuracy | t (clustered, 128 dates) |
|---|---|---|
| analog matcher | **50.50%** | +59.95 |
| random windows (control) | 51.18% | +55.96 |
| always say up (baseline) | 55.72% | +30.29 |
| **matcher − baseline** | **−5.22 pts** | **−3.19** |
| control − baseline | −4.54 pts | −3.24 |

The matcher and the random control are statistically indistinguishable.

**Magnitude carries no information either.** Correlation of predicted with
realised return +0.0027; cross-sectional IC −0.0071, t = −0.43.

**Confidence does not rescue it.** When all five analogs agreed on direction
(n=300), accuracy was 49.0% against a 51.3% baseline — still negative.

**Closer matches are not better.** The tightest quartile of matches had the
*worst* edge (−6.96 pts); the loosest had the best (−3.43). If the shape
metric were measuring anything, this table would slope the other way.

| match quality | n | hit rate | always-up | edge |
|---|---|---|---|---|
| closest 25% | 992 | 48.89% | 55.85% | −6.96 |
| 2nd quartile | 992 | 51.01% | 55.75% | −4.74 |
| 3rd quartile | 992 | 50.71% | 56.45% | −5.75 |
| loosest 25% | 992 | 51.41% | 54.84% | −3.43 |

**Predicting zero beats it.** The forecast is closer to the realised return
than a flat "no change" guess on only **40.9%** of forecasts.

## Why the chart still ships

"Here are the five stretches in this stock's history that most resemble now,
and here is exactly what followed each one" is a true and genuinely useful
statement. What the analogs mostly show is *disagreement* — NVDA's five
matches ranged from −9% to +49% over the following three months — and seeing
that spread is a better education in uncertainty than any point forecast.

So the tool is built to make the honest reading unavoidable: the accuracy
panel renders above the chart, is populated from the API response rather than
the page, and is attached to every response including errors. The number
cannot be separated from the picture.

## Reproduce

```
python tests/test_analog.py                        # 28 checks, incl. look-ahead
python research/analog/check_equivalence.py        # fast search == shipped code
python research/analog/sanity_known_effects.py     # momentum shows up
python research/analog/validate.py --tickers 400 --points 4000
```
