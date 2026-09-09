# Does analog pattern matching predict what a stock does next?

**No.** Directional accuracy 52.8% out of sample, against a 55.7% baseline
you get by assuming the stock goes up. The edge over that baseline is
**−2.8 points, t = −1.86** clustered by date — indistinguishable from zero,
and on the wrong side of it.

> **Correction (2026-09-09).** The first version of this file reported 50.5%.
> That figure was measured with the search ranging over all 400 tickers'
> histories, while `api/analog.py` searches only the queried ticker's own
> history — so the published number described a configuration that does not
> ship. Re-measured in the shipped configuration: 52.8%. The conclusion is
> unchanged; the number was wrong and is corrected throughout. Two look-ahead
> defects found in the same review are fixed and now have regression tests
> (`tests/test_analog.py` sections 2b and 4).

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

**No look-ahead, and no self-explanation.** At test date `t`, a candidate
window ending at `s` is admissible only if `s + 60 < t - 120` — both its own
outcome and the window itself must be finished before the query window even
opens. The weaker rule (`s + 60 <= t`) lets a match ending 60 days ago present
the right half of the query window as its own sequel. Enforced in
`_analog.find_analogs` and checked in `tests/test_analog.py`, including the
decisive test: tripling every future bar must not change the answer.

**Cross-symbol matches are gated by date, and fail closed.** Bar indices in
another series mean nothing in ours, so a foreign window is admitted only on
a date comparison. If no dates are supplied the series is skipped and listed
in `skipped_unverifiable` — the original code silently admitted it, and
returned a match from 357 bars *after* the as-of date.

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

**Same code, AND same configuration.** `validate.py` reimplements the search
in numpy for speed, and `check_equivalence.py` requires it to return identical
analogs to `api/_analog.py` on real ticker-dates. That check passed while the
published number was still wrong, because it verified the *algorithm* and not
the *search scope* — the endpoint searched one ticker's history and the
validation searched four hundred. Own-history is now the default and
`--cross-sectional` is an explicit opt-in.

**A known effect, to prove the test can find one.** 12-1 momentum on this same
panel: **IC +0.0188, t = +2.31** over 364 months. The harness detects a real
cross-sectional signal. Analog matching's IC is −0.0071, t = −0.43.

## Results

| | accuracy | t (clustered, 128 dates) |
|---|---|---|
| analog matcher | **52.84%** | +53.62 |
| random windows (control) | 51.62% | +55.18 |
| always say up (baseline) | 55.68% | +30.27 |
| **matcher − baseline** | **−2.83 pts** | **−1.86** |
| control − baseline | −4.06 pts | −2.78 |

The matcher and the random control are statistically indistinguishable.

**Magnitude carries no information either.** Correlation of predicted with
realised return −0.0325; cross-sectional IC −0.0181, t = −0.91.

**Confidence does not rescue it.** When all five analogs agreed on direction
(n=413), accuracy was 52.8% against a 55.7% baseline — still negative.

**Closer matches are not better.** The tightest quartile of matches beat the
baseline by −2.54; the loosest quartile was the only one positive (+0.92). If
the shape metric were measuring anything, this table would slope the other
way.

| match quality | n | hit rate | always-up | edge |
|---|---|---|---|---|
| closest 25% | 983 | 52.39% | 54.93% | −2.54 |
| 2nd quartile | 983 | 50.97% | 56.46% | −5.49 |
| 3rd quartile | 982 | 51.63% | 56.11% | −4.48 |
| loosest 25% | 983 | 56.15% | 55.24% | +0.92 |

**Predicting zero beats it.** The forecast is closer to the realised return
than a flat "no change" guess on only **42.4%** of forecasts.

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
python research/analog/validate.py --cross-sectional   # the other scope
```
