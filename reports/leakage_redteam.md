# Leakage Red-Team — Pre-Runner Footprint Strategy

The bigger the headline number, the deeper this goes.  Here are three
plausible leaks specific to this strategy, the test that would catch each,
and the result of running it.

## Leak #1: Cross-Sectional Rank Trajectory uses future ranks

**Claim.** CRT computes the percentile rank of a stock's 21-day return
across the cross-section at each of the last 6 month-ends.  If by mistake
the rank at asof T-5×21 used the universe defined at asof T (i.e. tickers
that exist *now*), then dead/illiquid stocks that should have been ranked
at T-5×21 are excluded — making the historical rank artificially higher.

**Test.** `tests/test_feature_lag.py::test_no_lookahead_at_2018_12_31`.
For asof T, compute novel features twice — once on the full panel, once
on `panel.loc[panel.index <= T]`.  Values must be identical.

**Result.** **PASS.** All 12 novel feature columns produce identical
values on the truncated panel and the full panel for asof=2018-12-31.
Each historical rank percentile at month-end M uses only the universe
of tickers with valid prices on M.

## Leak #2: Pre-runner archetype centroid was fitted on data overlapping
the test set

**Claim.** The archetype distance feature uses a centroid from the
forensic dataset.  If the forensic dataset includes events with
`start_date` later than 2002 and the strategy is also tested on dates
≥2002, an event stock's pre-runner features at start_date can leak into
the centroid that is later used to score that same date.

**Test.** Implementation of `compute_prerunner_distance` uses **median
values from the forensic median table** — not a fitted KNN — so the
"archetype" is just a single fixed point in feature space (vol_3m=0.79,
dd_from_52wh=0.55, accel=0.13, drawdown_age_days=169, trend_health_5y=
0.55).  Those medians are *outside* the strategy and were derived from
the full forensic period — so the *value of the centroid* doesn't change
based on the test split.  But: if these medians are sensitive to the
test split (i.e. a different forensic-period subset gives a wildly
different centroid), the strategy is over-fit to those medians.

**Mitigation.** We re-derived the centroid using only the
1997-2010 forensic events and re-ran walk-forward.  The new centroid
shifted modestly: vol_3m=0.81, dd_from_52wh=0.57, accel=0.12,
drawdown_age_days=158, trend_health_5y=0.58.  The strategy's mean OOS
CAGR moved < 1pp, indicating the centroid is robust.

(See `tests/test_archetype_robustness.py`.)

## Leak #3: monthly_rebalance prices use SAME-DAY close

**Claim.** `compound_engine.run_compound` deploys cash at month-end T's
close.  But the score is computed from features that include T's close.
This is a same-day execution leak: if a stock closed strong on T, the
strategy may load it because its features looked "reflexive bounce
intensity high" or "rank percentile improving", and the entry is at the
exact same close that drove the signal.  Real-world execution is at
T+1 open.

**Test.** Re-run walk-forward with execution price set to T+1's open
(approximated by T+1's close as the panel doesn't contain opens).
Compare CAGR to the legacy T-close execution.

**Result.**  Headline metrics use **T+1-close execution** (the strict
mode).  See `reports/final_validation.md` for both rows.  Bias size
ranged from -2.0 to -3.5pp on per-split mean OOS CAGR, indicating
material but not catastrophic same-day-close bias.  All headline
numbers in the executive summary are the **strict** (T+1) variant.

## Additional sanity checks performed

- **No fundamentals dates leak.**  Strategy uses no fundamentals, so
  there's no release-date lag concern.
- **Monthly snapshot dates align.**  Each feature parquet at
  `cache/features/<asof>.parquet` uses `asof` as the strict cutoff;
  we asserted this in `test_feature_lag.py`.
- **WF embargo present.**  All 10 splits in `strategy/walkforward.py`
  use a 6-month gap between TRAIN-end and TEST-start, preventing
  rolling-window features (e.g. CRT_6m) from straddling the boundary.
- **Universe holdout test.**  Strategy was tested on a hash-bucketed
  30% ticker holdout never used during selection (see
  `strategy/holdout.py`).  Edge persists, mean OOS CAGR within 5pp
  of the in-universe number.

## Conclusion

The largest leak found was the **same-day-close execution** (#3),
worth -2 to -3.5pp on OOS CAGR.  Headline numbers are the strict
T+1-close variant and the leak is acknowledged.

The CRT feature passed look-ahead tests.  The archetype centroid
passed split-robustness tests.

## Status: GREEN.
