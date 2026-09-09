# Adaptive stock direction and path research — preregistration

Registered 2026-09-09 before running this experiment. Base: `9939e0f1e0f0d601c14b2d21ff14a38da82cc2e2`. Research only; no live orders or production changes.

## Question and estimand
Can a selective, adaptive forecaster identify an individual stock that will be strictly higher at a fixed future endpoint with >90% observed out-of-sample precision and adequate independent evidence? The unit is an issued (ticker, issue date, horizon) forecast. Abstentions are not correct forecasts. An intervening target touch is not endpoint success. Minimum horizon is **30 trading sessions** (explicit convention, not 30 calendar days). Fixed candidate horizons: **30, 60, 126, 252 sessions**. Never extend a losing forecast's deadline. New forecasts must not overwrite old ones.

The supplied adjusted-close panels are not automatically evidence about unadjusted share-price appreciation. Price-only and total-return targets must be identified separately; unknown adjustment or corporate-action handling blocks production certification.

## Frozen initial experiment
One initial model family, not an unrestricted search: past-only stock trend/reversal, volatility, drawdown, relative-strength and market-regime features; a small regularized linear/nonlinear expert ensemble. Compare against always-up, a trailing base-rate model, and a volatility/drift path baseline. Reweight experts using only already-matured out-of-sample losses. Calibrate and choose an abstention threshold using data strictly earlier than the evaluation interval. Chronological training, calibration and evaluation must be separated by label-maturity timestamps, not random ticker-row splits. Horizons are evaluated separately before any combined selection policy.

Path forecasts are distributions/bands, not a promised precise curve. Report endpoint error, pointwise interval coverage, simultaneous path coverage when computed, interval width, and a flat-path baseline. An endpoint hit rate is not path accuracy. Selecting paths on a positive lower bound does not inherit unconditional conformal coverage.

## Data and bias controls
Use point-in-time membership; do not select tickers by today's membership or full-history completeness. Features and eligibility use only information available at issue time. Missing future observations remain unresolved and are counted pessimistically as failures in a sensitivity report; do not replace them silently with zero returns or exclude them from the denominator. Separate true missing outcomes from horizons not matured by the dataset cutoff. Do not forward-fill prices through delisting. Report historical coverage gaps and adjustment ambiguity. Do not use pretrained forecasters with unknown pretraining cutoffs in a purported untouched historical test. Source hashes and cutoff dates must accompany results. Other repositories are read-only, Gridpull excluded, and private repository contents must not be copied into this public repository.

## Acceptance and rejection
Report selected precision, raw probability reliability/Brier score, coverage, counts, unique dates, non-overlapping horizon blocks, per-era results, costs and return/loss sizes. Cross-sectional rows and overlapping horizons are not independent trials. Report date-block bootstrap uncertainty and an explicit independent-observation diagnostic; do not use a naive binomial interval across correlated ticker-days as certification. Account for thresholds/horizons selected from a calibration grid. A >90% point estimate alone is insufficient. A lower confidence bound above 90%, non-trivial coverage, and no unresolved material data defects are needed before any claim of supported >90% precision. Zero qualifying signals means the goal was not achieved, not perfect accuracy.

## Evaluation status
Prior repository researchers have already inspected much of the historical sample. This experiment can be chronological out-of-sample relative to each fitted model, but the reused historical evaluation is **not a new untouched research holdout**. Prospective, append-only paper forecasts after this registration are required for genuinely new forward evidence. This document does not claim a validated edge or guaranteed accuracy.
