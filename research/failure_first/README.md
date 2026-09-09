# Failure-first stock selection — executed research

**The >95% objective is NOT established.** The primary 95%-score policy emitted zero S&P forecasts and two Nasdaq forecasts, both wins. A less restrictive decomposition-only ablation emitted seven Nasdaq forecasts, all wins, but all seven originated in 2022 with overlapping one/two-year endpoints. Its next-close entry sensitivity is 6/7, not 7/7. None passed the separate historical evidence screen. See `RESULTS.md` for the complete interpretation.

This is an implemented experiment, not a new name for an untested idea. It is deliberately isolated from `research/adaptive_path/` and production. There is no brokerage connection or live recommendation endpoint.

## What was tested

Five separately fitted models classify mutually exclusive, exhaustive **price-observable endpoint failures**: unresolved endpoint; severe loss; other loss with nonpositive benchmark; other loss after an interim gain; remaining loss. These are not verified bankruptcy/dilution/earnings causes. All positive endpoints are successes even after a recovered drawdown. A separate interim-path risk head is an optional veto.

Methods are ordinary binary failure prediction, calibrated summed failure heads, a sum-plus-binary veto, that veto plus path risk, and a deliberately naive max-head control. The naive rule cannot be read as a valid probability: five disjoint 3% risks sum to 15%, not 3%. No independence of failure heads is assumed.

Each model is fitted annually using only earlier data, with a full-horizon purge separating training outcomes, calibration issue dates, and test timestamps. Thresholds and model parameters are fixed in `PROTOCOL.md`. Eight horizons are 30/60/90/126/180/252/504/756 NYSE sessions. Each stock receives its shortest qualifying horizon; at most one stock is emitted per five-session decision date. The deadline and reference price are locked, and the ticker cannot be reissued through that deadline.

All thresholds (.70/.80/.85/.90/.95/.975) and all variants are reported. A score cutoff is not the achieved accuracy. Zero picks is undefined accuracy, not 100%.

## Reproduce

Python 3.13, tested on Linux. Run from repository root:

```bash
python -m pip install -r research/failure_first/requirements.txt
python -m pytest tests/test_failure_first.py -q
python -m research.failure_first.download --output /tmp/ffv-inputs

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
python -m research.failure_first.run --inputs /tmp/ffv-inputs --output /tmp/ffv-sp500 --universe sp500
python -m research.failure_first.evaluate --directory /tmp/ffv-sp500
python -m research.failure_first.validate --directory /tmp/ffv-sp500 --inputs /tmp/ffv-inputs

python -m research.failure_first.run --inputs /tmp/ffv-inputs --output /tmp/ffv-ndx --universe ndx
python -m research.failure_first.evaluate --directory /tmp/ffv-ndx
python -m research.failure_first.validate --directory /tmp/ffv-ndx --inputs /tmp/ffv-inputs

# Fixed training-label permutation control; compare on the SAME 2024+ window.
python -m research.failure_first.run --inputs /tmp/ffv-inputs --output /tmp/ffv-null --universe sp500 --null --years 2024 2025 2026
python -m research.failure_first.evaluate --directory /tmp/ffv-null
python -m research.failure_first.validate --directory /tmp/ffv-null --inputs /tmp/ffv-inputs --skip-rerun
```

Repeat the null command with `--universe ndx` and a different output folder. Do not reuse a real-run output folder for a shuffled run. The `--skip-rerun` flag is necessary for null-run audits: the exact-fold repetition audit intentionally refits real rather than shuffled training labels.

The input downloader verifies all eight files against immutable commit references and SHA-256 hashes. Existing correctly hashed input files are reused. The `manifest.json` text may have different formatting between prior input exports; per-file content hashes are authoritative.

## Artifacts produced

`metadata.json`: configuration, source/input hashes, data limits, runtime. `fit_audit.json`: each annual/horizon training and calibration boundary. `features.parquet`: point-in-time eligible features. `outcomes_H.parquet`: every outcome including unresolved/pending labels. `risks_H.parquet`: predictions before selecting or attaching current outcomes. `picks.csv`: all issued policy forecasts. `decisions.csv`: issued and abstaining dates. `summary.json/csv`: complete policy grid, controls, uncertainty and sensitivities. `reliability.json`: all-candidate probability calibration. `validation.json`: mechanical audit and exact real-fold repetition/future-label attacks.

Raw intermediate Parquet files can be hundreds of MB; do not commit them. The GitHub Actions workflow independently downloads, tests, runs and audits both universes, then saves compact result artifacts for 30 days. Run serially on low-memory machines; the S&P panel and horizon labels can exhaust a 6GB machine when several jobs run concurrently.

## Validation versus prediction

50 new regression cases pass. They test model complement equivalence, union-risk arithmetic, exhaustive labels, exact endpoints versus touches, pending/missing handling, future-data attacks, three-way temporal purge, next-close entry, immutable deadlines, deterministic selection and no-pick behavior. These software tests do not establish predictive accuracy.

Local testing also passed the 31 tests in the prior SIEVE attachment: 81 total. SIEVE's saved prediction ledgers were independently recounted for comparison; the entire legacy SIEVE model was not refitted in this experiment. Its older single-random-stock benchmark differs from this experiment's exact expected same-date/horizon random rate.

## Material limitations

- S&P prices end 2026-03-20; Nasdaq prices end 2026-05-07. Neither is a current September scan.
- Mixed/dividend-adjusted data are not certified price-only appreciation. Terminal corporate-action proceeds are not verified.
- Historical constituent coverage is incomplete. This pilot uses 732 S&P-history tickers and 163 Nasdaq-history tickers, NOT the separate agent's 10,991-stock universe. These universes overlap.
- Historical outcomes were already studied in earlier projects. Annual chronological prediction prevents a fitted model from seeing its future; it does not turn previously researched data into an untouched scientific holdout.
- Date-block bootstrap and greedy non-overlap diagnostics do not prove independence. Perfect tiny cohorts can give degenerate bootstrap intervals; do not market those as certainty.
- Next-close and 20/50bp margin checks are sensitivities, not a fill-based portfolio return or net trading backtest.
- The SEC/event-based failure mechanisms remain untested here. No claim that every economic failure cause has been modeled.
