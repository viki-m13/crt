# Downside state recurrence research

**Executed research; no validated >95% predictor and no live recommendations.**

Read `RESULTS.md` and `CI_VERIFICATION.md`. All 49 fixed method/cutoff policies plus the adaptive95 diagnostic are available in the result artifacts. The original protocol is committed at `409753505c1c90d915ce9fbcd83b6840bd70a383:research/recurrence_downside/PROTOCOL.md`.

## Reproduce from a fresh output directory

Run from the repository root with Python 3.13:

```bash
python -m pip install -r research/failure_first/requirements.txt
python -m pytest tests/test_failure_first.py tests/test_recurrence_downside.py -q
python -m research.failure_first.download --output pinned-inputs
python -m research.recurrence_downside.run --inputs pinned-inputs --output results/sp500 --universe sp500
python -m research.recurrence_downside.validate --inputs pinned-inputs --directory results/sp500
python -m research.recurrence_downside.run --inputs pinned-inputs --output results/ndx --universe ndx
python -m research.recurrence_downside.validate --inputs pinned-inputs --directory results/ndx
```

The manifest pins eight PUBLIC CRT Parquet files by repository commit, byte length and SHA-256. S&P data end 2026-03-20; Nasdaq data end 2026-05-07. Historical membership and price/corporate-action coverage are incomplete. No SEC event extraction, new current-price feed, order placement or short P&L simulation is implemented.

For the frozen within-date label-permutation controls (stock-selection null, not an all-market null):

```bash
python -m research.recurrence_downside.run --inputs pinned-inputs --output results/null_sp500 --cache results/sp500 --universe sp500 --null --years 2024 2025 2026
python -m research.recurrence_downside.validate --inputs pinned-inputs --directory results/null_sp500 --cache results/sp500 --no-repeat
python -m research.recurrence_downside.run --inputs pinned-inputs --output results/null_ndx --cache results/ndx --universe ndx --null --years 2024 2025 2026
python -m research.recurrence_downside.validate --inputs pinned-inputs --directory results/null_ndx --cache results/ndx --no-repeat
```

Use `--cache` only with feature/label caches from the same exact input manifest and universe. This research cache is not a production cache-validation boundary. `--evaluate-only` replays saved annual forecasts; its runtime covers reporting only, not a new model fit. For an independent complete run use a fresh output directory without `--evaluate-only`.

## Result artifacts

Each real or null output directory includes `metadata.json`, `summary.csv/json`, `picks.csv`, `decisions.csv`, `fit_audit.json`, `reliability.json`, annual forecast Parquets and validation results. The user bundle excludes large regenerable feature/label/input caches but includes all annual forecasts and reporting files. The required shared `research/failure_first` code and exact input manifest are included.

Do not interpret a 0.95 score as 95% verified accuracy, unknown prices as declines, intermediate touches as endpoint success, overlapping forecasts as independent trials, or avoiding an interim squeeze as an executable short strategy. Borrow, recalls, dividends and fills are unobserved. Headlines should disclose counts and controls. Tests show code mechanics and reproducibility, not the existence of a forecasting edge.

The protocol was frozen before new outcomes were calculated. Previously inspected historical samples remain previously inspected; they are not a new final holdout. Work is isolated on `research/recurrence-downside-20260909`, stacked on the existing `research/downside-20260909` branch. Main and production are unchanged.
