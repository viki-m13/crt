# Data — Pre-Runner Footprint Strategy (FHtzX)

This strategy reuses the existing data layer in
`experiments/monthly_dca/cache/` rather than building new sources.

## What's in the cache

- `experiments/monthly_dca/cache/prices_extended.parquet` —
  daily adjusted close, 1995-01-03 → 2026-05-07, 1833 tickers
- `experiments/monthly_dca/cache/features/<asof>.parquet` — 353
  month-end snapshots with 67 base features + 12 novel features added
  by `strategy/features/novel_features.py` for FHtzX:
  - `crt_6m`, `crt_3m` — Cross-Sectional Rank Trajectory
  - `rank_now`, `rank_6m_ago`, `rank_3m_ago` — rank percentiles
  - `rbi_60`, `rbi_120` — Reflexive Bounce Intensity
  - `vol_asym_60`, `vol_asym_126` — squared-up / squared-down ratio
  - `cst_score` — Capitulation–Stabilization Transition
  - `vov_60` — Volatility of volatility (CV of rolling 5d vol)
  - `prerunner_dist` — Mahalanobis-style distance to pre-runner archetype

## Sources

- Prices: yfinance adjusted close, fetched and cached by
  `experiments/monthly_dca/extend_history.py` and refreshed daily by
  `experiments/monthly_dca/cron_daily_refresh.py`.
- Universe: union of recent S&P 500 constituents over the cache period.
- Survivorship: 1833 tickers; only 9 truly delisted.  Bias is partially
  compensated by Monte-Carlo synthetic-delisting overlay at α∈{0..20%}/yr.

## Limitations

- No volume data in the cached panel.  FHtzX did not pull volume; all
  features are price-only.
- No fundamentals; price-only by design.
- No PIT S&P 500 membership.  The universe is an approximation —
  flagged in `research/01_engine_audit.md` as the largest honesty
  gap, partially mitigated by the bias overlay and the 30%-bucket
  ticker holdout test.

## Frozen holdouts (for FHtzX)

- **Time holdout**: 2024-07-31 → 2026-04-30.  Strategy never trained
  on this window.
- **Universe holdout**: SHA-256 bucketing of tickers into mod-10
  buckets; buckets {7, 8, 9} are the holdout (≈30% of tickers).
  Used by `strategy/holdout.py::universe_holdout_run`.
