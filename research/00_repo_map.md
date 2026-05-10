# 00 — Repo Map

## High-level
This repo hosts several quant strategies + a static webapp at `docs/`. The
brief is about the **monthly stock-pick DCA** strategy under
`experiments/monthly_dca/`. Other strategies (`crypto/`, `strategies/`,
`max/`, `spreads/`) are out of scope for this invention.

## Existing strategy (V3 winner — what we must beat)
`experiments/monthly_dca/REPORT.md` describes the current production strategy:
`strategy_rotation k=5 monthly_rebalance`, a regime-conditional rotation
across 3 sub-strategies (`pullback_in_winner`, `explosive_winners`,
`quality_pullback`) gated by SPY regime indicators (`d_sma200`, `rsi_14`,
`mom_12_1`).

| Metric | Value |
|---|---|
| Full-window CAGR XIRR (2002-2024) | 35.37% |
| Walk-forward MEAN test CAGR (10 splits) | 40.47% |
| Walk-forward MEDIAN | 42.42% |
| Walk-forward MIN / MAX | -7.67% / +99.38% |
| Mean OOS edge vs SPY DCA | +25.83pp |
| α=4%/yr bias-corrected median CAGR | 28.63% |

The numbers in the cached sweep CSV reproduce exactly. **Reproduction OK —
engine produces the headline numbers from the cached features and panel.**

## Code layout (monthly_dca only)
- `compound_engine.py` — true compounding portfolio simulator. Daily walk;
  evaluates exits; deploys cash equally across top-K each month.
- `fast_engine.py` — feature loaders, XIRR, single-position simulator.
- `alpha_features.py`, `alpha2_features.py` — feature builders.
- `strategies_*.py` — score functions; each takes a per-asof feature
  DataFrame and returns a Series of scores.
- `cache/features/<asof>.parquet` — 353 month-end snapshots (1997-01 to
  2026-04) of 67 features × ~1641 tickers.
- `cache/prices_extended.parquet` — daily adjusted close panel,
  1995-01-03 → 2026-05-07, 8133 rows × 1833 tickers.
- `cache/sweep_monthly_rebalance.csv` — full sweep across 23 strategies × k.
- `cache/wf_forced_aggregate.csv` — 10-split walk-forward aggregates.
- `cron_daily_refresh.py` — nightly cron pulling fresh prices.
- `build_webapp_v3.py` — produces `experiments/docs/monthly-dca/data.json`
  consumed by `docs/monthly_dca.js` and rendered on the `/monthly-dca/`
  page.

## Data
- 1833 tickers, daily adjusted close only (no volume in panel — see audit).
- Universe is roughly the union of recent S&P 500 constituents.
  **Survivorship-biased** (only 9 truly-dead tickers in 30 years).
- No fundamentals (price-only by design).
- 67 cached features per ticker per month-end.

## What's good
- Disciplined PIT slicing in feature builders (`sub = panel.loc[panel.index <= asof]`).
- Round-trip costs (5 bp) baked into reported CAGR.
- 10-split walk-forward with regime-spanning test windows including GFC,
  COVID, 2022 bear, 2023-24 AI rally.
- Explicit synthetic-delisting Monte-Carlo overlay at α∈{0..20%}/yr to
  partially compensate for the survivorship-biased universe.
- Bench excluded from picking universe.

## What's gappy (see `01_engine_audit.md`)
- No actual point-in-time S&P 500 constituents history.
- Universe contains only 9 truly-delisted names — overwhelmingly survivors.
- No volume in the panel — limits accumulation-footprint signals.
- Same-day execution: signal uses prices ≤ T close; deploy at T close.
  Minor; should be next-day open.
- Walk-forward TRAIN/TEST splits don't have an embargo around the boundary.

## Project layout being added (this session)
```
research/
  00_repo_map.md                  # this file
  01_engine_audit.md
  02_invention.md                 # inventor's notebook
  forensics/                      # case studies of past runners
  exp_*.md                        # experiment writeups
  graveyard/                      # killed candidates
strategy/
  features/                       # new feature modules
  selection.py                    # final selection rule
backtests/
  runs/                           # run artifacts
reports/
  final_validation.md
  leakage_redteam.md
  mechanism.md
  executive_summary.md
tests/
  test_pit_membership.py
  test_feature_lag.py
  test_walkforward_splitter.py
  test_costs.py
```
