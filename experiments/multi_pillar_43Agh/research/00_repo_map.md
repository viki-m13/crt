# 00 — Repo Map

Date: 2026-05-10. Branch: `claude/multi-pillar-stock-strategy-43Agh`.

This file maps every part of the existing codebase that is relevant to the
multi-pillar stock strategy. It is the entry-point for new contributors.

---

## 1. Top-level layout

```
crt/
├── PLAN.md                     # original "rebound scanner" plan (legacy)
├── README.md                   # empty
├── api/analyze.py              # serverless analyze endpoint
├── docs/                       # the website (main page = index.html)
│   ├── index.html              # landing — links to monthly_dca, screener, etc
│   ├── monthly_dca.js          # the deployed monthly stock strategy
│   ├── screener.html           # daily-rebound scanner
│   ├── data/full.json          # webapp data fed by build_webapp_*
│   └── data/tickers/           # per-ticker JSON for detail views
├── experiments/                # all research code lives here
│   └── monthly_dca/            # THIS strategy's research tree
├── max/, spreads/, crypto/     # other research (ignore — out of scope)
├── strategies/                 # other strategies (credit_spread, stillpoint, touch_predict)
└── scripts/                    # daily_scan.py + extras (legacy single-day scanner)
```

The **deployed** monthly strategy is served from `docs/monthly_dca.js` →
`docs/data/full.json`. The **research** for that strategy is under
`experiments/monthly_dca/`. New research lives under `research/` (this file's
parent) plus `strategy/`, `backtests/`, `reports/`, `tests/`, `data/`.

---

## 2. The deployed strategy (V3 = "v6_minimal" cash-yield variant)

Specification (verified in `experiments/monthly_dca/v6/lib_engine.py`):

```
universe         = "sp500_pit"          # PIT S&P 500 membership
scorer           = "ml_3plus6"          # mean of ml_pred_3m and ml_pred_6m
regime_gate      = "tight"              # crash detector
k_normal         = k_recovery = k_bull = 3
weighting        = "ew"                 # equal-weight (V3) / "invvol" (V6)
hold_months      = 6
cost_bps         = 10                   # round-trip
cash_yield_yr    = 0.03                 # T-bill credit during cash months
```

Headline metrics on PIT S&P 500, 2003-09 → 2025-12 (parity-tested):

| Metric            | V3 deployed | V6 winner (invvol) | V7 safer (sl+CDI+TLT) |
|-------------------|------------:|-------------------:|----------------------:|
| Full-window CAGR  | **39.77%**  | 38.20%             | 29.57%                |
| Sharpe            | 0.955       | 0.971              | **1.105**             |
| MaxDD             | -49.83%     | -45.98%            | **-28.97%**           |
| WF mean CAGR      | 42.80%      | 42.48%             | 32.64%                |
| WF n_pos / 10     | 10          | 10                 | 10                    |
| WF n_beats_SPY/10 | 9           | 9                  | 9                     |

Source of truth: `experiments/monthly_dca/v6/results/v3_baseline_metrics.json`,
`v6_baseline_metrics.json`, `v7/results/v7_safer_metrics.json`.

The ~9,000 strategy variants explored in v6/v7 are a tightly explored
neighbourhood around the V3 deployed strategy. **What has NOT been tried at
scale**: stock-level (not just market-level) trend gates, failure-avoidance
filters built from forensic study of failures, novel-math features (TDA,
HMM state probs, transfer entropy), forensic archetype matching to known
pre-runner patterns, regime-conditional concentration (k variable by
signal agreement, not just regime).

Those four things are what the multi-pillar project introduces.

---

## 3. Data inventory

All under `experiments/monthly_dca/cache/`:

| File                                  | Shape         | Contents |
|---------------------------------------|---------------|----------|
| `prices_extended.parquet`             | 8133 × 1833   | Daily adj close, 1995-01-03 → 2026-05-07, 1833 tickers |
| `prices.parquet`                      | 2891 × 964    | Daily adj close, 2014-09-22 onwards (subset of above) |
| `monthly_returns_clean.parquet`       | 377 × 1833    | Month-end returns, bad-data masked, 1995-01 → 2026-05 |
| `delisted_panel.parquet`              | 3105 × 20     | Daily prices for delisted tickers (WM, FNMA, FMCC, SHLD, DDS, BBBY, FRO, SVB, …) |
| `fwd_returns.parquet`                 | ~50 MB        | Forward 1m/3m/6m returns per ticker per asof |
| `meta.parquet`                        | small         | Ticker metadata (sector, mcap snapshot) |
| `features/{YYYY-MM-DD}.parquet`       | 353 monthlies | 67-feature per-ticker frame, strict PIT (every value uses data ≤ asof) |
| `v2/ml_preds_v2.parquet`              | 372,218 × 7   | Walk-forward ML predictions: pred_1m, pred_3m, pred_6m for 1811 tickers |
| `v2/sp500_pit/sp500_membership_monthly.parquet` | 140,011 × 2 | PIT S&P 500 membership: 985 unique tickers, 280 monthlies starting 2003-01-31 |
| `v2/monthly_returns_clean.parquet`    | 377 × 1833    | (alias of top-level) |
| `v2/bad_data_tickers.json`            | small         | Tickers excluded for data-quality reasons |
| `v2/bad_month_cells_mask.parquet`     | 377 × 1833    | Per-cell boolean: bad data → masked to NaN in monthly_returns_clean |

Feature columns (67) include: `pullback_1y`, `mom_12_1`, `mom_6_1`, `mom_3y`,
`vol_1y`, `rsi_14`, `dd_from_52wh` (positive magnitude — engine flips sign
on load), `trend_health_5y`, `excess_5y_logret`, `recovery_rate`,
`fip_score`, `idio_mom_12_1`, `tight_consolidation_60`, `breakout_strength_60`,
`bb_width_pct`, `bb_width_contraction`, `multibagger_ratio_24m`,
`drawdown_age_days`, `quality_score_5y`, `tail_ratio_24m`, `mom_consistency_12m`,
`new_52wh`, plus SPY-specific (`d_sma200`, `max_below_200_streak`, etc.).
Full list in `experiments/monthly_dca/cache/features/2024-12-31.parquet`.

**Survivorship**: `delisted_panel.parquet` includes price series for
8 delisted tickers (WM, FNMA, FMCC, SHLD, DDS, BBBY, FRO, SVB, plus
others). The S&P PIT membership table is built from
`sp500_changes_since_2019.csv` + `sp500_hist_1996_2019.csv` (985 unique
tickers ever in the index). `monthly_returns_clean.parquet` covers 1833
tickers — a superset that includes most delisted names. **Not perfect
survivorship-bias-free, but materially better than yfinance-only.**

---

## 4. The simulation engine

Source: `experiments/monthly_dca/v6/lib_engine.py:simulate`.

Parity test (`run_baseline.py`) reproduces V3 numbers **bit-for-bit**:
```
V3 deployed:  cagr 0.39774062  sharpe 0.95536375  max_dd -0.49828619
V6 reproduces:cagr 0.39774062  sharpe 0.95536375  max_dd -0.49828619
```

Key engine properties (verified):

1. **PIT membership**: each month-end, the candidate set is filtered
   through `sp500_membership_monthly.parquet` (`lib_engine.py:202`).
2. **PIT features**: features at asof T are loaded from
   `features/{T.date}.parquet`, generated from data ≤ T
   (`backtester.py:compute_features`).
3. **Walk-forward ML**: predictions in `ml_preds_v2.parquet` were generated
   with annual retrain and a **7-month embargo** vs the 6-month forward
   target — verified in `experiments/monthly_dca/v2/ml_strategy.py:200-238`.
4. **Execution**: signal at month-end T → fill at next month-end's price
   (`monthly_returns[next_d, ticker]`). Effective lag ≈ 1 trading day if
   we interpret the rebal as "buy on first trading day of T+1 month".
5. **Costs**: 10 bps per rebalance × gross weight. Slippage not separately
   modelled (small).
6. **Delisting**: NaN return at month T → treated as **-100% return for
   that pick that month** (`lib_engine.py:546-548`). Honest, harsh.
7. **Cash credit**: 3%/yr in cash months (`v6_minimal`).

The v7 engine adds:
- Daily-resolution per-pick stop-loss (`v7/daily_stop_validator.py`)
- Conditional Drawdown Insurance (CDI) — dynamic SH overlay
- Permanent TLT sleeve

For the multi-pillar work we **build on top of v6's `simulate`**, not v7,
so we don't bake in the safety knobs before we've evaluated alternatives.

---

## 5. The walk-forward splits

10 splits used as the headline OOS metric (`lib_engine.py:578-589`):

| Split    | Train end | Test window         | Note            |
|----------|-----------|---------------------|-----------------|
| A1       | -         | 2011-01 → 2018-12   | Adaptive        |
| A2       | -         | 2015-01 → 2021-12   | Adaptive        |
| A3       | -         | 2018-01 → 2024-12   | Adaptive        |
| R1_GFC   | -         | 2008-01 → 2010-12   | GFC stress      |
| R2       | -         | 2011-01 → 2013-12   | Post-GFC bull   |
| R3       | -         | 2014-01 → 2016-12   | Mid-cycle       |
| R4       | -         | 2017-01 → 2019-12   | Late-cycle      |
| R5_COVID | -         | 2020-01 → 2022-12   | COVID stress    |
| R6_AI    | -         | 2023-01 → 2024-12   | AI rally        |
| STRICT   | -         | 2021-01 → 2024-12   | Strict-OOS test |

These are **test-only windows**. The ml_preds parquet is one big OOS panel
(every ticker-asof predicted using a model trained on data ≥ 7 months
older), so any split simply slices the panel.

**Frozen holdout for this project**: 2025-01-01 → 2026-05-07. We touch this
exactly once at Phase 5. Any tuning that uses this window invalidates the
project.

---

## 6. Web app integration points

The deployed monthly strategy is rendered by:
- `docs/monthly_dca.js` — page logic, table rendering
- `docs/data/full.json` — picks data (regenerated by
  `experiments/monthly_dca/build_webapp_json_v3.py`)
- `docs/index.html` — main page that links to monthly_dca

Wiring a new strategy means: (a) producing `docs/data/full.json` with the
new picks panel; (b) updating `monthly_dca.js` only if the schema changes;
(c) optionally updating `index.html` to add a sub-page link.

---

## 7. Where new code goes

```
research/                         # plans, experiments, write-ups (markdown)
  00_repo_map.md                  # this file
  01_engine_audit.md              # next file
  02_invention.md                 # design doc — multi-pillar architecture
  forensics/winners/              # Study A
  forensics/failures/             # Study B
  pillar_1_failure_avoidance/
  pillar_2_trend_regime/
  pillar_3_novel_math/
  pillar_4_archetype/
  pillar_5_composite/
  exp_NN_*.md                     # individual experiment write-ups
  graveyard/                      # things that didn't work, with the why
strategy/                         # production code
  features/                       # feature builders
  failure_filter.py
  regime.py
  trend.py
  novel_features.py
  archetype.py
  selection.py
  sizing.py
  config.yaml
backtests/runs/<ts>_<hash>/       # one folder per backtest run
backtests/experiment_log.csv      # rolling experiment ledger
backtests/pillar_decomposition.parquet
backtests/final_walkforward.parquet
reports/
  final_validation.md
  leakage_redteam.md
  mechanism.md
  decomposition.md
  executive_summary.md
tests/
  test_pit_membership.py
  test_feature_lag.py
  test_walkforward_splitter.py
  test_costs.py
  test_no_lookahead.py
data/
  raw/                            # external pulls (none used so far)
  cache/                          # symlink or shadow of experiments/.../cache
  README.md
```

The engine itself stays at `experiments/monthly_dca/v6/lib_engine.py` —
no fork. New strategies are just new `V6Config` configurations with extra
pillar overlays imported from `strategy/`.
