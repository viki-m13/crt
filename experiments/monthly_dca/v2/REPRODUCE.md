# Reproducing the v3 PIT S&P 500 strategy

This document is the single canonical reference for re-running every step of
the v3 PIT S&P 500 strategy from scratch, end to end. Every artifact is
derived from inputs that are checked into this repo.

---

## 0. Prerequisites

Python ≥ 3.10 with pandas, numpy, pyarrow, scikit-learn:

```bash
pip install pandas numpy pyarrow scikit-learn
```

All commands assume working directory = `/home/user/crt` (the repo root).

---

## 1. Data prerequisites (already cached)

The following are checked into the repo and form the inputs:

  - `experiments/monthly_dca/cache/v2/monthly_prices_clean.parquet` — clean
    month-end prices for 1,833 tickers, 1995-01 to 2026-05 (377 months).
  - `experiments/monthly_dca/cache/v2/monthly_returns_clean.parquet` — same
    panel as monthly returns, with [-100%, +200%] cap and bad-month masks.
  - `experiments/monthly_dca/cache/features/*.parquet` — 353 monthly feature
    panels, one per month-end 1997-01 to 2026-05.  67 columns each.
  - `experiments/monthly_dca/cache/v2/ml_preds_v2.parquet` — v2 GBM walk-forward
    predictions (annual retrain, 7-month embargo, 1m+3m+6m horizons), 372k
    rows over 268 months (2003-09 to 2025-12). **Used for backtest**.
  - `experiments/monthly_dca/cache/v2/ml_preds_live.parquet` — v2 GBM live
    predictions extending past the WF cutoff (385k rows, 2003-01 to
    2026-04). **Used for current-month picks only**.

If any of these are stale or missing, regenerate as follows:

  - Prices: `python3 -m experiments.monthly_dca.load_data` (rebuilds from
    `docs/data/tickers/*.json`).
  - Features: `python3 -m experiments.monthly_dca.cache_features` (depends
    on prices) and then `python3 -m experiments.monthly_dca.extra_features`.
  - GBM preds: `python3 -m experiments.monthly_dca.v2.ml_strategy` (slow:
    ~10 min for full walk-forward fit).

---

## 2. Build the PIT S&P 500 membership panel

```bash
python3 -m experiments.monthly_dca.v2.build_sp500_pit_membership
```

Inputs:
  - `cache/v2/sp500_pit/sp500_hist_1996_2019.csv` — daily PIT constituents
    1996-01 to 2019-01 (sourced from fja05680/sp500 GitHub repo).
  - `cache/v2/sp500_pit/sp500_changes_since_2019.csv` — add/remove change
    events 2019-01 onwards (110 events in current file). **Update this
    file manually as new index changes occur**.

Output:
  - `cache/v2/sp500_pit/sp500_membership_monthly.parquet` — one row per
    (asof, ticker) for each panel month-end, indicating membership.  Roughly
    500 members × 280 months = ~134k rows.

This step is fast (~10 seconds) and idempotent.  Re-run any time the
post-2019 changes file is updated, or when the panel extends to a new month.

---

## 3. Strategy sweeps (one-time, ~13 min)

These are the discovery sweeps that found the v3 winner config.  Re-run
only when the strategy logic, scorers, or hyperparameter space changes.

```bash
# Base sweep: 13 scorers × 5 K × 3 weightings × 3 gates × 3 holds = 1,755 variants.
python3 -m experiments.monthly_dca.v2.sp500_pit_strategy_sweep
# ~13 min on a single core.

# Focused sweep: multi-horizon scorers × 2 K × 2 holds × 2 weightings = 64 variants.
python3 -m experiments.monthly_dca.v2.sp500_pit_v3_focused_sweep
# ~30 sec.
```

Outputs:
  - `cache/v2/sp500_pit/sp500_pit_sweep_results.csv` (1,755 rows, base sweep).
  - `cache/v2/sp500_pit/sp500_pit_sweep_winner.json` (composite winner).
  - `cache/v2/sp500_pit/v3_focused_sweep.csv` (64 rows, focused).
  - `cache/v2/sp500_pit/sp500_pit_feature_ic.csv` (per-feature IC within S&P 500).

---

## 4. Validation pipeline (~3 min)

```bash
# Full validation of the v3 winner: WF, year-by-year, drawdowns, sub-periods,
# turnover, most-picked, bias overlay.
python3 -m experiments.monthly_dca.v2.sp500_pit_v3_validate "ml_3plus6|k3_3_3|ew|tight|h6|cap1.0"

# Generalisation: same v3 config across 5 universes (broader, non-SP500 PIT,
# random 500 × 5 seeds).
python3 -m experiments.monthly_dca.v2.sp500_pit_v3_generalize

# Parameter sensitivity: K ∈ {1..7}, hold ∈ {1..12}, 3 gates, 4 weightings,
# 7 cost levels.
python3 -m experiments.monthly_dca.v2.sp500_pit_v3_sensitivity
```

Outputs:
  - `cache/v2/sp500_pit/v3_ml_3plus6_*.csv` — winner full validation
    (equity curve, walkforward, yearly, sub-periods, drawdowns, most-picked,
    bias sensitivity).
  - `cache/v2/sp500_pit/v3_generalize.csv` + `v3_generalize_*_equity.csv`.
  - `cache/v2/sp500_pit/v3_winner_sensitivity.csv`.

---

## 5. Webapp data.json (daily, ~30 sec)

```bash
python3 -m experiments.monthly_dca.v2.build_webapp_v3_pit
```

Inputs:
  - `ml_preds_v2.parquet` (WF) — used for the historical equity curve
    (no look-ahead).
  - `ml_preds_live.parquet` (LIVE) — used **only** for the current
    basket pick at month-ends past the WF cutoff.
  - `cache/v2/sp500_pit/sp500_membership_monthly.parquet`.
  - All v3 validation CSVs from step 4.

Output:
  - `experiments/docs/monthly-dca/data.json` — consumed by
    `docs/monthly_dca.js`.

---

## 6. Cron / daily refresh

The complete daily refresh chains steps 1 (price update), 2-3 (features),
5 (PIT membership rollforward), 7 (data.json):

```bash
python3 -m experiments.monthly_dca.cron_daily_refresh
```

Suggested crontab (runs once daily at 21:30 NY time = 02:30 UTC):

```
30 2 * * * cd /path/to/crt && /usr/bin/python3 -m experiments.monthly_dca.cron_daily_refresh >> /var/log/dailystockguide.log 2>&1
```

The cron does NOT re-run the strategy sweeps (steps 3-4) — those are slow
and historical.  Run them manually whenever the strategy logic changes.

The cron does NOT retrain the GBM (`ml_strategy.py`) — that's a periodic
(monthly or annual) operation.  Add a separate cron entry or run manually.

---

## 7. Operational checklist

When does each artifact need to refresh?

| Artifact                                       | Trigger                                               | Cost (compute) |
|------------------------------------------------|-------------------------------------------------------|---------------:|
| `monthly_prices_clean.parquet`                 | Daily (new prices)                                    |    1 min       |
| `cache/features/*.parquet`                     | Daily (new month-ends, refresh recent 3)              |   30 sec       |
| `sp500_membership_monthly.parquet`             | Whenever S&P 500 changes (manual update)              |   10 sec       |
| `sp500_changes_since_2019.csv`                 | Manual: append new add/remove events as they occur    |    n/a         |
| `ml_preds_v2.parquet` (WF backtest predictions)| Annually (or when strategy logic changes)             |   ~10 min      |
| `ml_preds_live.parquet` (live predictions)     | Annually (or when strategy logic changes)             |   ~10 min      |
| Strategy sweep CSVs (`v3_*`)                   | When K/hold/scorer/regime gate logic changes          |   ~14 min      |
| `data.json` (webapp output)                    | Daily (cron)                                          |   30 sec       |

---

## 8. Files index

### Strategy code (all under `experiments/monthly_dca/v2/`)
- `ml_strategy.py` — the core walk-forward GBM trainer (v2 model used by v3).
- `build_sp500_pit_membership.py` — builds PIT S&P 500 panel.
- `sp500_pit_strategy_sweep.py` — base 1,755-variant sweep.
- `sp500_pit_v3_focused_sweep.py` — focused multi-horizon sweep (64 variants).
- `sp500_pit_v3_validate.py` — full validation harness.
- `sp500_pit_v3_generalize.py` — multi-universe generalisation.
- `sp500_pit_v3_sensitivity.py` — parameter sensitivity sweep.
- `sp500_pit_extended_sweep.py` — extended search space (4,608 variants;
  superseded by focused sweep + sensitivity).
- `sp500_pit_filter_backtest.py` — original v2 baseline (filter-only).
- `sp500_pit_retrain_backtest.py` — re-train baseline (academically clean).
- `sp500_pit_bias_overlay.py` — synthetic-delisting MC overlay.
- `build_webapp_v3_pit.py` — v3 webapp data builder.
- `build_webapp.py` — legacy v2 builder (kept for reference).

### Reports
- `experiments/monthly_dca/v2/SP500_PIT_ANALYSIS.md` — original v2-on-PIT
  analysis (the 80% → 15% collapse that motivated v3).
- `experiments/monthly_dca/v2/SP500_PIT_V3_REPORT.md` — v3 final strategy
  report (this is the canonical reference).
- `experiments/monthly_dca/v2/REPRODUCE.md` — this file.

### Data outputs (all under `experiments/monthly_dca/cache/v2/sp500_pit/`)

PIT raw data:
- `sp500_hist_1996_2019.csv` (7.5 MB)
- `sp500_changes_since_2019.csv`
- `sp500_today.csv` (sanity check)
- `sp500_membership_monthly.parquet`
- `sp500_membership_count.csv`

Sweep outputs:
- `sp500_pit_sweep_results.csv` (1,755 rows)
- `sp500_pit_sweep_winner.json`
- `sp500_pit_feature_ic.csv`
- `v3_focused_sweep.csv`

v3 validation outputs:
- `v3_ml_3plus6_summary.json`
- `v3_ml_3plus6_walkforward.csv`
- `v3_ml_3plus6_yearly.csv`
- `v3_ml_3plus6_sub_periods.csv`
- `v3_ml_3plus6_drawdowns.csv`
- `v3_ml_3plus6_most_picked.csv`
- `v3_ml_3plus6_bias_sensitivity.csv`
- `v3_ml_3plus6_333_ew_tight_h6_equity.csv`
- `v3_winner_sensitivity.csv`
- `v3_generalize.csv` + per-universe equity curves

### Webapp output
- `experiments/docs/monthly-dca/data.json` — consumed by
  `docs/monthly_dca.js`. Generated by `build_webapp_v3_pit.py`.
- `experiments/docs/monthly-dca/data.json.v2-backup` — backup of the
  previous v2 data.json before the v3 swap.

---

## 9. Adding a new constituent change post-2019

Edit `cache/v2/sp500_pit/sp500_changes_since_2019.csv` and append a row:

```
2026-05-15,"NEWADD","REMOVED"
```

Then re-run:
```bash
python3 -m experiments.monthly_dca.v2.build_sp500_pit_membership
python3 -m experiments.monthly_dca.v2.build_webapp_v3_pit
```

The next daily cron will pick up the change automatically.

---

## 10. Re-running the sweeps from scratch

If you change scorers, K combos, weightings, gates, or holds in
`sp500_pit_strategy_sweep.py` or `sp500_pit_v3_focused_sweep.py`, run:

```bash
# Base sweep (~13 min)
python3 -m experiments.monthly_dca.v2.sp500_pit_strategy_sweep

# Focused sweep (~30 sec)
python3 -m experiments.monthly_dca.v2.sp500_pit_v3_focused_sweep

# Pick the new winner from the focused sweep results, then validate it:
python3 -m experiments.monthly_dca.v2.sp500_pit_v3_validate "<new-winner-name>"
python3 -m experiments.monthly_dca.v2.sp500_pit_v3_sensitivity   # update if winner config changes
python3 -m experiments.monthly_dca.v2.sp500_pit_v3_generalize    # ditto
```

Then update `STRATEGY_SPEC` in `build_webapp_v3_pit.py` to point at the new
winner's parameters and re-run the builder.
