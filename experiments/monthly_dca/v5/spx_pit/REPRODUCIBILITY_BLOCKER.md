# Reproducibility blocker: the deployed E2 ml_preds is a frozen artifact

**Date:** 2026-05-17 · `train_ml_pit.py`, `monthly-dca-refresh.yml`.

## What was attempted

User asked to advance the live site to "today's date". The augmented
daily panel (`prices_extended_pit.parquet`) already extends to
2026-05-11, so the offline chain was re-run end-to-end
(`cache_features_pit` → `add_alpha_features_pit` → `build_panel_pit` →
`build_sp500_pit_panel_aug` → `train_ml_pit`) and the `fit_walkforward`
`train_end=2025-12-31` hard cap was fixed (extend to the panel's latest
asof — a correct, necessary fix kept in `train_ml_pit.py`). The retrain
then produced predictions through **2026-04-30** (410,800 rows).

## The blocker (critical, honest)

Regenerating the chain **does not reproduce the deployed model**.
Comparing the regenerated `ml_preds` to the committed (deployed) one on
the 405,439 overlapping historical `(asof, ticker)` rows:

- prediction correlation **0.968**, mean |Δ| 0.0074, max |Δ| 0.099
- **0 of 274** historical month-ends have identical predictions
- full-window backtest CAGR collapses **56.6% → 38.2%**

The K=2-per-sleeve picker is hypersensitive: a ~0.7% mean shift in the
rank-prediction reshuffles the top-2 and swings CAGR ~18pp. The drift
is inherent, not a code bug: `panel_cross_section_v3.parquet` and the
augmented `features/` dir were never version-controlled (gitignored),
and the daily price panel itself moves under data-vendor restatements
(split/adjusted-close revisions, FNSPID/yfinance backfill). The
committed `ml_preds.parquet` that produced the validated 56.6% E2 is
therefore effectively a **frozen artifact whose exact generating inputs
no longer exist**.

## Consequence

"Update the site to today" and "keep the exact validated 56.6% model"
are **mutually exclusive on this data** until the pipeline is made
reproducible. Re-running it to advance the date silently degrades the
strategy. This is a product/research decision, not a silent cron fix:

- **(A) Keep frozen** — site stays `as_of 2025-12-31` (honest but the
  live basket/dates are stale). The deployed numbers remain validated.
- **(B) Re-derive + re-freeze** — accept a freshly-rebuilt model
  (whatever CAGR fresh data yields, ~38–56%), re-run the full
  overfit/validation gauntlet on it, then re-freeze and advance the
  date. New public numbers.
- **(C) Reproducibility fix (correct long-term)** — commit
  `panel_cross_section_v3.parquet` (and pin the daily panel) so future
  appends are consistent with deployed history; only then can the
  monthly workflow extend the live month without changing the backtest.

## Actions taken (safe, no regression shipped)

1. **Restored** the validated 56.6% `ml_preds.parquet`,
   `sp500_pit_panel.parquet`, and `data.json` from `origin/main`. The
   live site is unchanged and correct (E2 56.6% / Sharpe 1.10).
2. **Regression guard added** to `monthly-dca-refresh.yml`: the job
   aborts (no commit) if the rebuilt full-window CAGR drifts > 5pp from
   the deployed 0.566 reference. The cron can **never silently deploy a
   degraded model** again.
3. Kept the `train_ml_pit` `train_end` fix (correct; only usable once
   reproducibility path C is done).

## Recommendation

Pursue **(C)**: version-control `panel_cross_section_v3.parquet` and
pin the augmented daily panel, then a controlled monthly extend can
advance the live month without disturbing the validated backtest. Until
then the site honestly stays `as_of 2025-12-31`; the regression guard
keeps it safe. This is a data-engineering reproducibility task, offered
for sign-off — not silently forced.
