# v8 — stock-selection rebuild research

Branch: `claude/rebuild-stock-selection-2qHxY`. Run identifier: **v8**.
Date: 2026-05-10. Author: Claude.

This directory contains the experimental sweep, weekly walk-forward
build, and final validation gauntlet that produced the
`reports/executive_summary.md` and `reports/final_validation.md`.

## Layout

```
v8/
├── README.md                       (this file)
├── run_tier1_sweep.py              concentration / horizon / scorer / regime
├── run_tier2_addons.py             single-knob and stacked add-ons on top
├── run_tier3_stacks.py             TLT-fallback stacks + staggered ensemble
├── run_validation_gauntlet.py      Phase 3+4 robustness + holdout + bias
├── results/                        every CSV/JSON output from the above
└── weekly/                         TRUE weekly walk-forward (KILLED)
    ├── build_weekly_features.py    weekly PIT feature panel (regenerable)
    ├── build_weekly_membership_and_spy.py
    ├── build_targets_alt.py        (started, not finished)
    ├── fit_weekly_gbm.py           walk-forward HistGBM at weekly cadence
    ├── lib_weekly.py               weekly simulator
    ├── run_weekly_baseline.py      13-variant weekly sweep
    ├── cache/                      weekly_preds.parquet (committed) + others
    │                                features_weekly.parquet is gitignored
    │                                because it's 337MB; rebuild via
    │                                build_weekly_features.py (~30s)
    └── results/                    weekly sweep CSVs
```

## Reproducing

From the repo root:

```bash
# Reproduce v3 baseline (sanity check vs deployed numbers)
python3 experiments/monthly_dca/v6/run_baseline.py

# Tier 1 sweep (~80s)
python3 experiments/monthly_dca/v8/run_tier1_sweep.py

# Tier 2 add-on sweep (~17s)
python3 experiments/monthly_dca/v8/run_tier2_addons.py

# Tier 3 stacks + staggered ensembles (~10s)
python3 experiments/monthly_dca/v8/run_tier3_stacks.py

# Final validation gauntlet (~30s)
python3 experiments/monthly_dca/v8/run_validation_gauntlet.py

# Weekly track (KILLED): expensive to rebuild
python3 experiments/monthly_dca/v8/weekly/build_weekly_features.py        # ~30s
python3 experiments/monthly_dca/v8/weekly/build_weekly_membership_and_spy.py
python3 experiments/monthly_dca/v8/weekly/fit_weekly_gbm.py               # ~3min
python3 experiments/monthly_dca/v8/weekly/run_weekly_baseline.py
```

## Headline result

`exp_02_winner` Pareto-improves the deployed v3 across every walk-
forward metric (WF mean OOS CAGR 50.16% vs 42.80%; Sharpe 1.08 vs
0.96; MaxDD -44.5% vs -49.8%; 10/10 vs 9/10 splits beat SPY) using:

- `scorer="ml_3plus6plus1"` (mean of 1m + 3m + 6m predictions)
- `regime_gate="safer"` (earlier crash trigger)
- `k=1` (top-pick concentration)
- `hold_months=1` (monthly rotation)
- `crash_fallback="tlt"` (long Treasuries during crash months)

See `reports/final_validation.md` for the full gauntlet results
including frozen holdout (-32% vs SPY +16% Jan 2025 → Apr 2026 — a
serious failure of recent OOS) and bias sensitivity (k=1 is fragile
to delisting; v3's k=3 is more robust). See
`reports/executive_summary.md` for the deploy-or-not recommendation.

The website was NOT updated as part of this run, per user instructions.
