# YLOka — Executive Summary

**Branch**: `claude/rebuild-stock-selection-YLOka`
**Mission**: Maximize OOS walk-forward CAGR & Sharpe for the main 3-stock 6-month basket on the front page.
**Session**: 2026-05-10 (Phase 0–2 cheap experiments).
**Website status**: NOT updated this session, per user direction.

## Bottom line

After Phase 0 audit and 19 cheap Phase-2 experiments, **no candidate clearly beats the v3 `ml_3plus6` baseline** under honest measurement. The two surface "wins" — H6 Donchian-130 (+8 pp CAGR) and H3 accel overlay (+1.6 pp CAGR) — are sample-of-1 and sample-of-2 artifacts dominated by 2016 (NVDA), 2020 (COVID rebound), and 2022 (rate-hike bear). They will not survive an honest holdout.

The single durable improvement is **trivial**: applying a 3% T-bill yield in cash months (currently 0%) gives +0.07 pp CAGR and a marginal MaxDD improvement at no implementation risk.

## Honest measurement

- **v3 baseline reproduces exactly**: full-OOS CAGR 39.77%, Sharpe 0.955, MaxDD -49.83%, 4 cash months, 268 months (2003-09 → 2025-12).
- **The published "42.80% mean across 10 walk-forward splits" headline oversells**: those 10 "splits" are time slices of one OOS curve, not 10 independent experiments. R1_GFC at 108.8% (buying the GFC bottom, 3-year window) pulls the mean up; median is 39.9% ≈ full-period CAGR.
- **The model's OOS retraining IS proper walk-forward** (Jan refit, 7-month embargo for 6m forward target). PIT membership ✅, survivorship ✅, no look-ahead ✅, bar-aligned execution ✅.

## Why the cheap experiments didn't break out

1. v3's GBM has already extracted ~80% of the price-only signal. Cheap downstream tweaks (weighting, K, hold, filters) re-arrange the same picks.
2. The model's score is well-calibrated as a rank but not as a magnitude. Conviction-weighting fails because the score gap doesn't predict relative outperformance.
3. ~600 v3-v7 prior variants on the same data + 19 here = significant multiple-testing exposure. Apparent +5 pp lifts driven by 1-2 outlier years are exactly what random search produces.
4. The K/h grid confirms K=3, h=6 is locally optimal on every dimension (within ±5 pp CAGR).

## Frozen holdout (not yet touched)

- **Reserved**: 2024-05 → 2026-04 (24 months). Will be queried ONCE in Phase 4 on the chosen winner.
- **No retuning** if it fails — pick next-best research-window candidate.

## What to do next session

The remaining unexplored directions all require either GBM retraining or new features (≥ 1 hour of compute). Ranked by expected value:

1. **H1 — multi-target ensemble**: train a 12m-rank GBM head and a top-quintile-classifier head; ensemble with 3m/6m by recent IC. Likely +1 to +3 pp CAGR. Cheapest unexplored direction.
2. **H8 — overnight/intraday return decomposition**: features from open/high/low (true-range, gap behaviour). Literature suggests overnight return dominates for momentum names — not currently in v3's 48-feature set.
3. **H10 — sector-residualized momentum**: requires PIT GICS sector tags (would need a one-time FactSet/Wikipedia/yfinance fetch).
4. **H9 — volume thrust persistence**: 5d up-volume ratio + breadth. Cheap once volume data is loaded.
5. **H7 — dispersion-conditional K**: K=3 in dispersed regimes, K=10 in compressed regimes. Cheap to test.

## Deliverables shipped this session

```
research/YLOka/
  INDEX.md                     ← session index
  00_repo_map.md               ← Phase 0 architecture map
  01_engine_audit.md           ← Phase 0 trustworthiness audit
  02_hypotheses.md             ← Phase 1: 17 ranked hypotheses
  exp_summary.md               ← Phase 2: results table + analysis
  graveyard/
    H2_conviction_sizing.md
    H3_accel_overlay.md
    H4_soft_cash_continuum.md
    H6_donchian_130_breakout.md
    K_hold_grid.md
strategy/YLOka/
  harness.py                   ← simulator + scorers + pickers + run logging
  run_experiments.py           ← Phase 2 sweep driver
backtests/YLOka/
  experiment_log.csv           ← append-only log of every run
  runs/<ts>_<name>_<hash>/
    manifest.json              ← config + metrics + git SHA
    equity.parquet             ← per-month equity + picks
tests/YLOka/
  test_pit_membership.py       ← 5 invariants, all PASS
  test_feature_lag.py          ← 3 invariants, all PASS
  test_harness_repro.py        ← v3 4-dp reproduction, PASS
reports/YLOka/
  executive_summary.md         ← this file
data/YLOka/
  pit_panel_with_scores.parquet  ← cached panel × ml_preds_v2
```

## Recommendations to the user

1. **Don't deploy any candidate from this session.** v3 stays as the production strategy.
2. **Update the headline claim** on the front page: replace "42.80% mean OOS CAGR across 10 walk-forward splits, 10/10 positive, 9/10 beat SPY" with the more honest "**Full-OOS CAGR 39.77% over 22 years, Sharpe 0.95, MaxDD -50%**" plus a non-overlapping rolling-5y CAGR distribution. (NOT done this session per user instruction; flagged for site-touching session.)
3. **Add T-bill yield to cash months in v3**: free 0.07 pp CAGR + 0.4 pp MaxDD improvement. One-line change.
4. **Lock the frozen holdout (2024-05 → 2026-04)** in code so no future agent or experiment can accidentally peek. Suggest a `tests/YLOka/test_no_holdout_peek.py` that fails CI if any new experiment-log entry has data after 2024-04.
5. **Plan next session for H1 + H8 + H10**: requires ~2-3 hours, most likely path to a real +CAGR.
