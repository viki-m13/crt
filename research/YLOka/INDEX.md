# YLOka — Session Index

**Branch**: `claude/rebuild-stock-selection-YLOka`
**Mission**: Maximize OOS Walk-Forward Performance for Main Stock Selection Strategy
**Start**: 2026-05-10
**Author tag**: All artifacts from this branch live under `*/YLOka/` to avoid colliding with concurrent agents working on the same mission.

## Layout (this branch's deliverables only)

```
research/YLOka/
  00_repo_map.md              ← Phase 0
  01_engine_audit.md          ← Phase 0
  02_hypotheses.md            ← Phase 1
  exp_NN_*.md                 ← Phase 2 experiment write-ups
  graveyard/                  ← failed experiment write-ups (kept for evidence)
strategy/YLOka/
  features/
  selection.py
  regime.py
  config.yaml
backtests/YLOka/
  runs/<timestamp>_<hash>/
  experiment_log.csv
reports/YLOka/
  final_validation.md
  leakage_redteam.md
  executive_summary.md
tests/YLOka/
  test_pit_membership.py
  test_feature_lag.py
  test_walkforward_splitter.py
  test_costs.py
data/YLOka/                   ← (placeholder; using existing experiments/monthly_dca/cache)
```

## Decisions made (Phase 0)

- **Frozen holdout**: 2024-05 → 2026-04 (24 months). Research happens on 2003-09 → 2024-04. Holdout is run **once** in Phase 4 and never re-tuned.
- **Data source**: existing price-only panel (no new data), per user direction.
- **Objective**: maximize walk-forward OOS CAGR; tie-break Sharpe; constraint MaxDD ≤ ~-50% (v3 ceiling).
- **Engine**: do not rewrite. Reuse `experiments/monthly_dca/v2/sp500_pit_extended_sweep.simulate_variant` and the cached `ml_preds_v2.parquet`. New strategies are scorers + portfolio rules layered on top.
- **Headline metric replacement (recommended; not yet pushed to website)**: full-OOS CAGR + a non-overlapping rolling 5y CAGR distribution — replacing the misleading "10 splits mean 42.80%" claim. Per user instruction, this session does NOT touch the website.

## Status

| Phase | Item | Status |
|---|---|---|
| 0 | Repo map | ✅ `research/YLOka/00_repo_map.md` |
| 0 | Engine audit | ✅ `research/YLOka/01_engine_audit.md` |
| 0 | Reproduce baseline OOS CAGR (39.77%) | ✅ Verified to 4 dp |
| 1 | Hypothesis list | ✅ `research/YLOka/02_hypotheses.md` (17 ranked) |
| 2 | Experiment harness | ✅ `strategy/YLOka/harness.py` (reproduces v3 to 4 dp) |
| 2 | Cheap experiments (19 runs) | ✅ `research/YLOka/exp_summary.md` + `backtests/YLOka/runs/` |
| 2 | Tests | ✅ `tests/YLOka/test_*.py` — all PASS |
| 2 | Executive summary | ✅ `reports/YLOka/executive_summary.md` |
| 3 | Hard-constraint check on a survivor | ❌ no surviving candidate this session |
| 4 | Frozen-holdout gauntlet | ⏳ deferred (no candidate to test yet) |
| — | Wire winner to main page | ❌ skipped per user direction (do not update website) |

## Key result

**No candidate beats v3 in cheap Phase-2 experiments.** Surface "wins" (H6 Donchian +8pp CAGR, H3 accel +1.6pp) are sample-of-1/2 artifacts driven by 2016 (NVDA), 2020 (COVID), and 2022 (rate-hike). v3 reproduction: **CAGR 39.77%, Sharpe 0.955, MaxDD -49.83%**, 268 months. Free improvement found: 3% T-bill yield in cash months → +0.07pp CAGR.

## Next session

Defer to H1 (multi-target ensemble with 12m + classifier heads), H8 (overnight/intraday split), H10 (sector-residualized momentum). All require GBM retraining or new feature pipelines — ~2-3 hours of compute. See `reports/YLOka/executive_summary.md` §"What to do next session".
