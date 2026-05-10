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
| 2a | Cheap experiments (19 runs) | ✅ `research/YLOka/exp_summary.md` |
| 2b | H1 multi-target ensemble (12m + cls heads, 12 variants) | ✅ `research/YLOka/exp_summary_session2.md` |
| 2b | H7 dispersion-conditional K (7 variants) | ✅ `research/YLOka/exp_summary_session2.md` |
| 2c | Session 3: 11 feature-based scorers | ✅ `research/YLOka/exp_summary_session3_4.md` |
| 2c | Session 4: 10 adaptive-IC / dynamic-hold / regime-K variants | ✅ `research/YLOka/exp_summary_session3_4.md` |
| 2 | Experiment harness | ✅ `strategy/YLOka/harness.py` (reproduces v3 to 4 dp) |
| 2 | Tests | ✅ `tests/YLOka/test_*.py` — all PASS |
| 2 | Executive summary | ✅ `reports/YLOka/executive_summary.md` |
| 3 | Hard-constraint check on a survivor | ❌ no surviving candidate |
| 4 | Frozen-holdout gauntlet | ⏳ deferred (no candidate to test) |
| — | Wire winner to main page | ❌ skipped per user direction |

## Key result

**80+ experiments across Sessions 1-4; v3 remains the local optimum.**

Sessions covered: H2 conviction sizing, H3 accel overlay, H4 soft-cash continuum, H6 Donchian-130 breakout, H1 multi-target ensemble (12m regressor + classifier heads, 12 variants), H7 dispersion-conditional K (7 variants), full K/h grid, scorer alternatives, cash-yield variant.

Surface "wins" — H6 +8pp, H3 +1.6pp, H7 disp_K23 +1pp — are all sample-of-1 to sample-of-3 artifacts from 2-3 specific years (2004, 2009, 2016, 2020, 2022). The same K-shrinkage mechanism (basket accidentally concentrated in NVDA-class winners during specific years) drives every "win". None will survive the frozen holdout.

The H1 multi-target ensemble — Session 1's most-likely-to-work direction — also fails: all 12 integration variants underperform by 0.05-9.1pp CAGR. Two new GBM heads (`pred_12m` and `pred_12m_cls`) trained walk-forward (22 years × 13-month embargo) produced no usable lift.

**v3 baseline**: CAGR 39.77%, Sharpe 0.955, MaxDD -49.83%, 268 months. **Only durable improvement found: +0.07 pp from a 3% T-bill yield in cash months.**

## Why nothing wins (concrete)

1. The 48-feature price-only space is saturated. v3 has been searched ~600 times in v4-v7 + 45 in YLOka.
2. The longer-horizon ML target dilutes signal rather than adding information.
3. Cross-sectional rank labels discard magnitude information the production model uses.
4. Apparent "wins" are sample-of-1 to sample-of-3 from outlier years; std of yearly diff ~20pp on the +1pp candidates.

## Next session — what would actually move the needle

Ranked, all within price-only / existing-data scope:

1. **Higher-resolution features from the daily price panel**: 5/10/20-day RSI, ATR-based vol, short-term mean-reversion residuals.
2. **Cross-sectional residualized momentum** (`mom_12_1 - β·SPY_mom`). In features as `idio_mom_12_1` but never tried as a primary scorer.
3. **Time-varying ensemble weights** based on rolling IC of each head.
4. **Conditional regime-specific GBMs** (separate model per crash/recovery/normal/bull regime).
5. **Stacked ensemble with linear blender** (small lasso on top of multiple ranks + regime indicators).

Out of price-only scope (need NEW data):
- OHLC bars → overnight vs intraday (H8).
- Volume → volume thrust + breadth (H9).
- PIT GICS → sector-neutralized momentum (H10).
