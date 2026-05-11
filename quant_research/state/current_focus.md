# Current Focus

**Updated**: 2026-05-11 (Session 1 — Bootstrap)

## This Session Plan

Bootstrap the quant_research/ framework from scratch:
1. ✅ Build clean, validated backtest engine (matches YLOka v3 to within 0.1pp)
2. ✅ Run Phase 2 baseline ladder (12 configs)
3. ✅ Data integrity checks (20/20 pass)
4. ✅ Vol targeting experiment (18 configs) → failed to improve Sharpe
5. ✅ Quick portfolio construction experiments (8 configs) → all fail vs baseline
6. ✅ Populate ideas backlog (22 ideas)
7. ⏳ Write STATE.md, commit

## Current Best

v3 ML signal (K=3, h=6m, tight crash gate):
- CAGR: 40.7% | Sharpe: 0.863 | MaxDD: -49.5% | 248 months (2003-09 → 2024-04)
- Sub-period Sharpes: [0.863, 1.004, 0.930] — all > 0.8, relatively stable

## Next Session Focus

**Priority: I02 — Asymmetric Loss GBM Retraining**
Walk-forward retrain the GBM with asymmetric loss (penalize large losers > large winners).
This is the highest-EV untried approach within the existing data.
Alternatively: **I03 — Meta-labeling secondary model**.

Time budget next session:
- 15 min: Set up LightGBM walk-forward training pipeline
- 35 min: Train 22-fold WF with 3 asymmetry levels (α=2, 5, 10)
- 10 min: Evaluate, journal, update STATE.md

## Key Open Questions

See STATE.md "Questions for V" section.
