# Current Focus

**Updated**: 2026-05-11 (Session 1 — Bootstrap)
**Time budget**: Bootstrap complete. Next session = 1 hour.

---

## This session: Bootstrap complete

Completed:
- Created quant_research/ directory structure
- Located and verified PIT universe and price data
- Ran Phase 2 baseline ladder (6 rungs + K variants)
- Analyzed gap to target
- Populated ideas backlog with 22 ranked ideas
- Wrote STATE.md and committed

---

## Next session plan (Session 2)

**Primary task**: Implement Idea 01 — Volatility-Targeted Meta-Labeling

The Sharpe gap (current ~1.06 vs target 2.0) is the binding constraint. The fastest path to
higher Sharpe is reducing the number of bad months (the v3 strategy has some terrible months
when concentrated picks all fall simultaneously).

### Hour budget:
- 00-10 min: Resume from STATE.md, read journal
- 10-30 min: Implement meta-labeling binary classifier (predict "basket beats cash?")
  - Features: trailing 3m portfolio vol, market breadth, v3 score spread, Chronos confidence
  - Model: LightGBM binary with purged CV
  - Target: basket return > 0.25% (=3% annual cash yield / 12)
- 30-50 min: Walk-forward backtest with meta-label filter
  - Compare: v3 only, v3 + meta-filter, v3 + Chronos, v3 + both
- 50-60 min: Journal, hypotheses log, STATE.md update, commit

### Secondary task (if time allows):
- Implement Idea 05 — Volatility targeting (adjust K to hit 15% annual vol target)
- This is a portfolio construction change, not a signal change

### Key question to answer:
Does a meta-model trained on monthly "basket success" have IC > 0.1 on OOS?
If yes, proceed with full validation. If no, log to dead_ends.md.

---

## Experiment naming convention (this project)

exp_001_meta_labeling — Session 2 primary
exp_002_vol_targeting — Session 2 secondary
exp_003_lambdamart_ranker — Session 3
exp_004_chronos_feature_gen — Session 3 (if Chronos re-inference done)
exp_005_broader_universe — Session 4

---

## Watchpoints

- The 30% CAGR v5 Chronos mentioned as baseline in CLAUDE.md is the YLOka v5 result under
  strict walk-forward. Our implementation shows ~45-47% WF mean, which is better.
- The Sharpe gap is NOT about CAGR but about vol reduction. Look at approaches that reduce
  monthly return variance, not just increase mean.
- Suspicious if any model shows Sharpe > 3.0 — trigger leakage re-audit immediately.
