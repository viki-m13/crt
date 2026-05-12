# STATE.md — Autonomous Quant Research Agent

**Updated**: 2026-05-11 (Session 1 — Bootstrap Complete)
**Status**: 🟡 Baseline established, no candidate yet

---

## Headline

**v5 Chronos + v3 GBM is the best known strategy at ~47% WF mean CAGR, Sharpe ~1.06.**
**CAGR gap to target: ~3pp. Sharpe gap to target: ~0.94 (binding constraint).**

---

## Current Best Metrics

| Metric | Value | Source | Window |
|---|---|---|---|
| Best WF mean CAGR | 47.2% | v5_chr_p70_q0.45_k3_invvol | 10 WF splits, 2003-2025 |
| Best full-window CAGR | 43.9% | v5_chr_p70_q0.45_k3_invvol | 2003-2025, 268 months |
| Best Sharpe (full) | 1.063 | v5_chr_p70_q0.45_k3_invvol | 2003-2025 |
| Best WF min CAGR | 23.1% | v5_chr_p70_q0.45_k3_invvol | worst of 10 splits |
| MaxDD | -48.4% | v5_chr_p70_q0.45_k3_invvol | 2003-2025 |
| Target CAGR | ≥ 50% | CLAUDE.md gate | WF OOS ≥ 10 years |
| Target Sharpe | ≥ 2.0 | CLAUDE.md gate | WF OOS annualized |

**Confidence interval**: Single strategy, no bootstrap CI computed yet. Need block bootstrap
(block=6, 1000 iterations) to estimate CI properly.

---

## Baseline Ladder (Session 1, OOS = 2008-09 to 2024-04, N=188 months)

| Rung | CAGR | Sharpe | MaxDD |
|---|---:|---:|---:|
| R1: 12-1 Momentum EW K=5 | 17.3% | 0.762 | -40.0% |
| R2: Momentum + Low-Vol | 8.3% | 0.639 | -23.8% |
| R3: + Trend Health quality | 7.2% | 0.551 | -22.2% |
| R4: + Regime gate | 6.6% | 0.602 | -22.3% |
| R5: OLS cross-sectional | 5.0% | 0.307 | -52.6% |
| R6: v3 GBM K=5 | 31.5% | 0.825 | -40.2% |
| **R6b: v3 GBM K=3** | **45.0%** | **0.914** | **-41.6%** |
| v3 GBM production (ref) | 39.8% | 0.953 | -49.8% |

**Key finding**: R1 (momentum) is the honest academic benchmark. v3 GBM K=3 is 2.7x better
CAGR at the cost of higher volatility. Adding quality/vol filters to momentum HURTS by diluting
the primary momentum signal. The GBM has already internalized all price-based factor information.

---

## Universe & Data

- **Universe chosen**: SPX (S&P 500 via sp500_membership_monthly.parquet, 2003-2026)
  - Rationale: 15+ years OOS data vs NDX which only has PIT membership from 2015
- **Price data**: prices_extended.parquet (1833 tickers, 1995-2026, daily adj close)
- **Feature cache**: pit_panel_full.parquet (47 features, 2003-2026, 268 monthly asofs)
- **Lockbox**: 2024-05-31 to present — SEALED, zero touches

---

## Experiment Log

| Exp ID | Description | Date | CAGR OOS | Sharpe OOS | Status |
|---|---|---|---|---|---|
| exp_000_baseline_ladder | Phase 2 baseline 5 rungs (buggy vol) | 2026-05-11 | 8.7% | 0.455 | archived |
| exp_000_baseline_ladder_v2 | Corrected 6-rung ladder w/ GBM ref | 2026-05-11 | 45.0% (R6b) | 0.914 | ✅ DONE |

---

## Hypotheses Tested (Total)

**11** (5 v1 rungs + 6 v2 rungs). DSR denominator: 11.
(Does NOT include 88+ from prior YLOka sessions, which used a different research structure.)

---

## Current Focus

**Session 2 plan**: Meta-labeling on v3 picks to reduce losing months → boost Sharpe.
See `state/current_focus.md` for full plan.

**Priority**: Idea 01 (Meta-labeling) → Idea 05 (Vol targeting) → Idea 02 (LambdaMART)

---

## Top 3 Next Steps

1. **[IMMEDIATE]** Implement binary meta-model: "Will the K=3 v3 basket beat cash this month?"
   Train LightGBM with purged CV on basket-level features (trailing vol, breadth, score spread).
   This directly targets Sharpe improvement by avoiding losing months.

2. **[SESSION 2-3]** Re-run Chronos inference on full SPX universe to enable Idea 03 (Chronos
   as GBM feature). The existing data only has Chronos predictions for NDX tickers (~20% overlap
   with SPX at any given date). Need full SPX coverage.

3. **[SESSION 3-4]** Implement LambdaMART ranker (ranking objective) as replacement for v3's
   regression objective. Expected to improve top-K capture rate directly.

---

## Dead Ends (Prior Sessions)

See `state/dead_ends.md` for 88+ failed experiments from YLOka sessions 1-5.

Summary: Price-only feature space with cross-sectional GBM regression is saturated. Anything
that adds noise to the v3 signal (additional factors, specialist models, pattern matching,
vertical classifier) degrades performance. The only durable improvement was +5pp from Chronos
filter (a genuinely new zero-shot model signal).

---

## ETA to Success

**Honest estimate: Unknown.** The Sharpe gap (current 1.06 vs target 2.0) is very large for
long-only monthly equity strategies. Historical evidence suggests Sharpe 2.0 is extremely
rare for this strategy type without:
- Fundamentals data (earnings, profitability)
- Higher-frequency signals
- Or a very powerful meta-labeling/regime-timing layer

**Most optimistic path** (if meta-labeling works): 3-4 sessions → CAGR ~48-52%, Sharpe ~1.3-1.5
Still unlikely to reach Sharpe 2.0 without new data.

**Questions for V**:
1. Are fundamentals (PIT earnings/profitability) data sources available?
2. Is the Sharpe 2.0 target firm, or is there flexibility if CAGR is 55%+ and Sharpe 1.5+?
3. Is the broader 1833-ticker universe acceptable (survivorship bias documented)?
4. GPU available for LSTM training?

---

## Lockbox Log

**Lockbox period**: 2024-05-31 to present
**Touches**: 0 of 2 allowed (per family)
**Status**: SEALED ✅
