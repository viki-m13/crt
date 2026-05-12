# Experiment 000 — Phase 2 Baseline Ladder v2

**Date**: 2026-05-11 21:23 UTC
**Universe**: SPX PIT (sp500_membership_monthly.parquet, 2003-01 to 2024-04)
**Features**: pit_panel_full.parquet (YLOka pre-computed, validated PIT)
**OOS window**: 2008-09-30 to 2024-04-30
**K**: 5 (also K=3 for R6b), **Cost**: 5.0 bps one-way

## Results (OOS)

| Rung | CAGR | Sharpe | MaxDD | N |
|---|---:|---:|---:|---:|
| R1_mom12_1 | 17.3% | 0.762 | -40.0% | 188 |
| R2_mom_lovol | 8.3% | 0.639 | -23.8% | 188 |
| R3_quality | 7.2% | 0.551 | -22.2% | 188 |
| R4_regime | 6.6% | 0.602 | -22.3% | 188 |
| R5_ols | 5.0% | 0.307 | -52.6% | 188 |
| R6_gbm_v3 | 31.5% | 0.825 | -40.2% | 188 |
| R6b_gbm_v3_K3 | 45.0% | 0.914 | -41.6% | 188 |
| v3_GBM_K3_prod (reference) | 39.77% | 0.953 | -49.83% | 248 |

**Best OOS Sharpe**: R6b_gbm_v3_K3 — CAGR 45.0%, Sharpe 0.914

## Gap Analysis

The success gate requires CAGR ≥ 50% and Sharpe ≥ 2.0 on walk-forward OOS.
The v3 GBM is the price-only ceiling at ~40% CAGR / Sharpe ~0.95 after 88 experiments.
The baseline ladder confirms this ceiling: momentum + quality + regime factors produce
significantly lower performance than a properly trained GBM on the same features.

## What's needed

1. **New data**: fundamentals (earnings, profitability, valuation) would provide the
   largest marginal information gain. Volume-based features are a close second.
2. **Better ML**: LSTM/Transformer on feature sequences could capture non-linear
   temporal patterns that a monthly cross-sectional GBM cannot.
3. **Alternative targets**: train on probability of >50% gain in 12m rather than
   raw return (right-tail asymmetric objective).
4. **Meta-labeling**: use v3 as primary selector, train a secondary classifier to
   take/skip each v3 pick based on contextual signals.

## Elapsed

24.3s
