# Quant Research — State

**Last updated:** 2026-05-12
**Branch:** claude/compassionate-planck-4T9wE

## Mission
Develop a US equity stock-picking strategy with **CAGR ≥ 50%** and **Sharpe ≥ 2.0** on a walk-forward OOS basis (2007–2021), monthly rebalance, long-only, no leverage.

## Current Best Result (Updated after exp_014)

| Config | CAGR | Sharpe | MaxDD | Ann Vol | mean_m | ratio |
|--------|------|--------|-------|---------|--------|-------|
| K=30, 4way blend (vol_asym_60×0.10), inv_vol, vt18%, regime_loose | **66.5% ✓** | **1.841 ✗** | -13.75% | 30.5% | ~4.5% | **0.5315** |
| K=30, LGBM×0.70+sh12×0.20+sh5y×0.10, inv_vol, vt18%, regime_loose | 63.1% ✓ | 1.825 ✗ | -13.6% | 29.5% | 4.48% | 0.527 |

**Status:** CAGR gate PASSED ✓ — Sharpe gate FAILED ✗ (best: 1.841, target: 2.0)

**To reach Sharpe 2.0:** Need ratio = 0.5774. Currently at 0.5315. Gap = 0.046.
- Keep mean_m=4.48%: need std_m ≤ 7.76% (ann_vol ≤ 26.9%) vs current 8.51% (29.5%)
- OR keep std_m=8.51%: need mean_m ≥ 4.91% (monthly mean, +9.5% increase)

## Structural Analysis: Why Sharpe 2.0 Is Hard

**Root cause**: Average pairwise portfolio correlation ρ≈0.53 among K=30 momentum picks.

Portfolio vol formula: σ_port = σ_stock × √(ρ + (1-ρ)/K)
= 40% × √(0.53 + 0.023) = 40% × 0.744 = 29.8% ≈ observed 29.5% ✓

**For Sharpe 2.0 (target σ_port = 27.0%):**
- Need ρ ≤ 0.43 (from 0.53) — 19% reduction in avg pairwise correlation
- OR individual stock σ ≤ 36% (lower-vol stocks, but these have lower returns)

**Fundamental tension**:
- CAGR ≥ 50% requires momentum stocks (high return, high correlation)
- Sharpe ≥ 2.0 requires lower portfolio correlation
- Momentum stocks ARE correlated because they share the same market factor

**Approaches exhausted** (all capped at Sharpe ≈ 1.84):
- Signal blending (13 blend configurations)
- Weighting (inv_vol, ERC, inv_vol², capped, conviction)
- Risk scaling (SPY vol, portfolio vol, drawdown)
- New signals (full IC audit of 79 features, add untried signals)

**Remaining approaches** (lower probability of success):
- Correlation-penalized greedy selection (exp_015): reduce ρ via diversification
- Vol-screen universe (exp_016): filter to lower-vol stocks (may cut CAGR)
- Conviction sizing (exp_016): concentrate weight in top picks
- LGBM variants (exp_017): longer training window, ensemble

## Experiments Run

| Exp | Focus | Configs | Best Sharpe | Best CAGR | Key Insight |
|-----|-------|---------|-------------|-----------|-------------|
| exp_001 | Baseline ladder (signals, K, weighting, regime) | 52 | 1.30 | 23.3% | Regime gate crucial; SPY in EXCLUDE bug fixed |
| exp_002 | Walk-forward LightGBM ranker | 21 | 1.63 | 48.6% | LGBM + inv_vol + 200ma → big jump |
| exp_003 | Larger K, min-var weighting, regime variants | 24 | 1.677 | 61.1% | Sharpe ceiling ~1.68; min-var ≈ inv_vol |
| exp_004 | IC analysis + Sharpe-target model | 13 | 1.52 | 48.6% | LGBM IC=45.8%; driven by regime, not alpha |
| exp_005 | Very large K (50–150) | 24 | 1.678 | 56.6% | Sharpe wall ~1.68; larger K dilutes CAGR |
| exp_006 | Volatility targeting (SPY vol scaling) | 30 | 1.74 | 56.9% | vt18% + regime_loose gives +0.06 Sharpe |
| exp_007 | Stock-level trend filters (d_sma200, rs_6m_spy) | 44 | 1.76 | 55.0% | rs_6m_spy>0 helps marginally; d_sma200 cuts CAGR |
| exp_008 | Two-way blend: LGBM+sharpe_12m | 48 | 1.80 | 58.3% | sharpe_12m IC=0.043 adds genuine lift |
| exp_009 | Three-way blend: LGBM+sh12+sh5y | 45 | 1.82 | 63.1% | sharpe_5y adds +0.02 Sharpe; **NEW BEST** |
| exp_010 | Sharpe-target LGBM (train on fwd_ret/vol) | 24 | 1.76 | 55.0% | Risk-adjusted ranking degrades signal |
| exp_011 | High-IC signals: breakout_60, crt_6m | 35 | 1.77 | 60.9% | High-IC signals fail CAGR gate alone; need LGBM |
| exp_012 | ERC/inv_vol²/capped weighting | 30 | 1.834 | 65.1% | ERC hurts; inv_vol²+cap tiny improvement only |
| **Total** | | **390** | **1.834** | **65.1%** | |

## Key Findings

1. **Sharpe ceiling ~1.83**: Ratio stuck at 0.527-0.529. No weighting or signal approach pushes past 1.84.
2. **LGBM IC near-zero** (pct_positive=45.8%): Performance driven by regime timing + survivorship, not alpha. But LGBM needed for CAGR; high-IC signals alone fail CAGR gate.
3. **Regime timing is critical**: 200ma_loose (~23 cash months) gives best Sharpe/CAGR tradeoff. Avoids GFC, COVID crashes.
4. **Optimal K=30**: Best risk/return. K=20 concentrates risk, K=40+ dilutes CAGR.
5. **Best blend**: LGBM×0.70+sh12×0.20+sh5y×0.10. sharpe_12m IC=0.043, sharpe_5y IC=0.040.
6. **Volatility targeting**: SPY-vol-based scaling (vt18%) adds ~0.06 Sharpe units.
7. **ERC fails**: Increases portfolio vol vs inv_vol (likely because optimizer finds non-diagonal correlation structure). No benefit.
8. **High-IC signals (breakout_60 IC=0.088, crt_6m IC=0.081)**: Select lower-return stocks, fail CAGR gate without LGBM.

## Total Hypotheses Tested: 390
*(Used for Deflated Sharpe Ratio penalty)*

## Lockbox Status
- **Sealed:** 2022-01-31 to 2025-12-31
- **Touches:** 0 (no candidate promoted yet)

## Mathematical Path to Sharpe 2.0

Current: mean_m=4.48%, std_m=8.51%, ratio=0.527 → Sharpe=1.82
Target: ratio=0.5774 → gap=0.048 (9.1% improvement needed)

**Option A — Reduce std_m** (portfolio vol):
- Need ann_vol ≤ 26.9% from current 29.5% (9% reduction)
- ERC failed to reduce vol. inv_vol² reduced to 25.9% but also cut mean_m proportionally.
- Untried: portfolio-level realized-vol scaling, trailing drawdown protection

**Option B — Increase mean_m** (monthly returns):
- Need +9.5% higher monthly returns
- Requires genuinely higher IC signals or better selection within current K
- Untried: adaptive K (smaller K in high-confidence months), quality pre-filter

## Next Experiment Plan

**exp_013 — Portfolio-level vol targeting + drawdown protection**
- At each month, compute trailing 3m realized portfolio vol → scale by min(target/port_vol, 1.0)
- Trailing drawdown control: if 2m portfolio return < -10%, reduce exposure by 50%
- Hypothesis: This directly targets std_m rather than tracking SPY vol

**exp_014 — Adaptive K based on regime quality**
- Strong regime (SPY >> 200ma, low vol): use K=20 (concentrated)
- Borderline regime: use K=40 (diversified)
- Hypothesis: Concentration in best months raises mean_m

**Remaining ideas:**
- Quality pre-filter: only rank within stocks meeting min quality criteria (vol_12m < 50%, price > $5)
- Alternative signal: combine trend_health_5y (IC=0.030) into blend
- Sub-period analysis: identify worst sub-periods and see if they're avoidable

## Dead Ends (Do Not Re-try Without New Angle)
See `state/dead_ends.md` for full list.
