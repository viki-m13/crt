# Quant Research — State

**Last updated:** 2026-05-11
**Branch:** claude/compassionate-planck-4T9wE

## Mission
Develop a US equity stock-picking strategy with **CAGR ≥ 50%** and **Sharpe ≥ 2.0** on a walk-forward OOS basis (2007–2021), monthly rebalance, long-only, no leverage.

## Current Best Result

| Config | CAGR | Sharpe | MaxDD | Ann Vol | Cash months |
|--------|------|--------|-------|---------|-------------|
| lgbm K=50 inv_vol + regime_200ma | 56.6% ✓ | 1.678 ✗ | -13.2% | 29.5% | 30/179 |
| lgbm K=40 inv_vol + regime_200ma | 61.1% ✓ | 1.677 ✗ | -14.5% | 31.6% | 30/179 |

**Status:** CAGR gate PASSED ✓ — Sharpe gate FAILED ✗ (best: 1.68, target: 2.0)

## Experiments Run

| Exp | Focus | Configs | Best Sharpe | Best CAGR |
|-----|-------|---------|-------------|-----------|
| exp_001 | Baseline ladder (signals, K, weighting, regime) | 52 | 1.30 | 23.3% |
| exp_002 | Walk-forward LightGBM ranker | 21 | 1.63 | 48.6% |
| exp_003 | Sharpe max: larger K, min-var, aggressive regime | 24 | 1.677 | 61.1% |
| exp_004 | IC analysis + Sharpe-target model | 13 | 1.52 | 48.6% |
| exp_005 | Very large K (50–150) + regime | 24 | 1.678 | 56.6% |
| **Total** | | **134** | **1.678** | **61.1%** |

## Key Findings

1. **Sharpe ceiling ~1.68**: No approach has broken through. The CAGR/Sharpe frontier tops out here.
2. **IC near-zero**: LGBM pct_positive=45.8%, composite=42.5%. Performance comes from regime timing + mild survivorship bias, not alpha.
3. **Regime timing works**: 200-day MA gate (30 cash months) reduces MaxDD from ~70% to ~13%. Critical contributor.
4. **Optimal K**: Around K=40-60 for the CAGR+Sharpe Pareto front. Very large K (>80) reduces CAGR faster than it reduces vol.
5. **Min-variance weighting**: Did not improve Sharpe vs inv_vol. Added complexity, similar results.

## Total Hypotheses Tested: 134
*(Used for Deflated Sharpe Ratio penalty)*

## Lockbox Status
- **Sealed:** 2022-01-31 to 2025-12-31
- **Touches:** 0 (no candidate promoted yet)

## Next Steps

1. **exp_006**: Volatility targeting — dynamically scale position size to target 15% ann portfolio vol
2. **exp_007**: Factor timing — rotate between momentum, quality, low-vol factors based on market regime
3. **exp_008**: Large-cap only (S&P 100) — reduce survivorship bias, check if honest Sharpe is still 1.5+
4. **exp_009**: Alternative regime — use VIX-based or credit spread regime instead of price-MA
5. Consider: sub-period stationarity check, DSR validation, PBO simulation

## ETA to Success
Unknown. Current approach shows a Sharpe ceiling of ~1.68. Breaching 2.0 likely requires:
- Genuinely predictive signal (IC > 0.08) currently missing
- Or a different universe/frequency with lower correlation structure
- Or a risk-management layer (vol targeting) that reduces portfolio vol below 20% while maintaining CAGR ≥ 50%
