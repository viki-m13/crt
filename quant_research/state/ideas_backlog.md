# Ideas Backlog

Updated after exp_014 (last best result).

## Current State
- Best Sharpe: 1.841 (gap = 0.159 to target 2.0)
- Best CAGR: 66.5%
- Ratio: 0.5315 (target 0.577, gap = 8.6%)
- Root cause: avg pairwise portfolio correlation ρ≈0.53 among K=30 momentum picks
- Signal blending ceiling confirmed: no blend combination breaks ratio > 0.532
- Full IC audit done: 79 features tested; untried high-IC = crt_3m, prerunner_dist, vol_asym_60 (added in exp_014)

## Active Experiments (RUNNING/STAGED)

### exp_015 (RUNNING — PID 14946): Correlation-Penalized Greedy Selection
Greedy stock selection penalizing correlation with already-selected picks.
Math: ρ=0.53→0.43 reduces portfolio σ from 29.5%→27% → Sharpe=2.0 target.
Risk: lower-correlation stocks may have lower returns, hurting mean_m and CAGR.
Expected runtime: ~45-60 min (LGBM load ~20 min + backtest ~30 configs).

### exp_016 (STAGED): Conviction-Based Sizing + Vol-Screen + Regime Quality
A) conviction α: weight by (blend_z^α × inv_vol), α=0.5-3.0
B) vol_screen: filter universe to vol_12m < threshold (0.35-0.60)
C) regime quality scaling: reduce exposure in borderline-pass months

### exp_017 (STAGED): LGBM Training Variants (60m/72m window + ensemble)
Hypothesis: Longer training window → more stable LGBM → higher IC.
Ensemble: average 48m + 60m + 72m models for variance reduction.
Expected runtime: ~90 min (3 separate LGBM caches).

### exp_018 (STAGED): Adaptive K by Regime Quality
Strong regime (quality=3): K=15 (concentrate in top picks)
Borderline regime (quality=1): K=40-50 (diversify to reduce variance)
Also tests: conviction weighting in quality-3 months only.

### exp_019 (STAGED): Hierarchical Risk Parity (HRP) Weighting
Unlike ERC (which INCREASED vol), HRP uses hierarchical clustering + bisection.
No matrix inversion. Known to outperform ERC out-of-sample.
Tests: pure HRP and blended HRP+inv_vol at different scales.

## Tier 2 — Lower Priority (try if Tier 1 experiments all fail)

### Regime-Adaptive Signal Weighting
Different blend weights per regime quality tier.
Quality=3: more LGBM weight (momentum signal stronger in bull market).
Quality=1: more vol_asym / sharpe_5y (defensive signals).
Expected gain: small (0.01-0.02 Sharpe), hard to tune without overfitting.

### dist_from_low_1y as Explicit Blend Component (IC=0.096, pct_pos=0.536)
Highest pct_positive of untried signals (53.6% correct direction).
Complements breakout signals — stocks far from 52w low are in confirmed uptrends.
Risk: probably already captured by LGBM. Marginal gain only.

### Two-Stage Model: LGBM Select → Sizing Model
Select top-50 via LGBM rank, then fit separate model (linreg/ridge) on features
to determine sizing weights within the 50. Independent signal path.
Risk: same features → limited added info. High implementation complexity.

## Dead Ends (see dead_ends.md)
ERC weighting (increased vol), min-var weighting, EW weighting,
Sharpe-target LGBM training, high-IC signals replacing LGBM,
very large K (>100 without vol targeting), PIT pred column,
dual/triple regime gates, IC-based feature weighting.

## Mathematical Reality Check

To reach Sharpe 2.0 from ratio=0.5315:
- Need ratio=0.5774 → gap=0.046 (8.6% improvement)
- Option A (raise mean_m): need +8.6% monthly return → requires genuinely better stock selection
- Option B (lower std_m): need -8.6% portfolio vol → requires ρ to drop from 0.53 to 0.43
- All 13 blend configs in exp_014 exhausted signal space: blending ceiling confirmed

Most likely paths to success:
1. exp_015 greedy selection: IF it reduces ρ from 0.53→0.43 WITHOUT killing mean_m
2. exp_016 vol-screen: IF lower-vol stocks still have mean_m ≥ 4.4%
3. Combination of multiple partial improvements from exp_015-019

Honest probability assessment: 20-30% chance any single exp reaches Sharpe 2.0.
Higher chance (~50%) that combining best elements from exp_015-019 reaches 1.95-2.0.
