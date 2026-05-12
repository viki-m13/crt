# Ideas Backlog

Ranked by expected Sharpe improvement × implementation effort. Updated after exp_013.

## Current State
- Best Sharpe: 1.836 (gap = 0.164 to target 2.0)
- Best CAGR: 65.1%
- Ratio: 0.530 (target 0.577, gap = 8.9%)
- Key constraint: mean_m/std_m ratio structurally limited by high within-portfolio correlation (ρ≈0.50)
- Full IC audit done: crt_3m IC=0.135, prerunner_dist IC=0.137, vol_asym_60 IC=0.105 (untried)

## Active/Planned Experiments

### exp_014 (running): New High-IC Signals in Blend
crt_3m, prerunner_dist, vol_asym_60 added to 3-way LGBM blend.

### exp_015 (written): Correlation-Penalized Greedy Selection
Greedy stock selection penalizing correlation with already-selected picks.
Math: ρ=0.50→0.30 reduces portfolio σ from 28.8%→22.7% → Sharpe=4.48%/6.55%×√12=2.37.
Risk: lower-correlation stocks may have lower returns, hurting CAGR.

## Remaining Tier 1 — High Impact

### Conviction-Based Sizing (within-K weight by score magnitude)
Weight within top-K by blend_score × inv_vol (not uniform inv_vol).
Higher-scoring picks get disproportionate weight → higher mean_m.
Expected gain: 0.05-0.15 Sharpe units.

### Vol-Screen Universe (vol_12m < 40%)
Only consider stocks with annual vol < 40% before ranking.
Math: avg σ=40%→33% → portfolio σ≈0.719×33%=23.7% → std_m=6.84%.
With mean_m=4.0%: Sharpe=4.0%/6.84%×√12=2.03 (if CAGR ≥ 50% holds).

### Regime Quality Exposure Scaling
In borderline-pass months (SPY just above 200ma): reduce exposure to 50%.
Expected: cuts std_m without reducing mean proportionally.

### Adaptive K (based on regime quality)
K=15-20 in strong bull regime, K=30-40 in borderline regime.
Concentration when signal is cleaner.

## Previously Explored (completed)
- exp_006: Vol targeting (SPY vol) → Sharpe 1.74 (+0.06 vs baseline)
- exp_007: Trend filters (d_sma200, rs_6m_spy) → Sharpe 1.76
- exp_008: Two-way blend LGBM+sh12 → Sharpe 1.80
- exp_009: Three-way blend LGBM+sh12+sh5y → Sharpe 1.82 (best pure signal)
- exp_010: Sharpe-target LGBM training → Sharpe 1.76 (worse)
- exp_011: High-IC signals (breakout, crt_6m) → fails CAGR gate alone
- exp_012: ERC/inv_vol²/capped weighting → Sharpe 1.834 (marginal)
- exp_013: Portfolio vol targeting + drawdown → Sharpe 1.836 (marginal)

## Dead Ends (see dead_ends.md)
ERC weighting, min-var, Sharpe-target LGBM, high-IC signals replacing LGBM,
EW weighting, very large K (>100), PIT pred column, dual/triple regime.
