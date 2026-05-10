# Session 5 — Regime-specialist GBMs — KILLED

Trained 6 walk-forward GBM heads (3 regimes {bull, normal, recovery} × 2 horizons {3m, 6m}). 22 train windows × 7-month embargo, 62 features (52 v3 + 10 runner-pattern). Each specialist trained on only-its-regime rows.

7 integration variants tested; all UNDERPERFORM v3 baseline (40.78% CAGR) by 7-14pp:

| variant | CAGR | Δ vs base |
|---|---:|---:|
| baseline | 40.78% | 0 |
| specialist_blend_03 | 33.11% | -7.7 |
| specialist_blend_07 | 32.95% | -7.8 |
| specialist_router | 30.53% | -10.3 |
| specialist_blend_05_K5 | 26.88% | -13.9 |

**Why specialists fail**:

1. **Training data fragmentation**. Bull specialist sees 39 months total of training data (vs 268 for the all-data baseline). By Jan-2003, bull-train is 15k rows; baseline is 200k. Less data = noisier GBM.
2. **Regime-label noise**. SPY-features `tight` classifier has hysteresis. Boundary months are misclassified, contaminating each specialist's training set.
3. **Out-of-regime prediction is worse than baseline's in-regime**. Specialists trained on only-r rows extrapolate poorly when feature distribution shifts (which happens regularly at regime transitions).
4. **Specialist MaxDD is consistently WORSE** (-59 to -70%) vs baseline (-50%). Specialists make more concentrated bets on names that work in-regime but blow up when regime shifts.
5. **The signal is mostly regime-invariant**. The all-data GBM's cross-regime generalization is itself the source of edge.

**Don't repeat**: per-regime training fragments the data and removes generalization. Future regime work should keep ONE all-data GBM and instead vary the gate/sizing/regime overlay outside the model. The crash gate is already optimal at fixed thresholds (4 cash months / 268).

**Specialist preds retained on main** for any future stacked-ensemble experiment that wants to use regime-specialist features as INPUT to a higher-level meta-model: `data/YLOka/ml_preds_{3m,6m}_{bull,normal,recovery}.parquet`.
