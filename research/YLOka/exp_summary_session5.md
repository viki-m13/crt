# Session 5 — Regime-specialist GBMs

**Branch**: `claude/rebuild-stock-selection-YLOka`
**Window**: research only (2003-09 → 2024-04, 248 months). Holdout (2024-05 → 2026-04) NOT touched.
**Baseline**: v3 `ml_3plus6` ew K=3 h=6 → CAGR **40.78%**, Sharpe **0.953**, MaxDD **-49.83%**.

## Hypothesis

A single GBM trained on all-regime data learns the AVERAGE predictive pattern. Different regimes (bull/normal/recovery) reward different signals. Train a specialist GBM per regime; route at test time.

## Implementation

Trained 6 walk-forward GBM heads (3 regimes × 2 horizons):
- For each regime r ∈ {bull, normal, recovery}:
  - For each horizon h ∈ {3m, 6m}:
    - 22 walk-forward train windows (Jan refit, expanding window, **7-month embargo**).
    - Training data restricted to rows where regime(asof) == r at training time.
    - 62 features (52 v3 baseline + 10 runner-pattern adds: `idio_mom_12_1`, `fip_score`, `tight_consolidation_60`, `breakout_strength_60`, `min_dd_60d`, `rsi_zone_score`, `acceleration_2y`, `mom_per_unit_vol_12`, `crt_3m`, `crt_6m`).
    - Predict for ALL test asofs (not just same-regime ones) so we can ensemble.

Output: 6 prediction parquets in `data/YLOka/ml_preds_{3m,6m}_{bull,normal,recovery}.parquet`.

## Regime distribution across 268 asofs (2003-01 → 2026-04)

| regime | count | share |
|---|---:|---:|
| recovery | 157 | 58.6% |
| normal | 60 | 22.4% |
| bull | 39 | 14.6% |
| crash | 12 | 4.5% |

Training-data sparsity by Jan-2003 first cutoff:
- recovery specialist: ~120 train rows × ~500 names = 60k rows by 2010, growing.
- bull specialist: 39 months total → ~30 train rows × ~500 names = 15k rows by 2010, smaller.
- crash specialist: SKIPPED (production strategy goes to cash).

## Results — all underperform baseline

| variant | CAGR | Sharpe | MaxDD | Δ vs base |
|---|---:|---:|---:|---:|
| **baseline** ml_3plus6 | **40.78%** | **0.953** | **-49.83%** | 0 |
| specialist_blend_03 (30% specialist) | 33.11% | 0.853 | -59.4% | -7.7 |
| specialist_blend_07 (70% specialist) | 32.95% | 0.937 | -65.8% | -7.8 |
| specialist_blend_05_cy (50% specialist + 3% cash yield) | 31.85% | 0.866 | -65.5% | -8.9 |
| specialist_blend_05 (50% specialist) | 31.78% | 0.865 | -65.8% | -9.0 |
| specialist_router (pure specialist) | 30.53% | 0.892 | -69.7% | -10.3 |
| specialist_rank_avg (rank ensemble) | 30.49% | 0.877 | -65.8% | -10.3 |
| specialist_blend_05_K5 (specialist + K=5) | 26.88% | 0.861 | -53.4% | -13.9 |

**Specialist GBMs UNDERPERFORM by 7-14pp CAGR with materially worse MaxDD across all integration variants.**

## Why specialists fail

1. **Training-data fragmentation**: bull specialist sees only 39 months of training data total over 22 years. By the first Jan-2003 cutoff, bull-specialist training set is ~15k rows; production v3 sees ~200k rows. Smaller training data = noisier GBM.
2. **Regime-label noise**: the `tight` SPY-features classifier has hysteresis. Months at regime boundaries get misclassified, mixing regime signals.
3. **Out-of-regime prediction is worse than in-regime baseline**: when the test month is in regime r, the specialist for r is queried — but specialists are trained on only-r rows, so they extrapolate poorly when feature distribution shifts (which happens regularly at regime transitions).
4. **The signal is mostly regime-invariant**: production v3's 3m+6m predictions implicitly handle regime via the 6m/12m momentum features and the regime gate. Forcing per-regime training removes the cross-regime generalization that the all-data GBM relies on.
5. **Specialist's MaxDD is consistently WORSE** (-59 to -70%) vs baseline (-50%) — the specialist makes more concentrated bets on names that work IN-regime but blow up when the regime shifts.

## Final disposition

**88+ experiments across 5 sessions. v3 is conclusively the local optimum** in:
- price-only feature space (no fundamentals, no options, no volume granularity, no PIT GICS).
- cached-prediction space (re-using `ml_preds_v2.parquet` 1m/3m/6m heads).
- single-GBM-architecture space (no deep learning, no sequence models).

| Hypothesis class | Sessions | Variants | Best | Verdict |
|---|---|---:|---|---|
| Conviction sizing / soft-cash / accel / Donchian | S1 | 19 | +1.6pp (sample-of-2) | KILL |
| Multi-target ensemble (12m + cls heads) | S2 | 12 | -0.05pp (no-op) | KILL |
| Dispersion-conditional K | S2 | 7 | +1pp (sample-of-3) | KILL |
| Feature-based scorers (idio, runner, fip, etc.) | S3 | 11 | -13.6pp | KILL |
| Adaptive IC / dynamic-hold / hold-period | S4 | 12 | -5.7pp (Sharpe +0.015 only) | KILL |
| Regime-conditional K | S4 | 4 | +2.3pp (sample-of-2) | KILL |
| Regime-specialist GBMs | S5 | 7 | -7.7pp | KILL |
| K/h grid | S1+S4 | 14 | K=3, h=6 locally optimal | confirms |
| **Cash yield 3% in cash months** | S1 | 1 | **+0.07pp** | **KEEP** |

**The only durable improvement is +0.07 pp from a 3% T-bill yield in cash months.** A one-line change to v3.

## What WOULD move the needle (for future sessions)

Out of price-only / existing-data scope (per user direction):
- **OHLC bars** → overnight vs intraday return decomposition.
- **Daily volume** → volume thrust, breadth confirmation, OBV regime indicators.
- **PIT GICS sectors** → properly sector-neutralized residual momentum (current `idio_mom_12_1` is residualized vs SPY only).
- **Options data** → IV skew, put-call ratio, term structure.
- **Short interest** → days-to-cover, change in SI, squeeze proxies.
- **Earnings calendars** → post-earnings drift quality filters, earnings-week avoidance.

Within scope but unexplored (heavy training cost):
- **Deep learning (LSTM / Transformer)** on sequences of feature rows. Would need PyTorch and ~30 min training cost per fold.
- **GBM with asymmetric loss** (penalize false positives on extreme losers more than false negatives on moderate winners) — could improve top-K capture under MaxDD constraint.
- **GBM trained on alternative target**: probability of >+30% in 6m (right-tail asymmetric); cumulative-rank-of-rank.
- **Stacked ensemble with linear blender** trained on per-asof rolling-window of (rank_p3m, rank_p6m, rank_p12m, rank_cls, regime indicators).

## Reusable artifacts retained on main

- `data/YLOka/ml_preds_{3m,6m}_{bull,normal,recovery}.parquet` — 6 specialist heads, ~3.7 MB each. **Reusable** for any future stacked-ensemble experiment without retraining.
- `data/YLOka/regime_labels.parquet` — per-asof regime labels.
- `data/YLOka/pit_panel_full.parquet` — PIT × all heads × features.
- `data/YLOka/rolling_ic.parquet` — per-asof 24m rolling IC.
- `data/YLOka/ml_preds_12m{,_cls}.parquet` — Session 2 12m heads.
- `data/YLOka/xs_dispersion.parquet` — XS momentum dispersion.

## Files

- `strategy/YLOka/train_regime_specialists.py` — walk-forward training pipeline.
- `strategy/YLOka/run_session5_experiments.py` — Session 5 driver.
- `strategy/YLOka/harness.py` — extended with 5 specialist scorers + `load_panel_specialist`.
- `backtests/YLOka/runs/` — 8 new manifests + equity curves.
