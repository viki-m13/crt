# Sessions 3-4 — feature-based scorers + adaptive ensemble + regime-K

**Branch**: `claude/rebuild-stock-selection-YLOka`
**Window**: research only (2003-09 → 2024-04, 248 months). Holdout (2024-05 → 2026-04) NOT touched.
**Baseline**: v3 `ml_3plus6` ew K=3 h=6 → CAGR **40.78%**, Sharpe **0.953**, MaxDD **-49.83%**.

## TL;DR

- **Session 3 (12 feature-based scorers)**: pure `idio_mom_12_1`, runner-footprint composite (multibagger × accel × breakout × idio_mom × fip), runner-gated ML, ML+runner additive blends, fip-gated ML, ML+cst, ML+breakout, ML+multibagger. **All UNDERPERFORM** the v3 baseline by 13-36 pp CAGR.
- **Session 4 (10 adaptive / regime-K variants)**: rolling-IC adaptive ensemble, IC-proportional weights, IC filter, dynamic-hold (rebalance early when picks fall below score quantile), monthly score check, K=1 / K=2 with longer holds, regime-conditional K (K=2 in bull / K=3 in others, etc.). **All UNDERPERFORM durably** — best apparent winner (`exp_92` K=2-bull/K=3-others +2.3pp CAGR) is the SAME K-shrinkage artifact: 6/22 years beat baseline, std of yearly diff 18pp, 2020 alone drives +73pp.
- **Aggregate across the 4 sessions: 80+ experiments tried; v3 remains the local optimum.**

## Session 3 — feature-based scorers

### Discovery: production v3 already uses the runner-pattern features

`experiments/monthly_dca/v2/ml_strategy.py:177-179` auto-detects features from the panel:
```python
feature_cols_raw = [c for c in big.columns
                    if c not in ("asof", "ticker") and not c.startswith("fwd_")
                    and not c.startswith("rank_target_")]
```
The panel includes all 79 cached features per asof, including `multibagger_ratio_24m`, `fip_score`, `breakout_strength_60`, `idio_mom_12_1`, `cst_score`, `tight_consolidation_60`, `prerunner_dist`, `crt_3m`, `crt_6m`, `rbi_60`, `rbi_120`, `vol_asym_*`. Production v3's GBM has direct access to all of them and learns their non-linear interactions.

This explains why hand-designed linear composites of these features cannot beat v3: **v3's GBM already extracts the signal these features carry, weighted optimally for the prediction target. Linear composites are a strictly weaker representation.**

### 12 scorer variants tested

| variant | CAGR | Sharpe | MaxDD | Δ CAGR vs base |
|---|---:|---:|---:|---:|
| **baseline** ml_3plus6 | **40.78%** | **0.953** | **-49.83%** | **0** |
| ml_plus_cst (+0.15 cst rank) | 27.22% | 0.719 | -50.4% | -13.6 |
| ml_plus_runner_weak (w=0.10) | 25.35% | 0.773 | -74.6% | -15.4 |
| ml_plus_multibagger (+0.15 rank) | 25.32% | 0.779 | -75.0% | -15.5 |
| runner_gated_ml (top-25% by RF) | 22.66% | 0.919 | -40.3% | -18.1 |
| ml_plus_runner (w=0.20) | 19.93% | 0.731 | -63.0% | -20.9 |
| fip_gate_ml | 19.82% | 0.717 | -66.4% | -21.0 |
| ml_plus_runner_strong (w=0.40) | 16.43% | 0.676 | -63.8% | -24.3 |
| ml_plus_breakout | 15.96% | 0.703 | -57.1% | -24.8 |
| idio_plus_ml (50/50 ranks) | 12.96% | 0.573 | -54.8% | -27.8 |
| **idio_mom_12_1 alone** | 8.16% | 0.404 | -83.7% | -32.6 |
| **runner_footprint composite** | 4.59% | 0.301 | -61.0% | -36.2 |

**Pattern**: the more weight given to the linear feature composites, the worse the result. `idio_mom` alone (8.16%) and the full runner_footprint composite (4.59%) are catastrophic — these features have low standalone IC for top-K selection and high cross-sectional correlation with already-priced names.

`runner_gated_ml` (use baseline ML score among top-25% of universe by runner footprint) is the best Sharpe (0.92, vs baseline 0.95) and best MaxDD (-40% vs -50%) — but at -18pp CAGR. Trade-off lands on the wrong side of the user's CAGR-max objective.

## Session 4 — adaptive ensemble + dynamic hold + regime-K

### Rolling 24-month IC by head (full research period)

| head | mean IC | std | min | max |
|---|---:|---:|---:|---:|
| pred_1m | 0.045 | 0.097 | -0.21 | 0.43 |
| pred_3m | 0.040 | 0.085 | -0.23 | 0.30 |
| pred_6m | 0.062 | 0.082 | -0.14 | 0.42 |
| pred_12m | **0.099** | 0.093 | -0.13 | 0.43 |

**Surprise**: `pred_12m` has the HIGHEST mean IC, yet H1 ensembles using it consistently underperformed. Resolution: **IC measures the rank correlation across the entire cross-section, while top-K capture is dominated by the extreme top of the score distribution**. Magnitudes of `pred_3m + pred_6m` produce sharper separation among the top 3 names than the rank ensemble can.

### 10 adaptive / regime variants tested

| variant | CAGR | Sharpe | MaxDD | Δ CAGR |
|---|---:|---:|---:|---:|
| **baseline** | **40.78%** | **0.953** | **-49.83%** | **0** |
| ml_3plus6_ic_filter (shrink in low-IC months) | 40.78% | 0.953 | -49.83% | 0 (no-op) |
| K_bull2_others3 (K=2 bull / K=3 others) | **43.07%** | **0.979** | -49.83% | **+2.3 (artifact)** |
| K_bull1_others3 | 40.40% | 0.928 | -52.5% | -0.4 |
| K_bull5_others3 | 39.77% | 0.943 | -49.83% | -1.0 |
| K_bull1_recovery2_normal3 | 35.99% | 0.831 | -57.0% | -4.8 |
| hold12 | 35.07% | 0.968 | -58.8% | -5.7 |
| K_norm1_bull3 | 34.90% | 0.726 | -72.6% | -5.9 |
| adaptive_ic_prop | 34.77% | 0.902 | -53.2% | -6.0 |
| adaptive_dyn (adaptive_ic + dyn hold) | 34.83% | 0.917 | -50.8% | -6.0 |
| dyn_hold_q90 | 34.23% | 0.904 | -55.5% | -6.6 |
| dyn_hold_q85 | 33.24% | 0.859 | -49.83% | -7.5 |
| K2_h12 | 33.11% | 0.854 | -72.5% | -7.7 |
| K5_bull2 | 32.98% | 0.882 | -59.1% | -7.8 |
| adaptive_ic | 32.60% | 0.859 | -53.2% | -8.2 |
| hold9 | 32.15% | 0.810 | -76.7% | -8.6 |
| dyn_hold_q75 | 31.84% | 0.830 | -64.4% | -8.9 |
| hold3 | 31.54% | 0.834 | -56.7% | -9.2 |
| K1 | 28.61% | 0.618 | -80.4% | -12.2 |
| K2_h12 | 33.11% | 0.854 | -72.5% | -7.7 |
| hold4 | 21.70% | 0.634 | -90.8% | -19.1 |
| K1_h12 | 19.16% | 0.588 | -86.5% | -21.6 |

### `exp_92` (K=2 bull / K=3 others) — verified as artifact

Surface result: +2.3 pp CAGR, +0.025 Sharpe, same MaxDD. Year-by-year diagnosis:

| year | base | exp_92 | diff (pp) | n_bull |
|---|---:|---:|---:|---:|
| 2004 | 27.7% | 57.2% | **+29.4** | 2 |
| 2017 | 44.8% | 49.1% | +4.3 | 2 |
| **2020** | **109.6%** | **182.9%** | **+73.3** | 3 |
| 2022 | 22.8% | 26.1% | +3.3 | 1 |
| 2013 | 80.1% | 56.5% | -23.6 | 2 |
| 2018 | -18.4% | -18.1% | +0.3 | 3 |
| 2021 | 65.8% | 50.6% | -15.2 | 3 |
| (15 other years) | | | 0.0 | varies |

- 6/22 years beat baseline.
- Median yearly diff: **0.0 pp**.
- Std of yearly diff: **18.0 pp**.
- 2020 alone (+73pp) drives nearly the entire +2.3pp full-period lift. 2020 was a single bull-regime fire that hit COVID rebound names in K=2 concentration; replicate-via-luck result.

**Same K-shrinkage trick** as H6 Donchian (Session 1) and H7 disp_K23 (Session 2). Three different conditional rules for "when to drop K to 2", three different "winners", all driven by 1-3 outlier years. The mechanism is portfolio concentration accidentally aligning with COVID-bounce / NVDA-class winners.

### `hold12` — best Sharpe but worse CAGR/MaxDD

`hold12` (12-month hold instead of 6) gives Sharpe **0.968** (vs baseline 0.953) at CAGR -5.7pp, MaxDD -58.8%. Sharpe up because turnover-related noise drops; CAGR down because the model's predictions decay over 6→12 months. Not a useful trade-off given CAGR-max objective.

## What this means

**v3 is the local optimum in the price-only / cached-prediction space.** Across 4 sessions and 80+ experiments, every tractable modification — ensembling, feature blending, regime conditioning, hold-period changes, K changes, dynamic exits, IC adaptive weighting — lands strictly worse than v3 OR is a sample-of-1-3 K-shrinkage artifact.

The remaining unexplored directions (without bringing in new data) all require non-trivial GBM retraining:

1. **Train a GBM on a different target**: e.g., probability of >+30% return in 6m (right-tail asymmetric); probability of <-15% drawdown in 6m (left-tail filter); cumulative-rank-of-rank target ("predict where this stock will rank 3 months from now"). Heavy: ~5-10 min per head + ensemble tuning.
2. **Train a regime-specific GBM**: separate models for bull/normal/recovery/crash, switch by gate. Heaviest: 4× training cost.
3. **Stacked ensemble with a learned linear blender**: train a small lasso on top of (rank_p3m, rank_p6m, rank_p12m, rank_cls, regime_indicators) per asof, with rolling-window training. Medium cost.
4. **Higher-resolution daily-bar features**: 5/10/20-day RSI, ATR-based vol, Parkinson if OHLC ever becomes available, short-MR residuals. Feature-engineering work.

What WOULD likely move the needle (out of price-only scope):
- OHLC bars → overnight vs intraday return decomposition (literature shows S&P500 winners' overnight return dominates).
- Volume → volume thrust + breadth confirmation features.
- PIT GICS sector tags → properly sector-neutralized residual momentum (current `idio_mom_12_1` is residualized vs SPY, not sector).

## Reusable artifacts retained for future sessions

- `data/YLOka/pit_panel_full.parquet` — 98k rows × 47 cols (PIT × all 4 prediction heads × 40 features). Loads in <1s. **Use this as the input panel for any future scorer experiment**.
- `data/YLOka/rolling_ic.parquet` — per-asof rolling 24m IC of each head. Useful for any IC-conditional logic.
- `data/YLOka/ml_preds_12m.parquet`, `data/YLOka/ml_preds_12m_cls.parquet` — Session 2 trained heads, reusable.

## Files

- `strategy/YLOka/harness.py` — extended with 14 new scorers + regime-K + dynamic-hold + adaptive-IC.
- `backtests/YLOka/runs/` — 30+ new manifests + equity curves for Session 3-4 experiments.
- `backtests/YLOka/experiment_log.csv` — append-only log.
