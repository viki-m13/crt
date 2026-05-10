# Session 2 — H1 (multi-target ensemble) + H7 (dispersion-conditional K)

**Branch**: `claude/rebuild-stock-selection-YLOka`
**Window**: research only (2003-09 → 2024-04, 248 months). Holdout (2024-05 → 2026-04) NOT touched.
**Baseline**: v3 `ml_3plus6` ew K=3 h=6 → CAGR **40.78%**, Sharpe **0.953**, MaxDD **-49.83%**.

## TL;DR

- **H1 multi-target ensemble** (added a walk-forward 12m-rank GBM head AND a top-quintile classifier head to v3's existing 1m/3m/6m heads): **all 12 integration variants UNDERPERFORM the baseline** by 0.05 to 9.1 pp CAGR.
- **H7 dispersion-conditional K**: 7 variants tested; only one (K=2 in wide dispersion, K=3 in narrow) gave +1.0 pp CAGR — but it's the same K-shrinkage trick as H6: 3 outlier years (2004 +29pp, 2009 +74pp, 2016 +28pp) drive the entire lift. **Not durable.**
- **Aggregate**: 45+ experiments run across this branch; **nothing cleanly beats v3** in the price-only feature space using cached GBM predictions. The remaining unexplored axes need NEW data (OHLC bars, volume, sector tags) which the user said are out of scope.

## H1 — Multi-target ensemble

### What was added

Trained two new walk-forward GBM heads to complement v3's existing 1m/3m/6m heads:
- **`pred_12m`** (regressor): cross-sectional rank of 12-month forward returns. 22 walk-forward train windows (Jan refit, expanding window, **13-month embargo**). 339,644 OOS predictions over 2003-01 → 2026-05.
- **`pred_12m_cls`** (classifier): probability of being top-quintile over the next 12 months. Same training schedule.

Both heads use the same 52 features as the production v3 model (`HistGradientBoostingRegressor` / `HistGradientBoostingClassifier` with 200 trees, depth 4, LR 0.05). Pipeline: `strategy/YLOka/train_12m_head.py`. Outputs: `data/YLOka/ml_preds_12m{,_cls}.parquet`.

### Integration variants tested

| variant | scorer | CAGR | Sharpe | MaxDD | Δ CAGR vs base |
|---|---|---:|---:|---:|---:|
| **baseline** | `(p3+p6)/2` raw | **40.78%** | **0.953** | **-49.83%** | **0** |
| ens_3_6_12 ew | `mean(rank(p3, p6, p12))` | 31.64% | 0.819 | -49.23% | **-9.1** |
| ens_3_6_12_cls | `0.7*ens + 0.3*rank(cls)` | 35.65% | 0.924 | -47.83% | **-5.1** |
| ens_3_6_12_invvol | `ens - 0.10*rank(vol)` | 31.64% | 0.819 | -49.23% | -9.1 |
| ens_36_12wt | `0.2*r3 + 0.4*r6 + 0.4*r12` | 33.40% | 0.846 | -49.23% | -7.4 |
| ens_3_6_12_cy | `ens + 3% cash yield` | 31.70% | 0.820 | -48.86% | -9.1 |
| ens_3_6_12_K5 | `ens, K=5` | 30.99% | 0.911 | -43.94% | -9.8 |
| ens_3_6_12_K2 | `ens, K=2` | 37.93% | 0.814 | -69.07% | -2.9 |
| **tilt_005** | `(p3+p6)/2 + 0.05·(rank(p12)-0.5)` | 35.69% | 0.872 | -49.83% | **-5.1** |
| tilt_015 | `(p3+p6)/2 + 0.15·(rank(p12)-0.5)` | 35.27% | 0.888 | -53.27% | -5.5 |
| **cls_filter_018** | `(p3+p6)/2 with cls < 0.18 → -∞` | 40.73% | 0.950 | -49.83% | **-0.05** |
| cls_filter_025 | `(p3+p6)/2 with cls < 0.25 → -∞` | 37.81% | 0.900 | -49.83% | -3.0 |
| cls_tilt | `(p3+p6)/2 + 0.05·(cls - 0.2)` | 39.73% | 0.921 | -49.83% | -1.1 |

**Best H1 result**: `cls_filter_018` is essentially a no-op (-0.05pp CAGR, identical Sharpe/MaxDD). Every other H1 variant is meaningfully worse.

### Why H1 failed

1. **Magnitude vs rank loss of information**: switching from raw `(pred_3m + pred_6m)/2` magnitudes to cross-sectional ranks erases the model's confidence calibration. The strong-conviction stocks lose their signal advantage when everything is rank-flattened to [0, 1].
2. **Longer horizon = lower IC**: the 12m head's effective predictive accuracy is materially lower than 3m/6m. Averaging it in dilutes the signal.
3. **The classifier head's information is mostly redundant** with the regressor heads. Cls filter at 0.18 is essentially a no-op because almost everything passes; tighter thresholds drop CAGR with no Sharpe improvement.
4. **Additive tilts** (tilt_005, tilt_015) confuse the score by mixing magnitude-scaled `pred_3m+pred_6m` with rank-scaled `pred_12m` — the rank-tilt dominates because it's the same scale across all months while the raw predictions vary in absolute magnitude.

This is **consistent with the v6 finding** (per `experiments/monthly_dca/v6/REPORT.md`) that "proprietary features GBM substantially worse than v3 baseline". Adding more model heads doesn't help.

## H7 — Dispersion-conditional K

### What was added

Computed per-asof XS dispersion of `mom_12_1` across the universe at every month-end (cached in `data/YLOka/xs_dispersion.parquet`). Hypothesis: when dispersion is wide, the model's top picks are most differentiated → trust them and concentrate (small K). When dispersion is narrow (everything moves together), picks are interchangeable → diversify (large K).

### Variants tested

| variant | K_low (narrow) | K_high (wide) | threshold | CAGR | Sharpe | MaxDD | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **baseline** | — | — | — | **40.78%** | **0.953** | **-49.83%** | 0 |
| disp_K35_p50 | 5 | 3 | p50 | 31.53% | 0.842 | -59.09% | -9.3 |
| disp_K310_p50 | 10 | 3 | p50 | 29.19% | 0.863 | -51.29% | -11.6 |
| disp_inv_K53_p50 | 3 | 5 | p50 | 39.05% | **0.956** | -49.83% | -1.7 |
| disp_K15_p50 | 5 | 1 | p50 | 30.33% | 0.707 | -60.40% | -10.5 |
| **disp_K23_p50** | 3 | 2 | p50 | **41.77%** | 0.932 | -49.83% | **+1.0** |
| disp_K35_p33 | 5 | 3 | p33 | 31.80% | 0.842 | -59.09% | -9.0 |
| disp_K35_p67 | 5 | 3 | p67 | 31.22% | 0.862 | -59.09% | -9.6 |

### Why even the +1pp lift isn't durable

`exp_43_disp_K23_p50` (K=2 wide / K=3 narrow) gives +1.0pp CAGR. Yearly diff vs baseline:

| year | base | exp_43 | diff (pp) |
|---|---:|---:|---:|
| 2003 | 8.0% | 8.3% | +0.2 |
| 2004 | 27.7% | 57.2% | **+29.4** |
| 2005 | 21.3% | 19.4% | -1.9 |
| 2006 | 26.4% | 20.0% | -6.4 |
| 2007 | 9.0% | 4.6% | -4.4 |
| 2008 | -17.5% | -17.5% | 0.0 |
| 2009 | 625.5% | 699.6% | **+74.1** |
| 2010 | 52.0% | 57.2% | +5.2 |
| 2013 | 80.1% | 56.4% | -23.7 |
| 2016 | 35.5% | 64.0% | **+28.5** |
| 2017 | 44.8% | 38.6% | -6.2 |
| 2020 | 109.6% | 105.5% | -4.2 |
| 2021 | 65.8% | 42.9% | -22.9 |

8/22 years beat baseline; median yearly diff 0.0 pp; std of yearly diff **19.7 pp**. The +1 pp lift comes from 3 outlier years (2004, 2009, 2016) that more than compensate for losses in 2013, 2017, 2021. This is the **same K-shrinkage trick as H6 Donchian**: when K=2 fires, that's portfolio concentration disguised as a regime signal. avg n_picks 2.56 vs baseline 2.95 — when wide-dispersion months coincide with K=2, you get accidental NVDA-style single-name concentration.

The inverted variant `disp_inv_K53_p50` (K=3 narrow, K=5 wide) gives marginally higher Sharpe (0.956 vs 0.953) at -1.7pp CAGR — DIVERSIFYING in wide-dispersion is mildly Sharpe-positive but CAGR-negative. Same kill verdict.

## Aggregate session diagnostic

Across 45+ experiments on this branch (19 in Session 1 cheap sweep + 12 H1 ensemble + 7+ H7 dispersion + a few setup runs), **the v3 baseline is not beaten cleanly on any dimension**:
- Highest CAGR among "real" candidates (excluding K-shrinkage trick): 40.85% (exp_08 cash-yield variant) — +0.07 pp.
- Highest Sharpe: 0.964 (exp_04 H3 accel) — sample-of-2.
- Lowest MaxDD: -43.9% (exp_25 ens_3_6_12_K5) — but CAGR -10pp.

## Why nothing wins (concrete diagnosis)

1. **The 48-feature price-only space is saturated**. v3 has been searched (~600 v4-v7 variants + 45 here = ~650). Random search through the same data hits the same local optimum.
2. **Adding longer-horizon ML targets dilutes the signal**, doesn't add information. The 1y forward return is dominated by macro/regime (which the model can't predict from per-stock features) rather than by per-stock signal.
3. **Cross-sectional rank label discards magnitude information** the production model uses. The ranking framework was a deliberate choice in v3 to avoid magnitude noise — but it caps how much information can be extracted.
4. **Apparent "wins" are sample-of-1 to sample-of-3** driven by single outlier years (2004, 2009, 2016, 2020, 2022). With 22 years of data and 45+ tests, finding 3-year wins is what random search produces.

## What would actually move the needle (next session)

Ranked by likely lift × feasibility, *given that price-only data is the hard constraint*:

1. **Higher-resolution features** from the daily price panel: 5/10/20-day RSI, ATR-based vol, Parkinson vol from H-L (would need OHLC, currently unavailable), short-term mean-reversion residuals.
2. **Cross-sectional residualized momentum**: `mom_12_1 - β·SPY_mom` per ticker. Already in features as `idio_mom_12_1` but not tried as a primary scorer in YLOka.
3. **Time-varying ensemble weights**: instead of constant `0.5·p3 + 0.5·p6`, use `IC_24m(p3) / Σ IC` weights. Adapts to which horizon is currently most predictive.
4. **Conditional regime-specific models**: train a SEPARATE GBM for crash / recovery / normal / bull regimes; pick the model's predictions per the active regime. Real complexity but potentially the biggest lift.
5. **Stacked ensemble with linear blender**: train a small lasso on top of (rank(p3), rank(p6), rank(p12), rank(cls), regime indicators) → final score. Lets the model learn weights instead of hand-tuning them.

All of these are still within price-only / existing-data scope. Each requires 1-3 hours of additional GBM/feature work.

## Files

- `strategy/YLOka/train_12m_head.py` — H1 walk-forward training pipeline.
- `strategy/YLOka/run_h1_experiments.py` — H1 experiment driver.
- `data/YLOka/ml_preds_12m.parquet` — 339,644 OOS 12m-rank predictions.
- `data/YLOka/ml_preds_12m_cls.parquet` — 339,644 OOS top-quintile classifier predictions.
- `data/YLOka/pit_panel_with_12m.parquet` — PIT panel × all 4 prediction heads.
- `data/YLOka/xs_dispersion.parquet` — per-asof XS dispersion of mom_12_1 (and mom_6_1, vol_12m).
- `backtests/YLOka/runs/` — 30+ new manifests + equity curves for Session 2 experiments.
- `backtests/YLOka/experiment_log.csv` — append-only log.
