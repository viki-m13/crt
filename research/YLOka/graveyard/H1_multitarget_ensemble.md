# H1 — Multi-target ensemble (12m + classifier heads) — KILLED

**Hypothesis**: train a 12m-rank GBM head and a top-quintile classifier head, ensemble with 3m/6m by recent IC. Different forward horizons should isolate different sources of edge. When all horizons agree, conviction is real; when they disagree, signal is noise.

**Implementation**:
- Trained `pred_12m` (regressor) and `pred_12m_cls` (classifier P(top-quintile)) using the same 52-feature production set.
- 22 walk-forward train windows (Jan refit, expanding window, 13-month embargo for 12m label).
- Cross-sectional rank labels (matches v3 framing).
- 339,644 OOS predictions per head over 2003-01 → 2026-05.

**Results** (12 integration variants tested, all UNDERPERFORM v3 baseline of CAGR 40.78%, Sharpe 0.953, MaxDD -49.83%):

| variant | CAGR | Sharpe | MaxDD | Δ CAGR |
|---|---:|---:|---:|---:|
| ens_3_6_12 ew (rank ensemble) | 31.64% | 0.819 | -49.23% | -9.1 |
| ens_3_6_12_cls (rank + cls weight) | 35.65% | 0.924 | -47.83% | -5.1 |
| ens_3_6_12_invvol | 31.64% | 0.819 | -49.23% | -9.1 |
| ens_36_12wt (long-bias) | 33.40% | 0.846 | -49.23% | -7.4 |
| tilt_005 (additive 12m rank) | 35.69% | 0.872 | -49.83% | -5.1 |
| tilt_015 (stronger additive) | 35.27% | 0.888 | -53.27% | -5.5 |
| cls_filter_018 (cls hard filter) | 40.73% | 0.950 | -49.83% | -0.05 |
| cls_filter_025 (tighter filter) | 37.81% | 0.900 | -49.83% | -3.0 |
| cls_tilt (additive cls) | 39.73% | 0.921 | -49.83% | -1.1 |

**Best result**: `cls_filter_018` is essentially a no-op. Every other variant materially underperforms.

**Why it failed**:

1. **Magnitude → rank loses information**. The baseline uses raw `(pred_3m + pred_6m)/2` magnitudes, which carry the model's confidence calibration. Switching to mean(rank(...)) flattens all months' top scores to ~1.0, erasing the relative-conviction signal.
2. **12m horizon is too long**. By 12 months the per-stock signal is dominated by macro/regime/idiosyncratic events the model can't predict from cross-sectional features. IC of `pred_12m` is materially lower than `pred_3m` or `pred_6m`.
3. **Classifier head is mostly redundant** with the regressor heads. The names that have high `pred_12m_cls` mostly overlap with high `pred_3m+pred_6m`, so adding classifier weight doesn't add information.
4. **Additive tilts confuse the score** by mixing magnitude-scaled `(p3+p6)/2` with rank-scaled `pred_12m` — the rank-tilt dominates because it's the same scale across all months.

**Consistent with prior finding**: `experiments/monthly_dca/v6/REPORT.md` documented that "proprietary features GBM substantially worse than v3 baseline". Adding more model heads doesn't help.

**Don't repeat in this form**: the v3 baseline architecture (raw magnitude average of 2 short horizons) is locally optimal. Real lift would require:
- **Different ML targets** (e.g., conditional probability of >X% return, downside-asymmetric losses).
- **Time-varying ensemble weights** based on rolling IC of each head.
- **Regime-specific models** trained separately for crash/recovery/normal/bull regimes.
- **Stacked ensemble** with a small lasso blender on top.

All deferred to a future session.

**Artifacts retained** (these CAN be reused for future ensemble experiments without retraining):
- `data/YLOka/ml_preds_12m.parquet` — 339,644 OOS 12m-rank predictions.
- `data/YLOka/ml_preds_12m_cls.parquet` — 339,644 OOS top-quintile classifier predictions.
