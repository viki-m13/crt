# Ideas Backlog (ranked by EV)

Priority 1: Must address the Sharpe gap (1.0 → 2.0). This is the critical blocker.

---

## TIER 1 — Sharpe-focused (high EV)

### 1. Volatility-targeted portfolio overlay [CURRENT FOCUS]
- **What**: Scale equity exposure by (vol_target / realized_portfolio_vol). When realized_vol is high, reduce equity fraction. When low, increase.
- **Implementation**: Rolling 3m and 6m realized vol of portfolio. Target 12% annual portfolio vol (1% monthly). Scale factor = 0.12/realized_vol_annualized, capped at 1.0 and floored at 0.
- **Why high EV**: Classic technique proven to improve Sharpe in many contexts. Reduces tail events without hurting CAGR proportionally (leverage is not used, only de-risk).
- **Estimated effect**: Vol 15% → 9%, CAGR penalty ~15-20%, but Sharpe 1.0 → 1.5-1.8

### 2. Sector-diversified picking (max 1 per GICS sector)
- **What**: Within top-scored picks, ensure no more than 1 stock from same sector.
- **Why**: Worst months (MBI/MTG/SLM in 2008) were pure sector concentration blowups.
- **Implementation**: Add sector labels to panel, then apply sector diversity constraint when picking K.
- **Estimated effect**: Reduces single-sector blowup risk. Should improve worst-month floor.

### 3. Quality-filtered selection (exclude distressed names)
- **What**: Gate selection on quality signals: exclude names with dd_from_52wh < -50%, or vol_1y > 0.8 (extremely high vol = likely distressed).
- **Why**: MBI, MTG, SLM in 2008 were momentum plays in financial distress. Quality filter would exclude.
- **Implementation**: Pre-filter picks using quality gate before top-K selection.
- **Estimated effect**: Reduces catastrophic blowup risk.

### 4. Invvol weighting at individual stock level
- **What**: Weight each of K picks inversely to its 60-day realized vol. High-vol picks get smaller weight.
- **Why**: Equalizes risk contribution across picks. Reduces impact of high-vol individual stocks.
- **Estimated effect**: Minor Sharpe improvement; best combined with vol targeting.

### 5. Two-level ensemble: momentum + quality
- **What**: Score = alpha * rank(ML_pred) + (1-alpha) * rank(quality_composite). quality_composite = trend_health_5y + frac_above_50dma_1y + sharpe_5y - vol_1y.
- **Why**: Pure momentum picks have high returns but high vol. Adding quality tilts toward smoother returners.

### 6. Monthly rebalance with crash gate only (hold=1)
- **What**: Rebalance every month, but only hold equity when crash gate = no crash.
- **Why**: Monthly rebalance allows faster crash exit. Transaction cost goes up (~5bps/month vs ~5bps/6mo = 12x more), but crash avoidance value may exceed cost.
- **Concern**: Costs matter a lot here. At 5bps per leg, monthly rebalance = 60bps/year extra vs 10bps/year.

---

## TIER 2 — Return-focused (moderate EV)

### 7. LightGBM cross-sectional ranker (walk-forward)
- **What**: Train LightGBM LambdaMART on all 47 features (from pit_panel_full) to rank stocks by next-month forward return. Walk-forward: retrain every 12 months using 36-month rolling window.
- **Why**: The current ML scores (pred_3m, pred_6m) are already GBM predictions. A fresh cross-sectional ranker might find better feature combinations for the 1-month horizon.
- **Risk**: With 47 features and small monthly panels (~450 stocks), risk of overfitting. Need strict CPCV.

### 8. Triple-horizon ensemble with Sharpe-aware scoring
- **What**: Score = pred_1m * (1/vol_1m) + pred_3m * (1/vol_3m) + pred_6m * (1/vol_6m), where vol_Xm is the predicted volatility at horizon X.
- **Why**: Returns-per-unit-risk scoring should select stocks that are better in risk-adjusted terms.

### 9. Momentum quality combo (Fama/French QMJ-style)
- **What**: Filter the top-momentum picks to require trend_health_5y > 0.6 AND sharpe_5y > 0.3.
- **Why**: Combines momentum (picks names with strong trends) with quality (picks names with consistent trends).

### 10. Regime-gated vol targeting
- **What**: Use 3 levels based on SPY regime: bull = 100% equity, normal = 80%, recovery = 100%, crash = 0%.
- **Why**: Instead of binary crash gate, use a stepped approach that reduces exposure in uncertain regimes.

### 11. Chronos-based features (time series embedding)
- **What**: Use v5 Chronos (or another time-series model) to generate 1-month forward return forecasts as an additional feature input to the ranking model.
- **Where**: Data is in experiments/monthly_dca/cache/v2/
- **Risk**: If Chronos outputs are already used (pred_1m, pred_3m, pred_6m ARE Chronos predictions based on harness comments), this is already baked in. Verify.

### 12. Factor-neutralized momentum
- **What**: Run cross-sectional OLS of mom_12_1 on size + beta + vol (to neutralize), use residual as signal.
- **Why**: Raw momentum picks are correlated with specific factor exposures. Residual momentum is cleaner.

### 13. Mean reversion overlay for entry timing
- **What**: Among top-ML picks, prefer those with recent short-term dip (e.g., ret_21d < 0 but mom_12_1 > 0.3). Buy the dip in a winner.
- **Estimated effect**: Lower entry prices, potentially better forward returns. High turnover risk.

---

## TIER 3 — Structural improvements (lower priority)

### 14. Combinatorial purged CV for feature selection
- **What**: Run CPCV on the full feature set to find the most robust subset for ranking.
- **Why**: 47 features may include noise; feature selection helps generalization.

### 15. Bayesian model averaging across top experiments
- **What**: Combine signals from top-5 experiments using BMA weights derived from out-of-sample Sharpe.
- **Why**: Diversifies model risk.

### 16. Walk-forward hyperparameter optimization with DSR
- **What**: Systematic grid search over (K, hold_months, score_fn) combinations, using Deflated Sharpe Ratio to penalize multiple testing.
- **Why**: Current grid is incomplete. May find better combination.

### 17. Fractional differentiation of price series
- **What**: Apply fractionally differenced price (d=0.4-0.6) as a stationarity-preserving memory feature.
- **Why**: Captures long-memory in price trends while remaining stationary.

### 18. VIX-equivalent breadth signal
- **What**: Use cross-sectional return dispersion as a market stress indicator (high dispersion = risky).
- **Why**: Better crash detection than SPY momentum alone.

### 19. Portfolio optimization (Markowitz MVO with constraints)
- **What**: Given top-10 candidate stocks, use rolling covariance matrix to find minimum-variance portfolio among them.
- **Why**: Better diversification than EW or invvol.
- **Risk**: Covariance estimation with 10 stocks and monthly data is noisy.

### 20. Meta-labeling (primary model selects, secondary decides take/skip)
- **What**: Use primary scorer to select candidates. Train binary classifier on whether each pick was actually profitable (using CPCV). Only take picks where classifier says yes.
- **Why**: Filters out systematic bad bets.

---

## Experiments needed to reach target:
- Sharpe gap: 1.0 → 2.0 (need 2x improvement) — CRITICAL
- CAGR gap: 48.73% → 50%+ — close, manageable
- Minimum viable combo: vol_targeting + quality_filter + sector_diversification + invvol_weights

## Hypotheses tested so far: ~108 (from YLOka sessions 1-5)
