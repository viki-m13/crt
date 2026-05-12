# Ideas Backlog

Ranked by expected information gain vs implementation cost.
Updated: 2026-05-11

**Baseline to beat**: v3 ML (K=3, h=6m, tight gate) → CAGR=40.7%, Sharpe=0.863
**Target**: CAGR ≥ 50%, Sharpe ≥ 2.0

---

## Tier 1: Highest EV (tackle first)

### I01 — Volatility Targeting / Risk Parity (EV: HIGH, Cost: MEDIUM)
At each rebalance, compute realized portfolio vol from trailing 3-month daily returns.
Scale position size to target 15% annual portfolio vol. Hold the rest in cash (earning T-bill rate).
When market vol is low → fully invested; when high → reduced exposure.
**Hypothesis**: Same picks as v3, but smoother returns → dramatically higher Sharpe.
**Expected impact**: CAGR may drop ~10pp, but Sharpe could 2x.
**Implementation**: Add `vol_target` parameter to BacktestConfig; need daily returns for vol estimation.
**Key risk**: Overfitting the vol target level (need to cross-validate on OOS).

### I02 — Asymmetric Loss GBM (EV: HIGH, Cost: HIGH)
Retrain the GBM with asymmetric loss: penalize false positives on large losers more than
false negatives on moderate winners. Forces the model to avoid catastrophic picks.
Asymmetric loss = max(pred - actual, 0)*w_down + max(actual - pred, 0)*w_up, w_down > w_up.
**Hypothesis**: Reduces MaxDD while maintaining CAGR; Sharpe improvement.
**Implementation**: Retrain GBM with LightGBM `huber` or custom asymmetric quantile loss.
Requires walk-forward retraining (22 folds × ~30 min = ~11 hours compute).
**Key risk**: Look-ahead in training data (must use embargo/purge CV carefully).

### I03 — Meta-Labeling: Skip/Take Filter (EV: HIGH, Cost: MEDIUM)
Primary model: v3 GBM picks top-K candidates.
Secondary model: classifier predicts P(next-6m return > 0) for each candidate.
Only take candidate if secondary model confidence > threshold.
Reduces number of positions taken → higher average quality.
**Hypothesis**: Same CAGR, fewer bad picks → lower vol → higher Sharpe.
**Features for secondary**: pred_3m, pred_6m, regime, recent XS dispersion, individual stock vol.
**Implementation**: Walk-forward train secondary on (features at T, sign(fwd_6m_ret)) pairs.
**Key risk**: Small training set for secondary; may overfit.

### I04 — Dynamic Position Sizing by Score Confidence (EV: MEDIUM-HIGH, Cost: LOW)
Rather than equal-weight K stocks, size proportionally to (pred_3m + pred_6m)^2.
High-confidence picks get larger weights; borderline picks get smaller.
**Hypothesis**: Concentrates capital in high-IC picks → better risk-adjusted returns.
**Implementation**: Softmax or square-proportional weighting. One-liner change.
**Key risk**: Overfit to in-sample score calibration.

### I05 — Stacked Ensemble: Time-Varying Model Weights (EV: MEDIUM-HIGH, Cost: MEDIUM)
Weight models by their rolling-window OOS IC (information coefficient).
At month T: w_i = softmax(IC_i_last_12m). High-IC model in recent months gets more weight.
Models: pred_3m, pred_6m, pred_12m (if available), mom_12_1, idio_mom.
**Hypothesis**: Adapts to changing market regimes better than fixed ensemble.
**Implementation**: Already have rolling_ic.parquet. Add IC-weighted scorer.
**Key risk**: IC estimation is noisy at 12m window; need longer horizon.

---

## Tier 2: Medium EV (after Tier 1 is exhausted)

### I06 — LSTM/GRU Sequential Predictor (EV: MEDIUM, Cost: VERY HIGH)
Train LSTM on sequences of 24 monthly feature vectors per stock, predict forward rank.
Cross-sectional: at month T, all stocks' feature sequences → rank scores.
**Hypothesis**: Captures multi-step momentum/reversal patterns missed by static GBM.
**Implementation**: Needs PyTorch. Walk-forward: 22 folds. Each fold: ~500 stocks × 24-step sequences.
~1h compute per fold. 22 folds = 22h. Likely needs GPU.
**Key risk**: Insufficient data for sequence models; 248 months × 500 stocks = 124k sequences.

### I07 — Regime-Conditional Hold Period (EV: MEDIUM, Cost: LOW)
During "bull" regime: h=3m (fresh momentum matters more).
During "recovery" regime: h=6m (let winners run).
During "normal" regime: h=6m.
**Hypothesis**: Different regimes have different momentum persistence.
**Implementation**: Add regime-conditional hold_months to BacktestConfig.
**Key risk**: Sample-of-1 in each regime makes this hard to validate.

### I08 — Alternative Crash Gate: Multi-Signal Composite (EV: MEDIUM, Cost: MEDIUM)
Combine: SPY_ret21d + SPY_mom6m + VIX-proxy (cross-sectional vol of all stocks) + breadth.
Soft weighting: cash_weight = sigmoid(risk_score). 
**Hypothesis**: More nuanced de-risking captures gradual slowdowns, not just crashes.
**Implementation**: Compute XS vol (already have vol_12m features); build composite risk score.
**Key risk**: More signals = more parameters = more overfitting risk.

### I09 — Sector-Neutral Residual Momentum (EV: MEDIUM, Cost: MEDIUM)
Compute residual momentum by removing sector-average momentum.
Pick top-K stocks with highest residual momentum (not raw momentum).
This avoids concentration in hot sectors.
**Hypothesis**: Lower correlated picks → lower portfolio vol → higher Sharpe.
**Implementation**: Need sector classification. Use SIC/GICS from another source?
PIT GICS not available but can approximate from price correlation clusters.
**Key risk**: Approximate sector classification adds noise.

### I10 — Stop-Loss at Individual Position Level (EV: MEDIUM, Cost: LOW)
If any position drops >20% from entry, sell it immediately (don't wait for rebalance).
Replace with next-best stock in the PIT universe.
**Hypothesis**: Limits catastrophic single-stock losses like ENRN, LEH, MBI-style blowups.
**Implementation**: Need intra-hold-period monitoring. Check if a position crossed -20% threshold.
**Key risk**: Increases turnover costs; may sell right before recovery.

### I11 — Concentration within Recovery (EV: MEDIUM, Cost: LOW)
In "recovery" regime, concentrate to K=1 (single best stock) or K=2.
In "normal" regime, use K=3.
In "bull" regime, use K=5.
**Hypothesis**: Recovery regimes have highest expected return for top momentum picks.
**Implementation**: Add regime-K override to BacktestConfig.
**Key risk**: Sample size for "recovery" regime is small (~60 months).

### I12 — Score-Based Exit (Anti-Momentum Stop) (EV: MEDIUM, Cost: LOW)
At each monthly check (within hold period), if a current holding drops below
the 50th percentile of the current universe scores, exit and replace.
**Hypothesis**: Avoids holding falling knives in the current basket.
**Implementation**: Monthly score check within hold period.
**Key risk**: Higher turnover and costs; tested and failed in YLOka (dynamic_hold).

---

## Tier 3: Lower EV / High Risk (explore after Tier 1-2)

### I13 — Fractional Differentiation of Price (EV: LOW-MEDIUM, Cost: MEDIUM)
Apply fractional differentiation (López de Prado) to price series.
d < 1 preserves memory; produces stationary but memory-preserving features.
Use as alternative to raw momentum features.
**Hypothesis**: Stationary version of momentum features may improve GBM training.

### I14 — Graph Neural Network on Sector Graph (EV: LOW, Cost: VERY HIGH)
Build a correlation/sector graph of S&P 500 stocks.
Train GNN to predict returns; propagate information across graph.
**Hypothesis**: Sector contagion/momentum captured.
**Key risk**: Almost certainly overfits without much more data.

### I15 — Bayesian Model Averaging Across Rungs (EV: LOW-MEDIUM, Cost: MEDIUM)
Weight all Phase 2 rungs and Phase 3 models by Bayesian posterior probability.
Use BGLS or simple BIC-weighted average.
**Hypothesis**: Reduces model risk; more robust ensemble.

### I16 — Monthly Rebalance with Momentum-Filtered Entry (EV: MEDIUM, Cost: LOW)
Monthly rebalance (h=1), but only take new positions if momentum is accelerating
(this month's rank percentile > last month's rank percentile + threshold).
**Hypothesis**: Enter only when momentum is strengthening, not coasting.

### I17 — Universe Reduction to NDX-Like Quality (EV: MEDIUM, Cost: LOW)
Filter universe to stocks with: vol_1y ≤ 40%, market_cap proxy (log_price) > median,
mom_12_1 > 0, trend_health_5y > 0.5.
Pick top-K from this filtered universe.
**Hypothesis**: Higher quality filter reduces blowup risk.

### I18 — Ensemble of v3 with Different K Values (EV: LOW, Cost: LOW)
Run v3 at K=1, K=3, K=5 simultaneously, combine in proportion 0.2:0.5:0.3.
**Hypothesis**: Diversification across K reduces variance without sacrificing much CAGR.

### I19 — Cash Yield Optimization (EV: VERY LOW, Cost: VERY LOW)
Add 4-5% T-bill yield to cash months (already partially implemented).
**Hypothesis**: Marginal Sharpe improvement during extended cash periods.

### I20 — Chronos Foundation Model as Feature Generator (EV: MEDIUM, Cost: HIGH)
Use v5 Chronos to generate per-stock next-period return forecasts.
Feed as additional features to the GBM ranker.
**Hypothesis**: Chronos captures time-series patterns missed by cross-sectional GBM.
**Implementation**: Need Chronos model setup and per-stock time series.

### I21 — Selection-Aware Meta-Learning (EV: MEDIUM, Cost: MEDIUM)
At month T, pick the model/rung that had best OOS performance in trailing 12 months.
This is itself a model that must be walked forward.
**Hypothesis**: Adapts to regime-conditional signal strength.

### I22 — Triple-Barrier Labeling for GBM Training (EV: LOW-MEDIUM, Cost: HIGH)
Train GBM to classify outcomes as: HIT_UPPER / HIT_LOWER / TIME_OUT.
Upper barrier: +30% in 6m. Lower barrier: -15% in 6m. Time-out: otherwise.
Pick stocks predicted to hit upper barrier.
**Hypothesis**: Better calibrated picks; avoids holds that time out at mediocre returns.

---

## Implementation Priority Queue

1. **I01** — Volatility targeting (fast win, high EV on Sharpe)
2. **I03** — Meta-labeling (medium effort, could reduce bad picks)
3. **I07** — Regime-conditional hold period (trivial code, test first)
4. **I04** — Score-proportional sizing (one line, test fast)
5. **I11** — Regime-conditional K (trivial, already in YLOka but try different values)
6. **I10** — Individual stop-loss (medium effort, test carefully)
7. **I05** — IC-weighted ensemble (already have IC data)
8. **I17** — Quality universe filter (trivial to implement)
9. **I02** — Asymmetric loss GBM (highest potential but highest cost)
10. **I06** — LSTM (only if all other options exhausted and compute available)
