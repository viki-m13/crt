# Ideas Backlog

Last updated: 2026-05-12. Ranked by expected EV (CAGR lift × robustness probability).

---

## Critical Context

**Baseline (v3 ML, k=3, EW, tight regime, 10bps, hold=6m):**
- Full CAGR: 39.8% | Sharpe: 0.955 | MaxDD: -49.8%
- WF mean CAGR: 42.8% | WF min CAGR: 14.5% | WF Sharpe: 1.03

**v5 Chronos (baseline + Chronos-bolt-tiny filter):**
- Full CAGR: 44.8% | Sharpe: 1.04 | WF mean CAGR: 45.9%

**Targets:** CAGR ≥ 50%, Sharpe ≥ 2.0 (annualized monthly)

**Theoretical ceiling analysis (honest):**
- Sharpe 2.0 with long-only monthly equities at 50% CAGR requires monthly std ≤ 5.96%.
- Current v3 portfolio has monthly std ≈ 10.3%. Must nearly halve portfolio vol.
- Best prior result (v8b, WITH leverage): Sharpe 1.12. Without leverage, ceiling likely ~1.5.
- Sharpe 2.0 may require either: (a) much more diversification OR (b) regime filtering so good 
  it eliminates nearly all negative months. Both are hard.
- Honest best-case: CAGR 50%+ with Sharpe 1.2-1.5. Sharpe 2.0 is aspirational.

---

## Tier 1 — Highest EV (do first)

### #1: Regime-conditional model ensemble with IC-tracking
**Hypothesis:** The ML model's IC varies strongly by market regime. In bull regimes, 
momentum is strong; in recovery regimes, mean-reversion is dominant. Training separate 
models per regime and routing predictions through a regime classifier should improve 
the IC and reduce drawdowns.
**Implementation:** Use `regime_labels.parquet` (YLOka), fit separate HistGBM per regime 
on features → forward_ret, use rolling regime classifier to weight at inference time.
**Expected lift:** +5-10pp CAGR, +0.1-0.2 Sharpe
**Risk:** Regime classification noise, small-sample in crash periods
**Data needed:** features/*.parquet, fwd_returns.parquet, regime_labels
**Work estimate:** 2-3 runs

### #2: LightGBM LambdaMART ranker with expanded features
**Hypothesis:** The v3 HistGBM predicts absolute returns; a LambdaMART ranker trained to 
maximize IC directly should produce better cross-sectional ranking.
**Implementation:** Use `lgb.train` with `objective='lambdarank'`, `ndcg_eval_at=[10,20]`.
Features: all 79 from feature panel. Target: decile rank within cross-section.
**Expected lift:** +3-8pp CAGR (uncertain, v8b found limited gains from LGB; this is a 
different objective and feature set)
**Risk:** Already tried, may repeat v8b failure
**Prior art:** v8b found new LGB regressor gave 17-22% WF CAGR (vs 42.8% v3). But 
rank objective not tried.
**Work estimate:** 1-2 runs

### #3: Information coefficient-weighted regime gate
**Hypothesis:** Only invest when the rolling 6-month IC ≥ threshold (model is "working"). 
When IC is low/negative, go to T-bills. This reduces invested months during signal-weak 
periods and increases Sharpe.
**Implementation:** Load rolling_ic.parquet, add IC filter to regime gate. Threshold 
sweeps: IC > 0, > 0.02, > 0.05.
**Expected lift:** Sharpe +0.1-0.3 at cost of -5-15pp CAGR (tradeoff)
**Risk:** IC measured in-sample (need proper embargoed IC)
**Work estimate:** 1 run (fast)

### #4: Sector-neutral portfolio construction
**Hypothesis:** The v3 picks tend to cluster in hot sectors (tech in 2020-24, energy in 
2008-22, financials in 2003-08). Within-sector clustering concentrates factor risk and 
inflates drawdowns. Picking top-1 from each GICS sector produces a more stable 
portfolio.
**Implementation:** Load sector labels from feature_panel_pit.parquet, pick top-1 by ML 
score within each sector, equal-weight across sectors. ~11 positions.
**Expected lift:** -10pp CAGR but +0.3-0.5 Sharpe (diversification trades return for smoothness)
**Risk:** May not reach 50% CAGR target without higher diversification gains
**Work estimate:** 1 run

### #5: Chronos ensemble expansion
**Hypothesis:** v5 Chronos-bolt-tiny adds +3-8pp CAGR as a confidence FILTER. Using 
it as an ADDITIONAL SCORE (blend with ML) and tuning at different horizons might add more.
**Implementation:** Load `ml_preds_3m_*.parquet` and `ml_preds_6m_*.parquet` from YLOka. 
Compute separate Chronos scores for bull/recovery/crash regimes. Blend with v3 score.
**Expected lift:** +2-5pp CAGR (diminishing from v5 as Chronos already tried)
**Risk:** Chronos bolt-tiny might be data-limited on the specific period
**Work estimate:** 1 run (existing predictions available)

### #6: k=2 concentration with regime-conditional hold
**Hypothesis:** v3 at k=3 misses some upside by holding 3 picks. k=2 with hold=3m 
might capture faster-rotating opportunities while keeping the top signal concentrated.
**Implementation:** sweep k ∈ {2,3} × hold_months ∈ {3,6} × weighting ∈ {ew, invvol}.
**Expected lift:** +5-10pp WF CAGR (based on v8 which found k=2 strong for longer holds)
**Risk:** k=2 fragile to single-stock disasters (v8b: -32% in 2025 holdout for k=1)
**Work estimate:** 1 run (8 variants, 10 min)

### #7: Stacked meta-learner with OOF cross-validation
**Hypothesis:** Train a meta-model to select which base model prediction to trust at 
each month (v3, Chronos-filtered v3, regime-regime models). Meta-model sees trailing 
IC, regime state, volatility regime.
**Implementation:** CPKF (combinatorial purged k-fold) at monthly level, train logistic 
meta on stacked predictions. Very careful embargo.
**Expected lift:** +3-6pp CAGR, +0.1-0.2 Sharpe
**Risk:** Complex, high overfitting risk at meta-level
**Work estimate:** 2 runs

---

## Tier 2 — Moderate EV

### #8: Triple-barrier labeling for ML training
**Hypothesis:** Training on {upper, lower, time-out} class labels instead of continuous 
forward returns might produce a cleaner signal by focusing on high-conviction moves.
**Implementation:** fwd_returns.parquet has ret__trail_35 (35% trailing stop), 
ret__tp100 (100% take-profit), ret__fixed_6m. Use these to label (win/loss/neutral).
**Work estimate:** 2-3 runs (new feature engineering)

### #9: Fractional differentiation of price series
**Hypothesis:** Price levels are nonstationary, causing lookahead in momentum features. 
Fractionally differentiated price (d ≈ 0.3-0.5) preserves more information than 
first-differencing while achieving stationarity.
**Implementation:** Implement fracdiff on log(price), verify stationarity with ADF.
Use as alternative feature inputs to ML models.
**Work estimate:** 2 runs

### #10: Volatility regime-conditional sizing
**Hypothesis:** Market realized vol is a leading indicator of drawdown. Scale down 
positions when market vol is elevated (realized 21-day vol > 1.5× 252-day avg). 
**Implementation:** vol signal = spy_vol_1y vs trailing_21d_vol from SPY returns. 
Scale: weight × min(1, target_vol / realized_vol).
**Expected lift:** Sharpe +0.15-0.25 at cost of -5-10pp CAGR
**Work estimate:** 1 run

### #11: NDX (Nasdaq-100) universe experiment
**Hypothesis:** NDX has a higher concentration of growth/momentum stocks. The same ML 
signal on NDX might produce higher alpha with better consistency (tech stocks tend to 
trend more consistently).
**Implementation:** Reuse ml_preds_v2 for NDX-eligible tickers only. Build PIT NDX 
membership from 2003 (from QQQ holdings historical data or reconstruction).
**Data limitation:** PIT NDX membership not in repo; approximate with ticker list.
**Work estimate:** 2 runs (data building + eval)

### #12: Quality-momentum composite with decay
**Hypothesis:** Momentum decays nonlinearly. Weighting mom_3m more than mom_12m (or 
vice versa in different regimes) captures better alpha.
**Implementation:** Blend mom_3m×0.6 + mom_12_1×0.4 for short-memory; reverse for 
longer memory. Combine with quality (sharpe_1y, trend_r2_12m). 
**Work estimate:** 1 run (factor engineering)

### #13: Earnings revision proxy signal
**Hypothesis:** The feature `earnings_drift_proxy` in the feature panel is not well 
utilized by the v3 model (it was added later). Training a model with this feature 
prominently should improve the earnings quality of picks.
**Implementation:** Retrain HistGBM with feature_importance analysis, then 
upweight earnings_drift_proxy in training.
**Work estimate:** 1 run

### #14: Beta-adjusted position sizing
**Hypothesis:** The top ML picks tend to have high beta. By sizing positions inversely 
to their beta (rather than invvol), we can achieve market-neutral return components 
while maintaining the alpha exposure.
**Implementation:** Load beta_2y from feature panel. Weight = 1/beta_2y, normalized.
**Work estimate:** 1 run (fast, same infrastructure)

---

## Tier 3 — Speculative / High-effort

### #15: Graph neural network on sector/correlation graph
**Hypothesis:** Stock returns propagate through sector correlation networks. A GNN 
trained on a monthly correlation graph + node features might capture network effects 
not visible in cross-sectional models.
**Implementation:** Build monthly correlation graph from prior 252-day returns. 
Node features = ML score + vol + momentum. EdgeConv or GraphSAGE. Walk-forward.
**Risk:** Prior work shows graph gains evaporate under ablation (CLAUDE.md).
**Work estimate:** 4-6 runs (complex infrastructure)

### #16: Transformer encoder for temporal feature sequences
**Hypothesis:** A transformer encoder ingesting 12-month sequences of per-stock features 
might capture temporal patterns invisible to the cross-sectional HistGBM.
**Implementation:** Build (stock, month) → next_month_rank training data. 
Encoder with 4 layers, 64-dim, 8 heads. Walk-forward fit on 5y windows.
**Risk:** Monthly cadence = only ~60 samples per stock per year. Very noisy.
**Work estimate:** 4-6 runs

### #17: Combinatorial purged CV + Bayesian model averaging
**Hypothesis:** CPCV across 3 base models (v3, regime-conditional, ranker) gives proper 
uncertainty estimates. BMA weights = CPCV Sharpe. Ensemble may smooth the returns.
**Work estimate:** 3-4 runs

### #18: Volume-of-volatility (VoV) based selection
**Hypothesis:** The `vov_60` feature (coefficient of variation of rolling 5-day vol) 
captures when stocks are about to break out of volatility consolidation. High VoV 
often precedes large directional moves.
**Implementation:** Interaction between high ML score AND high vov_60 — picks the 
"ML bull thesis that's about to resolve." 
**Work estimate:** 1 run (fast feature filter)

### #19: Holdout-aware feature selection
**Hypothesis:** The 79 features include some that over-fit to specific historical periods. 
Using SHAP feature importance from WF training to select the top 20 most CONSISTENT 
features (consistent SHAP across splits) should improve OOS stability.
**Work estimate:** 2 runs

### #20: Meta-labeling (primary + secondary model)
**Hypothesis:** Use v3 ML to select candidates, then train a secondary HistGBM to 
classify "take / skip" based on regime state, confidence, and feature context. The 
secondary model learns when v3 is more likely to be right.
**Implementation:** Primary: v3 top-10 candidates. Secondary: binary classifier 
trained on whether v3's top picks actually outperformed (OOF). Ensemble gate.
**Work estimate:** 2-3 runs

---

## Order of attack for next runs

1. #6 (k=2 concentration sweep) — fast, clear signal from v8 prior work
2. #3 (IC-weighted regime gate) — fast, directly targets Sharpe
3. #1 (regime-conditional model) — moderate, highest theoretical CAGR lift
4. #5 (Chronos ensemble) — fast (data exists)
5. #14 (beta-adjusted sizing) — fast one-liner
6. #4 (sector-neutral) — fast, Sharpe target
