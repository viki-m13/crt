# Ideas Backlog — Ranked by Expected Value

**Last updated**: 2026-05-11
**Best known baseline**: v5_chr_p70_q0.45_k3_invvol — WF mean CAGR 47.2%, Sharpe 1.06
**Target**: CAGR ≥ 50%, Sharpe ≥ 2.0
**Key bottleneck**: Sharpe gap of ~1.0 is the binding constraint; CAGR gap is ~3pp

---

## Tier 1 — Highest Expected Value (try first)

### Idea 01: Volatility-Targeted Meta-Labeling
**Hypothesis**: The v3 GBM + Chronos can be used as a SIGNAL. Train a binary meta-model:
"Will this basket of K picks beat cash next month?" Using features: trailing portfolio vol,
market breadth proxy, Chronos confidence level, v3 score spread, short-term momentum of picks.
If meta-model says NO → go to full cash. This should cut losing months substantially.
**Expected Sharpe lift**: +0.3 to +0.8 (if meta-model has IC > 0.1)
**Expected CAGR cost**: -2 to +5pp (more cash months, but avoids bad months)
**Effort**: Medium (train binary classifier with monthly features)
**Key risk**: Meta-model overfits. Must use combinatorial purged CV.

### Idea 02: LightGBM LambdaMART Ranker (Ranking Objective)
**Hypothesis**: Current v3 uses regression on 1m/3m/6m forward returns. A ranking objective
(LambdaMART on cross-sectional rank of forward returns) directly optimizes the picking task.
Should produce better top-K capture especially in the tails.
**Implementation**: lightgbm with objective="lambdarank" or "rank_xendcg". Use relative rank
within each cross-section (asof) as the label. 62+ features from pit_panel_full.
**Expected Sharpe lift**: +0.1 to +0.3
**Expected CAGR lift**: +2 to +8pp
**Effort**: Medium (modify training objective)
**Key risk**: Ranking objective may overfit to ordinal structure. Needs careful embargo.

### Idea 03: Chronos as Feature Generator (not filter)
**Hypothesis**: Instead of using Chronos p70 as a filter (binary keep/drop), use multiple
Chronos quantiles (p30, p50, p70, p90) as continuous features in the GBM. The shape of the
Chronos predictive distribution (spread between p30 and p70) captures forecast uncertainty.
High-confidence Chronos + high GBM = strongest signal.
**Implementation**: Score each SPX ticker with Chronos-bolt-tiny for multiple quantiles.
Feed all quantile values + distribution shape as GBM features. Retrain full walk-forward.
**Expected Sharpe lift**: +0.2 to +0.5
**Expected CAGR lift**: +3 to +10pp
**Effort**: High (need to re-run Chronos inference on full SPX universe, ~5-10 min CPU)
**Key risk**: Chronos is zero-shot — adding as GBM feature may overfit to Chronos noise.

### Idea 04: Triple-Barrier Labeling with Asymmetric Target
**Hypothesis**: Train on "probability of +30% gain before -15% loss" (triple-barrier with
asymmetric barriers). This trains GBM to find stocks with good risk/reward profiles,
not just high raw return. Should reduce picks that have high mean but also high loss prob.
**Implementation**: Compute triple-barrier labels from daily prices. Upper barrier: +30%,
Lower barrier: -15%, Time-out: 6 months. Train LightGBM on binary labels.
**Expected Sharpe lift**: +0.2 to +0.5 (reduced drawdown contribution)
**Expected CAGR cost**: -3 to +5pp
**Effort**: Medium (barrier computation from daily prices + GBM retrain)
**Key risk**: Asymmetric barrier choice needs to be defended and not tuned on OOS.

### Idea 05: Regime-Conditional Volatility Targeting
**Hypothesis**: When the GBM model's top picks have high trailing vol (indicating a volatile
market for those specific names), dynamically reduce K (fewer names, hence less idiosyncratic
risk). Target: portfolio daily vol = constant 15% annual. K varies from 1 to 10 based on
individual stock vol. This is NOT the market regime gate — it's a position-sizing rule.
**Implementation**: At each rebalance, set K and weights to target portfolio_vol = 15%.
Use inverse-vol + correlation approximation (diagonal covariance) for speed.
**Expected Sharpe lift**: +0.3 to +0.7 (vol stabilization is the primary Sharpe driver)
**Expected CAGR cost**: -5 to +3pp
**Effort**: Low (modify portfolio construction, not the signal)
**Key risk**: Diagonal covariance approximation ignores correlation. Full covariance needs
more history and introduces estimation error.

---

## Tier 2 — Strong Ideas, Secondary Priority

### Idea 06: Stack v3 + Chronos via Meta-Learner on Fold-OOF Predictions
**Hypothesis**: Train a meta-learner (Lasso or ElasticNet) that blends:
- v3 rank (pred_3m + pred_6m)
- Chronos_p70 rank
- Rolling 3m IC of each model
- Regime indicators
Using out-of-fold predictions (inner CPCV) to avoid leakage. The meta-learner learns
the optimal mixing ratio dynamically rather than fixing it.
**Expected Sharpe lift**: +0.1 to +0.3
**Effort**: Medium

### Idea 07: Earnings-Quality Proxy (Price-Only)
**Hypothesis**: Stocks with earnings beats show predictable post-earnings drift. We can
approximate earnings quality from PRICE patterns alone: a stock that gaps up on high vol
and immediately stabilizes (vs one that gaps up and reverses) is likely an earnings beat.
Features: gap day return, 5d return after a large gap, deviation from trend at earnings.
**Note**: This is speculative since we don't have earnings dates PIT.
**Expected Sharpe lift**: Unknown (genuinely new information if it works)
**Effort**: High (requires earnings-date identification from price discontinuities)

### Idea 08: Cross-Sectional Neutralization (Sector-Neutral Momentum)
**Hypothesis**: The v3 GBM picks concentrate in top-performing sectors. If we neutralize
by sector (GICS), within-sector momentum has better risk properties. Need sector data.
Can proxy GICS from the existing SIC-based categorization or from Yahoo Finance sector.
**Expected Sharpe lift**: +0.1 to +0.2 (more diversification reduces max sector drawdown)
**Expected CAGR cost**: -2 to -5pp (diversification cost)
**Effort**: Medium (need sector labels)

### Idea 09: Selection-Aware Ensembling (Model Rotation)
**Hypothesis**: At each month, pick the model that won the trailing 12-month OOS window.
Candidates: v3 alone, v3+Chronos, LambdaMART ranker, vol-targeted. This IS a model
and must be walked forward properly.
**Expected Sharpe lift**: +0.1 to +0.2 if models are regime-conditional
**Effort**: Low (post-hoc on existing model outputs)

### Idea 10: Fractional Differentiation of Price Series
**Hypothesis**: Fractionally differenced price series (d ≈ 0.3-0.5) preserves more memory
than log-returns while being stationary. Use as GBM features.
Features: fracDiff(price, d=0.3), fracDiff(price, d=0.5) over 6m/12m windows.
**Expected Sharpe lift**: +0.0 to +0.1 (marginal over existing momentum features)
**Effort**: Medium (implement fracDiff + integrate as feature)

### Idea 11: Broader Universe (1833 tickers) with Chronos
**Hypothesis**: v5 shows broader universe (1833 tickers) achieves ~52% WF mean CAGR with
Chronos filter. If the broader universe's PIT membership is defensible (survivorship concern:
only 9 of 1833 truly delisted), test the broader universe strategy formally.
**Expected CAGR**: ~52% (per v5 report), meeting the CAGR gate
**Expected Sharpe**: ~1.1-1.2 (marginally better than SPX due to dispersion)
**Effort**: Low (data already in cache, need formal PIT validation)
**Key risk**: Survivorship bias in 1833-ticker universe. Need to quantify impact.

### Idea 12: LSTM Sequence Model on Feature Panel
**Hypothesis**: Monthly sequence of 47 features per stock, LSTM predicts 1-month forward
rank. Captures temporal patterns (e.g., "accelerating momentum → overshoot") that
cross-sectional GBM cannot.
**Expected Sharpe lift**: Unknown (first attempt at deep learning on this data)
**Effort**: Very High (PyTorch, GPU preferred, ~1-2 hours training per fold)
**Key risk**: Overfitting with 268 time steps and many features.

### Idea 13: Chronos on Full SPX Universe (Inference)
**Hypothesis**: Run Chronos-bolt-tiny on the full SPX PIT universe (276 asofs × ~500 tickers).
The existing `ml_preds_chronos_ndx.parquet` is only NDX. SPX Chronos predictions would
cover the full universe. The script `score_chronos_bolt.py` exists but outputs were not
committed as a parquet (only CSVs).
**Effort**: Low-Medium (re-run score_chronos_bolt.py targeting SPX, ~5-10 min CPU)
**Note**: v5 report implies this was done (276 asofs × ~500 tickers) but the file wasn't
found in the cache. Need to regenerate or find it.

### Idea 14: Calibrated Probability Ensemble (Isotonic Regression)
**Hypothesis**: Calibrate v3 GBM's pred_3m and pred_6m outputs with isotonic regression
so they represent proper probabilities. Combine calibrated probabilities across horizons.
Use calibrated ensemble as ranker.
**Expected Sharpe lift**: +0.0 to +0.1
**Effort**: Low

### Idea 15: Long Window GBM with More Features
**Hypothesis**: Retrain GBM with 10-year lookback (vs current 5-year) and include all
47+ features from pit_panel_full.parquet. Check if longer training improves OOS.
**Expected CAGR lift**: +1 to +3pp
**Effort**: Medium (need to set up GBM training pipeline)

---

## Tier 3 — Longer-Term / Data-Limited Ideas

### Idea 16: Fundamentals (PIT Earnings Data)
**Requires**: New data fetch (yfinance earnings, Compustat, or similar)
Gross profitability, earnings surprise, forward P/E. These provide genuinely new information.
**Expected Sharpe lift**: +0.3 to +0.7 if quality signal is strong
**Effort**: Very High (new data, PIT handling, report lag)

### Idea 17: Short Interest from Public Sources
**Requires**: Monthly short interest data from FINRA or yfinance. Covers ~2010 to present.
Stocks with high and rising short interest AND strong GBM score = squeeze candidates.
**Effort**: High (data fetch, PIT alignment)

### Idea 18: Overnight vs Intraday Return Decomposition (OHLC)
**Requires**: Daily OHLC prices. prices_extended.parquet may be close-only.
Gap at open = overnight return = systematic factor. Within-day return = noise.
**Effort**: Medium (need O/H/L data)

### Idea 19: Graph Neural Network on Sector/Correlation Graph
**Hypothesis**: At each asof, build a correlation graph of daily returns. GNN aggregates
neighbor features. Sector structure regularizes the graph.
**Expected Sharpe lift**: Unknown, but likely small given GBM saturation
**Effort**: Very High (PyTorch Geometric, 20-fold training)
**Risk**: Most GNN gains evaporate under ablation vs non-graph version

### Idea 20: Macro Regime Conditioning (Observable Proxies)
**Hypothesis**: Cluster macro states using: realized vol of SPY, yield curve proxy (from
treasury data), credit spread proxy (HYG/LQD ratio or computed from high-vol stocks).
Train separate GBM per macro regime. At test time, soft-mix based on regime probability.
**Prior failure**: Session 5 YLOka tried price-regime specialists and they failed.
But macro-based regimes (credit + rates) may be more stable than momentum-based regimes.
**Effort**: High (macro data fetch or proxy construction)

### Idea 21: Bayesian Model Averaging (BMA) across all v3/v5 variants
**Hypothesis**: Weight v3, v3+Chronos, LambdaMART, vol-targeted models by their
posterior probability (from OOS Sharpe distribution). Use rolling 2-year evidence.
**Expected Sharpe lift**: +0.0 to +0.1 (marginal)
**Effort**: Low (post-hoc on existing results)

### Idea 22: NDX Universe with Longer Chronos History
**Hypothesis**: The Chronos NDX predictions go back to 1997 (353 asofs). The NDX PIT
membership only goes to 2015 (insufficient for 10-year WF gate). But: generate SPX
PIT membership for NDX-overlap tickers back to 2003, use Chronos predictions as signals
on the full SPX universe when ticker overlaps.
**Effort**: Medium (data alignment)

---

## Experiment Priority Queue (for next 3 sessions)

1. **Session 2**: Idea 01 (Meta-labeling) + Idea 05 (Vol targeting) — both target Sharpe directly
2. **Session 3**: Idea 02 (LambdaMART) + Idea 03 (Chronos features) — target both Sharpe and CAGR
3. **Session 4**: Idea 11 (Broader universe) — CAGR gate, fastest path to 50%+ CAGR
4. **Session 5**: Idea 13 (Chronos SPX inference) — prerequisite for Idea 03

## Tracking

Total hypotheses tested (this project, all sessions): 11 (5 from v1 ladder + 6 from v2 ladder rungs)
DSR deflation count: 11
