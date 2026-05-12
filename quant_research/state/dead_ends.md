# Dead Ends

Approaches that have been tried and failed. Do not retry without a substantively different formulation.

## 1. PIT Panel `pred` Column as Score Signal
- **Tried:** exp_001 — ml_score_v3, ml_plus_smooth using pred column from pit_panel_full.parquet
- **Result:** MaxDD = -97.5% without regime; with regime: Sharpe ≈ 0.43. Selects contrarian/reversion picks during GFC (GNW, MBI in Sept 2008) that then crash further.
- **Root cause:** pred column appears to be trained on look-forward labels with high false-positive rate for distressed securities.
- **Dead end until:** reformulation with entirely different target definition.

## 2. Min-Variance Portfolio Weighting
- **Tried:** exp_003 — K=20,30 with scipy min-var optimization
- **Result:** Same Sharpe (~1.63) as inv_vol weighting, more computation.
- **Root cause:** Covariance estimation is noisy with 1-year daily lookback on monthly-held K stocks. No benefit over simpler inv_vol.
- **Dead end unless:** Robust covariance estimator (Ledoit-Wolf shrinkage, factor model) is used.

## 3. Triple / Dual Regime (Complex Regime Gates)
- **Tried:** exp_003 — make_dual_regime(dsma=-0.02, mom6=-0.10) and make_triple_regime()
- **Result:** Same Sharpe as simple 200-day MA; sometimes worse CAGR due to more cash months.
- **Root cause:** Adding mom6 and vol conditions doesn't improve timing; 200-day MA already captures the key regime shifts.
- **Dead end unless:** Using a fundamentally different signal (VIX, credit spreads) rather than more SPY price conditions.

## 4. Sharpe-Target Model (exp_004)
- **Tried:** Training LightGBM with Sharpe_target label (Sharpe ≥ target threshold as binary class)
- **Result:** Sharpe 1.45–1.52 vs 1.63 baseline. Worse than regression on ranked returns.
- **Root cause:** Noisy binary label; threshold is arbitrary and leads to class imbalance. Cross-sectional rank regression works better.
- **Dead end:** Sharpe-binary classification is strictly dominated.

## 5. Very Large K (K > 100) Without Vol Targeting
- **Tried:** exp_005 — K=100,120,150 with inv_vol + regime
- **Result:** CAGR drops below 40% (fails CAGR gate), Sharpe also falls.
- **Root cause:** At K>100, stock selection signal is diluted by noise picks. Diversification benefit is exhausted by ~K=60-80.
- **Dead end unless:** Combined with vol targeting to allow higher expected Sharpe at lower portfolio vol.

## 6. IC-Based Feature Weighting
- **Tried:** exp_004 IC analysis — computing Spearman IC for each signal
- **Result:** IC uniformly near-zero; pct_positive = 42-47% for all signals.
- **Root cause:** Forward 1m returns are extremely noisy. Monthly IC of even the best signals is near zero.
- **Dead end:** Cannot improve prediction accuracy by re-weighting features based on IC scores. Need ensemble diversity, not better feature weights.

## 7. EW Weighting for Large K
- **Tried:** exp_001/002 — equal-weight at K=20,30,40
- **Result:** Consistently worse Sharpe than inv_vol by ~0.1-0.2.
- **Root cause:** Small-cap and high-vol picks are equally weighted, adding noise; inv_vol naturally tilts toward larger, lower-vol stocks.
- **Dead end:** EW is dominated by inv_vol for this universe.
