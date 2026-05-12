# Dead Ends

## From prior sessions (YLOka backtest experiments)

### 1. Conviction-spread weighting (softmax)
- Tried: conv_lambda=0.5, 1.0, etc.
- Result: CAGR 39.5%, Sharpe 0.87 — worse than EW baseline
- Reason: top predictions not reliably better than 2nd/3rd; added variance

### 2. Classifier hard filter (pred_12m_cls < 0.18/0.25)
- Tried: cls_filter_018, cls_filter_025
- Result: 40.7-37.8% CAGR, Sharpe 0.94-0.90 — marginal at best
- Reason: 12m classifier doesn't improve on 3m+6m ensemble enough

### 3. Dynamic hold (exit when score drops below quantile)
- Tried: dynamic_hold=True, quantile=0.85/0.75/0.90
- Result: hold3 CAGR=31.5%, hold4 CAGR=21.7%, hold9 CAGR=32.1%
- Reason: shorter holds increase turnover cost without proportionate return gain; CAGR hurt

### 4. 12m ensemble heads (ens_3_6_12)
- Tried: various ensemble weights including 3m+6m+12m
- Result: 35-37% CAGR, 0.88-0.92 Sharpe — worse than baseline
- Reason: 12m head covers too many events that happen outside portfolio window

### 5. Regime-specialist models (bull/normal/recovery specialists)
- Tried: specialist_router, specialist_blend_03/05/07, specialist_rank_avg
- Result: 30-43% CAGR, 0.85-0.98 Sharpe — inconsistent, mostly worse
- Reason: regime specialists overfit to regime labels; routing errors compound

### 6. Dispersion-conditional K
- Tried: K_high_dispersion=3, K_low_dispersion=5 at various thresholds
- Result: baseline-equivalent (40.78% CAGR, 0.95 Sharpe)
- Reason: dispersion signal too noisy at monthly frequency

### 7. Soft-cash overlay (continuous de-risking)
- Tried: soft_cash=True
- Result: 40.85% CAGR, 0.95 Sharpe — marginally different
- Reason: smooth de-risking adds overhead without clear improvement vs binary gate

### 8. K=1 concentration
- Tried: K=1 with various hold periods
- Result: 19-32% CAGR, 0.59-0.73 Sharpe — always worse
- Reason: idiosyncratic risk too high, single blowups are fatal

### 9. Adaptive IC weighting
- Tried: softmax/proportional weighting by rolling IC
- Result: baseline-equivalent (40.78% CAGR, 0.95 Sharpe)
- Reason: IC signal too noisy for reliable head selection at monthly freq

### 10. Pre-runner footprint composite
- Tried: runner_footprint, ml_plus_runner, ml_plus_runner_strong/weak
- Result: 30-35% CAGR, 0.82-0.87 Sharpe — worse than baseline
- Reason: pre-runner features (cst_score, rbi, vov) don't reliably predict next-month returns

## Summary of current best
- **Best result**: exp_05_donchian — 48.73% CAGR, 1.00 Sharpe, K=3, hold=6, Donchian130 filter
- **Baseline**: 40.78% CAGR, 0.95 Sharpe, K=3, hold=6, ml_3plus6 scorer
- **Gap to target**: Need ≥50% CAGR AND ≥2.0 Sharpe. Sharpe is 2x off.
