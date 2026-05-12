# Ideas Backlog

Ranked by expected Sharpe improvement × implementation effort (high value, low effort first).

## Tier 1 — High Confidence, Try Next

### 1. Volatility Targeting (exp_006)
Scale position size inversely with realized vol to target a fixed portfolio vol (e.g., 15% ann).
- Mechanism: negative vol-return correlation in equities means you scale down in bad times
- Implementation: `scale = min(target_vol / spy_vol_21d, 1.0)` applied to monthly return
- Expected Sharpe gain: 0.1–0.3 (empirical from academic literature on vol timing)
- Status: PENDING

### 2. Cross-Sector Momentum Rotation (exp_007)
Select top-3 sectors by 6m relative strength vs SPY, then top-K stocks within those sectors.
- Mechanism: sector-level momentum has IC > stock-level momentum (Graham-Harvey 2022)
- Implementation: compute sector rs_6m_spy, filter stocks to top-sector tickers before scoring
- Expected Sharpe gain: 0.1–0.2

### 3. Alternative Regime Signal: VIX or Credit Spread (exp_008)
Replace SPY 200-day MA gate with VIX level or IG credit spread threshold.
- VIX < 20: full invest; VIX 20-30: scale 0.5; VIX > 30: cash
- Credit spread: proxy via TLT/HYG ratio momentum
- Motivation: price-MA is lagging; vol-based regimes are more forward-looking
- Expected Sharpe gain: 0.05–0.15

### 4. Large-Cap Only Universe (exp_009)
Restrict to stocks with market cap implied by vol and price > $10B equivalent.
- Proxy: filter to stocks with vol_1y < 25% (large-caps are less volatile)
- Motivation: reduces survivorship bias, but may reduce CAGR
- Expected Sharpe gain: uncertain — may be honest correction downward

## Tier 2 — Medium Confidence

### 5. Quarterly Rebalance
Monthly rebalance costs 10bp/year in transactions. Quarterly drops cost to ~3.3bp/year.
- Expected vol effect: similar, turnover cost savings ~7bp/year, marginal
- Risk: less reactive to regime changes

### 6. Different LightGBM Objective
Switch from regression on ranked returns to binary classification: top quintile vs bottom.
- Intuition: cleaner labels, better calibrated probabilities
- Expected Sharpe gain: unclear

### 7. Longer Training Window (72m vs 48m)
More training data → better generalization, but model may be stale at end.
- Expected Sharpe gain: 0.0–0.1

### 8. Factor Timing
In bull regime (d_sma200 > 0.05): use momentum. In choppy regime: use quality/Sharpe.
- Two-factor blend with regime-conditional weights
- Expected Sharpe gain: 0.1–0.2

### 9. Momentum Reversal Filter
After a stock is up >100% in 12m, reduce weight (crowding proxy).
- Expected Sharpe gain: small, reduces crowding risk

### 10. Portfolio-Level Realized Vol Targeting
Instead of SPY vol as proxy, compute realized vol of actual portfolio from last 6m.
- Cleaner match between scale signal and actual portfolio vol
- Implementation complexity: medium (need to track portfolio returns history)

## Tier 3 — Speculative

### 11. Long/Short Market Neutral
Hedge market beta by shorting SPY proportional to portfolio beta.
- Problem: adds borrowing cost, still long-only for stock selection
- Likely reduces CAGR below 50% gate

### 12. Alternative Universe: NDX vs SPX
Switch from full 1833-ticker universe to NDX 100 only.
- NDX has higher growth tilt, may improve Sharpe via tech momentum
- Risk: 2000-2002 drawdown would be catastrophic (but OOS starts 2007)

### 13. Stop-Loss at Stock Level
Remove any position that falls >15% in a single month.
- Implementation: look ahead into monthly prices during rebalance
- Expected benefit: marginal in monthly context

### 14. Ensemble of Models
Average rankings from 3 different LightGBM seeds or hyperparameter sets.
- Expected Sharpe gain: variance reduction, small improvement

### 15. Time-Series Momentum (TSMOM)
Size position by sign and magnitude of own 12m return.
- Well-studied, but already proxied by mom_12_1 filter in lgbm_smooth
- Standalone TSMOM: expected Sharpe ~0.8-1.0 (not sufficient alone)
