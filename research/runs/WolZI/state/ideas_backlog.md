# Ideas Backlog — Ranked by Expected Value

Updated: 2026-05-11 (Run 1)

## Honest Assessment
Best achieved so far: CAGR 39.9%, Sharpe 0.89 (both well below 50%/2.0 gates).
The gap is large. Closing it requires either higher IC (new data/models) or better risk management.

---

## TIER 1 — Highest Expected Impact

**1. Walk-forward GBM retrained on 1m forward returns**
- The existing `pred` column was trained on 3m/6m targets. Retrain directly on 1m returns.
- Use 50-fold walk-forward with purged-embargoed CV (YLOka style but targeting 1m).
- Features: all 47 in pit_panel_full + cross-sectional ranks.
- Estimated IC improvement: 0.035 → 0.05 (possible). CAGR uplift: +10-15pp.
- Cost: ~30 min training per fold, ~15 hours total. Will need script caching.
- Status: NOT TRIED

**2. LSTM sequence model on monthly feature windows**
- Input: last 24 months of 47-feature rows per stock (sequences of length 24).
- Output: rank-score for next month's return.
- Training: expanding window walk-forward (not fitting within 1 hour).
- Expected IC: unclear, possibly 0.04-0.06 vs 1m return.
- Cost: high (GPU needed, 1-2h per fold). Need to check if torch available.
- Status: NOT TRIED

**3. Cross-sectional feature interactions (OLS or Ridge on ranks)**
- Input: 47 features × cross-sectional percentile ranks.
- Add interactions: d_sma50 × rsi_14, mom_3 × vol_12m, pred × trend_health_5y.
- Train expanding-window Ridge (low param count), refit monthly.
- Expected IC: possibly 0.04-0.05 vs 1m return.
- Cost: cheap (Ridge on 587 stocks × 47+47² interactions).
- Status: NOT TRIED (high priority for next run)

**4. Chronos time-series foundation model as feature generator**
- Use v5 Chronos to generate per-stock next-month probability predictions.
- Apply to the price series of each of the 587 stocks.
- Use Chronos output as an additional feature in the GBM composite.
- Status: NOT TRIED. Need to check if Chronos is installed.

**5. Better regime gate using cross-sectional market breadth**
- Current gate: SPY 200-day SMA + 1m return (simple, works).
- Better gate: add cross-sectional breadth (what % of stocks are above their 200 SMA).
- Also try: VIX-equivalent realized vol of SPY as crash predictor.
- When to go cash: breadth < 40% OR SPY vol > 30% annualized OR both.
- Status: NOT TRIED

---

## TIER 2 — Moderate Expected Impact

**6. Feature importance-weighted composite score**
- Current composite: equal-weight rank of 5 features.
- Better: weight by rolling IC of each feature vs next-month return.
- Adaptive weighting that shifts as regimes change.
- Status: NOT TRIED

**7. Asymmetric loss GBM (penalize false positives on losers)**
- Train GBM with higher penalty for selecting stocks that LOSE >20% in next month.
- Focus on extreme downside avoidance rather than return prediction.
- Status: NOT TRIED

**8. Meta-labeling approach**
- Primary model: existing GBM score selects top-K candidates.
- Secondary model: binary classifier "will this stock be in top-decile next month?"
- Only invest in candidates where secondary classifier says YES.
- Status: NOT TRIED

**9. Combinatorial purged k-fold CV for better GBM hyperparams**
- Current GBM (YLOka) uses walk-forward with fixed params.
- Use CPCV to find better num_leaves, num_iterations, min_child_samples.
- Status: NOT TRIED

**10. Sector-relative momentum scoring**
- Rank stocks WITHIN sector (GICS) rather than cross-sectionally.
- Then pick best sector(s) and top names within.
- Requires GICS mapping (might be available from existing panel).
- Status: NOT TRIED

---

## TIER 3 — Speculative but Creative

**11. Volatility-targeted position sizing**
- Scale equity exposure by target_vol / rolling_vol (last 3 months).
- When stock vol is high, reduce position size.
- Directly improves Sharpe by stabilizing monthly vol.
- Status: NOT TRIED. Easy to implement.

**12. Options-inspired ranking: implied momentum**
- Approximate market's expected vol using realized vol skewness.
- Use vol_asym_60 and vol_asym_126 (both in panel) as option-analog features.
- Status: PARTIALLY TRIED (IC analysis shows IR=-0.06 for vol_asym — not useful)

**13. Graph neural network on sector correlations**
- Build monthly correlation graph of stocks. GNN aggregates neighbor info.
- Requires PyTorch Geometric. Expensive to train. Probably overkill.
- Status: NOT TRIED

**14. Bayesian model averaging across top configurations**
- Weight predictions from top-5 models by their rolling OOS IC.
- Reduces sensitivity to any single model.
- Status: NOT TRIED

**15. Selection-aware ensembling (model that won trailing OOS)**
- Each month, select the model that had best IC in last 24 months.
- This is meta-learning over models, not stocks.
- Status: NOT TRIED

**16. Triple-barrier labels for GBM training**
- Instead of raw forward return, train on: hit +20% (upper), hit -20% (lower), or timeout.
- Focuses model on large moves rather than average return.
- Status: NOT TRIED

**17. Fractional differentiation of price series**
- Apply fractional differencing to preserve memory while achieving stationarity.
- Use as features feeding the GBM.
- Status: NOT TRIED

**18. Alternative universe: NDX instead of SPX**
- NDX has higher momentum (tech-heavy). More growth stocks.
- Risk: more volatile, fewer names.
- Status: NOT TRIED (would require building new data pipeline)

**19. Dynamic K based on signal confidence**
- When average score of top-K is very high: use K=1 or K=2.
- When signal is weak: use K=10 or K=15.
- Based on cross-sectional spread of scores.
- Status: NOT TRIED

**20. Cash position momentum: invest in T-bills vs equity based on relative momentum**
- Add Treasury momentum signal: if SPY-12m < T-bill-12m, hold T-bills.
- This is GTAA-style absolute momentum filter.
- Status: NOT TRIED

---

## Discarded (Do Not Revisit)

- d_sma50/rsi_14 as monthly rankers (negative IC vs next-month returns after date fix)
- GBM predictions with monthly rebalancing (IC 0.035 → max ~15% CAGR)
- Strict crash gate (was leakage-inflated; clean version no better than standard gate)
- Any feature from the buggy IC analysis (pre-date-fix results unreliable)
