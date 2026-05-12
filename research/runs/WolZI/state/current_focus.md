# Current Focus — Updated 2026-05-11 (Run 1 Complete)

## Status
Run 1 complete. Foundation built. Phase 2 baselines done. Initial Phase 3 explored.

## Next Run Plan (Run 2)

### Priority 1: Walk-forward GBM trained on 1m forward returns
**Goal**: Build a new GBM that explicitly targets 1-month forward cross-sectional rank, not 3-6m returns.
**Why**: The existing `pred` column has IC 0.035 vs 1m return. A model trained directly on 1m returns might achieve IC 0.05+. Each +0.01 IC improvement ≈ +3-5pp CAGR.
**Implementation plan**:
1. Load `monthly_returns_clean.parquet` → compute 1m forward return for each (asof, ticker) pair
2. Align with pit_panel_full features (47 features)
3. Run walk-forward GBM training: expanding window, 13-month embargo, refitting annually
4. Evaluate cross-sectional IC of new predictions vs next-month return
5. Run backtest with hold_engine using new predictions

**Expected time**: 45 minutes of compute.

### Priority 2: Feature interaction Ridge regression
**Goal**: Create a better composite score using interactions between features.
**Why**: Simple equal-weight ranks ignore interactions. Ridge on ranks + interactions is cheap and often boosts IC by 20-30%.
**Implementation**: 
1. At each asof, compute cross-sectional percentile ranks of ALL 47 features
2. Add top interaction pairs: pred × d_sma200, pred_12m × mom_3, etc.
3. Train expanding-window Ridge(alpha=100) to predict 1m forward return rank
4. Use as ranking signal

**Expected time**: 15 minutes of compute.

### Priority 3: Investigate why YLOka claims 40% CAGR vs my 27.9%
**Goal**: Understand the gap and identify if there's genuine improvement or if YLOka's backtest has issues.
**Approach**: Run the YLOka harness directly and compare outputs month-by-month.

## Time Budget for Run 2
- 5 min: Load state, review
- 20 min: Priority 1 (new GBM on 1m returns) — implement + run
- 15 min: Priority 2 (Ridge regression composite) — implement + run
- 10 min: Priority 3 (YLOka gap investigation)
- 10 min: Journal + STATE.md update + commit
