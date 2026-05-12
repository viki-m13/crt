# Dead Ends — Do Not Repeat

## DE-001: d_sma50/rsi_14 as monthly stock rankers
**Tried**: exp_001, exp_002 (RUN 1)
**Result**: CAGR 3-10%, Sharpe 0.3-0.5 (with correct date arithmetic)
**Why it failed**: These features have NEGATIVE IC (-0.11 to -0.16 IR) vs next-month returns when computed correctly. They predict short-term continuation (1-2 days) very well but mean-reversion at 1-month horizon. The originally-computed high IC (0.20) was an artifact of a date alignment bug where MonthEnd(1) from a non-month-end asof date produced a 1-2 day return window.
**Lesson**: Always verify IC with correct temporal alignment before building a strategy.

## DE-002: Composite of d_sma50+rsi_14+crt_3m+breakout_strength_60+mom_3
**Tried**: exp_002 rungs R4-R8 (RUN 1)
**Result**: CAGR 5-11%, Sharpe 0.5-0.8 (correct date arithmetic)
**Why it failed**: Same root cause as DE-001 — all constituent features have negative IC vs next-month return.

## DE-003: GBM pred with truly monthly rebalancing (h=1)
**Tried**: exp_003, exp_004 D-sweep (RUN 1)
**Result**: CAGR 15-17%, Sharpe 0.60 for K=3-7; CAGR 39.9% for K=1 (single stock)
**Why**: The existing GBM pred was trained on 3-6m targets. Monthly IC is low (0.035). Monthly turnover costs destroy much of the alpha. K=1 works because single-stock selection occasionally picks monster winners, but it's extremely volatile.
**Do not retry** without fundamentally different monthly-IC signal (IC > 0.10).

## DE-004: Strict regime gate (SPY below 200-day SMA at all = cash)
**Tried**: exp_004 B-section (RUN 1)
**Originally inflated result**: CAGR 49-84%, Sharpe 1.34-1.47 (LEAKAGE!)
**True result (no leakage)**: CAGR 21-28%, Sharpe 0.79-0.83
**Root cause**: Regime check was using NEXT month's SPY state (future data). Specifically, `_next_mr_date(asof - 1day)` returned next calendar month-end, not current.
**Corrected behavior**: Going to cash whenever SPY is below its 200-day SMA hurts CAGR (too much cash) without meaningful Sharpe improvement.

## DE-005: pred_12m with hold=6 (K=3+)
**Tried**: exp_003, exp_004 C-section (RUN 1)  
**Result**: CAGR 23-34%, Sharpe 0.79-0.89 — WORSE than pred (blend)
**Why**: pred_12m has higher IC vs 1-year return but lower IC vs 3-6m holding returns. The pred blend (1m+3m+6m) better matches the 6-month holding period.

## DE-006: YLOka-style GBM specialist models (session 5 of YLOka)
**Result**: UNDERPERFORMED by 7-14pp CAGR vs v3 baseline.
**Documented in**: research/YLOka/exp_summary_session5.md
**Do not retry** without fundamentally different regime detection.

## DE-007: Feature-based scorers replacing GBM (YLOka session 3)
**Result**: -13.6pp vs baseline for best scorer.
**Documented in**: research/YLOka/graveyard/Session3_feature_scorers.md
