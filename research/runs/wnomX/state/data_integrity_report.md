# Data Integrity Report

**Date**: 2026-05-11 (Session 1 Bootstrap)

## Data Sources Located

| File | Path | Shape | Date Range | Status |
|---|---|---|---|---|
| Daily prices (adj close) | experiments/monthly_dca/cache/prices_extended.parquet | (8133, 1833) | 1995-01-03 to 2026-05-07 | ✅ |
| SPX PIT membership | experiments/monthly_dca/cache/v2/sp500_pit/sp500_membership_monthly.parquet | (140011, 2) | 2003-01-31 to 2026-04-30 | ✅ |
| GBM predictions (v2) | experiments/monthly_dca/cache/v2/ml_preds_v2.parquet | (372218, 7) | 2003-09-30 to 2025-12-31 | ✅ |
| PIT feature panel | data/YLOka/pit_panel_full.parquet | (98059, 47) | 2003-09-30 to 2025-12-31 | ✅ |
| Chronos NDX preds | experiments/monthly_dca/v5/qqq_pit/ml_preds_chronos_ndx.parquet | (35375, 3) | 1997-01-31 to 2026-05-07 | ✅ (NDX only) |
| NDX PIT membership | experiments/monthly_dca/v5/qqq_pit/ndx_pit_membership_monthly.parquet | (12132, 2) | 2015-01-31 to 2026-05-31 | ⚠️ Only back to 2015 |
| NDX GBM predictions | experiments/monthly_dca/v5/qqq_pit/ml_preds_v2_ndx.parquet | (32966, 7) | 2003-09-30 to 2025-12-31 | ✅ |
| Regime labels | data/YLOka/regime_labels.parquet | - | - | ✅ |

## PIT Membership Spot Checks

**SPX membership spot check** (verified from known S&P 500 historical changes):
- Apple (AAPL): present from 2003-01-31 onward ✅
- Google (GOOG): would join in 2004 — need to verify
- Enron (ENRN): NOT in the 1833-ticker universe (delisted 2001) ⚠️ Survivorship gap

**Known limitation**: The 1833-ticker universe has only 9 truly delisted tickers. This represents
mild survivorship bias. The SPX PIT membership (sp500_membership_monthly.parquet) is more
reliable as a PIT-accurate universe — it tracks actual S&P 500 additions/deletions.

## Price Data Spot Checks

**AAPL 4:1 split August 2020**: 
- Price should drop ~75% from end of July to start of August 2020
- Need to verify adjusted prices show continuity (no step change)
- Status: ⚠️ Not verified manually (use price series continuity check)

**NVDA 4:1 split July 2021**: Similar check needed
**TSLA 5:1 split August 2020**: Similar check needed

## Feature Leakage Assessment

**pit_panel_full.parquet**: Features computed by prior YLOka sessions with explicit
PIT awareness (60-day report lag for fundamentals, monthly resample for price features).
Verified in YLOka research/01_engine_audit.md as PIT-correct.

**Forward returns (fwd_1m_ret in ml_preds_v2.parquet)**: These ARE forward returns
(price at t+1 month / price at t - 1). They are used as backtest evaluation, not as
features. No leakage concern.

**Key leakage guard**: The lockbox (2024-05 to 2026-05) has NEVER been queried.
The features in pit_panel_full only go to 2025-12-31, but we restrict all experiments
to RESEARCH_END = 2024-04-30.

## Assumptions Documented

1. **Universe**: S&P 500 (SPX) via sp500_membership_monthly.parquet (NOT broader 1833-ticker universe)
   - Justification: SPX has PIT data from 2003, giving 15+ years of OOS data (>10 year requirement)
   - NDX rejected: PIT data only from 2015 (barely 10 years, no margin)

2. **Cost model**: 5 bps one-way (round-trip 10 bps). No additional price impact modeled for K=3-5
   positions in large-cap SPX stocks. This is conservative for SPX names.

3. **Execution**: Executed at next-month open after month-end rebalance signal. Cost accounts
   for this by applying cost at rebalance date.

4. **Lookbox period**: 2024-05-31 to present. No algorithm in this project queries prices or
   features after 2024-04-30 (RESEARCH_END) during research phase.

5. **Split adjustments**: prices_extended.parquet uses Yahoo Finance adjusted closes. Splits
   and dividends are properly adjusted. This is the same data used in all prior YLOka sessions.

## Questions for V (open)

1. Are there any fundamental (earnings/profitability/valuation) data sources available?
   The current research is price-only, and adding quality fundamentals could provide
   genuinely new information to move Sharpe significantly.

2. Is Chronos-bolt-tiny available in the Python environment? Need to re-run Chronos
   inference on full SPX universe for Idea 03/13. The prior compute used it.

3. Is the 1833-ticker broader universe acceptable for the final strategy if properly
   documented? v5 report shows ~52% WF mean CAGR but survivorship bias is a concern.
