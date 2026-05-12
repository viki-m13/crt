# Quant Research — US Equity Stock Picking Strategy

**Mission**: Develop a US equity long-only monthly strategy achieving CAGR ≥50% and Sharpe ≥2.0 OOS.

## Quick Status

See `STATE.md` for current best results and next steps.

## Data Sources

- **Prices**: `experiments/monthly_dca/cache/prices_extended.parquet` — daily adjusted close, 1995-2026, 1833 tickers
- **Feature panel**: `data/YLOka/pit_panel_full.parquet` — monthly snapshots, 587 tickers, 2003-2025, 47 features including GBM predictions
- **Monthly returns**: `experiments/monthly_dca/cache/v2/monthly_returns_clean.parquet` — calendar month-end returns, 1833 tickers

## Key Design Decisions

- **Universe**: SPX-adjacent (587 tickers from existing panel). No true PIT membership.
- **Frequency**: Monthly or semi-annual rebalancing (YLOka-compatible).
- **Lockbox**: 2024-01-31 → 2025-12-31 (last 24 months of panel). NEVER TOUCH.
- **Research window**: 2003-09-30 → 2023-12-31 (244 months).

## Critical Lessons Learned (Run 1)

1. `pd.offsets.MonthEnd(1)` from a non-month-end date gives the SAME month's end (1-2 day return). Always use `next_month_end(asof)` instead.
2. Regime gate must use CURRENT calendar month-end SPY features (`asof + MonthEnd(0)`), not next month's.
3. IC analysis must use correctly-aligned next-month returns or it will be misleading.
4. d_sma50, rsi_14 have NEGATIVE IC vs next-month returns (they predict reversal, not continuation).
5. GBM pred (trained on 3-6m targets) has IC=0.035 vs next-1m return. Good for 6-month holding.

## Directory Layout

```
quant_research/
  STATE.md                   # Current status — updated every run
  README.md                  # This file
  state/
    journal.jsonl            # Append-only action log
    hypotheses_tested.jsonl  # Global hypothesis count for DSR
    lockbox_log.jsonl        # Lockbox touches — sacred
    ideas_backlog.md         # Ranked ideas for next runs
    dead_ends.md             # Failures with root causes
    current_focus.md         # What to do next run
  backtest/
    engine.py                # Monthly rebalancing engine (v1, use hold_engine for h>1)
    hold_engine.py           # Variable hold period engine (matches YLOka harness)
    metrics.py               # Sharpe, CAGR, DSR, block bootstrap
  experiments/
    exp_001_phase2_baselines/  # Phase 2 with buggy features (historical reference)
    exp_002_correct_baselines/ # Phase 2 with corrected date arithmetic
    exp_003_hold_period_baselines/ # Hold period sweep with GBM signals
    exp_004_improve_cagr_sharpe/  # Attempts to push toward gates
  notify/
    send_success.sh          # Email + local notification on WINNER
```
