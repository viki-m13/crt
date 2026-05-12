# Quant Research — US Equity Stock-Picking

## Mission
Develop a monthly-rebalance long-only US equity strategy targeting:
- CAGR ≥ 50% (OOS walk-forward)
- Sharpe ≥ 2.0 (annualized from monthly returns)

## Current Status
**PAUSED FOR GOAL CLARIFICATION** — See STATE.md.

After 224 hypotheses across 6 sessions, the dual target appears structurally
unreachable with price-only monthly long-only equities (see STATE.md for analysis).

## Data
- Universe: S&P 500 PIT membership (v2, ~450-600 stocks per month)
- Prices: extended monthly panel 1995-2026 (1833 tickers)
- Features: 47 price-derived features (momentum, vol, trend, pattern)
- ML predictions: pred_1m, pred_3m, pred_6m, pred_12m (v3 Chronos/GBM)
- Research window: 2003-09 → 2024-04 (248 months)
- Lockbox: 2024-05 → 2026-04 (24 months, NEVER touched)

## Best Result So Far
- Donchian130 filter, K=3, hold=6: **CAGR=48.73%, Sharpe=1.00, MDD=-57%**
- Beats v5 Chronos 30% baseline by 18.7 percentage points

## Key Codebase
- Backtest harness: `strategy/YLOka/harness.py`
- Pre-computed features: `data/YLOka/pit_panel_full.parquet`
- Backtest runs: `backtests/YLOka/runs/`
- Session 6 experiments: `quant_research/experiments/`
