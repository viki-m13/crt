# Quant Research — US Equity Stock-Picking Strategy

**Goal:** Long-only monthly US equity strategy with CAGR ≥ 50% and Sharpe ≥ 2.0 (OOS walk-forward).

## Quick Start

```bash
cd /home/user/crt

# Verify engine parity with v3 ground truth
python3 quant_research/backtest/engine.py  # (no main block, use as module)

# Run baseline ladder
python3 quant_research/experiments/exp_001_baseline_ladder/run.py

# Leakage audit
python3 quant_research/features/leakage_auditor.py
```

## Data Sources

| File | Description |
|---|---|
| `experiments/monthly_dca/cache/v2/ml_preds_v2.parquet` | v3 ML predictions (2003-09→2025-12, ~372K rows) |
| `experiments/monthly_dca/cache/v2/monthly_returns_clean.parquet` | Monthly returns, ME-indexed, 1833 tickers |
| `experiments/monthly_dca/cache/v2/sp500_pit/sp500_membership_monthly.parquet` | PIT S&P 500 membership |
| `experiments/monthly_dca/cache/features/*.parquet` | 79 features per stock per month-end (BME dates) |
| `data/YLOka/pit_panel_with_scores.parquet` | Alternative ML predictions (different model) |
| `data/YLOka/regime_labels.parquet` | Monthly bull/crash/recovery/normal labels |
| `data/YLOka/rolling_ic.parquet` | Rolling information coefficient by horizon |

## Architecture

```
quant_research/
  backtest/engine.py       # Core WF simulator (verified v3 parity: 39.4% vs 39.8%)
  features/ml_scorer.py    # Score functions wrapping ml_preds_v2 and YLOka
  features/leakage_auditor.py  # Feature leakage audit harness
  experiments/             # Numbered experiments (exp_NNN_slug/)
  state/                   # Journal, hypotheses count, lockbox log, ideas, dead ends
  STATE.md                 # Current status (updated every run)
```

## Benchmark

| Strategy | CAGR | Sharpe | Source |
|---|---|---|---|
| v3 ML baseline | 39.8% | 0.955 | v6/run_baseline.py (ground truth) |
| v5 + Chronos | 44.8% | 1.04 | experiments/monthly_dca/v5/ |
| Our engine (v3 equivalent) | 39.4% | 0.949 | backtest/engine.py parity check |

## Constraints (Non-Negotiable)
- Long-only, no leverage, no derivatives
- Monthly rebalance at month-end
- SPX universe (PIT membership)
- OOS walk-forward only — no tuning on WF set
- Lockbox: 2024-02 to 2026-05 (SEALED, 0 touches so far)
