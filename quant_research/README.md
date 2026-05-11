# Autonomous Quant Research — US Equity Stock-Picking Strategy

**Mission**: Develop a monthly-rebalance long-only US equity strategy achieving:
- CAGR ≥ 50% on walk-forward OOS (≥10 years)
- Sharpe ≥ 2.0 (annualized from monthly returns)
- Pass full validation gauntlet (see CLAUDE.md)

**Universe**: S&P 500 (SPX), PIT membership
**Data**: Price-only (prices_extended.parquet, 1995-2026, 1833 tickers)
**Lockbox**: 2024-05-31 to present (SEALED)

## Current Status

See `STATE.md` for full status.

Best known: v5_chr_p70_q0.45_k3_invvol — WF mean CAGR 47.2%, Sharpe 1.06
Target: CAGR ≥ 50%, Sharpe ≥ 2.0

## Layout

```
quant_research/
  STATE.md                      # current status (updated every session)
  README.md                     # this file
  state/
    journal.jsonl               # append-only action log
    hypotheses_tested.jsonl     # global hyperparam count for DSR
    lockbox_log.jsonl           # lockbox touches (sacred)
    ideas_backlog.md            # 22 ranked ideas
    dead_ends.md                # prior failed approaches (88+ from YLOka)
    current_focus.md            # current session plan
    data_integrity_report.md    # data sources and PIT checks
  data/
    cache/                      # feature caches (keyed by content hash)
  features/
    momentum.py                 # PIT momentum features
    leakage_audit.py            # automated leakage checker
  models/                       # model implementations (to be added)
  backtest/
    engine.py                   # walk-forward backtest engine
  experiments/
    exp_000_baseline_ladder/    # Phase 2 baseline (5 rungs + GBM ref)
  candidates/                   # candidates passing initial gates
  WINNER/                       # populated on full gauntlet pass
  notify/
    send_success.sh             # email notification on success
```

## Prior Research (YLOka Sessions 1-5)

88+ experiments on price-only features, all documented in `state/dead_ends.md`.
Key finding: The v3 GBM (LightGBM on 47 features, cross-sectional, monthly) is the
local optimum for price-only SPX. Adding Chronos zero-shot foundation model predictions
as a filter adds ~3-5pp CAGR (v5 result).

The path to Sharpe 2.0 requires either fundamentals or a strong meta-labeling layer.
