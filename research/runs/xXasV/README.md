# Quantitative Research — US Equity Stock Picker

**Mission**: Develop a long-only monthly-rebalance US equity strategy achieving
CAGR ≥ 50%, Sharpe ≥ 2.0 on strict walk-forward OOS.

**Universe**: S&P 500 (PIT membership)
**Frequency**: Monthly rebalance at month-end close
**Style**: Long-only, no leverage, DCA-compatible

## Quick Status

See `STATE.md` for current best metrics and next steps.

## Directory Layout

```
quant_research/
  STATE.md              ← current status (updated every session)
  README.md             ← this file
  state/
    journal.jsonl       ← append-only action log
    hypotheses_tested.jsonl  ← global DSR deflation counter
    lockbox_log.jsonl   ← lockbox touches (sacred)
    ideas_backlog.md    ← ranked ideas
    dead_ends.md        ← failed ideas with reasoning
    current_focus.md    ← this session's plan
  data/
    cache/              ← feature caches (keyed by content hash)
  backtest/
    engine.py           ← core backtest engine (validated vs YLOka v3)
    baseline_ladder.py  ← Phase 2 rungs
    data_integrity.py   ← integrity checks (20/20 pass)
  features/             ← feature engineering modules
  models/               ← model implementations
  experiments/
    exp_001_baseline_ladder/
    exp_002_vol_targeting/
  candidates/           ← promoted models awaiting gauntlet
  WINNER/               ← populated only on gauntlet pass
  notify/
    send_success.sh
```

## Key Data Paths

- PIT predictions: `data/YLOka/pit_panel_with_scores.parquet`
- Monthly returns: `experiments/monthly_dca/cache/v2/monthly_returns_clean.parquet`
- Feature snapshots: `experiments/monthly_dca/cache/features/YYYY-MM-DD.parquet`
- Daily prices: `experiments/monthly_dca/cache/prices_extended.parquet`

## Validated Baseline

`backtest/engine.py` reproduces YLOka v3 to within 0.1pp CAGR.
v3 benchmark: **40.7% CAGR, 0.863 Sharpe, -49.5% MaxDD** (248 months OOS WF).
