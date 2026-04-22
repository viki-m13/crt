# Step 39 FINAL: Signal Smoothing — Production Recommendation

**Date:** 2026-04-21
**Covers:** step39, step39b, step39c, step39d, step39e, step39f, step39g, step39h
**Status:** ADOPT SMA 12M smoothing as new CAP5 default (pending 128-ticker retest)

## One-line recommendation

**Change CAP5's monthly ranking to use the trailing 12-month average of
the `final` conviction score (vs current single-month reading).**

Delta: +0.90pp 20Y CAGR (17.41% → 18.31%), wins 5/5 rolling 10Y
windows vs SPY and baseline, jackknife-stable, zero extra capital,
minimal production change (one integer config parameter).

Code change is already landed in bt_core.py:
```python
cfg = StrategyConfig(..., smoothing_months=12)
```

## Validation summary (step36 battery, 97-ticker 20Y, 2006–2026)

| Variant | 20Y CAGR | Sharpe | MaxDD | 10Y median vs SPY | Annual wins vs SPY | GFC CAGR | Jackknife median |
|---|---|---|---|---|---|---|---|
| Baseline (current) | +17.41% | 1.34 | -46.15% | +9.51pp | 18/21 | +37.96% | 0.00pp |
| **SMA 12M (RECOMMEND)** | **+18.31%** | 1.35 | -47.51% | **+10.92pp** | 16/21 | +38.78% | 0.00pp |
| SMA 24M (aggressive) | +18.60% | 1.36 | -48.02% | +11.57pp | 16/21 | +39.21% | 0.00pp |
| SMA 60M (max) | +18.75% | 1.36 | -48.12% | — | — | — | — |

## Path to this conclusion

### step39 — initial finding
- Trailing 6M SMA of `final` improves CAGR +0.70pp, wins 10/10 rolling 10Y
- Monotonic improvement as window grows (2M→12M tested)
- Robust across GFC, bull market, post-COVID regimes

### step39c — sensitivity
- Works at every tested `top_n × cap` combination (3/5/7 × 3/5/7/10%)
- EMA loses to SMA — **older history carries real information**
- Hybrid (current + SMA blend) underperforms pure SMA

### step39d — long windows
- CAGR rises monotonically through SMA 60M (+1.34pp), 180M (+1.56pp)
- No catastrophic breakdown even at 15-year smoothing
- Rolling 5Y: SMA 6/12M win 11/15; 3Y wins 11/17
- Double-smoothing (SMA of SMA) does not help

### step39e — mechanism revealed
The mechanism is a **strategy-type shift**, not just noise reduction:

| Smoothing | Median 1Y prior return of picks | Strategy character |
|---|---|---|
| None | -19.10% | Deep mean-reversion (buy crash) |
| SMA 6M | -20.45% | Same but sharper |
| SMA 24M | -6.52% | Mild pullback |
| SMA 60M | **-0.34%** | Persistent quality, flat entry |

Long smoothing transforms "buy the deepest pullback" into "buy the
persistent mean-reversion franchise at fair value." SMA 60M concentrates
into 55/96 tickers (from 91/96 in baseline) with dramatically different
top picks (AMAT/CAT/GE vs NVDA/SMCI/NEM).

### step39f — smoothing × DCA-scaling matrix
| Variant | 20Y CAGR | 10Y robustness |
|---|---|---|
| flat + SMA 12M | +18.31% | 5/5 wins |
| **3x@-20dd + SMA 12M** | **+19.45%** | 2/5 wins (GFC-concentrated) |
| 3x@-20dd + SMA 24M | +19.38% | 2/5 |

Smoothing and dd-scaling ADD — you get both alphas. But dd-scaling
still carries its GFC dependency; it's not a robust overlay.

### step39g — defensive gate matrix
Tested sector_cap, rebound, zombie, score_threshold, min_score gates:
- **Zero defensive gate improves SMA 12M.** All either tie or hurt.
- sector_cap non-binding (smoothing already diversifies sectors).
- `score_thresh=50/70` is a no-op (smoothed scores exceed both).
- `zombie_lookback` LOSES -2.85pp — distressed-recovery is part of alpha.
- `rebound 63d bull-only` loses -0.32pp magnitude but wins 14/18 cal
  years (vs 10/18) — aesthetic tradeoff only.

### step39h — hold_days sensitivity (CRITICAL CAVEAT)
Smoothing's edge is concentrated at `hold_days=5000` (forever):

| hold_days | baseline CAGR | SMA 12M CAGR | Δ |
|---|---|---|---|
| 63 | +0.21% | +0.25% | +0.04pp |
| 252 | +0.95% | +1.20% | +0.24pp |
| 504 | +2.35% | +2.23% | **-0.11pp** |
| 1260 | +5.02% | +5.04% | +0.01pp |
| 2520 | +10.00% | +9.76% | **-0.24pp** |
| 5000 | +17.41% | +18.31% | +0.90pp |

**Smoothing is a hold-forever improvement.** For rotation strategies
(hold ≤ 2520d) the edge disappears or inverts. This is consistent with
the mechanism: smoothing picks "persistent quality franchises" which
compound indefinitely; if you sell them after 2-10Y you give up the
very thing that makes them win.

## What this means for production

1. **For CAP5** (our DCA strategy, hold-forever): **adopt SMA 12M**.
2. **For rotating strategies**: do NOT smooth — smoothing hurts.
3. **For the aggressive CAP5-crisis variant**: SMA 12M + 3x@-20dd
   gives +2.04pp but requires 22% more capital and retains GFC dependency.

## Migration guidance

Minimal change — one line in StrategyConfig instantiation:

```python
# Before
cfg = StrategyConfig(top_n=5, max_ticker_frac=0.05, ...)

# After
cfg = StrategyConfig(top_n=5, max_ticker_frac=0.05, smoothing_months=12, ...)
```

The rest of the pipeline is unchanged. SPY benchmark, DCA amount,
entry_delay, hold_days all identical. No new data sources required.
In live use, today's ranking is computed from the trailing 252 trading
days (~12 months) of `final` score values, which are already stored in
bt_ext.parquet.

## Unresolved / open items

1. **128-ticker retest**: regen in progress; validate SMA 12M holds on
   the expanded universe (31 new tickers across crypto-linked, growth,
   value, international ADRs).
2. **Out-of-sample**: 20Y backtest is in-sample. Adoption is a bet that
   the "persistent quality" bias continues to work.
3. **Transaction cost**: smoothing has lower monthly turnover (25-40%)
   than baseline (49%), so execution cost argues IN FAVOR of adoption.
4. **Pick stability**: SMA 12M top 5 are more stable — investors comparing
   this month's vs last month's email will see fewer surprise rotations.
   Psychological benefit.
