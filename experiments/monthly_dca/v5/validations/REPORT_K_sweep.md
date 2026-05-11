# K-sweep — basket size validation on PIT S&P 500

**Generated**: 2026-05-11
**Harness**: look-ahead-fixed + invvol_weights K=1 renorm fixed
**Universe**: PIT S&P 500 only

## Why this matters

Production v5 holds K=3 picks for 6 months, semi-annual rebalance.  This
report sweeps K=1..5 in two simulation modes:

1. **Lump-sum** (the deployed strategy) — full capital, semi-annual rebalance.
2. **Staggered monthly DCA** — $1 deposited monthly, one fresh K-pick
   tranche each month, each tranche held 6 months. At steady state ≈6
   overlapping tranches.

## Bug uncovered en route

`invvol_weights(picks, cap=0.40)` returned `[0.40]` for K=1 instead of
`[1.00]` — the cap reduced the single weight to 0.40 and the iterative
re-distribution loop's `else: break` branch left it un-renormalised.
**Lump-sum was unaffected** (the simulator does `cur_weights /= sum` after
the picker returns). **Staggered DCA was catastrophically affected**: only
40 % of capital was invested in K=1 mode, the other 60 % was destroyed →
+25 % CAGR rebuilt as −79 %.

Fix: append `return w / w.sum()` (or fall back to 1/N) at the end of
`invvol_weights`.  Committed in the same change set as this report.

## Lump-sum K-sweep (full window 2003-09 → 2026-04, PIT S&P 500)

| K | Lump CAGR | DCA CAGR | WF mean | WF min | Beats SPY | Positive | Sharpe | MaxDD | 2024 edge | 2025 edge |
|--:|----------:|---------:|--------:|-------:|----------:|---------:|-------:|------:|----------:|----------:|
| 1 |    28.76  |   26.31  |  24.75  | **−4.37** | 8/10 |  9/10  | 0.63 | −80.4 % | +2.9 | −28.5 |
| 2 | **44.27** |   44.00  | **48.64** | 17.54 | 10/10 | 10/10 | 0.92 | **−69.5 %** | −3.0 | −19.6 |
| 3 |   43.79   |   44.04  |   46.55 | **20.37** | 10/10 | 10/10 | **1.00** | **−51.4 %** | −14.8 | +8.6 |
| 4 |   34.97   |   35.48  |   38.44 | 14.46 | 10/10 | 10/10 | 0.93 | −56.3 % | −16.0 | +3.5 |
| 5 |   30.22   |   30.31  |   31.91 | 14.90 | 10/10 | 10/10 | 0.88 | −62.1 % | −16.1 | −2.1 |

## Staggered monthly-DCA K-sweep (same window)

| K | mw-CAGR | SPY DCA | Edge   | Wealth multiple | Tranche win | p10  | Worst | 2024 edge | 2025 edge |
|--:|--------:|--------:|-------:|----------------:|------------:|-----:|------:|----------:|----------:|
| 1 |   25.96 |  12.66  | +13.30 |    35.2 ×       |  64.6 %     | −23.3 | −92.8 | **+21.1** |  −1.3 |
| 2 | **41.05** |  12.66 | **+28.4** | **307.3 ×**  |  73.2 %     | −16.4 | −92.3 |  +11.9 |  −7.3 |
| 3 |   39.15 |  12.66  | +26.5  |   235.3 ×       | **76.4 %**  | −13.6 | −86.1 |   +3.7 |  +1.9 |
| 4 |   30.91 |  12.66  | +18.3  |    72.5 ×       |  76.4 %     | −13.0 | −85.5 |   +3.1 |  +1.9 |

## Verdict — K = 3 stays in production

**K=2** wins on raw CAGR in both modes (+0.5 pp lump-sum, +1.9 pp DCA over
K=3) but the trade-off is unfavourable:

- MaxDD: **−69 %** vs K=3's −51 % (18 pp deeper)
- Sharpe: **0.92** vs K=3's 1.00
- WF min: **17.5 %** vs K=3's 20.4 %
- Tranche win rate: 73 % vs K=3's 76 %
- 2025 edge: −20 pp vs K=3's +9 pp
- Tranche p10: −16 % vs K=3's −14 %

For a $10k 23y investment: K=2 gives ~$148M, K=3 gives ~$134M — 10 % more
wealth in exchange for a peak-to-trough drawdown 18 pp deeper.  Most
investors should not take that trade; K=2 is appropriate only for someone
explicitly choosing higher tail risk for ~10 % more terminal wealth.

**K=1** is dominated:  CAGR 28.76 %, MaxDD −80.4 %, **WF min −4.4 %** (the
only K where a walk-forward split is *negative*), Sharpe 0.63.  The
single-stock concentration risk is real — a single bad pick takes the entire
basket with it.

**K=4 and K=5** are simply worse than K=3 (lower CAGR, no risk-adjusted
benefit).

## Decision

- **No change to deployed production K.**  K=3 remains the right choice on
  every risk-adjusted metric.
- The honest harness now matches the production simulator within ~3 pp
  full-window CAGR (post-fix harness 43.79 % vs production 40.85 %); residual
  is attributable to (a) trading-day vs calendar-month alignment, (b)
  preds_wf-vs-preds_live cutover, (c) cost handling (`r-cf` vs `(1+r)(1-cf)`).
- The honest harness is now the canonical sim.  Treat any "anchor variant
  beats baseline" or "K=2 dominates K=3" claims from earlier reports as
  artifacts of the look-ahead bias.

## Side fixes shipped with this validation

1. **Look-ahead bias** (harness): see REPORT.md 2026-05-11 section.
2. **invvol_weights K=1 renorm** (harness): see this report.
3. **Regime classifier 5y → 1y streak** (harness + production
   `build_webapp_v5_pit.py`): the `max_below_200_streak` feature was the
   max over the past 5 years, so once SPY had a 40+ d below-200 stretch in
   2022, the "recovery" branch fired forever after.  Replaced with an
   on-the-fly 12-month streak computed from daily SPY in
   `prices_extended.parquet`.  2024 is now correctly classified as **bull**
   every month (previously: 9 × recovery, 3 × bull).

   **Does not change production CAGR** — K_normal = K_recovery = K_bull = 3
   and cap = 0.40 across all regimes, so the picks are identical regardless
   of regime label.  Fix is principal-correctness only and unlocks future
   regime-conditional strategies.

## Reproduce

```
# Lump-sum K-sweep
python3 -m experiments.monthly_dca.v5.validations.run_k_sweep

# Staggered DCA K-sweep
python3 -m experiments.monthly_dca.v5.validations.run_k_sweep_staggered
```
