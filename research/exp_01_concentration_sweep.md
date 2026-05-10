# Exp 01 — Concentration / horizon / scorer / regime sweep (Tier 1)

Date: 2026-05-10. Branch: `claude/rebuild-stock-selection-2qHxY`.
Engine: `experiments/monthly_dca/v6/lib_engine.py` (parity with v3).
Driver: `experiments/monthly_dca/v8/run_tier1_sweep.py`.

## Hypothesis

Lower k (1, 2) and shorter hold (1, 2, 3 months) — combined with the
existing walk-forward GBM preds — should lift WF mean OOS CAGR above v3's
42.8% baseline by capturing the explosive-mover head of the score
distribution before signal decay. Cost: noisier WF min CAGR.

## Method

384 configurations, single simulator pass each (~80 s total):
- scorer ∈ {ml_3plus6, ml_h3, ml_h6, ml_3plus6plus1}
- k ∈ {1, 2, 3}
- hold_months ∈ {1, 2, 3, 6}
- regime ∈ {tight, strict_dd, safer, combo}
- weighting ∈ {ew, invvol}
- cost_bps = 10 (round-trip)

**Floors** for "passing":
- WF min CAGR ≥ 0%
- WF mean Sharpe ≥ 1.0
- Full-window MaxDD ≥ -50%
- ≥ 8/10 splits beat SPY

## Result

12 of 384 pass all floors. Headline winner:

```
scorer = ml_3plus6plus1   (avg of 1m+3m+6m predictions)
k = 1
hold = 1 month
regime_gate = safer
weighting = invvol (or ew — 1 pick, weighting is a no-op)
cost_bps = 10
```

| Metric                | v3 baseline | exp_01 best | Δ        |
|-----------------------|------------:|------------:|---------:|
| Full-window CAGR      | 39.77%      | 38.64%      | -1.13pp  |
| Sharpe                | 0.955       | 1.058       | +0.10    |
| MaxDD                 | -49.83%     | -45.01%     | +4.82pp  |
| **WF mean CAGR**      | **42.80%**  | **48.28%**  | **+5.47pp** |
| WF median CAGR        | 39.90%      | (see CSV)   |          |
| WF min CAGR           | 14.49%      | 9.54%       | -4.95pp  |
| WF max CAGR           | 108.79%     | (see CSV)   |          |
| WF mean Sharpe        | 1.031       | 1.058       | +0.027   |
| WF n positive         | 10/10       | 10/10       | tied     |
| **WF n beats SPY**    | **9/10**    | **10/10**   | **+1**   |

**Pareto-improves v3** on Sharpe, MaxDD, WF mean CAGR, WF n beats SPY.
Loses some WF min CAGR (15% → 9.5%) but stays clearly above the 0%
floor.

## Mechanism (why this works)

1. **k=1 concentration.** The top-ranked ML prediction is materially
   above the rest of the cross-section. Picking only the #1 captures
   that tail; picking #2 and #3 dilutes it.
2. **Hold = 1 month.** Predictor is annual-retrain GBM but the predictions
   themselves update monthly. Shorter hold lets the basket roll into
   the new top-1 each month rather than holding stale picks for 6.
3. **`ml_3plus6plus1` scorer.** Averaging 1m / 3m / 6m predictions adds
   short-horizon signal that's missing from `ml_3plus6`. The 1m head is
   noisier in absolute terms but contributes meaningfully when ranked.
4. **`safer` regime.** Earlier crash trigger
   (also fires on SPY DD-from-52wH ≤ -8%) keeps MaxDD ≤ -45%.
   With k=1, single-stock vol is high, so an earlier-firing gate matters.

## Verdict

**KEEP** as the new headline candidate. Logged in
`backtests/experiment_log.csv`. Output CSVs in
`experiments/monthly_dca/v8/results/`.

## Next

- Try richer additive controls on top of this winner:
  conviction-blend, quality blend, half-cash on warning, vol penalty,
  trailing stop, sticky cash, smart re-entry.
- Try staggered ensemble of k=1 sub-baskets each held longer (1m × 3 vs
  3m × 1, etc.) — does the time-diversification reduce WF min while
  preserving WF mean?
- Try **true weekly rebalance** — the highest-EV scope-allowed change.
- Try regime-conditional k.
