# Executive summary — v8 stock-selection rebuild

Branch: `claude/rebuild-stock-selection-2qHxY`. Date: 2026-05-10.
Author: Claude (run identifier: `v8`).

## What we did

Reviewed the deployed main-page strategy ("V3-PIT-SP500", served via
`experiments/docs/monthly-dca/data.json` and `docs/monthly_dca.js`),
mapped the entire pipeline, audited the engine for survivorship and
look-ahead bias, reproduced the headline metrics exactly, and ran a
~450-config experimental sweep across concentration, holding horizon,
ML scorer choice, regime gate, weighting, and add-on risk controls. We
also attempted (and KILLED) a true weekly walk-forward.

## Headline outcome

Identified a strict Pareto-improvement over the deployed v3 strategy
that we are calling **exp_02 winner**:

```
ml_3plus6plus1, k=1, hold=1m, regime=safer, weighting=invvol,
crash_fallback=TLT, cost_bps=10
```

| Metric              | v3 deployed | exp_02 winner | Δ          |
|---------------------|------------:|--------------:|-----------:|
| WF mean OOS CAGR    | 42.80%      | **50.16%**    | **+7.36pp** |
| WF min OOS CAGR     | 14.49%      | 17.38%        | +2.89pp    |
| WF mean Sharpe      | 1.031       | 1.084         | +0.053     |
| Full backtest CAGR  | 39.77%      | 40.27%        | +0.50pp    |
| MaxDD               | -49.83%     | -44.49%       | +5.34pp    |
| WF beats SPY count  | 9/10        | **10/10**     | +1         |

Floors all cleared (WF min ≥ 0%, Sharpe ≥ 1.0, MaxDD ≥ -50%, ≥ 8/10
beats SPY).

## What changed vs v3

1. **Concentration** k=3 → **k=1**. Top-1 GBM-ranked S&P 500 pick
   each month.
2. **Hold** 6 months → **1 month**. Faster rotation captures the
   explosive head of the score distribution before signal decay.
3. **Scorer** mean of 3m+6m → **mean of 1m+3m+6m**. Adding the 1m head
   improves marginal IC.
4. **Regime gate** tight → **safer** (earlier crash trigger via
   SPY DD-from-52wH ≤ -8%).
5. **Crash fallback** cash → **TLT (long Treasuries)**. Reuses crash
   months as positive-return defensive allocation.

## Triple-digit goal — the honest answer

The user's stretch goal was **triple-digit OOS WF CAGR**. On the
PIT S&P 500 universe with monthly cadence and the user's scope rules
(k ≤ 3, no leverage, no down-cap), the empirical ceiling we hit is
**WF mean ~50%**. The same `exp_02` config applied off-scope to the
broader 1811-ticker universe hit **WF mean 119%** but with -76% MaxDD
and the universe carries explicit survivorship bias (it's
"tickers existing today", not PIT-correct). We do **not** claim the
119% number as a deployable result; it's recorded only as evidence
of where the alpha lives.

## Honest caveats (do not skip)

1. **Survivorship sensitivity at k=1 is catastrophic.** At an α=4%/yr
   synthetic delisting rate, the MC median bias-corrected CAGR
   collapses to -100% (a single wipe in ~268 months kills the
   compounding curve at k=1). The deployed v3 (k=3) absorbs this
   risk via diversification (its bias-corrected median CAGR at α=4%
   is +28.6%). **k=1 trades robustness for headline CAGR.**
2. **The 2025-01 → 2026-04 frozen holdout is brutal.** Strategy
   -32.4% vs SPY +16.0% over 12 months. This is the single-shot OOS
   test of the *selection*, and it failed badly. The deployed v3
   was equally hurt in 2025 (-32% calendar-year). Both strategies
   need a regime that hasn't been in this market.
3. **Weekly cadence is dead at this signal/cost level.** Built end-
   to-end (features, walk-forward GBM, simulator), tried 13 variants,
   0 passed floors. Best WF mean was 17% (less than half of monthly).
   Cost drag and horizon mismatch (4-week target on 1-week hold)
   were the killers. Records preserved in
   `experiments/monthly_dca/v8/weekly/`.

## Deliverables (where to find what)

- **Repo map / engine audit / hypotheses**: `research/00_repo_map.md`,
  `research/01_engine_audit.md`, `research/02_hypotheses.md`.
- **Per-experiment narratives**: `research/exp_01_concentration_sweep.md`,
  `research/exp_02_tlt_fallback.md`,
  `research/graveyard/exp_03_weekly_walkforward.md`.
- **Experiment log**: `backtests/experiment_log.csv`.
- **Sweep code**: `experiments/monthly_dca/v8/run_tier{1,2,3}_*.py`.
- **Weekly track (KILLED)**: `experiments/monthly_dca/v8/weekly/`.
- **Validation gauntlet**: `experiments/monthly_dca/v8/run_validation_gauntlet.py`.
- **Results CSVs**: `experiments/monthly_dca/v8/results/`.
- **Final validation report**: `reports/final_validation.md`.
- **This summary**: `reports/executive_summary.md`.

## Recommendations

1. **Do not deploy exp_02 winner alone**. The k=1 fragility under
   delisting and the 2025 holdout result combine into too much tail
   risk for a public-facing product. The deployed v3 (k=3) is more
   robust.
2. **Consider blending** exp_02 (k=1, fast rotation, TLT fallback) with
   v3 (k=3, slower, more diversified) at e.g. 50/50 weight. This is
   not tested in this run; it's the obvious next experiment. Expected:
   WF mean ~46%, MaxDD better than k=1, bias-correction more robust.
3. **For real triple-digit ambition**, change product scope: a
   bias-corrected smaller-cap PIT universe (Russell 2000 PIT, with a
   delisted-with-final-return data source) is the honest path. The
   119% result on the (biased) broader universe is a strong hint that
   the alpha exists down-cap if the data does too.
4. **The user explicitly asked us not to update the website.** No
   changes to `docs/index.html`, `docs/monthly_dca.js`, or
   `experiments/docs/monthly-dca/data.json` were made.

---

This branch is `claude/rebuild-stock-selection-2qHxY` and all v8
artifacts live under `experiments/monthly_dca/v8/`,
`research/`, `reports/`, and `backtests/`. To make this run easy to
identify alongside concurrent agents, every commit on this branch is
prefixed with `[v8]`.
