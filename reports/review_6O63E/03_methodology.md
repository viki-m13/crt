# 03 — Methodology

## Engine

All validation uses `experiments/monthly_dca/v6/lib_engine.py` — a clean
re-implementation of the v3 simulator that reproduces the deployed v3 metrics
exactly (verified to 16 decimal places against `v3_winner_summary.json`).

The engine supports all the knobs needed for A/B/C:

- `weighting`: ew | invvol | conv | softmax
- `k_normal`, `k_recovery`, `k_bull` (per-regime K)
- `cash_yield_yr`: T-bill credit while in cash
- `crash_fallback`: cash | spy | tlt
- Various risk knobs (stops, sticky cash, DD scaling) — not used by A/B/C

For Option C, the Chronos filter is applied as a panel pre-filter before the engine runs (filters down to stocks with `chronos_p70_rk ≥ 0.4`).

## Walk-forward splits

10 fixed (TRAIN, TEST) splits inherited from v3, covering 2003-09 → 2024-12:

```
A1       2011-2018  (post-GFC long bull)
A2       2015-2021  (mid-cycle through COVID)
A3       2018-2024  (modern through AI rally)
R1_GFC   2008-2010  (GFC + V-shaped recovery)
R2       2011-2013  (post-GFC consolidation)
R3       2014-2016  (range-bound + 2015 selloff)
R4       2017-2019  (low-vol bull + Dec 2018)
R5_COVID 2020-2022  (COVID + 2022 bear)
R6_AI    2023-2024  (AI rally, mega-cap dominance)
STRICT   2021-2024  (strict modern OOS)
```

These are *overlapping time slices of one OOS curve*, not 10 independent
experiments. The "WF mean 42.80%" headline can therefore overstate the
edge — `YLOka` correctly flagged that the median (39.9%) ≈ full-period CAGR
(39.77%) is the more honest deployment estimate.

## Universes tested

| Universe | # tickers | Source | Notes |
|---|---|---|---|
| `sp500_pit` | ~500 PIT | `sp500_membership_monthly.parquet` | Home, properly PIT |
| `broader` | 1,811 | full `ml_preds_v2.parquet` | Russell-3000 proxy; **not PIT — survivorship bias** |
| `non_sp500` | 1,579 | `broader` minus PIT-S&P-500 | Same survivor bias |
| `qqq_tech` | 92 | 99-name Nasdaq-100 list ∩ broader | Static (modern) composition |
| `iyw_tech` | 127 | iShares US Tech ETF holdings ∩ broader | Static |
| `tech_broad` | 212 | (QQQ ∪ IYW ∪ IGM extras) ∩ broader | Static, deduped |

Ticker lists are defined in `experiments/monthly_dca/v6/universes.py`.

**Honest caveat**: only `sp500_pit` is point-in-time. The other 5 use modern (2025-vintage) ticker lists applied retroactively to 2003-2025 history. Names that existed in earlier years but were not in the modern list aren't represented; names that have only existed since (say) 2015 are still "in scope" pre-2015. This biases backtests slightly upward.

## Walk-forward parameter selection (the key fairness test)

For Options A, B, C, each agent claims an "always-on" parameter:
- A: always invvol
- B: always kb=2 in bull regime
- C: always q=0.4 chronos filter

For each WF split, I:

1. Take the candidate parameter values (e.g., kb ∈ {1, 2, 3, 4, 5}).
2. Compute each variant's CAGR on **months strictly before the test split's start date**.
3. Choose the variant with the best train CAGR.
4. Measure the chosen variant's CAGR / Sharpe / MaxDD **on the test split**.

If the agent's "always-on" parameter is correct, the WF selector should
pick it on most/all splits, and the WF-honest test CAGR should match the
published lift. If the agent's parameter was sweep-overfit, the selector
picks something else on most splits and the WF-honest lift collapses
(possibly to negative).

Source code: `experiments/monthly_dca/v6/run_wf_selection.py` (for B and C)
and `run_wf_A_selection.py` (for A).

## Cross-universe matrix

Same six universes × six strategies (v3, A, B, C, A+C, B+C). Each
strategy is applied with **no re-tuning** — same K, hold, regime gate,
cost, cash yield as on home.

Source: `experiments/monthly_dca/v6/run_universe_matrix.py`.

## Holdout

The most recent 20 months (2024-05 → 2025-12) are evaluated separately as
a "fresh OOS" view. The training data ends at 2024-04 (the ml model
re-trained annually at January with 7-month embargo, so for test months
2025-01+ the model only saw data with `asof < 2024-06`).

Source: `experiments/monthly_dca/v6/run_universe_holdout.py`.

## Bias sensitivity (synthetic delisting)

For each (universe × strategy × α/yr), the monthly-returns matrix is
contaminated with a Bernoulli wipe (-100%) per cell with probability
`p_month = 1 - (1 - α)^(1/12)`. 20 MC iterations per (α, strategy).
Reported as p10 / median / p90 / mean CAGR.

α grid: {0, 4, 8, 12}% (the historical small/mid-cap rate is ~4%/yr).

Source: `experiments/monthly_dca/v6/run_universe_bias.py`.

## Reproducibility

All result CSVs are in `experiments/monthly_dca/v6/results/`. Re-run any
test with:

```bash
cd /home/user/crt/experiments/monthly_dca/v6
python3 run_wf_selection.py           # WF kb selection (B), WF q selection (C)
python3 run_wf_A_selection.py         # WF weighting selection (A)
python3 run_universe_matrix.py        # 6x6 matrix
python3 run_universe_holdout.py       # 2024-05 holdout per universe
python3 run_universe_bias.py          # bias sensitivity
python3 run_per_split_universe.py     # per-split decomposition
```

The Chronos prediction parquets (rebuildable on CPU in ~5 min for sp500
and ~12 min for broader) come from:

```bash
python3 experiments/monthly_dca/v5/score_chronos_v2.py        # sp500_pit
python3 experiments/monthly_dca/v5/score_chronos_universe.py  # broader (1811)
```
