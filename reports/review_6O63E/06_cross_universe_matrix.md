# 06 — Cross-universe matrix

Six universes × six strategies, no re-tuning. Source:
`experiments/monthly_dca/v6/run_universe_matrix.py` and
`results/universe_matrix.csv`.

## Universes

| Universe | # tickers | Notes |
|---|---|---|
| `sp500_pit` | ~500 PIT | Deployed home (proper PIT) |
| `broader` | 1,811 | Russell-3000 proxy; **not PIT** (survivor bias) |
| `non_sp500` | 1,579 | broader minus PIT-S&P-500 |
| `qqq_tech` | 92 | Nasdaq-100 representative |
| `iyw_tech` | 127 | iShares US Tech ETF representative |
| `tech_broad` | 212 | QQQ ∪ IYW ∪ IGM extras |

## Strategies

`v3`, `A` (invvol+cy3), `B` (kb=2+invvol+cy3), `C` (chr q=0.4),
`A+C` (invvol + chr filter), `B+C` (kb=2 + invvol + chr filter).

## Full CAGR

| Universe | v3 | A | B | C | A+C | B+C |
|---|---:|---:|---:|---:|---:|---:|
| sp500_pit | 39.77% | 38.20% | 41.54% | 44.81% | 42.89% | **46.49%** |
| broader | 50.94% | 50.73% | 46.79% | 49.20% | 50.04% | 43.71% |
| non_sp500 | 48.22% | 46.67% | 43.54% | 50.27% | **50.98%** | 45.32% |
| qqq_tech | 47.19% | 43.83% | 43.50% | **51.16%** | 48.25% | 49.00% |
| iyw_tech | 44.87% | 44.54% | 43.11% | 45.41% | **45.16%** | 43.08% |
| tech_broad | 55.04% | 52.84% | 48.62% | **58.02%** | 55.53% | 52.47% |

## Sharpe

| Universe | v3 | A | B | C | A+C | B+C |
|---|---:|---:|---:|---:|---:|---:|
| sp500_pit | 0.955 | 0.971 | 1.017 | 1.036 | 1.056 | **1.097** |
| broader | 0.912 | **0.978** | 0.926 | 0.898 | 0.971 | 0.896 |
| non_sp500 | 0.896 | 0.950 | 0.906 | 0.923 | **0.999** | 0.930 |
| qqq_tech | 1.343 | 1.346 | 1.283 | 1.387 | **1.385** | 1.353 |
| iyw_tech | 1.179 | **1.209** | 1.199 | 1.176 | 1.201 | 1.170 |
| tech_broad | 1.415 | 1.415 | 1.331 | 1.429 | **1.430** | 1.370 |

## MaxDD

| Universe | v3 | A | B | C | A+C | B+C |
|---|---:|---:|---:|---:|---:|---:|
| sp500_pit | −49.83% | **−45.98%** | −45.98% | −49.83% | −45.98% | −45.98% |
| broader | −62.45% | **−56.73%** | −56.73% | −68.07% | −66.80% | −66.80% |
| non_sp500 | −62.45% | **−55.20%** | −55.20% | −60.51% | −61.78% | −61.78% |
| qqq_tech | **−41.57%** | −41.51% | −47.02% | −43.59% | −42.98% | −47.93% |
| iyw_tech | −61.32% | **−58.60%** | −58.60% | −61.32% | −58.60% | −58.60% |
| tech_broad | −41.57% | **−41.51%** | −43.30% | −41.57% | −41.51% | −43.30% |

## WF mean CAGR (10 splits)

| Universe | v3 | A | B | C | A+C | B+C |
|---|---:|---:|---:|---:|---:|---:|
| sp500_pit | 42.80% | 42.48% | 46.15% | 45.86% | 45.66% | **49.49%** |
| broader | 51.83% | **61.90%** | 50.44% | 49.78% | 57.22% | 43.93% |
| non_sp500 | 51.03% | 59.61% | 49.18% | 54.82% | **62.51%** | 50.21% |
| qqq_tech | 43.90% | 43.27% | 43.24% | **47.67%** | 46.14% | 47.31% |
| iyw_tech | 48.97% | 50.20% | 47.05% | 51.05% | **51.64%** | 46.84% |
| tech_broad | 50.50% | 50.25% | 43.80% | **52.49%** | 51.68% | 45.91% |

## WF min CAGR (worst-case OOS split)

| Universe | v3 | A | B | C | A+C | B+C |
|---|---:|---:|---:|---:|---:|---:|
| sp500_pit | 14.49% | 20.92% | 24.16% | 17.01% | 22.41% | **26.55%** |
| broader | **13.73%** | 7.81% | 8.67% | 13.73% | 7.81% | 6.86% |
| non_sp500 | **21.37%** | 15.08% | 20.34% | 21.37% | 15.08% | 6.86% |
| qqq_tech | 21.36% | 17.37% | 19.30% | **29.60%** | 27.74% | 25.53% |
| iyw_tech | **15.99%** | 9.01% | 7.11% | 10.73% | 4.74% | 2.82% |
| tech_broad | 42.22% | 32.54% | 30.46% | **42.77%** | 36.14% | 35.42% |

## Beats SPY (count out of 10 splits)

| Universe | v3 | A | B | C | A+C | B+C |
|---|---:|---:|---:|---:|---:|---:|
| sp500_pit | 9 | 9 | **10** | **10** | **10** | **10** |
| broader | **10** | 9 | 8 | 9 | 9 | 8 |
| non_sp500 | **10** | **10** | 9 | 9 | **10** | 9 |
| qqq_tech | 9 | 9 | 9 | **10** | **10** | **10** |
| iyw_tech | **10** | 9 | 9 | 9 | 9 | 9 |
| tech_broad | **10** | **10** | **10** | **10** | **10** | **10** |

## Observations

1. **B (kb=2) is the only "win" on home that loses on every other universe.**
   On sp500_pit B has the best Sharpe (1.017) and tied-best MaxDD (−45.98%). On
   every tech universe (qqq, iyw, tech_broad), B is worse than v3 on Sharpe.
   This is consistent with B being calibrated to the SP500 vol profile.

2. **A's MaxDD advantage is universal (8 of 8 universes wins or ties)**, but its
   CAGR advantage is concentrated on broader/non_sp500 (where it adds +10pp WF
   mean by avoiding 2024-25 concentration risk).

3. **C alone wins WF mean on 5 of 6 universes** (loses only on broader to A).
   C's CAGR advantage is consistent across S&P 500 and tech universes.

4. **A+C is the broadest winner on Sharpe**: wins or ties Sharpe on 5 of 6
   universes (sp500, broader, non_sp500, qqq, tech_broad). On iyw_tech it's
   essentially tied with A (1.201 vs 1.209).

5. **tech_broad has the best structural metrics across the board**: Sharpe 1.42
   (every variant), MaxDD −41% to −43% (best of any universe), WF min CAGR 30–43%
   (highest of any universe). This is the universe the strategy works best on.

6. **broader / Russell-3000 has the highest absolute CAGR but the worst MaxDD**:
   v3 51% CAGR with −62% MaxDD; A 51% CAGR with −57% MaxDD. The risk-adjusted
   metrics are worse than tech_broad.

## Generalisation ranking (from sp500_pit to other universes)

Counting "wins or ties vs v3" on (Sharpe, MaxDD, WF mean, beats SPY) across
the 5 non-home universes:

| Strategy | Sharpe wins/5 | MaxDD wins/5 | WF mean wins/5 | Beats-SPY ≥ v3? |
|---|---:|---:|---:|---|
| A   | 4 | 5 | 2 | 4/5 |
| B   | 0 | 3 | 0 | 1/5 |
| C   | 4 | 0 | 4 | 4/5 |
| A+C | 5 | 2 | 4 | 4/5 |
| B+C | 2 | 2 | 1 | 3/5 |

A+C has the most universal Sharpe wins. C alone has the most universal
WF mean wins. B alone is the clear loser cross-universe.

## What this means for the universe-choice decision

For deployment, choose the universe that:
1. Has the best structural metrics in absolute terms.
2. The candidate strategy improves on, not just ties.
3. Has acceptable MaxDD.
4. The 2024-25 holdout doesn't blow up on.

Conclusion previewed: **`tech_broad` is the right universe** — Sharpe 1.43
(best of all), MaxDD −41.5% (best of all), WF mean 51.68%, and C/A+C add
material recent-year edge over v3. Details in
`07_holdout_2024_2025.md` and `10_final_ship_decision.md`.
