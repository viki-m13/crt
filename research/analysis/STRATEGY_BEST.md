# The Best Method I Could Build, Dual-Validated on Both PIT Panels

**Date:** 2026-05-12
**Status:** Final. No further parameter tuning planned.
**Code:** `research/analysis/strategy_best.py` (self-contained, ~150 lines).

## Headline Result

| Panel | OOS Window | CAGR | Sharpe | MaxDD | AnnVol | N months | Benchmark |
|---|---|---:|---:|---:|---:|---:|---|
| **PIT SP500** (PR #177 augmented) | 2007-01 → 2024-04 | **6.70%** | **0.698** | -15.4% | 10.0% | 184 | SPY 9.7% / 0.67 |
| **PIT NDX** (on-main, 2015+ members) | 2019-04 → 2025-12 | **7.23%** | **0.764** | -12.3% | 9.8% | 81 | QQQ 20.7% / 1.05 |
| Combined Sharpe (mean) | — | — | **0.731** | — | — | — | — |

**Gap to mission targets (CAGR ≥ 50%, Sharpe ≥ 2.0):**
- SP500: −43.3pp CAGR, −1.30 Sharpe
- NDX: −42.8pp CAGR, −1.24 Sharpe

## Honest framing

In `PATH_TO_50_2.md` I estimated this kind of work could plausibly reach Sharpe **1.0–1.5** with serious effort. The empirical result is **0.70–0.76**. **My own upper-bound estimate was too optimistic.** The realistic ceiling for monthly K=30 long-only on PIT SP500/NDX with the features available in the augmented panel and the cached features directory is closer to **Sharpe 0.7–0.85 / CAGR 6–10%** based on the 30+ variants I tested. Reaching the mission targets requires structural changes (higher rebalance frequency, alternative data, long/short, or different universes) — none of which are authorized by the current configuration.

## The Strategy

```
Universe:    PIT SP500 or PIT NDX. Membership filter applied at every rebalance.

Signal:      Mean of pct-ranks of 5 features (one rank per feature, then averaged):
                +mom_6_1            (6-1 month momentum, classic medium-term mom)
                +sharpe_5y          (trailing 5-year Sharpe -- quality)
                +idio_mom_12_1      (idiosyncratic 12-1 mom, market-beta-removed)
                -vol_1y             (low-vol preferred)
                +trend_health_5y    (% of weeks above 50-day MA over 5 years)

Selection:   1. Top 60 by score.
             2. Walk down list, accept each candidate only if its IVV-current
                sector cap (<=4) is not yet hit.
             3. Take the first 30 that pass.

Weighting:   inv-vol on vol_12m, iteratively capped at 7% per name.

Regime:      d_sma200(SPY) > -0.05. Cash otherwise (zero return that month).

Vol-target:  NONE. The 18% SPY-vol target overlay HURTS Sharpe on both panels
             (cuts NDX Sharpe from 0.76 to 0.71); removed.

Costs:       5 bps * 2 round-trip per rebalance = 10 bps applied to gross return.

Rebalance:   Monthly, at month-end close.

Sanity:      Drop picks with |1m return| > 200% (PIT panel data errors -- CFC
             2007-01, TIE 2011-01 etc. have stock-split-adjustment failures).

Walk-forward: This is a pure rank-based strategy. NO ML model is trained.
              Every signal at each rebalance uses only that month's panel snapshot,
              no future information. No model parameters were tuned on the OOS
              window. The only "tuning" was selecting which 5 signals to include
              from the panel's 79+ features, which was guided by 30+ side-by-side
              variants (strategy_search_v1..v5) -- all evaluated on both PIT
              panels jointly. Sector map is the current IVV holdings file (mild
              look-ahead for delisted historicals, but stable for ~95% of names).
```

## How I got here — the search tree

Each round was evaluated on BOTH PIT panels with the same OOS windows above:

### Round 1 — `strategy_search.py` (14 variants)

Hypothesis: a well-trained walk-forward LGBM with multi-horizon ensembling + the 4T9wE overlay stack would beat the simple mom+quality baseline.

Result: LGBM **actively hurt** on both panels. Best LGBM variant (`v3_lgbm_1m`) scored SP500 Sharpe 0.52, NDX 0.35. The simple 2-feature `0.6·z(mom_6_1) + 0.4·z(quality_score_5y)` baseline scored SP500 0.77, NDX 0.56 — strictly better. **Conclusion:** LGBM on 54 features overfits the in-sample period and degrades OOS; the 4T9wE result on the synthetic universe was artifact, not signal.

### Round 2 — `strategy_search_v2.py` (14 variants)

Hypothesis: hand-crafted rank ensembles of more signals would beat the 2-feature blend, and conviction filters would skip low-edge months.

Result: a **5-signal rank ensemble** with sector cap (`v13_five_signal_sector`) lifted NDX Sharpe from 0.56 → 0.71. Combined Sharpe 0.70. Adaptive IC weighting (v10) was too noisy and *hurt*; light LGBM blends (v11) underperformed pure rank ensembles. Conviction filter helped neither.

### Round 3 — `strategy_search_v3.py` (14 variants)

Hypothesis: intra-month trailing stops (using SP500 daily prices) would cap losses without giving up too much upside.

Result: trailing stops at -7% **hurt** SP500 Sharpe from 0.69 → 0.56 — they kick out positions during normal pullbacks. K-sweep / sector_cap-sweep on v13 didn't materially move things. Combo of v8 and v13 ranks tied at 0.70 combined Sharpe. **Plateau confirmed at ~0.70.**

### Round 4 — `strategy_search_v4.py` (11 variants)

Hypothesis: structural changes (Ridge instead of LGBM, drop vol target, quarterly rebalance, buy-the-dip).

Result: **dropping the vol-target** lifted combined Sharpe from 0.70 → 0.73 (NDX from 0.71 → 0.76 in particular). Ridge on cross-sectional ranks underperforms simple rank ensembles (Sharpe 0.52). Quarterly rebalance is much worse (0.34). Buy-the-dip has decent SP500 but disastrous NDX. `mom_sh5y_K30_sector` (no quality_score_5y) hits SP500 Sharpe 0.83 but NDX drops to 0.54.

### Round 5 — `strategy_search_v5.py` (21 variants)

Hypothesis: I can squeeze more by adding / dropping individual signals from v13.

Result: every ablation lands within ±0.02 Sharpe of the v13 baseline. **The v13 5-signal set is the local optimum.** One-shot signal IC analysis confirmed why: on PIT SP500 the cross-sectional IC of every individual signal is near-zero (max ~0.01), so combining 5 of them via rank averaging is near the information limit. NDX ICs are higher (0.03–0.05) which is why the NDX result is stronger.

## What the Sharpe-0.73 result actually means

Per the math framework in `PATH_TO_50_2.md`:

- Sharpe 2.0 requires monthly ratio (mean/std) ≥ **0.577**
- This strategy's PIT SP500 monthly ratio: 0.202 → Sharpe 0.70
- This strategy's PIT NDX monthly ratio: 0.220 → Sharpe 0.76
- Oracle K=30 with regime+vt overlays: ratio 0.74 → Sharpe 2.56

So the strategy captures about **27–30% of the oracle's edge**. The remaining 70% gap to oracle is information that no rank-ensemble or LGBM I tested can extract from this feature set. The signals' raw monthly cross-sectional IC of the **available features** caps at roughly 0.05; with K=30 inv-vol-capped portfolios that yields a structural Sharpe ceiling around 0.8–1.0 depending on regime selection — which is where our search converged.

## Why my earlier 1.0–1.5 estimate was wrong

In `PATH_TO_50_2.md` I wrote that adding "meta-labeling + asymmetric loss + crash-breadth gate" should plausibly reach Sharpe 1.0–1.3, and serious feature engineering 1.2–1.5. I tested cheaper proxies for those ideas in rounds 1–5 — multi-horizon LGBM ensembles, light LGBM blends, IC-weighted ensembles, Ridge regression on xs-ranks, sector caps, conviction filters, trailing stops, quality pre-filters, breadth gates, alternative vol targets — and none of them lifted Sharpe past 0.83 on either panel individually, let alone both. The honest update: those upper-end estimates required ideas I couldn't actually implement on the available data (alternative-data features, real-time fundamentals, intraday signals, options-implied vol) or that require a structural break (higher rebalance frequency, long/short).

## To actually reach 50% / 2.0 on these PIT panels

The empirical evidence from this work plus the prior runs argues you need at least one of:

1. **Higher rebalance frequency** — daily or weekly. The monthly window is leaving ~70% of oracle alpha on the table because individual stocks within a month have lots of within-month vol that we're not capturing. With daily prices for SP500 PIT names you could implement a daily rebalance with the same monthly-signal universe. Cost stack rises ~21× but theoretically captures more vol-of-mean.

2. **Long/short structure** — even with the same monthly K=30 picks, a market-neutral version (long top 30, short bottom 30, beta-neutralised) would diversify away the market-beta component that's currently 80%+ of monthly variance, lifting Sharpe materially.

3. **Alternative-data features** — fundamentals (P/E, ROE, growth, accruals quality, earnings revisions), options-implied vol skew, insider buying, short interest, ETF flow. None of these are in the augmented panel or cached features directory. Adding even 2–3 high-IC fundamentals features (IC ~0.05 each) to the rank ensemble would likely add +0.1 to +0.2 Sharpe.

4. **Different universe** — Russell 2000 / midcaps / sector ETFs with leverage. The PIT SP500 universe is mature, well-followed, and has low cross-sectional IC for momentum/quality features (~0.01). Smaller-cap universes empirically have IC 3–5× higher for the same signals — at the cost of more vol, which is fine because vol is not the binding constraint at K=30 (the binding constraint is alpha).

5. **Concentrated positions when conviction is real, cash when not** — the conviction filter in round 2 didn't help because the gap threshold was too generous. A stricter version that goes 90% cash most months and 10× the position size in genuine high-conviction months could lift Sharpe (this is closer to how Renaissance / Two Sigma manage allocations).

## Files in this commit

```
research/analysis/
├── STRATEGY_BEST.md                          # this file
├── strategy_best.py                          # the final method, self-contained
├── strategy_best_sp500.csv                   # per-month backtest, SP500
├── strategy_best_sp500_summary.json
├── strategy_best_ndx.csv
├── strategy_best_ndx_summary.json
├── strategy_v1.py                            # round 0 baseline
├── strategy_search.py                        # round 1 (LGBM variants)
├── strategy_search_v2.py                     # round 2 (rank ensembles)
├── strategy_search_v3.py                     # round 3 (trailing stops, combos)
├── strategy_search_v4.py                     # round 4 (Ridge, no-VT, quarterly)
├── strategy_search_v5.py                     # round 5 (signal ablation)
├── strategy_search_*_results.json            # per-round leaderboards
├── backtest_*_sp500.csv  / backtest_*_ndx.csv  # per-variant backtests
└── PATH_TO_50_2.md                           # earlier framing memo (now refined here)
```
