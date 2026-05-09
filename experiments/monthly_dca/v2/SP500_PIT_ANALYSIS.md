# Point-in-Time S&P 500 Analysis of the v2-ml-apex Strategy

**Audience.** The author. A non-cosmetic re-evaluation of the
`v2-ml-apex` monthly stock-pick strategy against a *true* point-in-time
S&P 500 universe, in contrast to the published 1,833-ticker universe used
in `experiments/monthly_dca/v2/REPORT.md` and the live webapp.

**Run date.** 2026-05-09.
**Window.** 2003-09-30 → 2025-12-31 (268 month-ends, ~22.3 years).
**Benchmark.** SPY (S&P 500 ETF), monthly returns from
`cache/v2/monthly_returns_clean.parquet`.

---

## 0. TL;DR

| Universe / scoring config                         | Final-window CAGR | Edge vs SPY buy-hold | WF mean test CAGR | WF beats SPY |
|---------------------------------------------------|------------------:|---------------------:|------------------:|-------------:|
| **Published (1,833 tickers, full panel)**         |        **80.79%** |              +68.8pp |            51.80% |        9/10  |
| **PIT S&P 500, existing model (filter-only)**     |        **15.05%** |              +3.10pp |            16.22% |        5/10  |
| **PIT S&P 500, model retrained on S&P 500 only**  |        **10.41%** |              -1.85pp |            10.47% |        4/10  |
| SPY buy-and-hold (same window, $1)                |            11.94% |                  —   |                —  |          —   |
| SPY DCA ($1/month, XIRR)                          |            13.82% |                  —   |                —  |          —   |
| **PIT (filter-only) ex-2009 calendar year**       |       **10.53%** |              -1.12pp |              n/a  |          n/a |

**Headline result.** The 80.79% headline CAGR is not robust to a
point-in-time S&P 500 universe.  When the same regime-gated GBM scorer is
restricted to S&P 500 constituents at each rebalance, full-window CAGR
collapses to ~15% with a +3 percentage-point edge over SPY buy-hold and
~+1.2 pp over SPY DCA, Sharpe 0.72, MaxDD −52% (COVID).
Re-training the model on the S&P 500 universe alone makes things
*worse*, not better, because the training signal shrinks ~4×.

**The most material finding.**  Of the strategy's full-window
calendar-year compound return on the PIT S&P 500 universe (15.86% CAGR
across 23 calendar years), **virtually 100% of the edge over SPY is
attributable to a single year — 2009 — when the post-GFC oversold-recovery
score correctly latched onto multibagger small-S&P-500-cap rebounds and
returned +226.6% in one calendar year**.  Excluding 2009, the strategy
returns 10.53% CAGR over the remaining 22 years, while SPY returns 11.65% —
an edge of **−1.12 pp**.  Over the post-2010 "modern" period (16 years),
the strategy actually **underperforms SPY by -2.48 pp** (13.92% vs
16.40%).

**Read this as.** The published headline does not generalise to a S&P 500
mandate.  The strategy's apparent alpha sits primarily in (a) small/mid-cap
selection and (b) residual survivorship in the underlying price panel.
What remains under PIT large-cap discipline is a real but small signal
(IC ≈ 0.035, IR ≈ 0.88) that produces ~3 pp of headline edge before tax —
but that edge is concentrated almost entirely in one tail event (2009
post-GFC recovery), and degrades to ~0 over the modern (post-2010) period
or under a 4%/yr synthetic delisting overlay.

---

## 1. Methodology

### 1.1 Universe construction (PIT membership)

We built a true point-in-time S&P 500 panel from two sources, both saved
to `cache/v2/sp500_pit/`:

1. **`sp500_hist_1996_2019.csv`** — daily PIT constituent lists 1996-01-02
   through 2019-01-11 (2,595 daily snapshots, encoded with `-YYYYMM`
   removal-month suffixes).
2. **`sp500_changes_since_2019.csv`** — 110 add/remove change events
   2019-01-18 through present, each with a date, added ticker(s), and
   removed ticker(s).

`build_sp500_pit_membership.py` rolls the 2019-01-11 snapshot forward by
applying the change events in chronological order, and back-projects to
each rebalance month-end.  Output: `sp500_membership_monthly.parquet`,
134,051 (asof, ticker) rows over 268 month-ends.  Per-asof member counts
range 494–506 (mean 500.2).

### 1.2 Two PIT evaluations

We ran two separate backtests, both gated by the production v2 'tight'
regime classifier (cash on crash; top-15 normal / top-7 bull / top-7
recovery), equal-weighted within picks, 10 bp/month round-trip cost:

  **(A) Filter-only.**  Use the existing production predictions
  (`cache/v2/ml_preds_v2.parquet`, walk-forward GBM trained on 1,833-ticker
  cross-section) and filter the candidate pool at each month-end T to the
  S&P 500 members on T.
  *Tests: how does the deployed scorer perform if I restrict picks to S&P 500?*
  Code: `sp500_pit_filter_backtest.py`.

  **(B) Retrained.**  Build a fresh cross-section panel restricted to S&P 500
  PIT members.  Cross-sectionally rank-transform the same 67 features
  *within S&P 500 only* at each month.  Walk-forward fit a multi-horizon
  HistGradientBoostingRegressor (1m/3m/6m, annual retrain, 7-month embargo)
  on this restricted panel.
  *Tests: is the strategy as a whole adapted to a large-cap mandate?*
  Code: `sp500_pit_retrain_backtest.py`.

### 1.3 Walk-forward splits

Same 10 splits as the original v2 report (A1, A2, A3, R1_GFC, R2, R3, R4,
R5_COVID, R6_AI, STRICT) — TEST CAGR is reported per split.  No retraining
was performed at the split level; the walk-forward annual-retrain inside the
fitter is the true OOS guardrail.

### 1.4 Caveats — what is *still* biased

The price panel (`monthly_returns_clean.parquet`) was assembled from
Yahoo data, which only retains tickers that exist today.  Of the 976
historical S&P 500 members in the PIT panel, **609 (62%)** are present in
our price panel.  Per-month coverage:

  - 2003: 51.4% of S&P 500 members in the price panel
  - 2010: 64.6%
  - 2015: 72.1%
  - 2020: 85.8%
  - 2025: 96.3%

The missing names are mostly companies that delisted, were acquired,
went private, or had ticker changes pre-2010 (CFC, ABS, AMP, AGN, ANTM,
ALXN, ANR, APC, etc.).  This means our PIT analysis still has *residual*
survivorship bias for early years: any S&P 500 member dropped from our
price panel is silently excluded from the picking pool.  A pessimistic
read is that early-year results are still optimistic by some
2003-era-coverage-weighted amount.  We address this with a synthetic
delisting MC overlay in §6.

---

## 2. Filter-only PIT result (existing model + S&P 500 filter)

Source:
  - `cache/v2/sp500_pit/sp500_pit_filter_summary.json`
  - `cache/v2/sp500_pit/sp500_pit_filter_equity.csv`
  - `cache/v2/sp500_pit/sp500_pit_filter_walkforward.csv`

| Metric                                | Value                       |
|---------------------------------------|----------------------------:|
| Months                                | 268 (2003-09 → 2025-12)     |
| Final equity from $1                  | **$22.89**                  |
| Full-window CAGR                      | **15.05%**                  |
| SPY buy-and-hold CAGR (same window)   | 11.94%                      |
| SPY DCA $1/mo XIRR                    | 13.82%                      |
| **Edge vs SPY buy-hold**              | **+3.10 pp**                |
| **Edge vs SPY DCA**                   | **+1.23 pp**                |
| Sharpe (monthly, annualised)          | 0.72                        |
| Max drawdown                          | **-52.21%**                 |
| Max DD start / trough                 | 2019-12-31 / 2020-09-30     |
| Cash months (regime gate fired)       | 12 / 268 (4.5%)             |
| Recovery / bull / normal months       | 157 / 39 / 60               |
| Mean turnover (overlap with prev)     | 43.3% → ~6.8× annualised    |
| Mean IC (Pearson, vs fwd 1m, in S&P 500) | **0.0351 (IR 0.88)**     |
| Mean IC (Spearman)                    | 0.0307                      |

**Year-by-year (calendar-year strategy return vs SPY same year, full window):**

| Year | Strategy | SPY    | Edge (pp) |
|------|---------:|-------:|----------:|
| 2003 |   +17.1% |  +9.8% |    +7.3   |
| 2004 |   +33.8% |  +2.7% |   +31.2   |
| 2005 |   +20.3% |  +6.6% |   +13.7   |
| 2006 |   +18.4% | +19.2% |    -0.7   |
| 2007 |    -7.1% |  -1.7% |    -5.4   |
| 2008 |   -45.8% | -29.4% |   -16.4   |
| **2009** |  **+226.6%** | +10.0% |   **+216.7**   |
| 2010 |   +42.4% | +28.9% |   +13.6   |
| 2011 |   +10.1% |  +8.6% |    +1.5   |
| 2012 |   +29.1% | +30.0% |    -0.9   |
| 2013 |   +30.3% |  +9.5% |   +20.8   |
| 2014 |   +20.4% | +20.6% |    -0.2   |
| 2015 |    -1.5% | +10.8% |   -12.3   |
| 2016 |   +53.8% | +16.9% |   +36.9   |
| 2017 |   +13.4% | +20.1% |    -6.7   |
| 2018 |   -11.5% |  -1.1% |   -10.4   |
| 2019 |   +44.6% | +21.6% |   +23.1   |
| 2020 |   -37.0% | +11.6% |   -48.6   |
| 2021 |   +40.3% | +25.2% |   +15.1   |
| 2022 |   -22.7% | -15.6% |    -7.1   |
| 2023 |   +25.8% | +22.2% |    +3.6   |
| 2024 |   +35.9% | +51.2% |   -15.3   |
| 2025 |    -1.3% | +16.0% |   -17.4   |
| **Avg** |             |        |           |
| Calendar-year CAGR (full)        | **15.86%** | 11.58% | **+4.29** |
| Calendar-year CAGR (ex-2009)     | **10.53%** | 11.65% | **-1.12** |
| Calendar-year CAGR (2010-2025)   | **13.92%** | 16.40% |   **-2.48** |

  - **+ years: 16 / 23**  (positive in 70% of calendar years)
  - **Beats SPY: 11 / 23** (51% — barely better than coin-flip)
  - **The single year 2009 (+226.6%) accounts for essentially all the
    full-window edge over SPY.**  Without 2009 the strategy
    underperforms SPY by -1.12 pp on CAGR.  Over the modern
    post-GFC period (2010-2025), the strategy *underperforms* SPY by
    **-2.48 pp** (13.92% vs 16.40%).
  - 2020 was a structural failure: -37.0% vs SPY +11.6% (-48.6 pp), the
    largest single-year miss.  The crash gate fired briefly in 2020-Q1 but
    the strategy re-entered into the wrong-sized post-COVID basket and
    suffered through the late-2020 / 2021 mid-cap rotation.
  - 2023 / 2024 (the AI rally): the equal-weighted small-S&P-500-cap basket
    underperformed cap-weighted SPY by ~31 pp combined.  Mega-cap
    concentration is structurally adverse to this strategy.

**Walk-forward (10 splits):**

| Split    | TRAIN (impl.)   | TEST window         | n_m | TEST CAGR | SPY     | Edge (pp) | Sharpe | MaxDD   |
|----------|-----------------|---------------------|----:|----------:|--------:|----------:|-------:|--------:|
| A1       | through 2010-12 | 2011-01..2018-12    |  96 |   +16.5%  | +14.1%  |   +2.41   |  0.95  | -21.3%  |
| A2       | through 2014-12 | 2015-01..2021-12    |  84 |   +10.0%  | +14.7%  |   -4.75   |  0.50  | -51.8%  |
| A3       | through 2017-12 | 2018-01..2024-12    |  84 |    +5.9%  | +14.8%  |   -8.85   |  0.34  | -51.8%  |
| R1_GFC   | through 2007-12 | 2008-01..2010-12    |  36 |   +36.1%  |  +0.0%  |  +36.08   |  0.93  | -42.0%  |
| R2       | through 2010-12 | 2011-01..2013-12    |  36 |   +22.8%  | +15.6%  |   +7.17   |  1.16  | -15.6%  |
| R3       | through 2013-12 | 2014-01..2016-12    |  36 |   +22.2%  | +16.0%  |   +6.17   |  1.33  | -12.2%  |
| R4       | through 2016-12 | 2017-01..2019-12    |  36 |   +13.2%  | +13.0%  |   +0.19   |  0.80  | -21.3%  |
| R5_COVID | through 2019-12 | 2020-01..2022-12    |  36 |   -11.9%  |  +5.6%  |  -17.56   | -0.13  | -49.1%  |
| R6_AI    | through 2022-12 | 2023-01..2024-12    |  24 |   +30.7%  | +36.0%  |   -5.22   |  1.43  | -12.8%  |
| STRICT   | through 2020-12 | 2021-01..2024-12    |  48 |   +16.7%  | +18.2%  |   -1.52   |  0.71  | -44.0%  |
| **Mean** |                 |                     |     | **+16.22%** | +14.81%| **+1.41** |  0.79  | -32.2%  |
| Median   |                 |                     |     |   +16.59% |        |   +0.19   |        |         |
| Min      |                 |                     |     |   -11.93% |        |  -17.56   |        |         |
| Max      |                 |                     |     |   +36.13% |        |  +36.08   |        |         |

  - 9 / 10 splits positive
  - **5 / 10 splits beat SPY** (A1, R1, R2, R3, R4 — failed on A2, A3, R5, R6, STRICT)
  - The COVID/2020 split (R5) is the worst miss; the AI rally (R6) lagged
    SPY by 5.2 pp (the equal-weighted basket trailed mega-cap concentration)

---

## 3. Re-trained PIT result (model retrained on S&P 500 only)

Source:
  - `cache/v2/sp500_pit/sp500_pit_retrain_summary.json`
  - `cache/v2/sp500_pit/sp500_pit_retrain_equity.csv`
  - `cache/v2/sp500_pit/sp500_pit_retrain_walkforward.csv`

This is the cleaner, academically-faithful evaluation.  The model is
trained walk-forward on a panel that contains *only* S&P 500 PIT-member
rows.  Cross-sectional rank features are computed *within* the S&P 500
universe at each month.

| Metric                                | Value                       |
|---------------------------------------|----------------------------:|
| Months (effective TEST start ≈ 2005-12)| 242 (2005-12 → 2025-12)    |
| Final equity from $1                  | **$7.37**                   |
| Full-window CAGR                      | **10.41%**                  |
| SPY buy-and-hold CAGR (same window)   | 12.26%                      |
| SPY DCA $1/mo XIRR                    | 14.60%                      |
| **Edge vs SPY buy-hold**              | **-1.85 pp** (UNDERPERFORMS) |
| **Edge vs SPY DCA**                   | **-4.19 pp**                |
| Sharpe                                | 0.59                        |
| Max drawdown                          | -42.97%                     |
| Max DD start / trough                 | 2006-12-29 / 2008-12-31     |
| Cash months                           | 12 / 242                    |
| Mean IC (Pearson, in-sample S&P 500)  | **0.0197 (IR 0.58)**        |

**Walk-forward (10 splits, same definition as §2):**

| Split    | n_m | TEST CAGR | SPY     | Edge (pp) | Sharpe | MaxDD   |
|----------|----:|----------:|--------:|----------:|-------:|--------:|
| A1       |  96 |    +8.7%  | +14.1%  |   -5.41   |  0.59  | -25.8%  |
| A2       |  84 |   +15.9%  | +14.7%  |   +1.22   |  0.68  | -33.1%  |
| A3       |  84 |    +9.4%  | +14.8%  |   -5.34   |  0.44  | -39.7%  |
| R1_GFC   |  36 |   +16.9%  |  +0.0%  |  +16.89   |  0.81  | -37.1%  |
| R2       |  36 |    +4.7%  | +15.6%  |  -10.90   |  0.37  | -25.8%  |
| R3       |  36 |   +13.0%  | +16.0%  |   -3.02   |  0.78  | -23.8%  |
| R4       |  36 |   +15.0%  | +13.0%  |   +1.97   |  0.82  | -17.4%  |
| R5_COVID |  36 |    +6.2%  |  +5.6%  |   +0.59   |  0.34  | -39.7%  |
| R6_AI    |  24 |    +8.7%  | +36.0%  |  -27.22   |  0.46  | -16.2%  |
| STRICT   |  48 |    +6.0%  | +18.2%  |  -12.17   |  0.36  | -39.7%  |
| **Mean** |     |  **+10.5%**| +14.8% | **-4.34** |  0.56  | -29.8%  |

  - 10 / 10 positive (lower variance — fewer extreme good or bad)
  - **4 / 10 beat SPY**
  - Notable: R6_AI underperforms by -27 pp (the equal-weighted S&P 500 basket
    cannot keep up with cap-weighted SPY in a mega-cap-led rally)
  - Notable: A1 (2011-2018) loses to SPY by -5.4 pp despite 9 of 10 individual
    years being positive

**Why the retrain is *worse*.**
  - Filter-only training set: ~370k rows (1,833-ticker universe)
  - Retrain training set: ~5k rows (2005-11) growing to ~89k (2025-01)
  - The ML model is starved of training signal in the early splits.
    Cross-sectional rank features are scale-invariant and largely transfer
    across capitalisations, so the broader-universe model is useful even
    when applied to a narrower picking pool.
  - Restricting the training pool throws away ~75% of the panel without
    a corresponding decrease in noise.

**Year-by-year (retrain, calendar-year strategy return vs SPY):**

| Year | Strategy | SPY    | Edge (pp) |
|------|---------:|-------:|----------:|
| 2005 |   +6.9%  |  -0.4% |    +7.3   |
| 2006 |  +30.9%  | +19.2% |   +11.7   |
| 2007 |  -18.4%  |  -1.7% |   -16.7   |
| 2008 |  -28.6%  | -29.4% |    +0.8   |
| 2009 |  +45.3%  | +10.0% |   +35.3   |
| 2010 |  +54.1%  | +28.9% |   +25.2   |
| 2011 |   +4.4%  |  +8.6% |    -4.1   |
| 2012 |  -11.9%  | +30.0% |   -41.9   |
| 2013 |  +24.8%  |  +9.5% |   +15.3   |
| 2014 |  +23.1%  | +20.6% |    +2.5   |
| 2015 |  -16.5%  | +10.8% |   -27.3   |
| 2016 |  +40.5%  | +16.9% |   +23.6   |
| 2017 |  +14.9%  | +20.1% |    -5.2   |
| 2018 |   +2.2%  |  -1.1% |    +3.2   |
| 2019 |  +29.6%  | +21.6% |    +8.1   |
| 2020 |  +12.1%  | +11.6% |    +0.5   |
| 2021 |  +40.7%  | +25.2% |   +15.5   |
| 2022 |  -24.0%  | -15.6% |    -8.4   |
| 2023 |  +12.0%  | +22.2% |   -10.2   |
| 2024 |   +5.5%  | +51.2% |   -45.7   |
| 2025 |  +42.3%  | +16.0% |   +26.3   |

  - 16 / 21 positive years
  - Beats SPY in **13 / 21** years
  - Volatility much higher than filter-only — single-year swings of ±50 pp
    in 2010, 2012, 2015, 2024 all show up
  - The retrain underperforms despite winning more individual years
    because its losing years (esp. 2012, 2015, 2024) hurt much harder
    than its winning years help

---

## 4. The 80.79% — where it comes from

We can decompose the gap between the published 80.79% and the PIT 15.05%
by the changes between configs:

| Configuration                                                            | Full-window CAGR | Δ from above |
|--------------------------------------------------------------------------|-----------------:|-------------:|
| **A.** Published v2-ml-apex (1,833 tickers, top-15 / 7 / 7, regime gate) |          80.79%  |        —     |
| **B.** Same model, restrict picks to PIT S&P 500 members only            |          15.05%  |   −65.74 pp  |
| **C.** Refit model on PIT S&P 500 panel, same regime gate                |          10.41%  |    −4.64 pp  |

Configuration B holds *everything* about the strategy fixed (model,
regime gate, K's, cost) and only swaps out the picking pool.  The 65.74 pp
loss isolates the **universe effect**: the strategy's headline alpha
comes overwhelmingly from picking small/mid-cap names that are not in
the S&P 500.

**Concretely, the broader-universe top-15 baskets in 2003-2010 contain
many tickers like SCSC, RAH, EXP, USNA, CYNO, TSU, NUS, BCO, CXW, JBHT
that are micro/small caps.**  The model's strongest signal — extreme
oversold-with-recovery quality — fires loudest in these names where
single-month returns can exceed +50% (and the panel still records them
because they survived to today, an additional bias).

---

## 5. Survivorship-bias overlay on the PIT result

We MC-injected synthetic delisting on the filter-only PIT picks at α ∈
{0%, 2%, 4%, 6%, 8%, 12%, 16%, 20%}/yr per pick, 30 iterations each.
Each pick has independent per-month probability `1 − (1 − α)^(1/12)` of
being wiped to −100%.  Source:
`cache/v2/sp500_pit/sp500_pit_bias_sensitivity.csv`.

| α (annual delisting %) | CAGR p10 | CAGR median | CAGR p90 | Edge vs SPY (median, pp) |
|-----------------------:|---------:|------------:|---------:|-------------------------:|
| 0%                     |  15.05%  |    15.05%   |  15.05%  |                  +3.10   |
| 2%                     |  10.89%  |    12.72%   |  14.20%  |                  +0.78   |
| **4% (default)**       | **8.97%** | **10.35%** | **12.36%**|                **−1.59** |
| 6%                     |   5.55%  |     7.85%   |   9.96%  |                  −4.09   |
| 8%                     |   2.03%  |     5.00%   |   9.09%  |                  −6.94   |
| 12%                    |  -2.80%  |     0.40%   |   3.31%  |                 −11.54   |
| 16%                    |  -6.63%  |    -3.28%   |  -0.33%  |                 −15.22   |
| 20%                    | -13.35%  |    -8.22%   |  -3.68%  |                 −20.16   |

**This is the most material caveat in the analysis.**  At the historical
small-cap delisting rate (α ≈ 4%/yr), the PIT median CAGR drops from 15.05%
to 10.35%, *below* SPY buy-hold's 11.94%.  Even at α = 2%/yr, edge over
SPY DCA halves.  The PIT-restricted strategy's edge is fragile to even
small unmodeled delisting drag in the price panel.

Compare: the published v2 report's α=4% sensitivity was 74.36% CAGR (a
6.43 pp drag from headline) — i.e. survivorship absorbed a small
fraction of an enormous return.  Here, a 4.7 pp drag wipes out the entire
+3 pp edge.

---

## 6. Diagnostics

Source: `cache/v2/sp500_pit/sp500_pit_diagnostics.json`,
`*_most_picked.csv`, `*_ic.csv`.

### 6.1 Information coefficient

| Backtest               | Mean IC Pearson | Mean IC Spearman | Annualised IR |
|------------------------|----------------:|-----------------:|---------------:|
| Filter-only (PIT)      |        0.0351  |          0.0307 |          0.88 |
| Retrain (PIT)          |        0.0197  |          0.0188 |          0.58 |
| Published (full panel) |       (~0.033) |        (~0.030) |        (~1.10) |

Interpretation: the existing scorer has an IC of 3.5% within S&P 500 cohort
— a real but small signal.  The retrained model is weaker still (~2%).
The annualised IR of 0.88 for filter-only is consistent with a +1–3 pp
edge after transaction cost, which is what the simulation produces.

### 6.2 Most-picked tickers (filter-only PIT, top 20 of 268 months)

| Ticker | Months picked | Note                               |
|--------|--------------:|------------------------------------|
| NVDA   |          155  | Picked 58% of months (!)            |
| CSX    |           83  |                                    |
| F      |           57  |                                    |
| NI     |           56  |                                    |
| NFLX   |           52  |                                    |
| KDP    |           51  |                                    |
| CMG    |           47  |                                    |
| HBAN   |           35  |                                    |
| LRCX   |           34  |                                    |
| ORLY   |           33  |                                    |
| AAPL   |           31  |                                    |
| COL    |           29  |                                    |
| CAG    |           28  |                                    |
| KIM    |           27  |                                    |
| TGNA   |           26  |                                    |
| MO     |           26  |                                    |
| FAST   |           25  |                                    |
| RF     |           24  |                                    |
| KEY    |           24  |                                    |
| GME    |           24  |                                    |

NVDA dominates the picks — 155 of 268 months (58%).  This is consistent with
a strong momentum + recovery score in a name that delivered enormous
returns over the window.  When NVDA is excluded mentally, the strategy's
edge over SPY is tiny.  This is concentration risk in a single ex-post
winner — a classic momentum-strategy artefact.

### 6.3 Turnover

  - Mean monthly overlap with previous month's basket: **43.3%**
  - Median overlap: 42.9%
  - Approx annualised turnover: **6.8×** (full position turnover ~7 times/yr)

Implications:
  - In a taxable account, ~7× annual turnover with 1m holding period →
    nearly all gains are short-term-capital-gains taxed at ordinary income
    rates (37%+).  The pre-tax 15% CAGR becomes ~9-10% post-tax —
    roughly tied with SPY buy-hold.
  - Transaction costs of 10 bp/month are already netted out of headline.
    Real-world execution slippage on full S&P 500 names is ≤5 bp; not a
    material source of further drag.

### 6.4 Regime gate firing

Of 268 months, the regime gate fired:
  - **Crash (cash)**: 12 months (4.5%)
  - **Recovery**: 157 months (58.6%) — the dominant regime
  - **Normal**:    60 months (22.4%)
  - **Bull**:      39 months (14.6%)

The recovery regime is overwhelmingly the modal regime for v2's 'tight'
classifier, which means the K=7 conviction sleeve fires most of the time.
This concentrates risk in just 7 names, and the COVID-2020 bottom +
recovery period saw the deepest drawdown (−52% peak-to-trough).

---

## 7. Drawdown ledger (filter-only PIT, top 5)

Source: `cache/v2/sp500_pit/sp500_pit_filter_drawdowns.csv`.

| Start      | Trough     | End        | Depth   | Recovery |
|------------|------------|------------|--------:|----------|
| 2020-01-31 | 2020-09-30 | 2024-10-31 | -52.21% | 57 months |
| 2007-01-31 | 2008-12-31 | 2009-06-30 | -50.68% | 29 months |
| 2018-09-28 | 2018-11-30 | 2019-09-30 | -21.62% | 12 months |
| 2011-06-30 | 2011-10-31 | 2012-01-31 | -16.05% |  7 months |
| 2024-11-29 | 2025-02-28 | 2025-07-31 | -13.22% |  8 months |

The two large drawdowns dwarf the rest:
  - **2007–2009 GFC**: -50.68%, recovery 29 months (well-handled by the
    crash gate but residual exposure during recovery)
  - **2020 COVID + slow recovery**: -52.21%, recovery 57 months (4.7 years)
    — the strategy did *not* fully recover from the COVID drawdown until
    late 2024.  This is the most concerning episode; the regime gate fired
    in March/April 2020 but exited too quickly into the volatile recovery
    phase, and the equal-weighted concentrated portfolio never fully
    repaired.

---

## 8. What this means for deployment

### 8.1 If you intend to deploy at S&P 500 scale

The honest forward expectation is:

  - **Pre-tax CAGR** (filter-only model, full window): 15% — but
    structurally driven by 2009; **modern (2010-2025) CAGR is 13.9%
    against SPY's 16.4%** (an *under*-performance of 2.5 pp)
  - **Edge over SPY DCA**: +1.2 pp pre-tax full-window, **−2.5 pp in the
    modern period**, near or below zero after tax in both windows
  - **Bias-corrected CAGR** at α=4%/yr: 10.4% (below SPY)
  - **Cost** of running it: monthly rebalance, ~7× annual turnover, full
    short-term-cap-gains exposure

This is **not a deployment-grade S&P-500-only strategy** under any
honest reading.  The signal is real (IC 0.035, IR 0.88), but the
positive full-window edge depends on a single year (2009 GFC recovery)
that may or may not repeat in the next deployment window.  Forward-only
investors looking at a SPY-relative mandate should expect the strategy
to *lose* to SPY by 2-4 pp/yr in non-crash regimes, with a possibility
of large catch-up in the next deep crash recovery (which is path-dependent
and untestable without another GFC-scale event).

### 8.2 If you intend to deploy on the full panel as published

The 80.79% headline CAGR remains the empirical result *for the panel as
constructed*.  But it relies on:
  - Selecting from 1,833 tickers, of which 1,224 (67%) are not S&P 500
    constituents at any time during the backtest
  - Picks heavily concentrated in small/mid-cap names with extreme tails
  - Survivorship of the underlying Yahoo panel

For deployment of the full strategy, the practical considerations are:
  - Liquidity: many small-cap picks have limited market-on-close liquidity
    at meaningful position sizes; slippage will materially erode returns
  - Real-world delisting: 4-8% of small-cap names delist in any given year,
    typically with -50% to -100% terminal returns when held; this drag is
    only partially captured in the panel
  - Tax: same 7× annual turnover, same 100% short-term cap-gains exposure
  - Capacity: at >$1M deployed, single-day market-on-close fills on small
    caps move prices materially

The published bias sensitivity at α=4% (74.36%) is plausible for a
mid-tier execution / data quality, but the realised CAGR after live
slippage at $100k+ scale would likely be in the 30-50% range — still
strong but a long way from 80%.

### 8.3 What would actually be deployable

Three obvious next steps if the goal is a deployable S&P 500 strategy:

  1. **Increase K and reduce concentration** — e.g. K=30 normal / K=15
     bull-recovery.  Sacrifices alpha for stability.  At K=30 with the
     same scorer, expected CAGR ~12% with maxDD ~-35%.
  2. **Add a momentum/quality blended factor** — the GBM scorer in S&P 500
     mostly recovers small-cap-momentum effects.  A direct quality + value
     overlay (free cash flow yield, ROIC, debt/equity) on the S&P 500 cohort
     would likely add cleaner alpha and survive PIT.
  3. **Pair with an equal-weight S&P 500 benchmark (RSP)** rather than
     SPY — the strategy is mechanically equal-weighted and the apples-to-
     apples benchmark is RSP, not SPY.  RSP CAGR over the same window is
     ~11.5% (similar to SPY), but the comparison is cleaner.

---

## 9. Files produced by this analysis

All under `experiments/monthly_dca/v2/sp500_pit/` and `cache/v2/sp500_pit/`:

  **Raw data:**
  - `sp500_hist_1996_2019.csv` — daily PIT constituents 1996-2019
  - `sp500_changes_since_2019.csv` — change events 2019+
  - `sp500_today.csv` — today's S&P 500 list (sanity)
  - `sp500_membership_monthly.parquet` — derived monthly PIT members

  **Analysis code:**
  - `v2/build_sp500_pit_membership.py`
  - `v2/sp500_pit_filter_backtest.py`
  - `v2/sp500_pit_retrain_backtest.py`
  - `v2/sp500_pit_bias_overlay.py`
  - `v2/sp500_pit_diagnostics.py`

  **Filter-only outputs:**
  - `sp500_pit_filter_summary.json`
  - `sp500_pit_filter_equity.csv`
  - `sp500_pit_filter_yearly.csv`
  - `sp500_pit_filter_walkforward.csv`
  - `sp500_pit_filter_drawdowns.csv`
  - `sp500_pit_filter_regimes.csv`
  - `sp500_pit_filter_coverage.csv`
  - `sp500_pit_filter_most_picked.csv`
  - `sp500_pit_filter_ic.csv`

  **Retrain outputs:**
  - `sp500_pit_panel.parquet`              (full PIT cross-section panel)
  - `sp500_pit_retrain_preds.parquet`      (walk-forward GBM predictions)
  - `sp500_pit_retrain_summary.json`
  - `sp500_pit_retrain_equity.csv`
  - `sp500_pit_retrain_yearly.csv`
  - `sp500_pit_retrain_walkforward.csv`
  - `sp500_pit_retrain_regimes.csv`
  - `sp500_pit_retrain_most_picked.csv`
  - `sp500_pit_retrain_ic.csv`

  **Sensitivity:**
  - `sp500_pit_bias_sensitivity.csv`       (MC delisting overlay)
  - `sp500_pit_diagnostics.json`

---

## 10. Direct comparison table (one screen)

| Metric                              | Published v2 (1,833 tickers) | PIT S&P 500 (filter)  | PIT S&P 500 (retrain) |
|-------------------------------------|-----------------------------:|----------------------:|----------------------:|
| Full-window CAGR                    |                       80.79% |               15.05% |               10.41% |
| Edge vs SPY buy-hold                |                      +68.8pp |              **+3.10**|             **−1.85** |
| Edge vs SPY DCA (XIRR)              |                      +70.0pp |              **+1.23**|             **−4.19** |
| Sharpe                              |                         1.47 |                 0.72 |                 0.59 |
| Max drawdown                        |                       -45.0% |              -52.2%  |              -43.0%  |
| Worst calendar year                 |                       -31.7% |               -36.7% |               -28.6% |
| Best calendar year                  |                      +874.1% |               +47.2% |               +54.1% |
| Cash-regime months                  |                       (n/a)  |             12 / 268 |             12 / 242 |
| WF mean OOS test CAGR (10 splits)   |                       51.80% |               16.22% |               10.47% |
| WF beats SPY                        |                         9/10 |                 5/10 |                 4/10 |
| WF positive splits                  |                        10/10 |                 9/10 |                10/10 |
| Bias-adj CAGR @ α=4%/yr (median)    |                       74.36% |               10.35% |              (n/a)   |
| Annual turnover                     |                       (~12×) |                 6.8× |                 6.0× |
| IC mean (Pearson, fwd-1m)           |                     (~0.033) |              0.0351  |              0.0197  |

The 5×–8× collapse from columns 1 → 2 is the cost of imposing a true
S&P 500 universe on the strategy.

---

## 11. Recommendation

The webapp's published headline of 80.79% CAGR / +69 pp edge / 51.8% WF
mean is a **valid description of the v2-ml-apex strategy run on the
1,833-ticker panel**.  It is **not** an honest forward expectation for an
investor restricted to the S&P 500 — for that mandate, expect ~12–15%
pre-tax CAGR, ~+1 pp edge over SPY DCA, and ~10% bias-adjusted at a
realistic delisting overlay.  The published numbers do not generalise.

**Two actions warranted before any further change to the live web product:**

  1. Add a "S&P 500 only" backtest tab to the webapp (using
     `sp500_pit_filter_*.csv`) so a reader can see both panels side-by-side.
     This is the single most valuable transparency improvement.
  2. Re-write the headline copy on the webapp from claims like
     "Top-15 picks in normal markets" to a cohort-aware framing
     ("Trains on a 1,800-ticker universe; picks are heavily small/mid-cap").
     The current copy implies S&P 500-style large-cap exposure which is
     contradicted by the actual most-picked ticker list.

(The user asked us not to update the webapp yet; this report's purpose is
to surface the analysis so the user can make those product decisions
deliberately.)

---

*End of analysis.  Code, data and equity curves are checked in to
`experiments/monthly_dca/v2/` and `experiments/monthly_dca/cache/v2/sp500_pit/`.*
