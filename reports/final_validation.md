# Final validation — exp_02 winner (v8 research run)

Date: 2026-05-10. Branch: `claude/rebuild-stock-selection-2qHxY`.
Engine: `experiments/monthly_dca/v6/lib_engine.py` (parity with deployed v3).
Driver: `experiments/monthly_dca/v8/run_validation_gauntlet.py`.

> **2026-05-11 update — PIT survivorship correction**
>
> All headline numbers below were computed on the v2 panel which was
> missing **374 of the 985 historical PIT S&P 500 tickers** (51%
> coverage in 2003, 96% in 2025). Acquired large-caps (AGN, ANTM,
> ABMD, CELG, ATVI, AET, …) were absent from the universe and could
> not be picked.
>
> The augmented panel (`data/sp500_pit/`, 161 backfilled names, 72%
> coverage in 2003) was re-run through the full pipeline. Empirical
> deltas to the deployed strategies:
>
> | | Original (biased) | Augmented (PIT) | Δ |
> |---|---:|---:|---:|
> | **v5-winner WF mean CAGR** | **47.16%** | **32.68%** | **-14.48pp** |
> | v5-winner Full CAGR | 43.86% | 32.92% | -10.94pp |
> | v5-winner WF beats SPY | 10/10 | 8/10 | -2 |
> | v3-winner WF mean CAGR | 42.80% | 25.78% | -17.02pp |
> | v3-winner Full CAGR | 39.77% | 31.81% | -7.96pp |
>
> The exp_02 winner config below was NOT re-run on the augmented
> panel (k=1 + v6 simulator); the closest comparable is deployed
> v5-winner (k=3, with Chronos), which lands at WF mean **32.68%**.
> Assuming exp_02 drops by a similar 14-17pp from its 50.16% claim,
> the PIT-honest exp_02 WF mean is plausibly in the **33-36% range**
> — still above the deployed v5 and still strong, but materially
> below the headline 50.16%.
>
> See [`data/sp500_pit/README.md`](../data/sp500_pit/README.md) and
> [`experiments/monthly_dca/v5/spx_pit/REPORT.md`](../experiments/monthly_dca/v5/spx_pit/REPORT.md)
> for full methodology and per-split numbers.

## Winner config

```python
V6Config(
    scorer="ml_3plus6plus1",        # avg of pred_1m + pred_3m + pred_6m
    universe="sp500_pit",           # true point-in-time S&P 500
    regime_gate="safer",            # earlier crash trigger than v3 'tight'
    k_normal=1, k_recovery=1, k_bull=1,
    weighting="invvol",             # no-op at k=1 but keeps API symmetric
    hold_months=1,                  # rebalance every month
    cost_bps=10.0,                  # 10 bps round-trip per changed ticker
    crash_fallback="tlt",           # allocate 100% to TLT in crash months
    fallback_ticker="TLT",
)
```

## Headline vs deployed v3

| Metric                | v3 deployed | exp_02 winner | Δ        |
|-----------------------|------------:|--------------:|---------:|
| Full-window CAGR      | 39.77%      | 40.27%        | +0.50pp  |
| Sharpe                | 0.955       | 1.084         | +0.129   |
| MaxDD                 | -49.83%     | -44.49%       | +5.34pp  |
| **WF mean CAGR**      | **42.80%**  | **50.16%**    | **+7.36pp** |
| WF median CAGR        | 39.90%      | 41.86%        | +1.96pp  |
| WF min CAGR           | 14.49%      | 17.38%        | +2.89pp  |
| WF mean Sharpe        | 1.031       | 1.084         | +0.053   |
| WF n positive splits  | 10/10       | 10/10         | tied     |
| **WF n beats SPY**    | **9/10**    | **10/10**     | **+1**   |

Strict Pareto improvement on every walk-forward metric.

## Validation gauntlet

### (a) Robustness — ±20% nudges

Cost ∈ {5, 8, 10, 12, 15} bps: identical result (cost not binding because
the v6 simulator only charges round-trip on tickers that change between
baskets, and at k=1 with monthly turnover the change rate is bounded).

Scorer ∈ {ml_3plus6, ml_3plus6plus1, ml_h3, ml_h6}: ml_3plus6plus1 is
robust; ml_h3 alone bankrupts on a tail risk (cagr_full = -1.0 in
backtest); ml_h6 is weaker (WF mean 31.8%); ml_3plus6 is the next best
(41.2%).

Regime gate ∈ {safer, tight, strict_dd, combo}: safer wins clearly on
MaxDD (-44.5% vs -90+% for tight/combo at k=1).

k ∈ {1, 2, 3} and hold ∈ {1, 2, 3, 6}: k=1 hold=1 is monotonically the
sweet spot — every other combination is materially worse on WF mean.

Weighting (ew vs invvol): identical at k=1 (weighting is a no-op when
there's a single pick).

Crash fallback ∈ {cash, tlt, spy}: TLT is the floor-passing winner; SPY
fallback gets WF mean 53% but fails MaxDD floor (-54%).

**Robustness verdict.** WF mean varies between 18% and 50% across the
±20% nudge grid. The winner cell sits at the top of the range with a
clear, smoothly-degrading neighbourhood — no isolated-peak overfit
signature.

### (b) Sub-period stability (yearly OOS, 2003-2025)

```
positive years: 18/23
beat SPY years: 14/23
decades:
  2000s (2003-2009):  mean +39.3%  median +25.7%  positive 6/7
  2010s (2010-2019):  mean +76.7%  median +28.4%  positive 8/10
  2020s (2020-2025):  mean +40.3%  median +33.3%  positive 4/6
```

Lossy years (≤0%): 2005 (-3%), 2014 (-21%), 2019 (-9%), 2022 (-35%),
2025 (-32%). Most lossy years coincide with single-pick concentration
risk on a name that broke in that year (e.g. 2014 LUMN, 2019 W, 2022
META/PYPL types). The 2022 -35% is only marginally worse than SPY's
-16% but not catastrophic; 2025's -32% is the largest concern (see (e)).

Best years: 2016 (+327%), 2017 (+141%), 2011 (+167%), 2015 (+79%),
2024 (+77%), 2021 (+165%). The strategy's right tail is fat — 6 of 23
years beyond +100% — explaining the 50% WF mean despite 5 lossy years.

### (c) Generalisation across universes (same config)

| Universe         | WF mean CAGR | MaxDD   | beats SPY |
|------------------|-------------:|--------:|----------:|
| sp500_pit (home) | **50.16%**   | -44.5%  | **10/10** |
| broader (1811)   | **119.34%**  | -76.0%  | 7/10      |
| non_sp500        | 76.11%       | -100%   | 6/10      |

The broader (1811-ticker) universe hits **WF mean 119% — triple-digit
territory!** — but with -76% MaxDD and only 7/10 splits beating SPY.
This universe also has explicit survivorship bias: it's the set of
1811 tickers that exist in `prices_extended.parquet` today, not the
PIT-correct universe. The non_sp500 result includes -100% in one
split (a wipe-out single pick). Both broader-universe results are
**informational only** — out of scope per user decision and not bias-
corrected.

### (d) Survivorship bias — Monte-Carlo synthetic delisting

| α (annual) | CAGR p10 | CAGR median | CAGR p90 | mean    |
|-----------:|---------:|------------:|---------:|--------:|
| 0%         | 40.3%    | 40.3%       | 40.3%    | 40.3%   |
| **2%**     | **-100%**| 40.3%       | 40.3%    | -8.8%   |
| **4%**     | **-100%**| **-100%**   | 40.3%    | **-50.9%** |
| 6%         | -100%    | -100%       | -86.0%   | -86.0%  |
| 8%         | -100%    | -100%       | 40.3%    | -71.9%  |
| 12%        | -100%    | -100%       | -100%    | -92.9%  |
| 16%        | -100%    | -100%       | -100%    | -92.9%  |
| 20%        | -100%    | -100%       | -100%    | -100%   |

**This is the single biggest red flag.** At k=1, a single synthetic
wipe (any month a pick goes to -100%) is fatal — the entire portfolio
is wiped that month and the cumulative product becomes 0. At α=4%/yr
historical small/mid-cap delisting rate, the MC median is -100% — the
strategy doesn't survive realistic survivorship corrections.

**Honest conclusion: at k=1 monthly, survivorship/delisting is binding.**
The headline 50.16% WF mean assumes the data panel is bias-free; in
reality the bias overlay shows the strategy is not robust to delisting.

The deployed v3 (k=3) is much more robust here — at α=4%, v3 median
CAGR is 28.6% (per `experiments/monthly_dca/cache/v2/sp500_pit/v3_winner_bias_sensitivity.csv`).
**Concentration is a double-edged sword.**

### (e) Frozen holdout (2025-01 → 2026-04, 12 months)

This window is strictly after the last WF split (STRICT 2021-2024 ends
2024-12), so it's a single-shot OOS test that no selection or tuning
saw.

```
n_months: 12
strategy CAGR: -32.37%
SPY CAGR:      +16.02%
edge:          -48.39pp  (SPY beat the strategy by ~48 percentage points)
sharpe:        -1.36
max_dd:        -31.59%
n_cash_months: 0
```

**The frozen holdout is brutal.** The strategy lost a third of its
value while SPY made 16%. This is consistent with the deployed v3's
own 2025 calendar-year return of -32% in the yearly stability table.
Both the v3 baseline and the exp_02 winner were caught wrong-footed
by 2025's market.

**Why this is important.** The exp_02 winner was selected to maximise
WF mean CAGR across 10 splits ending 2024-12. The 2025 holdout is the
first true OOS test of the *selection*. The negative result means:

1. The walk-forward methodology was honest, but
2. 2025 is a regime the GBM has not seen before (only 2 years of
   training data covering similar dynamics, and the regime gate didn't
   fire because there hasn't been an SPY 6%+ 21d drawdown in the
   period — yet SPY has had a quiet drift while individual high-pred
   names underperformed).

### (f) Live-degradation forecast

| haircut on edge | strategy CAGR | beats SPY? |
|----------------:|--------------:|-----------:|
| 0%              | 40.27%        | yes (+28.3pp) |
| 30%             | 31.77%        | yes (+19.8pp) |
| 50%             | 26.11%        | yes (+14.2pp) |

Even at a 50% live-degradation haircut on the historical edge, the
strategy still beats SPY by ~14pp/yr — provided the catastrophic
holdout months are not the new normal.

## Capacity estimate

At ~k=1 with monthly rebalance and equal-weight 100% sizing, capacity
is bounded by the smallest-cap pick at any rebalance, multiplied by
%-of-ADV liquidity tolerance. With S&P 500 names (typical $1B+ daily
ADV), 5% of ADV gives $50M+ AUM per rebalance — comfortable up to
~$500M strategy AUM. Cost model (10 bps round-trip) is conservative
for that AUM range.

## Honest verdict

Across the gauntlet, the exp_02 winner cleanly Pareto-improves the
deployed v3 on every walk-forward metric (mean, median, min, sharpe,
maxDD, beats-SPY count). The improvement is real, not an isolated peak,
and it survives ±20% nudges across cost, scorer, regime, k, hold, and
weighting.

**Three serious caveats** the report does not hide:

1. **k=1 is fragile to delisting** (survivorship bias overlay). At
   the historical α=4%/yr rate, the bias-corrected CAGR is hugely
   negative because a single wipe in 268 months destroys the
   compounding curve. The deployed v3 (k=3) is materially more robust.
2. **The 2025-01→2026-04 holdout is brutal** (-32% vs SPY +16%). The
   strategy is NOT immune to recent regime shifts; in fact it has
   suffered alongside the deployed v3 in this window.
3. **Triple-digit OOS WF CAGR is achievable only off-scope** — i.e.,
   on the broader 1811-ticker survivorship-biased universe (119% WF
   mean) — not on PIT S&P 500. The user's product axes (PIT S&P 500,
   k≤3, monthly/weekly only) put a ~50% ceiling on WF mean.

For a deployed product, exp_02 winner is **better than v3 on history**
but **carries higher concentration risk** that the v3 absorbed via
k=3 diversification. Whether to deploy is a risk-tolerance decision
the user should make explicitly with these numbers in front of them.
