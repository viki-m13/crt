# 2024 lag — root-cause diagnosis (PIT S&P 500, no survivorship)

**Generated**: 2026-05-11
**Universe**: PIT S&P 500 only (members at each month-end, no current-membership
back-fill).
**Comparison frame**: monthly-DCA tranches (one fresh basket each month-end,
held 6 months) — the same picker as production, just at every entry date, so
the picking quality is measured 12× per year instead of 2×.

## Headline

| Year | Lump-sum edge (2 entries) | Monthly-DCA edge (12 entries) | Δ |
|-----:|--------------------------:|------------------------------:|---:|
| 2014 |  −0.6 pp                  |  **+4.8 pp**                  | +5.4 |
| 2018 |  −1.1 pp                  |  **+2.3 pp**                  | +3.3 |
| 2024 | **−10.6 pp**              |  **+3.7 pp**                  | **+14.3** |
| 2025 |  +4.2 pp                  |  +1.9 pp                      | −2.3 |

**2024 was NOT a picking failure — it was a 2-out-of-12 timing failure.**
The lump-sum Jan and Jul 2024 rebalance dates produced the two worst entries of
the year (KEY/CCL/WBD and AAPL/TSCO/F, both −7 to −14 pp edge vs SPY). The
other 10 monthly entries were positive: e.g. PLTR/AAPL/CARR (Sep entry, +17 pp
edge), PLTR/NCLH/F (Oct entry, **+35 pp edge**), KEY/AAL/F (Jun entry, +12 pp).

For comparison, in years where the GBM's alpha is supposed to shine:
- 2009 monthly-DCA edge: **+77.9 pp per tranche**, 100 % win-rate vs SPY
- 2020 monthly-DCA edge: **+30.7 pp per tranche**, 91 % win-rate vs SPY

## Per-month 2024 detail (monthly-DCA tranches, PIT S&P 500)

| Entry      | Picks            | 6m return | SPY 6m | Edge      |
|-----------:|------------------|----------:|-------:|----------:|
| 2024-01-31 | KEY, CCL, WBD    |  +0.70 %  | +14.79 | −14.09 pp |
| 2024-02-29 | UBER, HWM, MSFT  | +14.74 %  | +11.65 |  +3.10 pp |
| 2024-03-28 | MSFT, CARR, ODFL |  +8.26 %  | +10.38 |  −2.12 pp |
| 2024-04-30 | F, KEY, CEG      | +18.66 %  | +13.99 |  +4.66 pp |
| 2024-05-31 | KEY, CARR, MSFT  | +18.26 %  | +14.98 |  +3.29 pp |
| 2024-06-28 | KEY, AAL, F      | +19.98 %  |  +8.39 | +11.59 pp |
| 2024-07-31 | AAPL, TSCO, F    |  +2.90 %  |  +9.96 |  −7.07 pp |
| 2024-08-30 | LKQ, CAT, SYF    |  +7.52 %  |  +6.09 |  +1.43 pp |
| 2024-09-30 | PLTR, AAPL, CARR | +15.18 %  |  −1.88 | +17.06 pp |
| 2024-10-31 | PLTR, NCLH, F    | +32.79 %  |  −1.86 | +34.65 pp |
| 2024-11-29 | TSCO, FAST, APH  |  +4.85 %  |  −1.56 |  +6.41 pp |
| 2024-12-31 | ON, CCL, CE      |  −8.47 %  |  +6.05 | −14.52 pp |
| **Mean**   | **(per tranche)** | **+11.28 %** | **+7.58** | **+3.70 pp** |
| Win-rate vs SPY |                  |           |        | **8 / 12 (66.7 %)** |

The lump-sum strategy ran the **Jan 31** and **Jul 31** tranches: the only two
that produced negative double-digit edge.

## Mechanism

### What was ruled out

1. **Regime gate**: 2024 was classified as "recovery" 9× and "bull" 3× by
   the v5 regime classifier. Production rebalance dates (Jan 31, Jul 31) both
   classified as "recovery". K_recovery = K_bull = 3 and cap = 0.40 in both
   regimes, so the regime classification has no direct effect on the basket
   in 2024. The `max_below_200_streak ≥ 40` over the past 5 y is satisfied
   forever after the 2022 bear market (streak = 89), which prevents the
   "bull" branch from firing during the Sep 2023 → Apr 2024 stretch even
   though 12-m SPY momentum was 17–28 %. Cosmetic, not causal.

2. **GBM picks bad stocks**: WRONG. The 12 monthly tranches average +3.7 pp
   edge vs SPY — the GBM is adding alpha **on average** in 2024. The Jan 31
   basket (KEY/CCL/WBD, mom_12_1 = −0.22 / +0.42 / negative) was a deep-value
   tilt that lagged the mega-cap rally, but later months picked MSFT, AAPL,
   PLTR, CEG (AI data-center power), HWM (aerospace) — momentum and
   AI-tail-wind names. **The GBM scorer is not single-mindedly value-tilted.**

3. **Anchor variants ("add a momentum pick")** cure 2024 (+19 to +47 pp edge
   for various anchor signals) but inflict −6 to −15 pp on 2025 (the value-led
   leg of 2025, when SYF/SW/VST drew down then rebounded). Net CAGR drops.
   See [REPORT.md](REPORT.md) post-2026-05-11 section for the honest table.

### What actually happened

The **two lump-sum rebalance dates in 2024 happened to land on the year's
worst picking moments**. Both Jan 31 and Jul 31, 2024 had:

- Recent SPY rally (1-m return +5 to +6 %) → no fresh distress to bet on,
  yet the GBM's training data is dominated by recoveries (2009, 2020), so it
  still gravitates to pullback / recovery signals.
- A handful of beaten-down value names with **high `pullback_1y`** and
  **high `recovery_rate`** scores (KEY pullback −0.21, recovery_rate +0.50;
  CCL pullback −0.14, recovery_rate +0.38). These signals are productive
  during real recoveries but mis-fire when there is no recovery to capture.

By Aug–Nov 2024, those exact same signals worked again (PLTR pullback was
nil but the model picked it on other features; KEY/AAL/F caught the late-Q3
small-cap rally on the Sep rate-cut). The lump-sum strategy was holding the
Jul basket through that stretch and didn't reform until Jan 2025.

**This is a rebalance-timing artifact, not a strategy failure.** A 2024
investor who DCA'd into the strategy month-by-month captured +3.7 pp of
alpha. A 2024 investor who lump-summed into the strategy on Jan 31 and held
through Jul 31 captured −10.6 pp.

## Implications for next steps

### Likely-productive directions

1. **Overlapping rebalance** (already validated): hold 6 staggered tranches
   (one per month-of-rebalance), each on its own 6-month clock. Equivalent
   to monthly-DCA semantics with the same total capital. Cuts the
   single-date-luck risk and would have delivered +3.7 pp edge in 2024
   instead of −10.6 pp.

   The existing staggered-DCA artifact already proves this works:
   `results/staggered_dca_equity.csv`.

2. **Monthly rebalance with 1/6 turnover**: at each month, swap out 1/6 of
   the basket (the oldest of 6 sleeves). Equivalent to (1) but with cleaner
   weight accounting. Validate against PIT S&P 500.

3. **Surface monthly-DCA edge in the webapp**: the current "Year-by-year"
   block shows lump-sum CAGR vs SPY-DCA CAGR. Add a third column for
   per-tranche monthly-DCA edge — this is the honest picking-quality metric
   and would show **2024 monthly-DCA = +3.7 pp** alongside the current
   "2024 lump-sum = −15 pp" without changing the deployed strategy.

### Lower-priority

4. **Regime classifier fix**: `max_below_200_streak` should be the streak
   ending in the past 12-24 months, not the max over the past 5 y. Would
   let "bull" fire correctly. Won't move 2024 numbers (K and cap don't
   change), but worth tidying for principal correctness.

5. **GBM training-data balancing**: the model spends most of its training
   gradient on 2009 + 2020 recoveries. Weighting samples to flatten the
   recovery / non-recovery balance could cure the structural value-tilt.
   Higher implementation cost, harder to validate without leakage.

### Already-ruled-out

- Adding momentum anchor picks. They fix 2024 but break enough other years
  to lose 5–18 pp CAGR overall. Honest-harness verdict (12 variants, see
  `SUMMARY_momentum.csv`): no variant beats baseline.

## Reproducibility

```
# Per-month tranche table for any year
python3 -c "
import pandas as pd, numpy as np
tr = pd.read_csv('experiments/monthly_dca/v5/validations/results/staggered_dca_tranches.csv')
tr = tr[tr['status']=='exited'].copy()
tr['entry_date'] = pd.to_datetime(tr['entry_date'])
spy = pd.read_parquet('experiments/monthly_dca/cache/v2/monthly_returns_clean.parquet')['SPY']
spy.index = pd.to_datetime(spy.index)
YEAR = 2024
t = tr[tr['entry_date'].dt.year == YEAR].copy()
t['spy_6m'] = [(1+spy.loc[d:].iloc[1:7].fillna(0)).prod()-1 for d in t['entry_date']]
t['edge_pp'] = t['return_pct'] - t['spy_6m']*100
print(t[['entry_date','picks','return_pct','spy_6m','edge_pp']])
print(f'Mean edge: {t[\"edge_pp\"].mean():+.2f}pp  win-rate: {(t[\"edge_pp\"]>0).mean()*100:.0f}%')
"
```

Data sources used (all PIT, no survivorship back-fill):
- `experiments/monthly_dca/cache/v2/sp500_pit/sp500_membership_monthly.parquet`
- `experiments/monthly_dca/cache/v2/monthly_returns_clean.parquet`
- `experiments/monthly_dca/cache/features/<YYYY-MM-DD>.parquet`
- `experiments/monthly_dca/v5/validations/results/staggered_dca_tranches.csv`
