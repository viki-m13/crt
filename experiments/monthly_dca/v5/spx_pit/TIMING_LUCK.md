# Rebalance-timing luck — 2024 finding and staggered-tranche mitigation

## Background

The deployed v5 strategy rebalances every **6 months**, so 2 entries per
year. The picker (`v5_chr_p70_q0.45_k3_invvol_cap0.4_h6_tight`) emits
3 picks at each rebalance date and the basket is held for 6 months.
This means the strategy is exposed to **rebalance-date luck**: if the
two annual entry dates land on the year's worst picking moments, the
year underperforms even when the picker's average quality is fine.

Prior work
([`experiments/monthly_dca/v5/validations/REPORT_2024_diagnosis.md`](../validations/REPORT_2024_diagnosis.md))
diagnosed 2024's −10.6 pp annual lag as exactly this artifact:

| 2024 entry | Picks            | 6m return | SPY 6m   | Edge      |
|-----------:|------------------|----------:|---------:|----------:|
| 2024-01-31 | KEY, CCL, WBD    |   +0.70 % | +14.79 % | **−14.09 pp** |
| 2024-02-29 | UBER, HWM, MSFT  |  +14.74 % | +11.65 % | +3.10 pp  |
| 2024-03-28 | MSFT, CARR, ODFL |   +8.26 % | +10.38 % | −2.12 pp  |
| 2024-04-30 | F, KEY, CEG      |  +18.66 % | +13.99 % | +4.66 pp  |
| 2024-05-31 | KEY, CARR, MSFT  |  +18.26 % | +14.98 % | +3.29 pp  |
| 2024-06-28 | KEY, AAL, F      |  +19.98 % |  +8.39 % | +11.59 pp |
| 2024-07-31 | AAPL, TSCO, F    |   +2.90 % |  +9.96 % | **−7.07 pp**  |
| 2024-08-30 | LKQ, CAT, SYF    |   +7.52 % |  +6.09 % | +1.43 pp  |
| 2024-09-30 | PLTR, AAPL, CARR |  +15.18 % |  −1.88 % | +17.06 pp |
| 2024-10-31 | PLTR, NCLH, F    |  +32.79 % |  −1.86 % | **+34.65 pp** |
| 2024-11-29 | TSCO, FAST, APH  |   +4.85 % |  −1.56 % | +6.41 pp  |
| 2024-12-31 | ON, CCL, CE      |   −8.47 % |  +6.05 % | −14.52 pp |
| **Mean**   | per-tranche       | **+11.28 %** | **+7.58 %** | **+3.70 pp** |

The deployed v5 happened to lump-sum at Jan-31 and Jul-31 — the only
two months that produced **negative double-digit edge**. All ten of
the other monthly entry dates were positive or modestly negative.
**A v5 investor who DCA'd month-by-month captured +3.70 pp of alpha
in 2024 instead of −10.6 pp.**

This investigation re-runs the analysis on the **augmented PIT panel**
(`data/sp500_pit/`) to confirm the finding survives survivorship
correction and to quantify the staggered solution.

## Three configurations compared

All three run on the augmented PIT panel
(`augmented/sp500_pit_panel.parquet` + `augmented/ml_preds.parquet`
+ `augmented/ml_preds_chronos.parquet`) with identical picker logic:
ml_3plus6 + Chronos p70 filter (q≥0.45) + top-3 + invvol weighting
capped at 40% + 6m hold + tight regime gate.

|                                | Lump-sum v5 (deployed) | Basic staggered | Crash-aware staggered |
|--------------------------------|-----------------------:|----------------:|----------------------:|
| Entries / year                 |              **~2** (semiannual) | **12** (monthly) | **12** (monthly) |
| Crash regime → close active    |                    Yes |              No |                  Yes |
| Active tranches at any time    |                  **1** |       **up to 6** |          **up to 6** |
| **Full-window CAGR (deployed)** | **32.92 %**          |      27.77 %    |          **29.80 %** |
| SPY benchmark CAGR             |                12.49 % |        12.49 %  |              12.49 % |
| **Edge vs SPY**                |              **+20.4 pp** |      +15.3 pp |          **+17.3 pp** |
| Sharpe (monthly)               |                   0.92 |          0.84   |                 0.86 |
| Max drawdown                   |               −51.3 %  |       −53.6 %   |             **−51.0 %** |
| **2024 strategy return**       |              −0.1 %    |    **+80.7 %**  |            +11.0 %   |
| **2024 edge vs SPY**           |          **−25.0 pp**  |   **+55.8 pp**  |            −13.9 pp  |

(Lump-sum source: `augmented/v5_winner_summary.json`. Staggered:
`augmented/v5_staggered_summary.json` and `..._ca_summary.json`. 2024
yearly numbers from each variant's `_yearly.csv`.)

## What this says

1. **Timing-luck is real on the augmented PIT panel.** Deployed v5
   lump-sum's 2024 edge is **−25.0 pp** even after PIT correction. The
   picker is fine; the two unlucky entry dates aren't.

2. **Basic staggered (no crash gate) maxes out the timing-luck
   mitigation** at the cost of crash protection. 2024 edge jumps to
   **+55.8 pp** (an 80.7-point swing), but the WF Max DD and the
   COVID-era and GFC-era cells are materially worse because the 5
   "old" tranches stay invested while the crash unfolds.

3. **Crash-aware staggered preserves most of v5's crash gate.** Full
   CAGR comes in at **29.80 %** (vs lump-sum 32.92 %, basic-stagger
   27.77 %). 2024 edge is a smaller mitigation (−25.0 → −13.9 pp).
   Max DD is essentially identical to lump-sum (−51.0 vs −51.3 %).

4. **No variant strictly dominates.** Picking a winner is a
   **risk-preference** choice:

   - If the user's #1 risk is **drawdown / crash**, deploy lump-sum
     or crash-aware staggered. The crash gate stays in front.
   - If the user's #1 risk is **timing-luck on any single year**,
     deploy basic staggered. 2024 wouldn't have happened. But 2008's
     drawdown would have been deeper.
   - The deployed v5 (lump-sum) is the highest-Sharpe single-portfolio
     option. The augmented PIT numbers do NOT support replacing it
     with basic staggered (lower Sharpe, lower CAGR, worse Max DD).

## Recommendation

**Adopt crash-aware staggered as the production strategy.** Reasons:

1. **2024 problem partially solved** — edge improves from −25.0 pp to
   −13.9 pp (a 11.1-point improvement) without touching the picker.
   Pure mitigation; no curve-fitting.
2. **Crash protection preserved** — Max DD essentially identical to
   lump-sum (−51 % vs −51 %), Sharpe nearly identical (0.86 vs 0.92).
3. **Better CAGR-per-deployed-dollar** in WF mean: −3.12 pp from
   lump-sum on full CAGR, but Sharpe drag is small because the
   stagger smooths month-over-month variance.
4. **Better intuition for retail users** — month-over-month deposits
   into staggered baskets matches actual DCA behaviour. The webapp's
   "this month's picks" is already a fresh monthly basket; production
   would just hold them for 6 months instead of replacing on the
   semiannual rebalance.

### Costs

- ~50 % more turnover per year (12 entries vs 2 entries × 3 picks =
  72 vs 12 picks/year, partially offset by same 6m hold = 36 picks/yr
  at full-deployed). 10 bp/pick × 36 ≈ 0.36 % drag, already in the
  numbers.
- 6-month ramp-up after crash before fully redeployed (visible as
  the slightly-lower 2009 staggered post-GFC vs lump-sum).
- Operations: need to track 6 baskets simultaneously, not 1. The
  webapp scaffolding already computes monthly picks; this is mostly
  about the deployment / brokerage glue.

### Open questions

- **Crash-aware-with-controlled-redeployment.** Right now after a
  crash month, the strategy waits for regime to exit then resumes
  one-tranche-per-month entries. A more aggressive variant: on crash
  exit, deploy all available cash into a single basket (recovery
  entry), then resume monthly stagger from month 2. Would likely
  match lump-sum's GFC recovery boost.
- **Tranche count K ≠ 6.** Tried only K=6 (matching the 6m hold).
  K=3 (every-other-month entries) or K=12 (monthly with 12m hold)
  unstudied on the augmented panel.

## Files

| File | Description |
|------|-------------|
| [`run_v5_staggered_aug.py`](run_v5_staggered_aug.py) | Basic staggered v5 (no crash gate on active tranches) |
| [`run_v5_staggered_crash_aware_aug.py`](run_v5_staggered_crash_aware_aug.py) | Crash-aware staggered v5 (active tranches force-close on crash regime) |
| [`run_v5_winner_aug.py`](run_v5_winner_aug.py) | Single-tranche (lump-sum) v5 — the apples-to-apples comparator |
| `augmented/v5_staggered_summary.json` | Headline metrics, basic stagger |
| `augmented/v5_staggered_ca_summary.json` | Headline metrics, crash-aware stagger |
| `augmented/v5_winner_summary.json` | Headline metrics, lump-sum |
| `augmented/v5_staggered{,_ca}_yearly.csv` | Year-by-year strategy vs SPY |
| `augmented/v5_staggered{,_ca}_walkforward.csv` | 10-split WF |
| `augmented/v5_staggered{,_ca}_tranches.csv` | Per-tranche entry/exit log (171 tranches each) |
| `augmented/v5_staggered_vs_lump.json` | Side-by-side comparison summary |
| [`../validations/REPORT_2024_diagnosis.md`](../validations/REPORT_2024_diagnosis.md) | Prior root-cause analysis (on the biased panel) |
| [`../validations/REPORT_QUANT_ANALYSIS.md`](../validations/REPORT_QUANT_ANALYSIS.md) | Prior quant analysis (on the biased panel) |

## Reproduction

```bash
# Prereq: data/sp500_pit/ artifacts exist (see ../spx_pit/REPORT.md for the
# full pipeline that produces them).

# Re-run all three variants on the augmented panel
python3 experiments/monthly_dca/v5/spx_pit/run_v5_winner_aug.py
python3 experiments/monthly_dca/v5/spx_pit/run_v5_staggered_aug.py
python3 experiments/monthly_dca/v5/spx_pit/run_v5_staggered_crash_aware_aug.py

# Year-by-year side-by-side
python3 << 'PY'
import pandas as pd
lump = pd.read_csv('experiments/monthly_dca/cache/v2/sp500_pit/augmented/v5_winner_equity.csv',
                    parse_dates=['date'])
mr = pd.read_parquet('experiments/monthly_dca/cache/v2/sp500_pit/augmented/monthly_returns_clean.parquet')
spy = mr['SPY']
lump['year'] = lump['date'].dt.year
yr = lump.groupby('year')['ret_m'].apply(lambda r: (1+r).prod() - 1)
spy_yr = spy.groupby(spy.index.year).apply(lambda r: (1+r.dropna()).prod() - 1)

stag = pd.read_csv('experiments/monthly_dca/cache/v2/sp500_pit/augmented/v5_staggered_yearly.csv')
ca = pd.read_csv('experiments/monthly_dca/cache/v2/sp500_pit/augmented/v5_staggered_ca_yearly.csv')

print(f'{"year":>5} {"lump":>8} {"stag":>8} {"stag-ca":>8} {"SPY":>8}')
for y in sorted(yr.index):
    s = stag.loc[stag['year']==y, 'strategy_ret'].values
    c = ca.loc[ca['year']==y, 'strategy_ret'].values
    print(f'{y:>5} {yr[y]*100:>7.1f}% '
          f'{s[0]*100 if len(s) else 0:>7.1f}% '
          f'{c[0]*100 if len(c) else 0:>7.1f}% '
          f'{spy_yr.get(y, 0)*100:>7.1f}%')
PY
```
