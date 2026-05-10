# Session 4 — Adaptive IC / dynamic hold / regime-K — KILLED

10 variants tested. Three findings:

## 1. Rolling-IC adaptive ensemble UNDERPERFORMS

| head | mean rolling-24m IC |
|---|---:|
| pred_1m | 0.045 |
| pred_3m | 0.040 |
| pred_6m | 0.062 |
| pred_12m | **0.099** |

12m has highest mean IC across all asofs, BUT softmax-weighted ensemble using IC weights (`adaptive_ic`) drops CAGR to 32.6%, vs baseline 40.78%. Resolution: **IC measures rank correlation across the entire cross-section, while top-K capture is dominated by the extreme top of the score distribution. Magnitudes of `pred_3m + pred_6m` produce sharper top-K separation than rank-based ensembles can.**

## 2. Dynamic-hold UNDERPERFORMS

`dyn_hold_q90` (rebalance early if any pick falls below 90th percentile of current scores) drops CAGR to 34.2%, MaxDD -55.5%. The "intelligent exit" trigger frequently fires on noise; whipsaw cost > capture benefit. Same pattern as v7 trailing stops (already documented in v7/REPORT_V7.md).

## 3. Regime-K — `K_bull2_others3` is the same K-shrinkage artifact

Surface result: CAGR **+2.3pp**, Sharpe **+0.025**, same MaxDD. Year-by-year diagnosis:

| year | base | exp_92 | diff (pp) |
|---|---:|---:|---:|
| **2020** | 109.6% | 182.9% | **+73.3** |
| 2004 | 27.7% | 57.2% | +29.4 |
| 2017 | 44.8% | 49.1% | +4.3 |
| 2013 | 80.1% | 56.5% | -23.6 |
| 2021 | 65.8% | 50.6% | -15.2 |
| (17 other years) | | | 0 to ±5 |

- 6/22 years beat baseline.
- Median yearly diff: **0.0 pp**.
- Std of yearly diff: **18.0 pp**.
- 2020 alone (+73pp) drives nearly the entire +2.3pp full-period lift.

**Same K-shrinkage trick** as H6 Donchian (Session 1) and H7 disp_K23 (Session 2). Three different conditional rules for "when to drop K to 2", three different "winners", all driven by 1-3 outlier years (2004, 2009, 2016, 2020 alternately). The mechanism is portfolio concentration accidentally aligning with COVID-bounce / NVDA-class winners, not a real regime signal.

**Don't repeat**: Any conditional K-shrinkage will produce noisy +1-3pp CAGR with std-of-yearly-diff ~18-20pp. The lift is sample-of-1-3 across the 22-year window. Future regime work should hold K fixed and vary something else (weighting, gate sensitivity, score blend).

## What hold-period sensitivity reveals

Tested holds {3, 4, 6, 9, 12} months:
- hold_months=4 collapses to CAGR 21.7% (way more rebalances per regime cycle, costs eat returns).
- hold_months=12 gives Sharpe 0.97 (slightly better than baseline 0.95) but CAGR -5.7pp and MaxDD -58.8% (worse) — model predictions decay past 6 months.
- **Baseline 6-month hold is locally optimal.**

## What `K=1` reveals

- K=1 alone: CAGR 28.6%, Sharpe 0.62, MaxDD -80.4%.
- K=1 with 12m hold: CAGR 19.2%, MaxDD -86.5%.

Single-name basket has too much idiosyncratic risk for the model's accuracy level. The +2-3pp CAGR vs baseline that K=1 occasionally captures via lucky picks is dwarfed by the wrong-pick penalty.
