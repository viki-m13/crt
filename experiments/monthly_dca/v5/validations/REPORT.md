# v5 strategy — 6-variant validation (improving lagging years)

**Generated**: 2026-05-11
**Universe**: PIT S&P 500 (985 tickers across 2003-2026)
**Walk-forward**: identical 10 splits to v5 production (A1, A2, A3, R1_GFC, R2, R3, R4, R5_COVID, R6_AI, STRICT)

## Baseline reproduction sanity check

| Metric              | Production v5 | Harness baseline | Δ        |
|---------------------|--------------:|-----------------:|---------:|
| Full-window CAGR    |    43.86 %    |    42.60 %       | −1.26 pp |
| WF mean CAGR        |    47.16 %    |    45.62 %       | −1.54 pp |
| WF min CAGR         |    23.08 %    |    20.01 %       | −3.07 pp |
| WF beat-SPY count   |    10 / 10    |    10 / 10       | ✓        |
| WF positive count   |    10 / 10    |    10 / 10       | ✓        |

Small residual (≈1.5 pp on CAGR) comes from (a) production using
`ml_preds_live.parquet` for the latest 1-4 months past the WF cutoff while the
harness only uses `ml_preds_v2.parquet`, (b) production's multiplicative cost
handling `(1+r)(1-cf)` vs the harness's additive `r-cf`. Both are immaterial
for **relative** comparison across variants on the same harness.

## Headline summary (full window 2003-09 → 2026-04)

| #   | Variant                  | Lump CAGR | DCA CAGR | WF mean | WF min | WF mean edge | Beats SPY | Sharpe | MaxDD   |
|----:|--------------------------|----------:|---------:|--------:|-------:|-------------:|----------:|-------:|--------:|
|  0  | **baseline**             | **42.60** |   43.37  |  45.62  |  20.01 |   +33.23 pp  |  10 / 10  |  0.98  | −61.4 % |
|  1  | concentration_overlay    |   41.80   |   42.69  |  44.94  |  18.64 |   +32.55 pp  |  10 / 10  |  0.97  | −61.4 % |
|  2  | sector_diversification   |   38.27   |   38.85  |  40.95  |  20.01 |   +28.56 pp  |  10 / 10  |  0.88  | −64.2 % |
|  3  | cap_loose_bull           |   42.60   |   43.37  |  45.62  |  20.01 |   +33.23 pp  |  10 / 10  |  0.98  | −61.4 % |
|  4  | ensemble                 |   11.10   |   10.54  |  11.97  |  −5.10 |    −0.42 pp  |   3 / 10  |  0.51  | −60.8 % |
|  5  | **anchor (K=2 + anchor)** |   40.93  |   41.45  | **46.72** | 20.26 |  **+34.33 pp** | **10 / 10** | **1.07** | **−52.5 %** |

## Lagging-year impact (the year edge — strat minus SPY)

| Year | baseline | conc-overlay | sector-div | cap-loose | ensemble | **anchor** |
|-----:|---------:|-------------:|-----------:|----------:|---------:|-----------:|
| 2014 |   −0.83  |    −0.83     |   −0.83    |  similar  |  +6.88   |  **+3.34** |
| 2018 |  +23.84  |   +23.84     |  +23.84    |  similar  | +11.64   |  +16.08    |
| 2024 |  −35.37  |   −35.37     |  −35.37    |  similar  |  −5.45   | **+53.73** |
| 2025 |   +5.57  |    +5.57     |   +5.57    |  similar  | −10.28   |   +1.63    |

## What each variant does and why

### 1. concentration_overlay (SPY sleeve when dispersion is in bottom 30 %)

Hypothesis: when cross-sectional dispersion in PIT-SP500 monthly returns is in the bottom 30 % of its 36-month rolling distribution (mega-cap-led market), blend in a 25 % SPY sleeve to capture cap-weighted upside the K=3 basket misses.

**Verdict: marginally negative.** WF mean drops 0.68 pp; DCA-CAGR drops 0.68 pp. The SPY sleeve dilutes alpha without specifically capturing the leader. Dispersion-based regime detection lags the actual narrow-leadership periods.

### 2. sector_diversification (≤ 2 picks per GICS sector)

Hypothesis: avoid all 3 picks in one sector (typical late-cycle behaviour) to break single-sector concentration risk.

**Verdict: pure drag.** WF mean drops 4.67 pp; full-window CAGR drops 4.33 pp. The constraint forces the picker away from the top-scored stocks more often than not. The current basket HWM/SYF/PH (2 industrials + 1 financial) already passes ≤2-per-sector, so the variant only bites when the model has 3 strong picks in one sector — which is when concentration is alpha, not risk.

### 3. cap_loose_bull (per-pick cap 0.55 in bull regime)

Hypothesis: in obvious bull markets, let the highest-conviction pick get more weight.

**Verdict: identical to baseline** (no effect detectable). The "bull" regime gate (SPY mom_12_1 ≥ 10 % AND dsma200 > 0) doesn't actually fire often enough to materially shift weights — and in 2024 the regime was "recovery", not "bull", so the cap stayed at 0.40.

### 4. ensemble (mean rank of v2 + v6 + pattern_sim + vertical)

Hypothesis: averaging cross-sectional rank across multiple ML signals reduces single-model overfit.

**Verdict: large negative.** WF mean drops 33.6 pp to 11.97 %, 3/10 beat SPY. Diluting the (already-good) v2 GBM score with weaker / less-validated signals (pattern_sim, vertical) reverses most of v5's edge. The Chronos filter is already the ensemble (it gates the GBM); adding ranking ensembles on top is over-engineering.

### 5. **anchor (K = 2 alpha + 1 mega-cap anchor)**  ⬅ winner

Hypothesis: keep 2 picks from the v5 alpha process (GBM + Chronos), and add 1 "anchor" = highest 12-1 momentum within the eligible PIT pool. Anchor naturally captures mega-cap leaders during narrow rallies.

**Verdict: improves where it matters most.**
- WF mean **+1.10 pp** (45.62 → 46.72), WF mean edge **+1.10 pp** (33.23 → 34.33), 10/10 beat SPY.
- **Sharpe 1.07** vs 0.98 (+9 %).
- **MaxDD −52.5 %** vs −61.4 % (−9 pp shallower).
- **2024 +53.73 pp edge** vs −35.37 pp — solves the lagging-year problem.
- Trade-off: gives up GFC-recovery upside (R1_GFC −46 pp vs baseline). The mega-cap anchor under-bets the explosive small-cap rebound. Net: anchor is **better in 5 of 10 splits and worse in 5**, but the wins are bigger.

Picks in 2024 — anchor variant:
- H1: KEY, CCL, **NVDA (40 %)** ← NVDA selected as highest 12-1 momentum.
- H2: AAPL, TSCO, VST.

## DCA vs lump-sum metric

Every variant reports both. The **DCA-into-strategy CAGR is consistently within ±1 pp of the lump-sum CAGR** across all variants. Reason: the strategy's monthly returns are roughly stationary across the 22-year backtest, so DCA-IRR and lump-sum CAGR converge. DCA-into-SPY is +12.71 % vs lump-sum SPY +11.09 % — the SPY sequence-of-returns matters a bit more.

For the website: it's safe to show either metric to the user — they're nearly identical for the strategy. The "vs SPY DCA" framing on the homepage is honest.

## Files saved

```
experiments/monthly_dca/v5/validations/
  harness.py                 # shared simulator + WF + DCA-CAGR
  run_all.py                 # runs all 6 variants end-to-end
  results/
    SUMMARY.csv              # one row per variant
    REPORT.md                # this file (also at parent dir)
    baseline.json + .csv     # per-variant JSON report + monthly equity
    concentration_overlay.json + .csv
    sector_diversification.json + .csv
    cap_loose_bull.json + .csv
    ensemble.json + .csv
    anchor.json + .csv
```

Reproduce:
```bash
python3 -m experiments.monthly_dca.v5.validations.run_all
```

## Recommendation

Ship the **anchor variant** to production with K=2 alpha + 1 mega-cap anchor. Expected impact:
- Same 10/10 walk-forward beat-SPY record.
- Better risk-adjusted return (Sharpe 1.07 vs 0.98, MaxDD −52 % vs −61 %).
- Resolves the 2024-style narrow-leadership lag.
- Trade-off accepted: gives up some upside in explosive small-cap recovery regimes (GFC-bottom-style) in exchange for cleaner mega-cap-led performance — the more common modern regime.

## Staggered monthly-tranche DCA (proper retail DCA)

The previous experiments all assume a **single lump-sum** deployed at t=0 with
internal 6-month basket rotations. The user-requested variant matches how a
retail investor would actually deploy: deposit $X every month-end into a new
3-stock v5 basket, hold each tranche exactly 6 months, recycle proceeds into
the current month's tranche. At steady state, up to 6 overlapping tranches
are active at any time (one started each of the last 6 months).

### Result

| Metric                          |    Value |
|---------------------------------|---------:|
| n months                        |      272 |
| Total deposits                  |    $ 272 |
| Final NAV (strategy)            | $ 64,004 |
| Final NAV (SPY DCA, same flow)  |  $ 1,393 |
| Multiple (strategy)             |  235.3 × |
| Multiple (SPY DCA)              |    5.12× |
| **Money-weighted CAGR**         | **39.15 %** |
| Money-weighted CAGR (SPY DCA)   |  12.66 % |
| **Edge over SPY DCA**           | **+26.49 pp** |

### Tranche distribution (254 closed tranches)

| Stat                  |  Value |
|-----------------------|-------:|
| Win rate (6m return > 0) | 76.4 % |
| Mean 6m return        | +16.6 % |
| Median 6m return      | +12.7 % |

### Best / worst tranches

| Entry → Exit                    | Picks                 | 6m return |
|---------------------------------|----------------------|----------:|
| 2009-03 → 2009-09 (GFC bottom)  | GNW, TGNA, MTW       | **+372.46 %** |
| 2009-06 → 2009-12               | MTW, F, TGNA         | +167.83 % |
| 2009-04 → 2009-10               | GNW, FITB, RF        | +129.84 % |
| 2008-12 → 2009-06               | MBI, GNW, THC        | +117.95 % |
| 2020-09 → 2021-03               | FTI, OXY, DVN        | +116.66 % |
| ⋮                               |                      |           |
| 2022-03 → 2022-09 (2022 bear)   | CFG, UAA, WDC        | −37.96 %  |
| 2008-01 → 2008-07               | MBI, MTG, AMZN       | −42.36 %  |
| 2007-12 → 2008-06               | AMZN, NVDA, MBI      | −51.19 %  |
| 2008-07 → 2009-01 (GFC)         | FMCC, MTG, FNMA      | −85.50 %  |
| 2008-08 → 2009-02 (GFC)         | FMCC, FNMA, SLM      | −86.14 %  |

### Interpretation

- Money-weighted CAGR of **39.15 %** is the right number to quote a user who
  contributes a fixed amount monthly. It's 3.5 pp below the lump-sum-baseline
  CAGR (42.60 %), because early deposits had more compounding time at the
  strategy's high rate — but it's still **far above SPY DCA (12.66 %)**.
- Edge over SPY DCA is **+26.5 pp**, consistent with the lump-sum edge of
  +31.5 pp.
- **76 % of 6-month tranches end positive** — a strong reproduction of the
  strategy's hit rate at the tranche level.
- The worst tranches all cluster in the GFC (Aug 2008, Jul 2008, Jan 2008 —
  Fannie / Freddie / SLM, MBI / AMZN). These are the months where the model
  picked deep-value financials that subsequently went to zero. The crash gate
  catches the WORST 2-3 months but does miss several months of the slide.
- The best tranches are the inverse: deep-value financial / cyclical names
  entered AT the March 2009 trough — multi-bagger 6-month holds.

### Files

```
experiments/monthly_dca/v5/validations/
  run_staggered_dca.py              # standalone script
  results/
    staggered_dca.json              # headline JSON
    staggered_dca_equity.csv        # 272 months: cum deposits, active tranches, NAV
    staggered_dca_tranches.csv      # every tranche: entry, exit, picks, return
```

Reproduce: `python3 -m experiments.monthly_dca.v5.validations.run_staggered_dca`
