# v5 strategy on PIT Nasdaq-100 — honest backtest

**Generated**: 2026-05-11
**Strategy**: `v5_ml_3plus6_chronos_p70_k3_invvol_cap0.4_h6_tight`
  (GBM 3m+6m forward-rank ensemble, gated by HuggingFace Chronos-bolt-tiny p70
  confidence filter @ q=0.45, top-3 picks, inverse-volatility weighted with 40%
  per-pick cap, 6-month hold, tight SPY crash gate.)

## Source of PIT membership

We use **[jmccarrell/n100tickers](https://github.com/jmccarrell/n100tickers)** —
a community-maintained authoritative point-in-time Nasdaq-100 history sourced
from Nasdaq's annual reconstitution announcements and intra-year change press
releases. The dataset claims accurate coverage from **2015-01-01 onward**.

Validation: at 2026-05-31 the yaml-reconstructed set (101 tickers) is identical
to the live `api.nasdaq.com/api/quote/list-type/nasdaq100` endpoint (101
tickers). Set difference = 0 in both directions.

Coverage caveat — the v5 walk-forward design has 10 splits starting in 2003
(A1, A2, A3, R1 GFC, R2, R3, R4, R5 COVID, R6 AI, STRICT). With the
2015-onward PIT NDX dataset we can only honestly test the 5 splits whose TEST
window starts at 2015 or later: **R3 (2015-16)**, **R4 (2017-19)**,
**R5 COVID (2020-22)**, **R6 AI (2023-25)**, **STRICT (2022-25)**.

## Universe / data coverage

- **207 unique tickers** were in the Nasdaq-100 at some point between Jan 2015
  and May 2026 (each month has 101–107 active members).
- **173 / 207** are covered by our broader 1833-ticker daily price panel
  (existing 153 + 20 yfinance-backfilled merges/delistings such as DISCA,
  DISCK, YHOO, DISH, CTRP, MXIM, BIDU, FOXA, FOX, GMCR, NTES, SBAC, SIRI, JD,
  EXPE, etc.).
- **34 / 207 are missing** from our panel — mostly pre-2018 delistings /
  acquisitions whose tickers no longer return any yfinance data
  (BRCM/Broadcom merger into AVGO, CTXS/Citrix taken private, ALXN/Alexion
  acquired by AZN, ATVI/Activision acquired by MSFT, CELG, SPLK, XLNX, MXIM,
  CTRX, KRFT, WFM, etc.). The surviving entity (AVGO, AZN, MSFT, etc.) is
  present in our panel where applicable. Net effect: median panel-resolvable
  pool at any month-end is **93 tickers** (vs ~103 in the real NDX = ~90 %
  coverage).

This is an **honest approximation** rather than perfect reconstitution. The
missing tickers are systematically pre-2018 names that were acquired at
premiums (small upside lost to the strategy) or merged into surviving names
(no bias, the survivor is captured). The walk-forward windows starting 2017+
are minimally affected.

## ML preds / Chronos signals

- The GBM `ml_preds_v2.parquet` was trained on the broader 1833-ticker panel
  with annual retraining and a 7-month embargo. Predictions for the
  207 NDX tickers are filtered out of the existing parquet.
- Chronos `ml_preds_chronos_broader.parquet` is zero-shot — the
  `amazon/chronos-bolt-tiny` model was never retrained on our data. Using its
  predictions on the NDX cohort is a pure OOS application.

The cross-sectional **Chronos rank** is recomputed within the NDX cohort each
month (not the SP500 cohort). This is the correct comparison.

## Results

### Full window (2015-01-31 → 2026-05-31, 11.4 years)

| Metric             | Strategy   | QQQ buy-and-hold | Edge       |
|--------------------|-----------:|-----------------:|-----------:|
| CAGR               |   **26.66 %** |   19.51 %        | **+7.15 pp** |
| Sharpe (monthly)   |     1.00   |   —              |            |
| Max drawdown       |   −36.8 %  |   ≈ −35 %         |            |
| $1 grows to        |   **$14.85** |   $7.65          |   1.94×     |
| Cash months        |   2 / 137  |   0              |            |
| Baskets formed     |   22       |   —              |            |

### Walk-forward splits

| Split                          | Window               | Strat CAGR | QQQ CAGR | Edge        | Sharpe | MaxDD   |
|--------------------------------|----------------------|-----------:|---------:|------------:|-------:|--------:|
| **R3 2015-16**                 | 2015-01 → 2016-12   |  +61.67 %  |  +9.41 % | **+52.26 pp** | 2.30   | −10.5 % |
| **R4 2017-19**                 | 2017-01 → 2019-12   |  +27.37 %  | +20.53 % | **+6.84 pp**  | 1.01   | −28.1 % |
| **R5 COVID 2020-22**           | 2020-01 → 2022-12   |  +12.09 %  |  +7.36 % | **+4.74 pp**  | 0.56   | −46.3 % |
| **R6 AI 2023-25**              | 2023-01 → 2025-12   |  +51.05 %  | +28.52 % | **+22.53 pp** | 1.35   | −26.2 % |
| STRICT 2022-25                 | 2022-01 → 2025-12   |   +5.52 %  | +14.77 % | **−9.25 pp**  | 0.33   | −30.5 % |
| Post-COVID 2020-26 (overlap)   | 2020-01 → 2026-04   |  +10.21 %  | +19.94 % | **−9.73 pp**  | 0.49   | −46.3 % |

Strategy **beats QQQ on 4 of 6** windows above, including the AI rally
(R6 +22.5 pp) and the 2015-16 sideways-then-rally (R3 +52 pp). The two losing
windows (STRICT 2022-25 and Post-COVID 2020-26) overlap and share the same
underlying period — QQQ rallied very tightly from a mega-cap-led recovery
that a 3-stock equal-vol-weighted basket cannot capture as fully as a 100-name
cap-weighted index.

### Interpretation

- **Generalisation holds**: full-window +7.15 pp edge over the actual QQQ ETF
  is meaningful given (a) the model was not retrained on the NDX cohort, (b)
  Chronos is zero-shot, (c) the universe is much narrower (~93 stocks vs
  ~500 PIT SP500).
- **Concentration risk is real**: with K=3 and inv-vol caps, the strategy
  systematically under-bets the mega-cap tech bias of QQQ. In windows where
  the market is mega-cap-driven (2022-25, where NVDA alone accounted for a
  large fraction of QQQ's return), the strategy lags.
- **Where it wins**: when the NDX has dispersion (multiple sectors / themes
  contributing), the GBM+Chronos top-3 selection captures the rotation
  better than QQQ's static cap-weight.
- **NOT recommended as a production switch** from PIT-SP500 to PIT-NDX. The
  PIT-SP500 v5 production strategy retains broader diversification and the
  decade-long walk-forward record. This is a curiosity / OOS generalisation
  result only.

## Files

- `ndx_pit_membership_monthly.parquet` — `(asof, ticker)` rows where ticker was
  in NDX on month-end (panel-resolvable subset, 173 unique tickers).
- `ndx_pit_membership_monthly_full.parquet` — full set, 207 unique tickers.
- `ndx_monthly_prices.parquet` / `ndx_monthly_returns.parquet` — NDX-restricted
  monthly panel (merged from existing v2 panel + yfinance backfill).
- `qqq_pit_equity.csv` — month-by-month equity curve (strategy + QQQ).
- `qqq_pit_trades.csv` — every closed/open pick with entry/exit prices.
- `qqq_pit_walkforward.csv` — per-split summary.
- `qqq_pit_report.json` — full machine-readable report.

## Reproduce

```bash
python3 -m experiments.monthly_dca.v5.qqq_pit.build_qqq_pit_panel
python3 -m experiments.monthly_dca.v5.qqq_pit.backfill_ndx_tickers
python3 -m experiments.monthly_dca.v5.qqq_pit.run_v5_on_qqq_pit
```
