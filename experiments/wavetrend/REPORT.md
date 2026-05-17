# WaveTrend on PIT S&P 500 / NDX — findings

**Separate experiment**, not wired into anything deployed. Reproduce with
`python3 experiments/wavetrend/run_wavetrend_pit.py` (~6 min, CPU).

## What was tested

The repo-root `wavetrend` file is a WaveTrend (Pine `wt1/wt2`) pyramiding
strategy with an RSI exit, optimised by Bayesian search to **maximise trade
win-rate**, on a hand-picked yfinance universe.

Three problems were fixed before trusting any number:

1. **Look-ahead universe bug.** The original picks "underperformers vs SPY"
   using *full-sample final equity* — it cannot be known at trade time.
   Replaced with **PIT index membership** (S&P 500 and Nasdaq-100) from the
   repo's existing point-in-time data. A name is only tradable while it is an
   index member; it is force-exited on removal/delisting.
2. **Survivorship.** Runs on `prices_extended_pit.parquet` (1994 tickers,
   PIT-corrected). Caveat: panel is **adjusted close only**, so Pine's
   `ap = hlc3` is approximated by close. 10 bps round-trip cost applied.
3. **The win-rate objective is a trap.** It rewards "never sell at a loss",
   producing a strategy that barely trades and badly lags the market.
   Replaced with a risk-adjusted objective `Sharpe + 0.5·CAGR + 3·min(0,
   MDD+0.45)` with a ≥40-trade floor, **selected only on a train window**
   (S&P 500 2003–2013; NDX 2015–2020), holdout never used for tuning. The
   naive win-rate run is kept side-by-side as the control.

## Result 1 — the win-rate objective is empirically a trap (S&P 500 PIT)

| | Honest risk-adjusted | Naive win-rate (original) | SPY |
|---|---:|---:|---:|
| Full CAGR | **17.6%** | 4.3% | 11.7% |
| Sharpe | **1.06** | 0.78 | — |
| Holdout CAGR (2014+) | **22.8%** | 7.1% | — |
| Holdout Sharpe | **1.29** | 0.97 | — |
| Trades | 185 | 16 | — |
| Win rate | 69.7% | **87.5%** | — |
| WF splits beating SPY | **10/10** | 2/10 | — |

The 87.5%-win-rate config trades 16 times in 23 years and returns 4.3%/yr —
the exact failure the original objective invites. The honest config's
**holdout Sharpe (1.29) exceeds its train Sharpe** → not overfit.

## Result 2 — it does NOT beat the deployed strategy standalone, and NDX fails

- Honest WaveTrend SP500: 17.6% CAGR vs deployed **v5 ≈ 39.8%**. Not a
  replacement.
- NDX PIT (2015–2026, only 137 months): honest config holdout CAGR **2.4%**,
  Sharpe 0.46, **0/8** vs QQQ. A buy-the-dip mean-reverter structurally lags
  a momentum mega-cap index (QQQ 19.4%). **WaveTrend does not generalize to
  NDX** — reported honestly, not hidden.

## Result 3 — the real value: a near-orthogonal diversifying sleeve for the SP500 book

WaveTrend's monthly returns vs the deployed v5 stream
(`augmented/v5_winner_equity.csv`, 253 common months):

- **corr = 0.054** full; max |corr| across WF splits 0.38 (stable).
  This is a *lower* correlation than every existing diversifier the repo has
  banked — carry (0.23), quality second-sleeve (0.19), mn-composite (−0.19) —
  and unlike mn-composite it has a **positive standalone Sharpe (1.06)**.

Static v5 + WaveTrend blends (same metric code as the v5 winner):

| blend | CAGR | Sharpe | Max DD | WF mean Sharpe | WF min Sharpe |
|---|---:|---:|---:|---:|---:|
| v5 alone | 39.8% | 0.91 | −77% | 1.20 | 0.62 |
| 80% v5 + 20% wt | 37.5% | 1.00 | −69% | 1.35 | 0.68 |
| **70% v5 + 30% wt** | **35.9%** | **1.05** | **−65%** | **1.45** | **0.72** |
| **60% v5 + 40% wt** | **34.1%** | **1.11** | **−61%** | **1.56** | **0.77** |
| 50% v5 + 50% wt | 32.0% | 1.19 | −57% | 1.69 | 0.78 |

(Max DD is on the raw monthly stream; the deployed book reports the
accumulating-DCA DD which is shallower — directionally the same.)

## Recommendation

Do **not** deploy WaveTrend standalone and do **not** touch the deployed E2
strategy. The actionable finding is the sleeve: a **~30–40% WaveTrend /
60–70% v5** mix on the S&P 500 book lifts Sharpe 0.91 → 1.05–1.11, raises
WF-min Sharpe 0.62 → 0.77, and cuts the tail ~−77% → ~−61%, costing ~4–6pp
CAGR. It sits on the same orthogonal-sleeve lever the deployed E2 already
uses, and is the strongest low-correlation, positive-Sharpe diversifier
found so far. Natural next step (not done here): walk-forward-validate the
blend weight and stack it with the existing carry/quality sleeves rather
than v5 alone.

### Caveats
- Close-only proxy for `hlc3` (no intraday H/L in the PIT panel).
- 213 OTC bankruptcy-"Q" names still missing → residual tail-risk
  understatement (same limitation as the rest of the PIT work).
- NDX did not generalize — the edge is S&P-500-specific here.
- Blend weight shown is a full-sample read; treat the 30–40% band as
  indicative until walk-forward-validated.

## Files
- `wavetrend_pit.py` — leakage-free sim + indicators + metrics
- `run_wavetrend_pit.py` — train/holdout optimiser + sleeve study
- `wavetrend_pit_results.json` — full metrics (both universes, both objectives)
- `wt_sp500_monthly_returns.csv`, `wt_sp500_trades.csv`, `wt_equity.png`
- `run.log` — full optimisation trace
