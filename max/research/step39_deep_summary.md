# Step 39 DEEP DIVE: Signal Smoothing (step 39c/d/e)

**Date:** 2026-04-21
**Follow-on from:** step39_summary.md (trailing 6M SMA confirmed robust winner)

## Headline: smoothing window curve is MONOTONIC through 60-180M

| Smoothing | 20Y CAGR | Δ vs incumbent | Sharpe | MaxDD |
|---|---|---|---|---|
| current (incumbent) | +17.41% | — | 1.34 | -46.15% |
| SMA 6M | +18.10% | +0.70pp | 1.35 | -47.33% |
| SMA 12M | +18.31% | +0.90pp | 1.35 | -47.51% |
| SMA 18M | +18.32% | +0.92pp | 1.35 | -47.58% |
| SMA 24M | +18.60% | +1.19pp | 1.36 | -48.02% |
| SMA 48M | +18.73% | +1.32pp | 1.36 | -47.97% |
| SMA 60M | +18.75% | +1.34pp | 1.36 | -48.12% |
| SMA 96M | +18.90% | +1.49pp | 1.36 | -48.39% |
| SMA 120M | +18.94% | +1.53pp | 1.36 | -48.35% |
| SMA 180M | +18.97% | +1.56pp | 1.36 | -48.36% |

**No ceiling at 15-year smoothing.** Diminishing returns past 60M, but
no catastrophic breakdown.

## SMA 60M robustness — WINS 10/10 ROLLING 10Y

| Window | Incumbent | SMA 60M | Δ |
|---|---|---|---|
| 2006-04 → 2016-04 | +459.29% | +460.16% | +0.87pp |
| 2007-04 → 2017-04 | +40.31% | +41.48% | +1.18pp |
| 2008-04 → 2018-04 | +37.55% | +39.55% | +1.99pp |
| 2009-04 → 2019-04 | +37.89% | +39.30% | +1.41pp |
| 2010-04 → 2020-04 | +30.49% | +31.66% | +1.17pp |
| 2011-04 → 2021-04 | +30.90% | +32.35% | +1.45pp |
| 2012-04 → 2022-04 | +29.15% | +30.68% | +1.53pp |
| 2013-04 → 2023-05 | +27.07% | +28.99% | +1.91pp |
| 2014-04 → 2024-05 | +31.68% | +33.74% | +2.06pp |
| 2015-04 → 2025-05 | +30.30% | +32.75% | +2.45pp |

**Every rolling 10Y window is a win. Edge actually grows in recent
years (+2.45pp for the most recent 10Y window).**

## MECHANISM REVEAL: smoothing changes the strategy type

| Smoothing | 1Y prior return at pick (median) | Strategy character |
|---|---|---|
| incumbent | **-19.10%** | Deep mean-reversion |
| SMA 6M | -20.45% | Same, slightly sharper |
| SMA 24M | -6.52% | Mild pullback |
| SMA 60M | **-0.34%** | Persistent-quality, flat prior |

Long smoothing **fundamentally changes** the strategy from "buy the
deepest pullback" to "buy the persistent quality franchise that is
currently reasonably priced." This is a quality+value bias, not
momentum (picks are not up big) and not pullback-chasing (picks are
not down big).

## Pick concentration

| | Incumbent | SMA 60M |
|---|---|---|
| Unique tickers over 20Y | 91/96 | **55/96** |
| Monthly turnover | 48.7% | 39.2% |
| Top picks | NEM, NVDA, SPG, SMCI, GILD | **AMAT, CAT, GE, SPG, COP** |

Smoothing concentrates into ~55 persistently-good mean-reversion
franchises. The long-window winners (AMAT, CAT, GE, SPG, COP)
represent steadier compounding businesses than the incumbent's big
winners (NVDA, SMCI) whose selection is driven by month-specific
signal spikes.

## Sector composition shift

| Sector | Incumbent | SMA 60M |
|---|---|---|
| Tech | 23.1% | 22.1% |
| Fin | 12.0% | **21.5%** |
| Energy | 10.3% | 11.6% |
| Indust | 9.0% | 12.3% |
| Health | 10.2% | 8.0% |

Fin exposure nearly doubles under SMA 60M — suggesting bank/insurance
tickers score consistently well on the mean-reversion signal but get
passed over by the noisy incumbent.

## Static all-history ranking (ablation)

If we use the all-history mean `final` score (no time-varying signal
at all), we get +17.50% CAGR (barely above incumbent, worse Sharpe).
This proves smoothing's edge is NOT just "pick perennial winners" —
the time-varying smoothed signal is still tracking real dynamics.

## Sensitivity (SMA 6M across top_n × cap)

SMA 6M wins at EVERY top_n/cap combination tested:

| top_n | cap=3% | cap=5% | cap=7% | cap=10% |
|---|---|---|---|---|
| 3 | +0.44pp | +0.81pp | +1.57pp | +1.45pp |
| 5 | +0.46pp | +0.70pp | +1.42pp | +1.47pp |
| 7 | +0.48pp | +1.24pp | +1.42pp | +1.39pp |

No combination loses.

## Rolling 5Y and 3Y (smoothing holds in shorter windows)

- SMA 6M wins **11/15** rolling 5Y and **11/17** rolling 3Y windows
- SMA 24M wins 10/15 rolling 5Y, 11/17 rolling 3Y
- Losses cluster in short windows that start RIGHT after a major
  turning point (2009-2012, 2018-2021), where smoothing overweights
  obsolete pre-crisis signal

## Ablation: double-smoothing doesn't help

| Double | CAGR |
|---|---|
| SMA(6M)→SMA(6M) | +18.16% |
| SMA(6M)→SMA(12M) | +18.11% |
| SMA(12M)→SMA(24M) | +18.09% |

Single-pass SMA at the target window is best. Additional smoothing
just loses information.

## EMA vs SMA

EMA (exponential weighted, same effective window) consistently **loses
to SMA** by 0.1-0.2pp. The fresher-weighting bias of EMA hurts rather
than helps. This is additional evidence that older history is
MEANINGFUL for ranking, not just noise to be downweighted.

## RECOMMENDATION — revised from step39_summary

**Production default: SMA 12M or SMA 24M.**

Why not SMA 60M+ despite the higher CAGR?
- **Warm-up**: SMA 60M needs 5 years before the window is full. In
  live deployment starting today, the first 5 years use partial data.
- **Regime change risk**: If the universe or market character changes
  fundamentally (e.g., a major structural shift), 60M lag is too slow
  to respond.
- **MaxDD slightly worse**: 48.1% vs 46.2% — the extra 0.6pp/yr CAGR
  doesn't justify the higher tail risk for most investors.
- **SMA 12M captures 60% of the alpha** (+0.90pp vs +1.56pp cap) with
  only 1.4pp extra MaxDD.
- **SMA 24M captures 75% of the alpha** (+1.19pp) with a reasonable
  2-year warm-up.

**Production decision**: ship SMA **12M** as the new CAP5 default.
Document SMA 24M as the aggressive variant for users with long horizons
and tolerance for slightly higher tail risk.

## Caveats

1. All tests on 97-ticker, 20Y universe. Retest on 128-ticker once
   regen completes.
2. Long-window smoothing effectively creates a "quality franchise"
   bias. If a formerly-good ticker structurally deteriorates (e.g.
   company breakup, fraud), smoothing delays removal by N/2 months.
3. Tested backtest-only. No transaction cost or slippage modeling —
   SMA 60M has higher turnover than SMA 24M (39% vs 27%), so
   execution cost may erode some gain in live trading.
4. Pick universe shift is real — if you're philosophically committed
   to "buying deep dips", long-window smoothing is a different
   strategy. It co-opts the same score engine but operationally it's
   closer to "quality-tilted DCA into persistent compounders."

## Next

- Retest SMA 6/12/24M on 128-ticker universe once regen completes
- Run step36 validation battery on SMA 12M variant (jackknife, bootstrap)
- Consider productionizing as default in bt_core.simulate()
