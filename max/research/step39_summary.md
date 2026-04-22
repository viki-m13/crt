# Step 39: Signal Smoothing — ROBUST WINNER

**Date:** 2026-04-22
**Hypothesis:** Smoothing the conviction score over trailing months
should reduce noise and prefer tickers with persistently high signals.

## Headline result

| Variant | 20Y CAGR | Sharpe | 10Y median | 10Y wins vs incumbent |
|---|---|---|---|---|
| CAP5 current (incumbent) | +17.41% | 1.34 | +31.29% | — |
| CAP5 trailing 2M avg | +17.72% | 1.34 | +31.44% | **10/10** |
| CAP5 trailing 3M avg | +17.80% | 1.35 | +31.65% | 9/10 |
| **CAP5 trailing 6M avg** | **+18.10%** | **1.35** | **+32.05%** | **10/10** |
| CAP5 trailing 12M avg | +18.29% | 1.35 | +32.17% | **10/10** |

**Monotonic improvement as smoothing window grows.**

## Robustness

Unlike step 38 (DCA scaling) which was a GFC artifact, signal
smoothing holds up across every 10Y window tested and across regimes:

| Period | incumbent | 6M trailing | Δ |
|---|---|---|---|
| GFC 2008-2012 | +41.66% | +41.60% | -0.06pp (tie) |
| Post-GFC bull 2012-2020 | +32.01% | +32.34% | +0.33pp |
| Post-COVID 2020-2026 | +36.15% | +37.67% | +1.52pp |

Calendar-year wins (18 years since 2008): 10/18 (56%) — a small
per-year edge that compounds robustly.

## Mechanism

The per-month `final` score is noisy. A ticker that scores 95 this
month may score 60 next month for reasons that don't reflect
persistent quality. Smoothing prefers tickers with CONSISTENTLY high
signals, which correlates with:

1. **Persistence of mean-reversion setup** — tickers that stay washed
   out for multiple months (extended pullback without capitulation)
   tend to rebound more robustly than one-month anomalies.
2. **Reduced signal whipsaw** — fewer flip-flops between picks month-
   to-month, so each DCA dollar is deployed into higher-conviction
   names.
3. **Quality filtering** — the smoothing effectively acts as a low-
   pass filter on quality drift; true quality changes slowly, noise
   is high frequency.

## Capital requirements

**Zero extra capital.** Unlike DCA-scaling which requires 20-70% more
dollars invested over 20Y, signal smoothing is a pure ranking change.
Same monthly $1000 DCA, same top-5 selection — just smoother scores.

## Decision

**ADOPT trailing 6M smoothing** as the new CAP5 default. Rationale:
- Robust gain across every 10Y window tested
- Clean mechanism, no capital commitment
- +0.70pp CAGR over 20Y compounds to significant dollar improvement
- 6M is a reasonable middle ground; 12M gives slightly more but may
  be too slow to react to regime changes (though not tested)
- Same entry_delay, top_n, cap — minimal production changes required.

## Caveats

- Tested on 97-ticker, 20Y universe. Needs retest on 128-ticker (step 35).
- 20Y includes the full GFC-to-AI-boom cycle. A regime with fundamentally
  different dynamics could break the smoothing edge.
- Smoothing increases exposure to stale signals: if a quality deterioration
  happens (e.g. a ticker fundamentally breaks), smoothing delays its
  removal from pick lists by ~6 months.

## Next steps

- Run full validation battery (step36) on CAP5-smooth6M
- Retest on 128-ticker universe (step35) once regen completes
- If still robust, consider production adoption
