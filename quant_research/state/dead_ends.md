# Dead Ends

Approaches confirmed to NOT work (from prior codebase research, not our own experiments yet).
Do not re-try without a substantively different formulation.

---

## From v6 sweep (9,114 variants):
- **Stricter regime gates (faber, faber_lite, strict_dd, combo)**: -5 to -22pp CAGR
- **Crash fallback to SPY or TLT**: -5 to -7pp CAGR, MaxDD WORSE
- **SPY DD-based gross scaling**: -7pp CAGR
- **Trailing stop on portfolio drawdown**: catastrophic (CAGR <5%)
- **Sticky cash re-entry**: -10 to -20pp CAGR
- **Smart re-entry**: -9pp CAGR
- **Pullback filter (deep pullback stocks)**: -10 to -13pp CAGR
- **Pick-momentum filter**: -13pp CAGR
- **Vol-penalty on score**: -2 to -14pp CAGR
- **Quality blend (multiply by trend_health)**: -3 to -14pp CAGR
- **K=4 or K=5 with v3 signal**: -4 to -8pp CAGR
- **Conv/Softmax weighting**: Sharpe -0.20, MaxDD worse

## From v8b (>1,000 variants):
- **Pure 12-1 momentum**: 19% WF CAGR (too weak)
- **Idiosyncratic momentum**: 18% WF CAGR (same)
- **Stage-2 trend (Weinstein-style)**: 11% WF CAGR
- **Tight-consolidation breakout**: 0.6% WF CAGR (too few setups)
- **New LightGBM ranker (80 features)**: 17-22% WF CAGR (raw regressor objective is inferior to v3's HistGBM on this panel size)
- **Banger top-decile classifier**: 17-24% WF CAGR
- **ML + LGB blend (50/50)**: 30-38% WF CAGR (LGB drags ML down)
- **Filter ML picks below-200dma**: 26% WF CAGR (over-filters)
- **Filter ML picks by pullback>50%**: 23-26% WF CAGR
- **Heavy DD-scaling de-leverage**: 30-47% WF CAGR

## From v5:
- **Chronos-bolt-mini, IBM TTM as filters**: degrade WF CAGR
- **Lag-Llama**: impractical on CPU (11min/20 forecasts)
- **Any model larger than bolt-tiny**: too slow for full panel

## Our own experiment (exp_001):
- **Cross-sectional OLS composite**: -0.6% CAGR, essentially random. The OLS weights are highly 
  unstable across 5-year rolling windows and the signal is too noisy.
- **Low-vol filter on momentum (R2)**: HURTS performance vs pure momentum. High-vol stocks 
  are the big winners in this universe.
- **200-day MA regime gate (binary)**: Hurts with v3 signal (16 cash months vs 4 for tight gate).
  Too restrictive.
