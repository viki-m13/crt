# Dead Ends

## Session 1

### Volatility Targeting (I01)
- **What**: Scale equity exposure to target annual vol (vt∈{10-25%}, lb∈{3,6,12m})
- **Result**: Best: CAGR=31.2%, Sharpe=0.860 (vs baseline 40.7%, 0.863). CAGR -9pp, Sharpe -0.003
- **Why it failed**: The v3 crash gate already handles the worst drawdowns (exit to cash in crash regime).
  Vol targeting during non-crash periods scales back the early-recovery months, which ARE the high-return
  periods in this strategy. Net: sacrifices CAGR without improving Sharpe.
- **Verdict**: DEAD for this architecture. Would only help if there were NO crash gate (bare momentum).

### Score-Proportional Weighting (I04)
- **What**: Weight positions proportionally to score^2 instead of EW
- **Result**: 40.8% CAGR, 0.860 Sharpe — essentially identical to baseline
- **Why**: With K=3, the weight difference between EW and score-prop is minimal (~33%±10%)
- **Verdict**: DEAD

### Invvol Weighting (I04b)
- **Result**: Identical to baseline — same picks, negligible weighting difference
- **Verdict**: DEAD

### K=2 / K=1 (I11 reduced K)
- K=2: CAGR=37.0%, Sharpe=0.737 — worse than K=3 (higher idiosyncratic risk)
- K=1: CAGR=28.5%, Sharpe=0.563 — much worse (extreme single-stock concentration)
- **Verdict**: DEAD. K=3 is the optimum for this signal/universe

### Quality Filter on ML signal (I17)
- **What**: Filter to sharpe_1y > 0 AND above-median trend_health_5y before ML scoring
- **Result**: Identical to baseline — the ML signal already implicitly filters for quality
- **Verdict**: DEAD

### Shorter Hold Period (h=3m, h=2m) (I07 variant)
- h=3: CAGR=31.3%, Sharpe=0.736 — worse
- h=2: CAGR=21.2%, Sharpe=0.588 — much worse (high turnover + worse momentum timing)
- **Verdict**: DEAD for h < 6m. h=6m is optimal for this signal.

### Longer Hold Period (h=12m) (I07 variant)
- h=12: CAGR=35.1%, Sharpe=0.863 — same Sharpe but -5.6pp CAGR
- **Verdict**: DEAD — no improvement over h=6m

### Phase 2 Rung 1-4 (Pure JT Momentum without ML)
- Best: JT + quality + regime gate, CAGR=9.2%, Sharpe=0.423
- Far below ML baseline. JT momentum without the GBM signal is a much weaker predictor.
- **Verdict**: DEAD as standalone. Useful only as baseline sanity check.

## Inherited from YLOka Research (do not re-test without new twist)

See `/home/user/crt/research/YLOka/graveyard/` for full list. Key ones:
- Conviction sizing (soft-max weighting by score z-score): -0.05pp CAGR
- Donchian-130 breakout filter: +8pp sample-of-1 (not reproducible)
- Multi-target ensemble (12m + classifier heads): -0.05 to -9.1pp
- Dispersion-conditional K (H7): +1pp sample-of-3
- Adaptive IC / dynamic-hold: -5.7pp
- Regime-specialist GBMs: -7.7pp to -13.9pp
- Runner footprint features: -13.6pp
