# Dead Ends — Do Not Repeat

All experiments from prior YLOka sessions (branches `claude/rebuild-stock-selection-YLOka`):

## Session 1 — Basic structural experiments (19 variants)
- Conviction sizing (softmax tilt on scores): +0 to +1.6pp CAGR, sample-of-2 artifact. KILL.
- Soft-cash continuum: no improvement over hard regime gate. KILL.
- Acceleration overlay (2y price acceleration filter): +1.6pp from 2 specific years (2009, 2016). KILL.
- Donchian-130 breakout filter: +8pp apparent, pure sample artifact from NVDA concentration in 2 years. KILL.
- K/h grid: K=3, h=6 confirmed locally optimal for v3-style GBM scorer. CONFIRMED.
- Cash yield 3% in cash months: +0.07pp durable. KEEP (but tiny).

## Session 2 — Multi-target ensemble, dispersion-K (19 variants)
- 12m regressor head: walk-forward trained GBM on 12m fwd returns. All variants -0.05 to -9.1pp. KILL.
- Classification head (top-quartile binary): similar degradation. KILL.
- Dispersion-conditional K (K=2 high-disp, K=5 low-disp): +1pp from 3 years. KILL.
- Ensemble blending (3m+6m+12m): no improvement over 3m+6m. KILL.

## Session 3 — Feature-based scorers (11 variants)
- Idiosyncratic momentum (SPY-residualized): -13.6pp CAGR as standalone. Too noisy at K=3. KILL.
- Runner-pattern distance: -5 to -8pp. KILL.
- FIP score (freefall-in-progress): as tilt, no sustained lift. KILL.
- RSI zone score: -3 to -5pp. KILL.
- CRT (cross-sectional rank trajectory): -4pp. KILL.
- Breakout strength / Donchian: already tried in S1. KILL.

## Session 4 — Adaptive IC, dynamic hold, regime-K (10 variants)
- Adaptive IC-weighted ensemble (rolling 24m IC as mixing weight): +0.015 Sharpe only, no CAGR gain. KILL.
- Dynamic hold (keep names while score in top-85%): -5.7pp CAGR (turnover penalty dominates). KILL.
- Monthly score check with dynamic hold: even worse. KILL.
- Regime-conditional K (K_bull=1, K_normal=3): +2.3pp from 2 years. KILL.

## Session 5 — Regime-specialist GBMs (8 variants)
- Per-regime GBMs (bull/normal/recovery × 3m/6m): all variants -7.7 to -13.9pp CAGR. KILL.
  Reasons: training data fragmentation (only ~30 months of bull-regime data), regime-label noise,
  out-of-regime extrapolation failure, worse MaxDD than baseline.

## Structural limitations of existing data
- Price-only feature space is SATURATED after 88+ experiments and v3.
- No volume → no volume-based signals.
- No fundamentals → no earnings, profitability, or valuation signals.
- No PIT GICS → no proper sector neutralization.
- No options data → no IV signals.
- Survivorship: 1833 tickers, only 9 truly delisted. Partial bias overlay attempted.
