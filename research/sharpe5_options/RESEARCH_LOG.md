# Sharpe-5+ Options Strategy Research — Honesty Ledger

**Started:** 2026-08-08 04:20 UTC · **Budget:** 8 hours autonomous research
**Goal:** invent an options strategy with annualized Sharpe ≥ 5, validated honestly and thoroughly.

## Pre-registered methodology (committed BEFORE any strategy results)

**Data.** Real EOD option quotes (bid/ask/IV/greeks) from the DoltHub
`post-no-preference/options` database: S&P-500/Dow components + a few ETFs,
2019-02 → 2026-08. Sampling cadence is ~weekly in 2019 and ~Mon/Wed/Fri from
2020 on. Chains carry ~3 monthly-ish expirations per day. This window contains
the COVID crash (Feb-Mar 2020), the 2022 bear, the Aug 2024 vol spike, and the
2025 tariff shock — a fair tail-risk sample.

**Fills (pessimistic by default).**
- OPEN: sell at **bid**, buy at **ask** (worst side of EOD spread). No mid fills.
- CLOSE before expiry: same worst-side rule.
- Expiry settlement: intrinsic value at parity-extracted spot on expiration
  observation date. Quotes with bid=0 cannot be sold; crossed/absurd quotes dropped.
- Sensitivity to fills reported (worse: bid−1 tick; better: mid) but the
  HEADLINE number is always worst-side.

**Spot.** Extracted per (date,symbol) via put-call parity on near-ATM pairs
(median of C−P+K over the 5 strikes closest to ATM, nearest expiry ≥ 5 days).
Sanity-checked for continuity; days failing checks dropped.

**Capital & Sharpe.** P&L is computed on a fixed notional capital base that must
cover worst-case loss of every open position (full max-loss reserve for defined
risk; for undefined-risk shorts, a reserve of 15% notional, marked daily).
Sharpe = mean/std of per-observation portfolio returns × √(obs per year),
using the actual observation calendar (~150/yr). Risk-free rate subtracted
(T-bill ~ constant per year table). No Sharpe on stale/flat segments: days with
no positions still count as 0-return observations if the strategy is "live".

**Anti-overfitting protocol.**
1. **Split:** development = 2019-02 → 2024-12. **Holdout = 2025-01 → 2026-08,
   touched only by the final candidate (once per family, logged).**
2. Every experiment (including failures) appended to this log with its dev-set
   result. Trial count feeds the Deflated Sharpe Ratio (Bailey & López de Prado)
   of the final candidate.
3. Walk-forward inside dev where parameters are fitted: params chosen on data
   strictly before each test year.
4. Bootstrap 95% CI on Sharpe (stationary block bootstrap, 2000 resamples).
5. Red-team pass on the final candidate: lookahead audit, quote-staleness audit,
   fill realism, capacity, margin realism, regime dependence.

**Honesty commitments.** If nothing honestly clears Sharpe 5, the report says so
and presents the best honest result with its CI. No metric shopping: Sharpe is
on total portfolio returns including cash drag, not per-trade or "when active".

---

## Experiment log

(appended chronologically; every trial counts toward DSR trial count N)
