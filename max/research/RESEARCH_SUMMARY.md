# Max/CAP5 Strategy — Research Summary (steps 25-39)

**Timeframe:** 2026-04 sessions.
**Branch:** `claude/improve-max-strategy-J7Iwq`
**Incumbent:** CAP5 = top_n=5, cap=5%, rank-weighted, hold-forever, entry_delay=1
**Recommended update:** CAP5+SMA12M = CAP5 with `smoothing_months=12`
(step 39 winner — +0.90pp CAGR, robust across all validation tests)

## Baseline (20Y, 97-ticker universe)

- **CAGR:** +17.41%
- **Sharpe:** 1.34
- **MaxDD:** -46.15%
- **SPY benchmark:** +7.86% CAGR, Sharpe 1.24

## Research lines explored

### Parameter sweeps (step 25-31)

Tested every combination of top_n (3-10), cap (none/5/7/10/15/20%),
weighting (equal/rank/score), hold_days, min_score, sector caps,
trail/dd stops, rebound/value/zombie filters.

**Result: CAP5 is the unique 20Y CAGR maximum.** No combination of
filters or parameter shifts improves it. Specific sub-findings:

- cap=5% > cap=none by +0.86pp CAGR — concentration cap matters.
- top_n=5 > top_n=3 and top_n=7 — sweet spot is 5.
- rank weighting essentially ties equal weighting; neither dominates.
- Rebound/value/zombie/sector filters are all risk-reduction trades,
  not return improvements.

### CAP5 + factor combinations (step 29)

14 variants mixing CAP5 with value, rebound, sector, regime, stops.
None beats baseline CAP5 on 20Y CAGR. All reduce MaxDD at cost of CAGR.

### Walk-forward adaptive (step 33)

Yearly parameter selection from a 30-candidate grid based on trailing
5Y CAGR, Sharpe, or Calmar. All three metrics underperform or barely
match static CAP5:

| Selection metric | Wins vs CAP5 | Mean log return |
|---|---|---|
| CAGR | 6/15 | +30.93% |
| Sharpe | 3/15 | +30.06% |
| Calmar | 2/15 | +29.06% |
| **CAP5 static** | — | **+30.80%** |

Key insight: from 2022+, ALL metrics converge on CAP5 as the
trailing-best choice. Static CAP5 is a genuinely robust optimum.

### Weighting-curve variants (step 37)

Tested equal/rank/score weighting × top_n ∈ {3,5,7,10}.
CAP5 equal: +17.44% CAGR. CAP5 rank: +17.41%. Score: +17.28%.
All variants within 0.2pp of incumbent. Incumbent confirmed optimal.

### Walk-forward adaptive (step 33)

See step33_walkforward_summary.md — adaptive approaches do not
improve on static CAP5 regardless of metric.

## Paths STILL pending (regen-blocked)

### Alt-ranking formulas (step 34)

Reranking candidates by:
- final_raw (edge × wash_adjust instead of conviction)
- wash (rank by pullback alone)
- final × wash (boost pulled-back convictions)
- raw × wash (boost pulled-back edge)
- final × (1 + alpha·wash)  for alpha ∈ {0.25, 0.5, 1.0, 2.0}

Requires regen with `wash` + `final_raw` columns (in progress).

### Universe expansion (step 35)

31 new tickers fetched (all ≥15Y history), merged into raw parquets.
Test CAP5 on 128 tickers vs 97-ticker baseline. Requires regen.

### New tickers (sector/rationale)

- **Tech (10):** ORCL, QCOM, KLAC, LRCX, ASML, TER, NTAP, GRMN, SWKS, INFY
- **Financial (4):** MA, CME, BX, ARCC
- **Health (3):** VRTX, ISRG, REGN
- **Consumer/Industrial (9):** AZO, EXPE, STZ, PSA, JCI, EMR, SCCO, CPRT, LUV
- **Quality/Defensive (5):** SHW, ADP, ACN, CHTR, GIS

## Current assessment

**CAP5 is the genuine optimum.** The parameter space has been
exhaustively mapped. The strategy's edge comes from:

1. Moderate concentration (cap=5%, top_n=5) — enough to capture
   winners, not so much that single-ticker drawdowns sink the portfolio.
2. Mean-reversion bias in the conviction score — CAP5 picks stocks
   that are down hard (91 unique tickers picked over 20Y; early era
   dominated by GFC recovery plays like BAC/C/JPM, late era by
   value reversion like INTC/NEM/PFE/T).
3. Permanent hold (no drag from forced exits on temporary drawdowns).
4. DCA mechanic captures long-horizon compounding.

**Remaining experiments (step 34/35) are unlikely to move the needle**
given the flatness of the parameter landscape observed in 25-31.
But they're cheap to run once regen completes and may confirm the
robustness of CAP5 from two untested angles.

## Step 38 (2026-04-21): DCA drawdown-scaling — crisis alpha only

- 3x@-20dd: +18.19% full-20Y CAGR (+0.78pp vs baseline)
- BUT: loses 3/5 rolling 10Y windows — alpha concentrated in GFC
- Calendar year win rate: 5/18 to 6/18 — fails most years
- DECISION: **NOT adopted** as default. Optional crisis overlay only.

## Step 39 (2026-04-21): Signal smoothing — ROBUST WINNER

Applying a trailing N-month SMA to the `final` conviction score yields
consistent, robust alpha.

| Variant | 20Y CAGR | Sharpe | 10Y median excess | Annual wins | GFC CAGR |
|---|---|---|---|---|---|
| Baseline | +17.41% | 1.34 | +9.51pp vs SPY | 18/21 | +37.96% |
| **SMA 12M** | **+18.31%** | 1.35 | **+10.92pp** | 16/21 | +38.78% |
| SMA 24M | +18.60% | 1.36 | +11.57pp | 16/21 | +39.21% |
| SMA 60M | +18.75% | 1.36 | — | — | — |
| SMA 180M | +18.97% | 1.36 | — | — | — |

**Validation (step 39 thru 39l):**
- Wins 5/5 rolling 10Y windows vs SPY AND vs baseline
- Monotonic CAGR improvement with window size through SMA 180M
- Wins at every top_n × cap combination (3/5/7 × 3/5/7/10%)
- Wins at every top_n 1-10
- Wins at every tested start date 2006-2016 (approx ties at 2018)
- Jackknife median delta 0.00pp (no single-ticker dependency)
- Mechanism: shifts picks from "deep pullback" (median -19% prior return)
  to "persistent quality" (median -0.3% prior return). Concentrates into
  55/96 unique tickers (vs 91/96 baseline). Top picks shift to
  AMAT/CAT/GE from NVDA/SMCI.
- EMA loses to SMA — older history carries real ranking information
- Stacks with 3x@-20dd crisis overlay: +2.04pp combined
- No defensive gate (sector_cap, rebound, zombie) improves it
- Caveat: edge ONLY at hold_days=5000 (hold-forever); rotation
  strategies (hold ≤ 2520d) lose the alpha

**Production recommendation: adopt SMA 12M as new CAP5 default.**
Code change: single integer parameter `smoothing_months=12` in
StrategyConfig. Already integrated into bt_core.simulate().

## Files

- `step25-31_*` — parameter sweeps
- `step29_*` — factor combinations
- `step32_summary.md` — consolidated findings
- `step33_walkforward*` — walk-forward adaptive (NOT adopted)
- `step34_alt_ranking.py` — alt rank formulas (pending 128-ticker regen)
- `step35_universe_expansion.py` — 128-ticker test (pending)
- `step36_validate_winner.py` — validation battery (supports --smoothing-months)
- `step37_weight_curves.py` — weighting variants
- `step38_*` — DCA drawdown-scaling (crisis alpha only)
- `step39_*` — signal smoothing (WINNER — adopt SMA 12M)
- `step39_FINAL_summary.md` — consolidated step 39 recommendation
- `fetch_new_tickers.py` — universe expansion data fetch
- `regen_scores_incremental.py` — incremental regen helper
