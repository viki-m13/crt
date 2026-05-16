# Scoping: a second uncorrelated alpha sleeve (path to higher Sharpe)

**Question.** Phase 11 proved blended Sharpe > 2.0 is unreachable from
the single v5 price-momentum sleeve (Sharpe ~0.91) plus the panel's
low-Sharpe diversifiers — the 2-asset tangency bound caps it at ~1.14.
The only honest route to materially higher Sharpe is a **second alpha
sleeve that is genuinely uncorrelated** to v5. This document scopes
what that takes, grounded in an empirical probe (Phase 12).

## Empirical finding — price-only second sleeves are NOT uncorrelated enough

`probe_second_sleeve.py` built 6 candidate price-only sleeves from
features already in the augmented panel (same K=2, invvol cap 0.40,
rule-based rebalance; no Chronos so it doesn't inherit v5's DNA), and
measured each one's correlation to the deployed v5 return stream:

| Candidate sleeve | signal | Sharpe | vol | MaxDD | **corr → v5** | blend tangency SR* | WF-mean SR |
|---|---|---:|---:|---:|---:|---:|---:|
| quality | quality_score_5y | 0.81 | 19% | -47% | **+0.19** | 1.12 | 0.88 |
| lowvol | -vol_1y | 0.75 | 17% | -21% | **+0.15** | 1.10 | 0.67 |
| deep_pullback | pullback_5y | 0.57 | 72% | -76% | +0.36 | 0.95 | 0.44 |
| meanrev_st | -ret_21d | 0.50 | 66% | -95% | +0.33 | 0.93 | 0.59 |
| value_ltr | -mom_5y | 0.47 | 60% | -86% | +0.50 | 0.91 | 0.54 |
| lowbeta | -beta_2y | 0.44 | 57% | -67% | **-0.01** | 1.01 | **-0.07** |

Plus no-crash-gate and cross-asset variants:

| Variant | Sharpe | corr → v5 | blend SR* |
|---|---:|---:|---:|
| quality, no crash gate | 0.69 | +0.22 | 1.04 |
| lowvol, no crash gate | 0.73 | +0.16 | 1.08 |
| cross-asset SPY/TLT dual-mom | 0.55 | +0.17 | 0.99 |

**The best buildable second sleeve gets the blend to ~1.1 — barely
above v5's own 0.91, nowhere near 2.0.**

## Why price-only sleeves can't be uncorrelated enough (structural)

1. **Shared market beta.** Every long-only equity sleeve on the S&P
   500 carries broad-market beta. Even "low-correlation" factor
   sleeves (quality, low-vol) floor at ρ ≈ +0.15-0.20 to v5 because
   in any broad sell-off they all fall together.
2. **Shared crash gate.** v5 and the natural sleeve designs both go to
   cash in the same SPY-defined crash regime. Removing the gate from
   the second sleeve barely helps (ρ +0.16-0.22) and costs it ~10pp
   of Max DD.
3. **Cross-asset trend isn't independent either.** SPY/TLT dual-
   momentum (ρ +0.17) co-moves with v5 because its risk-on/off switch
   is driven by the same equity regime v5's gate reacts to.
4. **The one truly uncorrelated candidate has no edge.** lowbeta is
   ρ = -0.01 but WF-mean Sharpe is **negative** (-0.07) — it does not
   survive out-of-sample. Zero correlation with zero alpha is useless.

**Conclusion: a genuinely uncorrelated, OOS-robust sleeve cannot be
built from this repo's price-only S&P 500 data. It requires a
different return *driver* — i.e., a different data family.**

## The honest math — how many sleeves to reach a Sharpe target

For N equal-Sharpe (SR), pairwise-correlation-ρ sleeves, the
optimally-blended Sharpe is approximately:

```
SR_blend ≈ SR * sqrt( N / (1 + (N-1)*ρ) )
```

Starting from SR ≈ 0.9 per sleeve:

| Target | ρ = 0.0 | ρ = 0.15 | ρ = 0.30 |
|--------|--------:|---------:|---------:|
| 1.3    | 2 sleeves | 3 | 6 |
| 1.5    | 3 | 4 | 12 |
| **2.0**| **5** | **9** | impractical |

So even with *perfectly* uncorrelated sleeves you need ~5 independent
Sharpe-0.9 alpha sources for 2.0. At the realistic ρ ≈ 0.15 you'd need
~9. **Sharpe ~1.3-1.5 is a plausible 2-3 year research target;
2.0 is not a realistic honest goal for a long-only equity book.**

## Candidate second-sleeve programs (ranked by honest feasibility)

Ranked by `independence potential × OOS-robustness ÷ build cost`.
"New data" = not currently in the repo (the user's constraint).

### Tier 1 — highest ROI, needs fundamentals data

1. **Earnings-revision / post-earnings-drift sleeve.**
   - Signal: analyst EPS-estimate revisions + earnings-surprise drift.
   - Driver: information diffusion around fundamentals — genuinely
     orthogonal to price-trend (different cause).
   - Data needed: quarterly EPS estimates + actuals (e.g.
     Sharadar SF1, or free-tier Financial Modeling Prep / Finnhub).
   - Expected: Sharpe ~0.7-1.0, ρ to v5 ~0.05-0.15 (lower than any
     price sleeve because the driver is fundamental, not price).
   - Effort: medium. Build a point-in-time fundamentals panel
     (filing-date lagged, no look-ahead), score, walk-forward.
   - Risk: PIT fundamentals are hard to source cleanly; restatement
     and filing-lag bugs are the classic trap.

2. **Quality/profitability factor (fundamental, not the price-proxy).**
   - Signal: gross-profitability, ROIC, accruals, asset growth.
   - Driver: balance-sheet quality — orthogonal to momentum.
   - Data: annual/quarterly fundamentals (same source as #1).
   - Expected: Sharpe ~0.5-0.8, ρ ~0.10. Lower-vol → helps the blend
     denominator even at modest Sharpe.

### Tier 2 — moderate, uses broader market data

3. **Cross-asset carry / risk-parity overlay.**
   - Signal: carry across equities/bonds/commodities/FX + vol-scaled
     risk parity.
   - Driver: term-structure / carry — different from equity selection.
   - Data: a handful of liquid futures or ETF proxies (some already
     here: TLT; would add DBC/GLD/UUP/commodity ETFs).
   - Expected: Sharpe ~0.6-0.9, ρ to v5 ~0.0-0.1 (it's a different
     asset class). The cleanest *true* decorrelation available.
   - Effort: low-medium. ~6-10 ETF return series + carry/vol scaling.

4. **Defensive vol-risk-premium sleeve (options).**
   - Signal: short-vol / put-write style premium harvested with a
     crash circuit-breaker.
   - Driver: variance risk premium — orthogonal to cross-section.
   - Data: options (SPX/VIX) — new data, harder.
   - Expected: Sharpe ~0.8-1.2 but fat left tail; needs careful
     crash gating to be honest.

### Tier 3 — research-grade, high effort / high uncertainty

5. **Alternative-data sentiment** (news/web/supply-chain). Genuinely
   orthogonal but data is expensive, noisy, and short-history — hard
   to validate honestly walk-forward.
6. **Statistical-arbitrage / pairs** within the S&P 500. Market-
   neutral so ρ ≈ 0 by construction, but capacity-constrained and
   execution-sensitive at the monthly cadence this product uses.

## Recommended phased plan

**Phase A (cheapest true decorrelation — do first):**
Build the **cross-asset carry/risk-parity sleeve** (Tier 2 #3). It
only needs ETF return series, gives the cleanest ρ≈0 to v5, and is
fully validatable with the existing walk-forward harness. Target: a
Sharpe ~0.7 sleeve at ρ≈0.05 → blended with v5 ≈ **1.15-1.25**, with
materially lower Max DD. Realistic in days, not months.

**Phase B (the real Sharpe lever — fundamental sleeve):**
Source a point-in-time fundamentals panel and build the
**earnings-revision sleeve** (Tier 1 #1). This is the orthogonal
*alpha* (not just risk reduction). Target Sharpe ~0.8 at ρ≈0.10 →
3-sleeve blend (v5 + carry + earnings) ≈ **1.35-1.5**.

**Phase C (stretch):** add the fundamental-quality sleeve (Tier 1 #2)
for a 4-sleeve blend ≈ **1.5-1.6**. This is the realistic honest
ceiling for a long-biased product. **2.0 would require a
market-neutral statarb or options-VRP sleeve** with its own
infrastructure and capacity limits — a separate product, not an
overlay.

## Validation protocol (mandatory for every new sleeve)

1. **PIT discipline.** Any fundamental data must be lagged to its
   actual filing/availability date — never period-end. This is the
   #1 source of fake alpha.
2. **Walk-forward only.** Same 10-split harness; the sleeve's model
   trained only on data older than (test − embargo). No per-split
   parameter fitting.
3. **Correlation stability.** Report ρ to v5 *per walk-forward
   split*, not just full-sample. A sleeve whose decorrelation only
   holds in-sample is worthless. Require |ρ| < 0.25 in every split.
4. **Blend is fixed, not optimized.** Combine sleeves at fixed
   risk-parity (1/vol) weights — never per-period mean-variance
   optimized (that overfits the covariance).
5. **MC delisting overlay** on the blended stream, same as v5.
6. **The honest bar:** a new sleeve is adopted only if the *blended*
   WF-mean Sharpe rises AND WF-min Sharpe does not fall, on the
   augmented PIT panel, with all params fixed.

## Bottom line

- A second *price-only* sleeve cannot decorrelate enough (proven
  empirically: best blend ~1.1). 
- The honest path is **different data families**: cross-asset carry
  (cheap, true ρ≈0, do first) then a **PIT-fundamentals
  earnings-revision sleeve** (the real orthogonal alpha).
- Realistic honest ceiling for this long-biased product is
  **Sharpe ~1.3-1.6 with 3-4 genuinely independent sleeves**.
  **2.0 is not an honest target** without a separate market-neutral
  or options-VRP strategy.

## Files
- `probe_second_sleeve.py` — the empirical orthogonality probe
- `augmented/second_sleeve_probe.csv` — per-candidate metrics + corr
- `augmented/second_sleeve_streams.csv` — audited return streams
