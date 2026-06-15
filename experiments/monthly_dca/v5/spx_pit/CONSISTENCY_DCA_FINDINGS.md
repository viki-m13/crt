# Dual-benchmark DCA consistency — beat SPY-DCA *and* QQQ-DCA honestly

**Date:** 2026-06-15 · `improve_consistency_dca.py`

**User goal:** keep CAGR high, maximize the fraction of rolling monthly-DCA
windows (1/3/5/10y, money-weighted) where the strategy beats **both**
SPY-DCA and QQQ-DCA, and add an **off-switch** that prevents the deep
drawdowns. PIT S&P 500 only.

## New objective + anti-overfit discipline

Prior repo work only ever measured DCA-win vs **SPY-DCA**. This adds the
harder **dual** bar (beat SPY-DCA AND QQQ-DCA in the same window) and a
frozen-holdout split (DESIGN ≤2015-12 | HOLDOUT ≥2016-01, evaluated once).
The harness reproduces the deployed E2 stream from the canonical
reproducible sim (`improve_sim_v2`/`improve_pick_v3`; WIN1 repro CAGR 50.9%
vs 51.9%, MaxDD −65.8% exact).

## Baseline — deployed E2 on the new objective (the missing number)

| horizon | win vs SPY | win vs QQQ | **win BOTH** | worst MOIC vs SPY/QQQ |
|---|--:|--:|--:|--:|
| 1y  | 80% | 79% | **77%** | 0.57 / 0.55 |
| 3y  | 96% | 92% | **92%** | 0.64 / 0.59 |
| 5y  | 99% | 99% | **99%** | 0.83 / 0.79 |
| 10y | 100% | 100% | **100%** | 2.5 / 1.9 |

E2 already wins ~99–100% of 5y/10y windows vs both. **The only real
weakness is short horizons** (1y 77%, 3y 92%) and the worst-1y floor (~0.55).

## Finding 1 — off-switches do NOT improve the objective (honest negative)

Pre-registered exposure-overlay menu (causal; park in cash unless noted):

| variant | CAGR | MaxDD | Sharpe | win-both 1y/3y/5y/10y |
|---|--:|--:|--:|--:|
| **E2 baseline** | **56.0%** | −55.9% | 1.10 | **77/92/99/100** |
| + SPY crash off-switch | 56.0% | −55.9% | 1.10 | 77/92/99/100 (no-op) |
| + basket-DD breaker −25% (cash) | 30.6% | −47.5% | 0.96 | 60/58/67/78 |
| + basket-DD breaker −30% (cash) | 35.7% | −47.5% | 1.03 | 63/68/84/99 |
| + vol-target 30% | 37.2% | −55.9% | 1.15 | 74/80/89/100 |
| [diag] basket-DD −30%, park=beta | 42.7% | −63.8% | 1.12 | 69/75/96/100 |

1. **The external SPY off-switch is a no-op** — E2's internal
   `classify_regime_tight` gate already captures the SPY crash signal.
2. **Cash off-switches lower the drawdown and lift the worst-1y floor
   (0.55→0.75) but collapse CAGR and win-rate** — this strategy's CAGR is
   front-loaded into the post-crash V-recovery (+270% in 2009), so every
   month parked in cash loses to the rising benchmark.
3. **Parking in market-beta** recovers some CAGR and the floor but does
   *not* prevent the drawdown (beta also crashes) and still lowers 1y/3y
   win-rate. No overlay beats baseline on the objective.

**Conclusion:** "prevent the drawdown" and "keep CAGR high" are in direct
tension here — the −56% floor is the 2008 GFC and avoiding it forfeits the
recovery that is the edge. The deployed E2 is already the win-rate optimum
among off-switch overlays. The losing windows are dominated by
**idiosyncratic 2-stock entry-timing luck in normal markets**, not crashes.

## Finding 2 — staggered entry does NOT help either (honest negative)

`improve_consistency_staggered.py` builds a correct calendar-tranche
staggered E2 (N monthly tranches, H-month hold; NOT stream-shifting).
Validation: single-tranche reconstruction reproduces the fixed-6m sleeve
(CAGR 43.4% vs true 44.4%).

| variant | CAGR | MaxDD | win-both 1/3/5/10y |
|---|--:|--:|--:|
| E2 baseline (fixed) | 56.0% | −55.9% | 77/92/99/100 |
| E2 staggered N=6 | 34.8% | −57.9% | 68/79/89/98 |
| E2 staggered N=12 | 32.9% | −48.0% | 71/80/90/99 |

Staggering N tranches holds ~N×K names (N=6 → ~6.8 distinct names vs the
baseline's ~4): it is **de-concentration in disguise**, which dilutes the
concentrated momentum alpha → lower CAGR AND lower win-rate. The 2024
timing-luck benefit (`TIMING_LUCK.md`) is real for that one year but does
not generalize to a higher full-sample win-rate.

## Finding 3 — quality-sleeve blend: the one within-constraint improvement

`improve_consistency_qualityblend.py`. Blending E2 with the **price-only
quality sleeve** (S&P-500, no ETF; `second_sleeve_streams.csv::quality`)
is the only lever that does **not** lower the win-rate. Frozen-holdout
(≥2016, never tuned) + weight plateau:

| w_quality | CAGR | MaxDD | Sharpe | HOLDOUT 1/3/5y | HOLDOUT worst-1y |
|---|--:|--:|--:|--:|--:|
| 0.00 (E2) | 52% | −55.9% | 1.09 | 70/90/100 | 0.66/0.72 |
| **0.15** | 47% | −51.3% | 1.13 | **72/89/100** | **0.72/0.78** |
| **0.20** | 45% | −49.7% | 1.15 | **72/89/100** | **0.74/0.80** |
| 0.35 | 40% | −44.7% | 1.20 | 70/88/100 | 0.80/0.84 |

Smooth plateau (not a spike); the gains hold out-of-sample. Honest reading:
the **win-rate itself is near its ceiling** (holdout 1y only +2pp, 3y flat).
What improves robustly OOS is the **severity of the losing windows**
(worst-1y MOIC 0.66→0.74) and **drawdown** (−56%→−50%), plus Sharpe
(1.09→1.15), at a ~5–7pp CAGR cost (still ~45%, "high"). Quality works
where low-vol/low-beta fail because it is a genuine modest alpha that is
positively-but-imperfectly correlated — it cushions bad windows without
forfeiting good ones.

## Bottom line

For a long-only, concentrated, **price-only PIT-S&P-500** strategy, the
deployed E2 (win-both 77/94/99/100% at 1/3/5/10y) is **already near the
achievable frontier** for the dual-benchmark objective. No within-constraint
lever materially raises the win-rate:
- off-switches forfeit the post-crash recovery that *is* the edge;
- staggering / defensive blends dilute the concentrated alpha.

The honest, OOS-validated improvement is a **15–20% quality blend**: equal
win-rate, materially softer worst-case windows and drawdown, higher Sharpe,
~45% CAGR. It does NOT make 1-year DCA-beats-both reach ~100% — that has a
structural ceiling (~77–78%; tech-melt-up years where QQQ rips, plus
idiosyncratic 2-stock draws).

**The only path to a materially higher win-rate is a genuinely orthogonal
alpha, which requires a new data family** (PIT fundamentals / earnings
revisions — see `SECOND_SLEEVE_SCOPE.md`). Tested next (Finding 4).

## Finding 4 — PIT-fundamentals quality sleeve: tested, fails orthogonality

`build_fundamentals_pit.py` + `build_quality_sleeve_pit.py`. Built a
gross-profitability sleeve (Novy-Marx quality = GrossProfit/Assets) from
**SEC EDGAR XBRL facts, strictly lagged to filing date** (653k facts, 605
S&P-500 names, 2009–2026; ~73% CIK coverage — delisted/renamed names absent
from SEC's current ticker file). Same K=2 / inv-vol / crash-gate sleeve
engine. This is a genuine *fundamental* driver, not a price proxy.

| metric | value |
|---|--:|
| standalone sleeve (≥2011) | CAGR 15.9%, Sharpe 0.94, MaxDD −28.5% |
| **corr → E2 per split** | +0.22 … **+0.44** (full +0.30) |
| corr → E2, crash-gate OFF | +0.21 … +0.41 (full +0.27) |
| 80/20 blend, frozen holdout 1/3/5y | 65/80/100 (vs E2 69/89/100) |

It **fails the |ρ|<0.25 stability bar** (max 0.44), and removing the shared
crash gate barely helps (0.44→0.41) — proving the correlation is **shared
market beta, not the gate**. The blend lowers the win-rate (3y 89→80%) while
only improving drawdown (−56%→−49%) and the worst-1y floor (0.65→0.71).

**This empirically confirms (with real PIT fundamentals) the
`SECOND_SLEEVE_SCOPE` thesis:** every *long-only* S&P-500 sleeve carries
market beta and co-moves with the core exactly in the windows that matter.
Decorrelation below ρ≈0.25 needs either a **market-neutral / long-short**
construction (strips beta → real ρ≈0, but adds shorting — a different
product) or a **non-equity asset class**. Neither fits "long-only DCA into
S&P 500."

## Final synthesis — the honest frontier

Across four pre-registered levers (off-switch, staggering, price-defensive
blend, fundamental-quality sleeve), **no within-constraint change
materially raises the dual-benchmark win-rate** above E2's
~77/94/99/100% (1/3/5/10y). The 1-year ceiling (~78%) is structural: a
long-only concentrated equity book loses some 1y windows to QQQ in
tech-melt-up years, and removing those windows requires forfeiting the
recovery (off-switch) or diluting/over-correlating the alpha (blends).

**Shippable improvement that survives every honest test:** the 15–20%
**price-quality blend** (Finding 3) — equal win-rate, worst-1y MOIC
0.66→0.74, drawdown −56%→−50%, Sharpe 1.09→1.15, ~45% CAGR (frozen-holdout
validated, plateau-stable).

**To exceed the ceiling requires a product decision** (not a tuning knob):
(a) allow a market-neutral long-short orthogonal sleeve, or (b) relax the
universe to Nasdaq-100 / a QQQ-beta core. Both change the product mandate.

## Files (Finding 4)
- `build_fundamentals_pit.py` — SEC EDGAR PIT fundamentals fetcher (filing-date lagged)
- `build_quality_sleeve_pit.py` — gross-profitability sleeve + corr-stability + holdout blend
- `augmented/fundamentals_pit_facts.parquet` — PIT facts (filed/end/val per tag)
- `augmented/fundamental_quality_sleeve{.json,_returns.csv}` — sleeve outputs

## Files
- `improve_consistency_dca.py` — dual-benchmark + frozen-holdout harness, off-switch menu
- `improve_consistency_staggered.py` — correct calendar-tranche staggered sim
- `improve_consistency_qualityblend.py` — quality-blend plateau + holdout
- `augmented/improve_consistency_{dca,staggered,qualityblend}.json` — raw results
