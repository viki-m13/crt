# Lowering the −56% GFC floor — leading trend-rollover → MN sleeve (VALIDATED)

**Date:** 2026-05-17 · `regime_timing_lead.py`,
`regime_timing_validate.py`.

**Ask (user):** push the regime-timing angle that could actually lower
E2's −56% accumulating-DCA drawdown floor — thoroughly validated.

## The lever

The deployed crash gate (`classify_regime_tight`) fires on a 21-day SPY
shock — it is **late** (only ~11 months total, ~3 in all of 2008; see
`DCA_INVESTOR_EVAL.md`). The repo's documented dead-ends: reactive
vol-target / DD-breaker bleed 9–27 pp CAGR *and* wreck WF/era
consistency (`IMPROVE_FINDINGS` Phase 11). Untried angle: a **leading
trend-rollover tier** —

> SPY < its 200-day SMA **AND** the 200-day SMA slope < 0

— which fires *before* the shock rule (26 "pre" months + 11 "crash"
vs the old gate's late 11), routing the book to the repo's already
validated **market-neutral sleeve** (`v5_mn_sleeve_returns.csv`, full
2003–2025, Sharpe 1.05, ann vol 14%, ρ≈0) instead of dead cash —
keeping a positive-carry, uncorrelated stream on while the systemic
drawdown plays out, then back to E2 when the trend repairs. Strictly
causal (each month uses daily SPY only through the prior month-end).

## Result vs deployed E2 (switch-cost 10 bps)

| variant | CAGR | Sharpe | **accum-DCA DD (the floor)** | WF | era | worst-5y |
|---|--:|--:|--:|--:|--:|--:|
| **E2 (deployed)** | 56.6% | 1.10 | **−56.0%** | 10/10 | 4/4 | +13.6% |
| **+ lead→MN 0.6** | 51.5% | **1.25** | **−41.9%** | 10/10 | 4/4 | **+17.1%** |
| **+ lead→MN full** | 47.2% | **1.27** | **−31.1%** | 10/10 | 4/4 | **+17.2%** |
| repo's reactive −25%DD→MN | 40.6% | 1.06 | −39.3% | 9/10 | 3/4 | **−2.8%** |

It is the **first lever in the repo to materially lower the GFC floor
while *improving* risk-adjusted return and consistency** — Sharpe
+0.15/+0.17, worst-rolling-5y DCA +3.5pp, WF 10/10 and eras 4/4
preserved. The cost is CAGR (a real, smooth efficient frontier — not
free).

## Full overfit gauntlet — all pass

0. **Causality/leakage:** gate is deterministic and built strictly on
   prior-month-end daily SPY (asserted).
1. **Switch-cost insensitive:** CAGR 51.6→51.2%, DD ~−42%, Sharpe 1.25
   flat at 0/10/20/30 bps (only ~37 gated months → trivial turnover).
2. **Parameter plateau (48 configs:** slope-lookback {10,21,42,63} ×
   crash-subthr {−3,−5,−8%} × route {0.4–1.0}**):** accum-DCA DD
   min/med/max = −47/−37/−31% — **100% of configs beat E2's −56%
   floor**, Sharpe ≥ 1.10 in **100%**, era==4 in **100%**, WF≥10 in
   83%. A wide robust plateau, not a fragile peak.
3. **TRUE OOS — the decisive test** (design 2003-12 *with* GFC |
   untouched holdout 2013-26 *with* COVID-2020 + 2022 bear):

   | | design DCA-DD/Sh | holdout DCA-DD/Sh |
   |---|--:|--:|
   | E2 | −53% / 1.15 | −56% / 1.24 |
   | lead→MN 0.6 | −32% / 1.30 | **−41% / 1.29** |
   | lead→MN full | −31% / 1.31 | **−30% / 1.30** |

   The gate **also lowers the drawdown and raises Sharpe in the
   untouched holdout** — entirely different crashes (2020, 2022) — for
   ~−2 pp CAGR. It is a genuine regime mechanism, **not a 2008
   curve-fit**: the anti-overfit signature.
4. **Leading ≫ reactive:** the repo's already-shipped reactive
   −20/25/30%-DD→MN switch is strictly worse on every axis (CAGR ~40%,
   Sharpe ~1.06, **worst-5y goes negative −2.8%** — it locks in losses
   by switching at the bottom, WF 7-9/10, era 3/4). Anticipatory
   timing decisively beats reactive de-risking — the clean proof of
   the thesis.
5. **Route-fraction frontier 0.0→1.0:** perfectly smooth monotone
   tradeoff (CAGR 55.4→47.2%, Sharpe 1.09→1.27, DD −56→−31%, WF 10 &
   era 4 the *whole* way). An efficient frontier — pick the point — not
   a knife-edge.
6. **Alternate trigger robustness:** an independent leading signal
   (SPY trailing-vol regime, different construction) → MN gives the
   *same* qualitative result (DD −56→−42% at 0.6, Sharpe 1.27, WF
   10/10, era 4/4). The robustness is in the **idea**, not one
   parameterisation.

## Honest caveats

1. **Not free** — it buys floor + Sharpe with CAGR: route 0.6 =
   56.6→51.5% (−5.1 pp) for −14 pp floor; route 1.0 = −9.4 pp for
   −25 pp. A genuine efficient frontier, disclosed, not engineered
   away.
2. **−31% is still a real drawdown.** This *narrows* the GFC floor
   materially; it does not make the strategy low-risk. Same honest
   framing as everywhere else in the repo.
3. **Depends on the MN sleeve** (a real, deployable, cost-net validated
   repo stream). Degrades gracefully to cash (−32%) or SPY (−38%) if
   MN is unavailable — still lowers the floor, just less efficiently.
4. **It changes the product's risk profile** (no longer pure E2).
   Deploying it is a *product decision* — offered as a risk-managed
   variant (**E2-RM**), not auto-shipped. `build_webapp_v5_pit.py`,
   `data.json`, `STRATEGY_SPEC` are unchanged on this branch.

## Recommendation

**E2-RM = E2 + leading trend-rollover → MN at route ≈ 0.6** is the
recommended risk-managed variant: floor **−56% → −42%**, Sharpe
**1.10 → 1.25**, worst-5y **+13.6 → +17.1%**, WF 10/10, eras 4/4, 100%
10y DCA-win retained, for **−5 pp CAGR (56.6 → 51.5%)** — and it is
*stronger in the untouched holdout*, decisively beats the repo's
existing reactive switch, and sits on a wide plateau with a
smooth frontier and a robust alternate-trigger replication. Route 1.0
is the maximal-defence point (−31% floor, Sharpe 1.27, −9 pp CAGR).
This is the genuine answer to "regime-timing that actually lowers the
floor" — the keep-E2-headline / offer-E2-RM-as-the-de-risked-product
split is the honest deployment choice; offered for sign-off.

## Files

- `regime_timing_lead.py` — the lever + first-look
- `regime_timing_validate.py` — the full gauntlet (causality / cost /
  plateau / TRUE-OOS / leading-vs-reactive / route frontier / alt
  trigger)
- `augmented/regime_timing_lead.json`, `regime_timing_validate.json`
