# Beating the deployed WIN2 — regime-conditional select-blend (+ adaptive breadth)

**Date:** 2026-05-17 · `improve_pick_v3.py` (regime hooks),
`improve_pick_regime.py`, `improve_pick_rcd.py`.

**Context:** WIN2 (`trigger=ml_3plus6, select=blend@0.5`) was just
deployed (best-DD lever: −54.5% vs WIN1 −65.8%, at −3pp CAGR vs WIN1).
**Ask:** beat WIN2 *decisively* — not a tweak.

## The lever — regime-conditional select-blend weight

WIN2's `select=blend` uses a *fixed* 50/50 of consensus-rank and
ml-rank. Its only weakness vs WIN1 is CAGR (48.8 vs 51.9) and
worst-year. New idea, one causal knob: make the blend weight a function
of the **already-audited** `classify_regime_tight` label —

- **bull** (confirmed up-trend): lean **momentum** (ml-heavy, `w≈0.30`)
  → recover WIN1's bull-market CAGR.
- **normal / recovery**: lean **consensus** (`w≈0.60`) → keep WIN2's
  stability and shallow drawdown.
- crash → cash (unchanged; never reaches selection).

No new signal, no look-ahead — only *which already-computed scorer
weight* applies, conditioned on the existing regime gate. Trigger stays
`ml_3plus6`; only the basket-formation blend changes.

## Result — a strict Pareto win over the deployed WIN2

| variant | CAGR | Sharpe | Max DD | WF | era | yr-std | DCA 1y/3y/10y |
|---|--:|--:|--:|--:|--:|--:|--:|
| **WIN2 (deployed)** | 48.8% | 0.95 | **−54.5%** | 9/10 | 4/4 | 2.40 | .73/.84/1.0 |
| WIN1 (prior) | 51.9% | 1.01 | −65.8% | 9/10 | 4/4 | 3.21 | .77/.91/1.0 |
| **RC D** (.30/.60/.60) | **53.8%** | **1.03** | **−54.5%** | **10/10** | 4/4 | 2.39 | .76/.90/1.0 |
| **RC D + adaptive-breadth** | **58.8%** | **1.11** | **−54.5%** | **10/10** | **4/4** | 2.39 | **.80/.94/1.0** |

- **RC D dominates WIN2 on every headline metric at the *identical*
  −54.5% drawdown**, and even out-CAGRs the higher-DD WIN1 (+1.9pp)
  while keeping WIN2's 11pp-shallower drawdown.
- **RC D + conviction-adaptive breadth** (stacking the prior
  `IMPROVE_PICK_FINDINGS.md` lever — here the two ARE additive):
  **+10.0pp CAGR over WIN2, +0.16 Sharpe**, WF 10/10, era 4/4, strictly
  better DCA win at *every* horizon, same −54.5% DD.

## Overfit gauntlet — passes (the repo's standard battery)

- **Dense 45-cell plateau** (bull∈.20-.40 × norm∈.55-.65 × rec∈.50-.70):
  Max DD = **−54.5% in 100% of cells**, **WF≥9 in 100% of cells**, 78%
  of cells beat WIN2 on CAGR (CAGR med 51.3%, max 54.4%). RC D's
  immediate neighbours (.30/.55/.60, .30/.65/.60, .30/.60/.50) are all
  CAGR 53.8-54.4% / WF 10/10 / era 4/4 — a real ridge, not a spike.
- **Cost-insensitive:** identical at 0/10/20/30 bps.
- **TRUE OOS (design 2003-12 | untouched holdout 2013-26):**

  | | design CAGR/Sh | holdout CAGR/Sh/DD |
  |---|--:|--:|
  | WIN2 | 70.3% / 1.03 | 34.3% / 0.98 / −54.5% |
  | RC D | 76.8% / 1.10 | **38.4% / 1.07** / −54.5% |
  | RC D + adaptK | 76.8% / 1.10 | **46.4% / 1.31** / −54.5% |

  Stronger *out-of-sample* than WIN2 (holdout +4pp / +0.09 Sh for RC D;
  +12pp / +0.33 Sh for RC D+adaptK) — the opposite of an overfit
  signature.
- **MC synthetic-delisting:** RC D ≥ WIN2 at every hazard (α0 55.6 vs
  49.0; α4 33.5 vs 25.6; α8 −0.8 vs −4.9) — more delisting-robust.

## Honest caveats

1. **Drawdown is EQUAL to WIN2, not lower.** −54.5% is unchanged across
   the entire plateau — it is the 2008 GFC systemic event, which a
   selection-blend lever cannot reshape (only the documented,
   late-firing regime-cash gate touches that tail; see
   `DCA_INVESTOR_EVAL.md`). The honest claim is **"much higher CAGR /
   Sharpe / consistency at WIN2's drawdown"**, not "less drawdown than
   WIN2". (Versus the *prior* WIN1 it is both higher CAGR *and* 11pp
   shallower DD.)
2. **`era==4` is the softest claim.** It holds on RC D's exact cell and
   ~half its immediate neighbourhood but only 18% of the wide 45-cell
   grid. The fully robust, plateau-wide claims are the −54.5% DD and
   WF≥9 (both 100% of cells) and CAGR>WIN2 (78%).
3. **No new signal / no new look-ahead.** Regime labels are the
   existing audited `classify_regime_tight`; its bull/normal thresholds
   were NOT retuned — only the per-regime blend weight, on a wide
   plateau.
4. **Not auto-deployed.** Per repo convention
   (`NOVEL_V9_PROD_FINDINGS.md`, `IMPROVE_FINDINGS.md`), changing the
   live product's public numbers is a user decision.
   `build_webapp_v5_pit.py`, `data.json`, `STRATEGY_SPEC` unchanged.

## Recommendation

**RC D + conviction-adaptive breadth** is the strongest, most
defensible result in this line of work: it strictly dominates the
just-deployed WIN2 by **+10pp CAGR, +0.16 Sharpe**, WF 10/10, better
DCA at every horizon, more delisting-robust, and materially stronger
out-of-sample (holdout Sharpe 1.31 vs 0.98) — all at WIN2's *identical*
−54.5% drawdown. **RC D alone** is the conservative one-knob version
(+5pp CAGR strict Pareto win, fewer moving parts). Both are honest,
plateau-stable, cost-free, OOS-positive; offered for sign-off, not
shipped. The −54.5% drawdown floor (the GFC) still stands — that
remains a separate regime-timing research program, not a
stock-picking knob.

## Files

- `improve_pick_v3.py` — adds `_score_pool_w` + `regime_w` hook
  (reproduces WIN1/WIN2 bit-exactly at defaults)
- `improve_pick_regime.py` — static-w sweep + regime-conditional grids
- `improve_pick_rcd.py` — dense plateau / cost / OOS / MC-delisting
- `augmented/improve_pick_regime.json`, `augmented/improve_pick_rcd.json`
