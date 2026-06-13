# CreditFloor v3 ("Sigma-Clear") — Validation Report

**Date:** 2026-06-12.
**Scope:** the single credit-spread strategy now deployed at `/spreads/`
(engine `v3-sigmaclear` in `strategies/credit_spread/`).

This document records the full validation protocol, every material
finding (including the negative ones), the frozen rule set, and its
honest out-of-sample performance. Read it before trusting any number
on the webapp.

---

## 1. The headline result, stated honestly

The requested property — *"identifies credit spreads daily, for any
duration, that are profitable under conservative fills and with 100%
accuracy never close in the money"* — is **empirically unattainable**,
and this repository now contains the receipts (§4). What is shipped
instead is the best operating point we could find and validate on the
honest frontier:

| | Design window 2008–2018 | Validation 2019–2026 (untouched) |
|---|---|---|
| Independent trades (deduped) | 554 | 1,508 |
| Losses | **0** (incl. the 2008 GFC) | **9** |
| Win rate | 100% in-design | **99.4%** |
| Net P&L (1 contract/trade, conservative fills, commissions) | +$5,931 | **+$18,174** |
| Profitable years | 11 / 11 | **7 / 8** (2020: −$1,386) |
| Worst single trade | — | −$2,218 |

Per published rung (overlapping daily re-publications included):
3,536 resolved validation rungs, 18 losses → 99.49%. The 95%
Clopper–Pearson interval on the deduped validation loss rate is
roughly **0.3%–1.1% per trade**. That — not zero — is the honest
number, and the engine's published copy now says so.

All resolution is **close-at-expiry vs the short strike** ("the spread
never finishes in the money"), matching the live log exactly.

---

## 2. Why the previous "100%" was an illusion (diagnosis)

The legacy CreditFloor engine claimed a pooled walk-forward OOS win
rate of exactly 100% (162,728 tests). Three independent measurements
showed the claim does not transfer forward:

1. **Live scoreboard:** 36 losses in 7,000 resolutions (99.49%) between
   2026-03-20 and 2026-06-05.
2. **Point-in-time daily replay** (`replay.py`, new): re-running the
   exact live protocol day-by-day over 2021–2026 — as-of-date
   eligibility, publish, resolve at the snapped expiry close — gives
   **4,489 losses in 597,849 resolutions (99.25%)**. The fold backtest
   grades eligibility *after* seeing the full fold history; the replay
   grades it the only way that matters, as-of each publish date.
3. **Anatomy of the 36 live losses:** every one was a far-OTM
   (≥15.8% buffer), short-DTE (≤14 session) rung on a name about to
   have an idiosyncratic event (A, SEB, VNT, OSK, BIO, LSTR, RVTY,
   ALSN — earnings/M&A moves of +17% to −22%). Several were
   "tradeable" only on paper; SEB has **no listed options at all**.

Protocol and data bugs found and fixed along the way:

- **Expiry snap-up bug:** the engine certified an h-session window but
  assigned the first standard expiry *on or after* the session target —
  up to ~18 sessions further out. An h=21 signal published 2020-01-23
  was assigned the 2020-03-20 expiry (57 calendar days vs the 29 it
  certified) and rode straight into the COVID crash. Expiries now snap
  **down** (`covered_options_expiry`): the certified window always
  covers the actual trade window.
- **Split-seam corruption:** the append-only price backfill stitched
  post-split adjusted closes onto pre-split history (BKNG 25:1, KLAC
  10:1, CVNA, CAR, APLS) — fake −96% "moves" in the live panel. The
  panel is now rebuilt from scratch on a single adjustment basis every
  scan, and `backfill_prices.py` gained a seam-integrity check.
- **NYSE calendar cache** started at 2014, silently corrupting any
  deep-history expiry math (fixed: 1980+).
- **No optionability, staleness, or history-depth gates** (fixed: all
  three fail closed).

## 3. The validation instrument

`replay.py` simulates the live protocol bit-for-bit and was verified
against the actual live log: on the overlapping window it reproduces
**698 of 698** live signal rungs for the smoke universe, with identical
resolutions (the only diffs: 10 strikes ~1% off from one ticker's
dividend-adjustment seam, and live-pendings that fresher data can now
resolve). Conservative fills are modeled per rung (Black-Scholes + put
smile at IV = 1.3× realized vol, tenor-dependent haircut to below-mid,
$0.05 minimum bid-ask, $1.32/contract commissions), plus a stress
variant at IV = 1.0× realized (no volatility risk premium at all).

Three replay panels were used:

| Panel | Window | Purpose |
|---|---|---|
| 2015+ site panel, folds 2020+ | 2021–2026 | reproduce live protocol |
| 2015+ site panel, folds 2018+ | 2019–2020 | COVID stress probe |
| **Full history (period=max, 1962+), folds 2006+** | **2008–2026** | design + final validation |

## 4. The impossibility result

Each intermediate construction that looked perfect on its design data
failed the next, harder test:

| Construction | Looked perfect on | Failed on |
|---|---|---|
| Legacy conformal (claimed 100%) | fold backtest | live (99.49%), replay (99.25%) |
| v2 margin layer (1.6× history max + 3σ floor + caps + fill gate) | 2021–2024 design AND untouched 2025–2026 holdout (0 losses / 2,312) | 2019–2020 replay: **63 losses / 10,491** (COVID *and* ordinary 2019) |
| Any rule requiring full-history clearance | — | **the empty set**: on the full-history panel, rungs that clear the all-time worst move with margin AND carry ≥$0.05 conservative net premium number **zero** across 18 years × 960 optionable names × 8 durations |

The last row is the structural truth: premium exists only within
~2.5–3σ of spot, and over any multi-year window single names breach
that distance ~0.5% of the time (earnings, M&A, Archegos, COVID).
Every engine in this repo that ever showed "100% + profitable" did so
by truncating history, ignoring fills, or grading itself after the
fact. Three is enough; we stopped building the fourth.

## 5. The frozen v3 rule ("Sigma-Clear")

Certification (unchanged conformal machinery, now on full history,
folds 2006+): per (ticker, side ∈ {put, call}, h ∈ {7, 10, 14, 21, 42,
63, 126, 252}, variant ∈ {plain, regime}) — every walk-forward fold
100% clean at the certified buffer, ≥50 pooled OOS tests, certified
buffer ≤ 25%.

Publication (the v3 layer; all gates fail closed):

```
published buffer  b = 2.5 · σ60_daily · √h + 1%
history clearance b ≥ 0.8 × (worst h-day move in the ticker's full history)
caps              b ≤ 25% (h ≤ 21)   /   45% (h ≥ 42)
expiry            latest standard weekly/monthly expiry ≤ h sessions out
                  (snap-DOWN; certified window covers the trade)
tradeability      conservative net fill ≥ $0.05/share after commissions
underlying        has a listed options chain (optionable.json)
history depth     ≥ ~10 years listed (3,652 calendar days)
freshness         series ≤ 5 sessions stale
```

Design grid (k_sigma × history-clearance, on 2008–2018 only):
(2.5, 0.8) was the zero-loss cell with the most volume — 1,139
published rungs spanning every design year including 316 in 2008.
Neighboring cells: (2.5, 0.6): 6 losses; (2.2, 0.8): 17; (2.0, 0.8): 80.

### Validation (single pass, frozen, 2019–2026)

Per year (published rungs / losses): 2019: 620/6 · 2020: 411/6 ·
2021: 288/1 · 2022: 613/2 · 2023: 273/0 · 2024: 437/0 · 2025: 511/0 ·
2026: 383/3.

The 18 losses (9 independent trades): MKTX (Sep 2019 momentum unwind),
MA, TDG, GNRC (Mar 2020 COVID), DKS (Nov 2021), EOG, CHTR (2022),
GWRE calls (Feb 2026 acquisition pop +38%), AVGO (Jun 2026). All are
exactly the irreducible tail: systemic crash plus single-name events.

Deduped P&L by year, $ per 1 contract per independent trade,
conservative fills, commissions included (full table in
`validate_v3.py` output): every year 2008–2019 positive, 2020 −$1,386,
2021–2026 all positive. Total 2008–2026: **+$24,105 on 2,062 trades**
(credit $32,571, losses paid $8,465).

## 6. Caveats that remain (no way around them with this data)

- **Survivorship:** the replay universe is today's ~960 optionable
  names; tickers that delisted before 2026 are absent. Deep-replay
  loss rates are therefore optimistic at the margin. The append-only
  live log (`live_log.json`) is the unbiased forward measure — losses
  there stay forever, and the legacy engine's 36 losses remain visible.
- **Modeled fills:** no historical options quotes are in the loop. The
  fill model is deliberately below-mid with tenor haircuts and
  commissions, and each published rung also carries a stress fill at
  bare realized vol; but it is a model.
- **Correlated rungs:** daily re-publication of the same trade idea
  produces overlapping windows. Both per-rung and deduped counts are
  reported everywhere; size positions off the deduped numbers.
- **The 0.8× history-clearance floor sits *below* the all-time worst
  move.** That is what makes premium possible, and what makes losses
  possible. The certified (never-breached) strike is published
  alongside for transparency; it is usually untradeable.

## 7. Early-exit and liquidity overlays (tested 2026-06-13)

Two natural proposals for pushing the win rate to 100% were simulated
on the frozen validation trade set (`exit_sim.py`):

**Early exit ("sell before it goes in the money").** Rule: close the
spread when the underlying's close crosses a trigger near the short
strike; filled at the *next* session's close (a close can't be acted
on before it prints), paying BS spread value at 1.5×-stressed IV plus
15% slippage. Result on 2019–2026 (1,508 trades):

| Exit rule | Losing trades | Win rate | P&L | Worst trade |
|---|---|---|---|---|
| Hold to expiry | 9 | 99.40% | +$18,174 | −$2,218 |
| Exit at strike touch | **16** | 98.94% | +$16,945 | −$1,680 |
| Exit at strike +3% | **24** | 98.41% | +$17,069 | −$1,680 |
| Exit at strike +5% | **32** | 97.88% | +$17,001 | −$1,634 |

Early exit moves the win rate **away** from 100%, not toward it. The
mechanism is option pricing, not an execution detail: by the time spot
reaches the short strike, buying the vertical back costs ~0.3–0.5× the
width versus the few cents of credit collected — **every stop is a
realized loss** — and stops also fire on trades that would have
expired worthless (11–53 whipsaws costing $3.9k–$6.5k of P&L). What
stops do buy is severity capping (worst trade −$2,218 → −$1,680, a
thinner tail): a legitimate *risk-sizing* overlay, not a win-rate one.
Gap events (GWRE's +38% M&A pop) move straight through any close-based
trigger.

**GTC take-profit buybacks** (resting buy-to-close limit at a fraction
of the entry credit, filled when the model mid trades through the limit
by half the bid-ask): at 50% of credit the order fills on 1,501 of
1,508 validation trades and rescues just **2 of the 9 losses**
(99.40% → 99.54%) while **destroying 77% of the P&L**
(+$18,174 → +$4,164) — every thin far-OTM credit is halved and pays a
second round of commissions, while the actual losses (immediate
adverse moves: MA/TDG Mar-2020, AVGO, GWRE) never decay through the
take-profit zone first. At 25% of credit: zero rescues, P&L +$10,731.
Worst trade unchanged (−$2,218) in both. GTC *stop* orders are the
early-exit rows above with worse real-world fills (options stops
trigger off stressed wide quotes and fill at the ask; gaps open beyond
the stop). GTC *entry* orders (resting sell-to-open at a richer
credit) are adverse selection by construction: they fill exactly when
the spread richens — the stock moving toward the strike or vol
exploding — so they concentrate fills in the eventual losers.

**GTC entry at a richer credit** (resting sell-to-open at a limit above
the publish-day fill, same strike and expiry): the cleanest adverse-
selection measurement in the project. Because strike and expiry are
fixed, the expiry outcome of a filled order is identical to the
original trade's — only *which* trades fill changes. **All 9 losers
fill at every limit level** (a spread richens precisely when the stock
moves toward the strike or vol explodes — i.e. on its way to losing),
while most winners never richen enough to fill:

| Entry rule | Fills | Losses | Win rate | P&L |
|---|---|---|---|---|
| At publish (market) | 1,508 | 9 | 99.40% | +$18,174 |
| GTC limit @ 1.25× credit | 271 | 9 | 96.68% | **−$3,140** |
| GTC limit @ 1.50× credit | 185 | 9 | 95.14% | −$4,247 |
| GTC limit @ 2.00× credit | 122 | 9 | 92.62% | −$4,838 |

The resting order is a machine for selecting exactly the trades that
are about to go wrong; "more credit per trade" is the market charging
fair freight for risk that has already arrived.

**Crash wing (the structure that actually helps).** Instead of trying
to exit a loss after it starts (always pays the repriced threat), make
deep breaches convex: buy **half a unit** of a long option one width
past the long leg (same expiry), attached only when the net credit
still clears the $0.05 floor after paying for it (wings priced at ask
+ commission, credited only expiry intrinsic — conservative, since
mid-crash marks would be far higher):

| Structure | Design 08–18 | Validation losses | Win rate | P&L | Worst trade |
|---|---|---|---|---|---|
| Plain vertical | 0 / 554 | 9 / 1,508 | 99.40% | +$18,174 | −$2,218 |
| + conditional half-wing | **0 / 554** | **8** | **99.47%** | +$16,809 | **−$1,585** |

The design-window zero-loss invariant is preserved (no new losses
anywhere), the worst case drops 29%, and the deepest breach in the
validation set — TDG's −44% COVID move — flips from −$2,218 to
**+$1,209** (a 2008/2020-style systemic crash makes the wing book
anti-fragile rather than maximally damaged). Cost: 7.5% of P&L.
Shallow breaches (stock pinned between the strikes, e.g. MKTX, DKS)
remain small losses — no overlay reaches 100%, for the §4 reasons.
Published rungs now carry the wing as `crash_wing` metadata (strike,
ratio, cost, net credit after wing) whenever it is affordable. Full
ratio-1.0 wings or wings at the long leg (1×2 backspreads) were also
tested and rejected: their drag floods the thin credits with new small
losses (up to 743 of them) for little extra protection.

**Liquidity filter** (90-day average daily dollar volume): shrinks the
book faster than it removes losses, and the loss *rate* worsens at the
strictest cut:

| ADV filter | Validation trades | Losses | Loss rate |
|---|---|---|---|
| none | 1,508 | 9 | 0.60% |
| ≥ $300M/day | 793 | 4 | 0.50% |
| ≥ $1B/day | 246 | 2 | **0.81%** |

The two losers surviving the strictest filter are **Mastercard**
(COVID, Mar 2020) and **Broadcom** (−20% in days, Jun 2026) — among
the most liquid stocks and options markets on earth. Liquidity buys
better fills (it is why the optionability gate exists); it does not
buy immunity from events. The combination (≥$300M ADV + strike-touch
stop) lands at 98.61% with lower P&L than holding to expiry.

## 8. Reproduction

```bash
cd strategies/credit_spread
python3 fetch_full_history.py            # full-history panel (~3 min)
python3 fetch_optionable.py              # listed-options map (~8 min)
CS_DATA_DIR=$PWD/cache_full CS_FOLD_START=2006 CS_REPLAY_START=2008-01-02 \
  CS_SNAP=down CS_HORIZONS=7,10,14,21,42,63,126,252 \
  CS_REPLAY_OUT=replay_rows_full.csv.gz python3 replay.py   # ~4.5 min
python3 validate_v3.py                   # design/validation/P&L tables
python3 scan.py                          # live scan + publish
```
python3 exit_sim.py                      # early-exit / liquidity overlays (§7)
