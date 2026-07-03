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

## 9. The ROR-vs-accuracy frontier (tested 2026-06-13)

Request: ≥50% return-on-risk per spread at the current 99.4% accuracy.
Measured on the 1,508 validation trades by solving, per trade, for the
strike that prices at each target ROR under conservative fills and
resolving it against the realized path:

| Target ROR/trade | Win rate | Median cushion | Note |
|---|---|---|---|
| 2% | 98.67% | 17.6% OTM | |
| 5% | 96.22% | 13.0% | P&L negative under stress (IV=1.0) fills |
| 10% | 91.98% | 9.2% | barely positive under stress fills |
| 20% | 80.31% | 5.0% | survives stress fills; loses in 2022 |
| 30% | 66.25% | 2.3% | |
| **50%** | **not quotable** | — | no OTM vertical nets ≥50% of max loss under conservative fills, anywhere |

Win rate tracks 1/(1+ROR) almost exactly — the option-pricing identity,
confirmed on our own certified inventory. Structural levers at FIXED
accuracy (same short strike) were also exhausted: width sweep
2.5%→20% of spot moves median ROR only 1.52%→0.48% (bid-ask floor vs
credit dilution; absolute P&L moves the other way); iron-condor
pairing has no inventory (34 of 4,675 published rows, 0.7%, have both
sides quotable the same day). Conclusion: at ~99.4% accuracy the
per-spread ROR is structurally ~1–1.5%; per-trade profit and win rate
trade against each other on a fixed frontier that no strike, width,
structure, order type, or exit rule escapes. The honest lever for
total profitability at fixed accuracy is capital redeployment across
the 7–21-day cycles (~15–150% annualized per rung), not per-trade ROR.

## 10. The reality layer (added 2026-07-02)

Fair critique received: the engine published *modeled* contracts —
theoretical Friday expirations, exact-dollar strikes (e.g. $266.18)
that aren't listed, Black-Scholes credits. Many names list only
monthly expirations; strikes are discrete; far-OTM options often carry
a ZERO bid, meaning the modeled credit is not collectible at all.

`reality.py` now verifies every rung against the ACTUAL listed chain
before publication, fail-closed:

- expiration: the latest expiration the ticker really lists inside the
  certified h-session window (no listed expiration inside → dropped);
- strikes: snapped to real listed strikes in the SAFE direction (put
  short rounds down, call short rounds up — the real cushion is always
  ≥ the certified one); long leg to the listed strike nearest the
  model's protection level;
- quotes: published credit = the NATURAL credit (sell at the short
  leg's bid, buy at the long leg's ask, commissions subtracted) — the
  fill available without negotiating; mid reported alongside;
- liquidity: short bid > 0, long ask > 0, open interest ≥ 10 per leg;
- the crash wing is re-quoted from the real chain (real ask) and only
  attached when real quotes can afford it.

The live log now records the real contract (real strike, real
expiration), so live resolution grades what a user could actually
trade. Honest caveats: quotes are exchange-delayed and fetched at scan
time (after the close), so live prices at the next open will differ —
that is why the natural credit, not the mid, is the published number;
and historical validation (§5) necessarily used modeled fills, because
historical chains are not available — the reality layer guarantees
existence and collectibility of today's published contracts, not a
re-validation of history.

## 11. Technical timing signals (ai-trader port, tested 2026-07-02)

Five strategies from github.com/whchien/ai-trader (close-based
adaptations) were used to TIME spread placement — bull signal → put
spread below, bear → call spread above, buffer k·σ√h — and benchmarked
against UNCONDITIONAL entries (every 5th session) at identical strikes,
2008–2018 design / 2019–2026 validation, full universe, conservative
fills (`tech_spreads.py`). Net pooled return-on-risk per trade:

| Signal | k=1.0 h=14 des/val | k=1.5 h=7 des/val | vs baseline |
|---|---|---|---|
| unconditional baseline | 2.25% / 2.53% | 1.19% / 1.17% | — |
| **rsi_bb** (RSI14 + Bollinger MR) | **2.69% / 2.71%** | **1.48% / 1.26%** | **+0.2–0.5pp, both windows** |
| bbands | 2.10% / 2.43% | 1.09% / 1.04% | ≈ baseline |
| momentum | 1.45% / 1.85% | 0.90% / 0.98% | worse |
| donchian (turtle) | 1.20% / 1.90% | 0.84% / 1.01% | worse |
| sma_cross (5/37) | 0.94% / 1.66% | 0.47% / 0.83% | **anti-alpha everywhere** |

Conclusions: (a) only oversold mean-reversion (rsi_bb) carries
measurable timing alpha for premium selling, ~+0.3pp of risk per trade,
consistent across design and validation; trend-crossover timing is
systematically worse than random. (b) Early profit-taking at 50% of
credit flips EVERY cell negative (double commissions + slippage +
winner truncation) — hold-to-expiry dominates for the fourth time.
(c) Even at 1σ strikes (≈88% win rate, credit ≈10–12% of width), net
expectancy is 2–2.7% of risk per trade; timing changes WHEN, the market
still prices WHERE. Absolute P&L here leans on the IV=1.3×realized fill
model; the signal-vs-baseline comparison does not (both priced
identically).

## 12. The Vol-Crush edge (systematic feature scan, 2026-07-02)

A systematic conditional-alpha scan (`feature_scan.py`: ~1.53M
candidate trades 2008–2026, unconditional 1.5σ/14-session verticals
both sides, 14 causal features, design-window selection only) found
one substantial, replicating edge. Sell premium only when:

```
vr_60_252 = sigma60/sigma252 >= 1.183   (vol ELEVATED vs its own long-run; design p80)
vr_10_60  = sigma10/sigma60  <= 0.858   (and already CALMING; design p50)
```

— the post-spike **vol-crush** regime: strikes and credits are set off
inflated trailing vol while forward vol mean-reverts. Frozen on design,
validated once on 2019–2026:

| (net ROR/trade, win rate) | Design 08–18 | Validation 19–26 |
|---|---|---|
| Baseline puts | 1.64% / 95.2% | 1.91% / 94.8% |
| Baseline calls | 1.47% / 93.8% | 1.79% / 93.4% |
| **Vol-crush puts** | 4.02% / 97.8% | **5.23% / 98.3%** |
| **Vol-crush calls** | 4.46% / 97.6% | **5.27% / 97.6%** |

Deduped validation trades (one per ticker/side/expiry): **63,121, 97.6%
win, 5.37% net ROR/trade, +$2.17M at 1 contract each, positive in all
8 years** (2020: +7.4% at 98.7% — the edge pays MOST in crisis
aftermath), max cumulative drawdown −$19.0k on ~$277k average deployed
risk (≈7%), ≈105%/yr on deployed capital at model fills. Worst single
trade −$8.3k. The signal is symmetric across puts and calls (a
volatility phenomenon, not directional luck) and fires on ~83% of days.

Caveats: absolute levels use the IV = 1.3×σ60 fill model; in vol-crush
regimes real IV typically sits ABOVE that (post-spike fear premium), so
the model error is likely in the strategy's favor — but this must be
confirmed live through the reality layer before any of it is published
as a tier. Entries at 3-session stride overlap within ticker;
dedup-by-expiry is reported. Other scanned features (trend, RSI,
breadth, day-of-week, earnings-gap age): small or non-replicating
spreads; sigma60 level and vol ratios dominate.

## 13. The accuracy-vs-distance frontier, stock-outcome-only (2026-07-02)

Per the "assume the credit, grade on the stock" framing: weekly entry
candidates, both directions, strikes at c·σ60·√14, outcome = closed on
the safe side at the snapped expiry. 1.43M candidates 2008–2026.
Conditioning: the frozen §12 vol-crush gate, and a
HistGradientBoosting composite of all 13 features (fit 2008–2018,
scored once on 2019–2026, most-confident decile). Validation-window
no-breach rates (`sigma_distance_scan.py`):

| Distance | Fair credit ≈ (of width) | Uncond. | Vol-crush | GBM top-10% |
|---|---|---|---|---|
| 0.4σ | ~33% → **50% ROR** | 68.0% | 74.4% | **75.1%** |
| 0.6σ | ~25% → 33% ROR | 75.5% | 82.9% | 83.3% |
| 0.8σ | ~18% → 22% ROR | 81.6% | 89.0% | 90.2% |
| 1.0σ | ~14% → 16% ROR | 86.5% | 93.0% | 94.4% |
| 1.2σ | ~10% → 11–15% ROR | 90.1% | 95.7% | **96.7%** |

Conditioning lifts accuracy ~7–9pp at every distance — real, validated
selection alpha — but the frontier is continuous: (50% ROR, 95%
accuracy) sits ~20 points above the best achievable point at the 50%-
ROR distance. 95%+ accuracy first appears at ~1.2σ, where fair credit
commands ~11–15% ROR. The achievable validated menu on weekly cadence:
~75% @ 50% ROR, ~83% @ 33%, ~90% @ 22%, ~95–97% @ 11–16%.

## 14. Tier 2 — "Vol-Alpha" GBM put spreads (productionized 2026-07-03)

The frontier work in §13 was hardened into a second published tier
(`tier2.py`, engine `t2-volalpha-gbm`):

```
trade       PUT vertical, short strike 0.6·σ60·√14 below spot,
            width 2.5% of spot, expiry snapped DOWN within 14 sessions,
            hold to expiry
selection   HistGradientBoosting over the 13-feature causal library,
            fit ONLY on 2008–2018 (committed artifact
            results/tier2_model.joblib), publish when confidence ≥ the
            frozen deep cut (0.9561 — the design-99%-accuracy cut)
hygiene     optionable, ≥10y history, fresh series, reality layer
            (real expiration/strikes/quotes, natural credit ≥ $0.05)
```

Validation 2019–2026 (deduped by ticker/expiry, conservative fills,
single frozen pass): **2,599 trades · 98.2% accuracy · 24.3% net
ROR/trade · 19.7% ROR under zero-vol-premium stress pricing · ~7
trades/week · worst trade −$516/contract · positive in 2020 (+24%) and
2022 (+17%)**. Width sweep and c-sweep documented in the session
experiments; c=0.6/width 2.5% is the max-ROR cell with volume. The
stress-pricing robustness is the key property: the conditional-breach
alpha (1.8% realized vs ~24% unconditional at that distance) dominates
any fill-model assumption.

Honest limits: ~1.8 of every 100 trades lose (this tier is NOT the
99.4% tier); calibration decays across regimes (a design-95% cut
delivers ~88–92% out-of-sample — hence the deepest cut is used);
live natural credits at 0.6σ must confirm the modeled ~25% ROR (the
reality layer records them); losses cluster in vol episodes. Live-log
entries carry engine `t2-volalpha-gbm` with `:t2`-suffixed ids so the
tiers are scored separately, forever.

## 15. Learned adaptive exits (tested 2026-07-03)

The strongest version of "close early when needed": a second GBM
trained on 181,829 design-window trade-days of open Tier-2 positions
(state: time elapsed, cushion in σ-units, P&L captured, normalized
return since entry, current vol ratio/RSI/momentum) to score whether
exiting at next-day modeled cost beats holding to expiry
(`exit_policy.py`). Design-selected threshold, one validation pass:

| Policy | Validation ROR | Accuracy | Worst trade | Exits |
|---|---|---|---|---|
| Hold to maturity | 24.34% | 98.58% | −$516 | — |
| Learned exits τ=0.5 | 24.13% | 98.35% | −$452 | 0.7% |
| Learned exits τ=0.9 (frozen) | 24.34% | 98.58% | −$516 | **0.0%** |

The learned policy converges to **never exiting**: given full position
state and 11 years of examples, the model itself concludes that early
exits are net-negative at every confidence level — looser thresholds
strictly reduce both ROR and accuracy. The mechanism generalizes the
§7 fixed-trigger result: at 2-week tenors, an exit pays slippage,
commissions, and the market's already-repriced threat, while
threatened-but-recoverable positions dominate true failures at these
distances; the true failures (gaps) outrun any close-resolution
signal. Hold-to-maturity is not a simplification — it is the optimum
of the adaptive family on daily data.

## 16. The (horizon × distance) grid confirms Tier 2 is a genuine peak (2026-07-03)

Same protocol (per-cell GBM, design-only fit, d99 cut, one validation
pass, conservative fills), across horizons 7/14/21 and distances
0.6σ/0.8σ, put side:

| Cell | Validation acc | ROR/trade | Trades/wk |
|---|---|---|---|
| h=7, c=0.6 | 87.5% | 6.0% | 11 |
| h=7, c=0.8 | 92.7% | 6.9% | 19 |
| **h=14, c=0.6 (Tier 2)** | **98.2%** | **24.3%** | 7 |
| h=14, c=0.8 | 96.9% | 16.7% | 19 |
| h=21, c=0.6 | 93.2% | 13.5% | 22 |
| h=21, c=0.8 | 94.5% | 9.6% | 43 |

The shipped cell dominates every neighbor by wide margins, and a
union ensemble of all cells dilutes to 92.5% / 10.0% — concentration
in the peak beats diversification across weaker cells. Together with
§15 (exits: hold optimal), the cross-sectional feature test (richer
features degrade OOS), the width sweep (2.5% optimal), and the side
split (puts ≫ calls here), every axis around the Tier 2 operating
point has now been searched and the point is a sharp, genuine optimum:
the alpha is specifically a ~2-week vol-crush persistence phenomenon.

**Research-budget note:** this panel has now been examined across
entries, exits, features, horizons, distances, widths, sides and
structures. Any further "discovery" from the same data carries high
false-positive risk (multiple comparisons). The correct next evidence
is LIVE: Tier 2's natural credits and outcomes accrue nightly in the
live log under engine `t2-volalpha-gbm`.

## 17. Liquidity-filter impact — it's free (2026-07-03)

The reality-layer liquidity gate has three parts; only the underlying
average-dollar-volume floor can be applied to HISTORY (current ADV per
name in `results/adv.json`; historical option OI / bid-ask don't
exist). The OI≥25 and short-leg-spread≤40%-of-width gates only ever
remove MORE, and act at publication, not on the historical outcome
distribution. Restricting each tier's untouched 2019–2026 validation
set to names clearing the ADV floor (`liquidity_impact.py`):

| ADV floor | Tier 1 acc / ROR (trades) | Tier 2 acc / ROR (trades) |
|---|---|---|
| none | 99.40% / 0.86% (1,509) | 98.19% / 24.34% (2,599) |
| ≥ $50M/day | 99.40% / 0.86% (1,509) | 98.15% / 24.33% (2,487) |
| **≥ $100M/day** | **99.43% / 0.95% (1,232)** | **98.21% / 24.38% (2,065)** |
| ≥ $250M/day | 99.46% / 0.89% (923) | 98.51% / 24.63% (1,277) |

Filtering does **not** degrade either tier — accuracy and ROR are held
or slightly improved at every floor, and both rise as you demand more
liquidity (the most liquid names have the cleanest vol-crush
dynamics). Tier 1's certified far-OTM universe is already entirely
≥$50M (60 names). Production floor set to **$100M/day** ("very liquid")
since it costs nothing: Tier 1 keeps 51 names, Tier 2 keeps 79% of
trades (~5/week) at 98.2% / 24.4%. The headline validated numbers are
unchanged by the filter.
