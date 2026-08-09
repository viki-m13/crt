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

### 04:30-05:00 UTC — infrastructure + data quality

- DoltHub `option_chain` verified: real EOD bid/ask/IV/greeks, 2019-02→2026-08,
  ~weekly (2019) then ~M/W/F cadence; ~3 monthly expirations per chain.
- Parity spots validate against known closes (AAPL 186 Mar-2019, SPY 240.3 on
  2020-03-18 ✓). All 61 universe names remained quoted through the COVID crash
  with sane spreads (SPY ATM 2.8% spread/mid at the worst).
- **Spread-cost map (2024-06-03, ATM/5%OTM spread ÷ mid):** SPY 0.7/1.9%,
  NVDA 1.2/1.8%, MU 2.4/2.9%, AAPL 2.8/4.8%, KO 6/26%, GE 13/10%, JNJ 16/33%,
  XLF >100% (quote junk). ⇒ worst-side round trips are only viable on a
  liquid tier; wide names only via hold-to-expiry (pay half-spread once).
- Engine smoke test (toy SPY strangle, 2019): mechanics verified; loses money
  at worst-side fills, as a no-edge strategy should.

### Trials 1-17 (partial data 2019→mid-2021 only, dev): baseline screens

All unconditional premium-selling sleeves NEGATIVE after worst-side fills in
this crash-heavy partial window (scrSharpe: short_straddle all −0.84, SPY
iron_condor −0.41, SPY short_straddle_dh +0.35 best). Signal ICs vs
short_straddle_dh returns (Spearman, t over dates): **credit_yield +5.6**,
**mom8 +5.1**, **term(iv_back−iv_front) −4.0** (inversion → richer short
returns; consistent with earnings-crush harvesting), ivrv +1.75 (weak).
→ Sleeve hypotheses: (A) index VRP regime-gated; (B) single-name earnings
IV-crush via term-inversion detector, short-hold; (C) cross-sectional
credit_yield/mom rank long-short; (D) skew-conditioned credit spreads.
NOTE: partial-window numbers; will re-run on full data. Trial count so far
N≈17 (6 signals × ~1 + 17 sleeve/universe combos).

### Trials 18-71 (partial data 2019→mid-2022): study2 sleeve screens

45 configs + 9 corrected-C configs. Window = worst-case for short vol.
- **C long-short vol EXPOSED AS FAKE**: with the long leg priced at real ask
  (not the negated-short shortcut), C_ls_cy+mom_liq falls +2.32 → **−1.64**.
  Long ATM straddle legs at worst-side fills + hedge costs bleed ~5%/mo.
  Lesson: long options only as far-OTM wings, never as symmetric alpha legs.
- Survivors (scrSharpe, worst window): **C_shortonly_cy+mom_liq +1.23**
  (hit 70%), **A_credit_putspread SPY uncond +0.92** (hit 92% incl COVID),
  **A_short_straddle_dh_contango +0.91**, **B_str25_idioinv>0.06_liq +0.90**.
- Dead: calendars (double-spread entry + back-leg exit spread), weekend theta
  (−2.2, hit 2.5% — EOD spread ≫ 2-day theta), call spreads on low skew
  (−2.0), B with 2-obs exits (round-trip spread ≫ 2-day crush capture).
  → All premium harvesting must be HOLD-TO-EXPIRY; exits only for risk.
### Trial 72 — Study 3: intra-chain smile RV → DEAD (efficient smile)

Fitted vega-weighted quadratic smiles per (sym,exp) on the liquid tier;
residual σ ≈ 2.2 vol pts, 95th pct ≈ 1 vol pt. The only ≥2.5-pt-rich quotes
are far-OTM 3-40¢ options with vega ≤0.035 → est. edge ~$0.08/spread vs $8
minimum after worst-side entry. **Zero tradable candidates over 580 dates.**
Conclusion: EOD quotes on liquid names are smile-efficient within costs;
no market-making-adjacent family exists in this data. (Event-loop sleeve
A_ps calibration: engine Sharpe 0.70 vs screen 0.92 → marks deflate ~25%.)

### 06:15 UTC — FULL DATASET COMPLETE

**1254 observation dates, 2019-02-09 → 2026-08-06, 128 symbols, 0 fetch
failures.** Per-year obs: 2019:47, 2020:155, 2021:154, 2022:151, 2023:153,
2024:182, 2025:259, 2026:153. Everything below this line uses the full panel;
everything above used a partial (2019→2022) cache and is superseded except
where noted as a structural lesson (C long-leg ask fills, smile efficiency,
exit-spread dominance) — those hold regardless of window.

Slate for full data: A (SPY putspread, straddle_dh w/ contango gate),
B (strangle25 idio_inv .06/.09 liq), C (short-only quintile cy+mom liq),
D (put_skewrich_liq). Running trial count N≈71.

### Trials 73-118 — FULL PANEL (1254 dates), uncorrected spot

**Every sleeve family collapses on the full panel.** Best is
A_credit_putspread_uncond at **+0.57** screening (was +0.92 on the partial
window); the previously-best C_shortonly_cy+mom_liq goes **+1.23 → −0.42**.
Baselines: short_straddle_dh SPY −0.15, all-names −1.68; iron_condor
all-names −3.55 (wide-spread names destroy the credit). Signal ICs *hold* and
even strengthen with more data (credit_yield t=+8.4, term t=−7.9, mom8 t=+5.6,
ivrv t=+4.0) — the signals rank cross-sectionally, but ranking does not
overcome the half-spread paid on entry.

**Study 4 dispersion (Trial ~119): DEAD both directions.** Vega-matched short
SPY straddle vs long 36-name component basket: **−0.44**. Reverse: **−0.60**.
Index short alone: −0.11. Implied correlation (computed point-in-time from the
chains) median 0.25. Conditioning on rho does not rescue it: rho∈[0.4,0.6)
gives disp −1.74 / rev +0.52 on only 124 obs. The correlation premium is real
in the IV data but is smaller than the two-sided spread cost of trading 37
straddles.

### Methodology correction (applies to everything above)

Audit of my own spot extraction found a genuine bias: `C−P+K` recovers the
**forward** F = S·e^{(r−q)T}, not the spot. Whichever expiry is nearest sets
the tenor, so "spot" drifts with it — measured **+15 bps at 2024 rates**,
+1 bp under 2021 ZIRP, −4 bps in the COVID crash. That drift sits between a
trade's entry and its settlement and biases settlement payouts (it flatters
put spreads, roughly cancels for straddles). Fixed by fitting ln F = ln S + cT
across the expiration curve and taking the intercept, recovering spot and
carry from the quotes alone. Corrected SPY on 2024-06-03: 528.33 (true close
~527.8) vs naive forward 528.98. **All full-panel numbers were re-run on
corrected spots; the uncorrected log is kept at
cache/rebuild_uncorrected_spot.log for comparison.**

### Trials 119-164 — FULL PANEL, CORRECTED SPOT (definitive)

The correction behaved exactly as predicted: it deflated put spreads (which it
had been flattering) and left straddles ~unchanged. Conclusions are unchanged.

| sleeve | corrected | uncorrected |
|---|---:|---:|
| A_credit_putspread_uncond (best of all) | **+0.53** | +0.57 |
| A_credit_putspread_cont+mom | +0.41 | +0.45 |
| B_str25_idioinv>0.09_liq | +0.30 | +0.24 |
| A_short_strangle25_uncond | +0.03 | +0.01 |
| A_short_straddle_dh_uncond | −0.16 | −0.15 |
| A_iron_condor_uncond | −0.15 | −0.11 |
| dispersion (short idx / long comps) | **−0.43** | −0.44 |
| reverse dispersion | −0.61 | −0.60 |
| index short only | −0.10 | −0.11 |

Implied correlation median 0.25 (mean 0.30). The rho∈[0.4,0.6) reverse-
dispersion cell at +0.55 is on 124 observations and is not treated as a
finding — it is one cell of a conditioning grid and would need its own
out-of-sample test to mean anything.

### Trials 165-170 — Study 6: short-dated (8-14 DTE), held to expiry

No sub-week expirations exist in this DB; 8-14 DTE is the shortest tenor and
is present on ~94% of dates. SPY: putspread **+0.57**, strangle25 +0.35,
straddle +0.17. Liquid tier: all negative (−0.11 to −0.27). Denser theta does
not outrun the spread except on the single tightest-quoted instrument, and
even there it only matches the monthly version.

### Study 5 — the ceiling (the central result)

IC=+0.0371 (t=+5.54, 1237 dates) on the liquid tier. 38 names, mean pairwise
return correlation +0.185, **effective independent names 13.7** (eigenvalue
participation ratio, 36% of nominal) → 164 independent bets/yr →
**IR ceiling 0.48 gross of costs**. Sharpe 5 needs BR=18,130/yr (**111x**
available) or IC=0.391 (**11x** measured). The breadth deficit, not signal
quality, is what makes 5 unreachable.

### Event-loop finals (dev) — the only numbers reported as results

| sleeve | Sharpe | CI95 | maxDD | DSR |
|---|---:|---|---:|---:|
| A_ps_6_12 | 0.63 | (−0.17, 3.14) | −36% | 0.0002 |
| A_ps | 0.57 | (−0.18, 1.99) | −38% | 0.00 |
| A_ps_mom | 0.54 | (−0.14, 1.81) | −19% | 0.0001 |
| **ensemble (inv-vol, frozen dev weights)** | **0.74** | **(−0.11, 1.96)** | **−15%** | 0.0001 |

Per-year for A_ps_6_12: 2019 **12.18**, 2020 0.36, 2021 **15.90**, 2022 0.00,
2023 7.74, 2024 8.87 — full period **0.63**. Any single calm year clears
Sharpe 5 comfortably; the full window does not. This is the manufacturing
mechanism, demonstrated on real fills.

**Sleeve C_short_wings excluded from the ensemble: equity went non-positive
(−500% DD).** Sizing capital/f_pos by a narrow defined-risk width levered the
position through zero. Its inverse-vol weight was only 0.005, but a busted
curve has no meaningful return series and must not be blended in at any
weight.

### FINAL — holdout touched once (2025-01 → 2026-08)

Exclusion rule applied on DEV ONLY (C_short bust 2022-03-04, C_short_wings
bust 2020-07-31 — both in-sample decisions). B_str25 KEPT despite going
insolvent 2025-05-12, because dropping it would hide the holdout's finding.

| | Sharpe | CAGR | maxDD | CI95 |
|---|---:|---:|---:|---|
| DEV 2019-2024 | +0.57 | +6.6% | −15% | (−0.28, 1.80) |
| **HOLDOUT 2025-2026** | **−1.12** | **−28.1%** | **−43%** | (−2.05, 0.25) |
| FULL | −0.24 | −1.9% | −43% | (−0.80, 0.68) |

**The candidate failed out of sample.** Sleeve correlations confirm the
breadth diagnosis: the five put-spread variants correlate 0.71-0.94 with each
other — five configurations of one bet, not five bets.

### Illusion audit — identical trades, different conventions

| convention | Sharpe |
|---|---:|
| honest (worst-side, weekly marks, margin, full period) | +0.79 |
| 2021 only | **+15.94** |
| 2019 only | **+17.02** |
| 2021 + per-trade ×√149 | +6.51 |
| 2023 + per-trade ×√151 | +7.48 |
| 2021 + per-trade + drop worst 2% | +15.86 |
| per-trade ×√164 (full period) | +1.76 (3.3x) |
| drop worst 1% of weeks (4 weeks) | +1.35 (2.5x) |

Final trial count N≈170. Deflated Sharpe of every candidate ≈ 0.

---

## SECOND PASS — attacking the ceiling's own terms (trials 171-210)

IR = IC x sqrt(BR) names its levers. Pass 1 used one signal, one tenor, one
direction. Pass 2 attacked each, plus the one family with unbounded Sharpe.

### Trial 171 — Study 7: BOX SPREADS (defined payoff = true arbitrage) — DEAD

114,649 boxes at worst-side fills. Implied lending rate: **median −99.9%**,
99th pct −4.8%, vs T-bill +3.2%. Boxes lending above T-bill+2%: **0.023%**.
Hard arbitrage (cost<=0 for guaranteed positive payoff): **10 of 114,649
(0.0087%)** — and all 10 are in a **single name (F)** with median width $2.00,
i.e. stale quotes on a low-priced stock, not opportunity. Crossing four spreads
swamps any financing dislocation. **The only family with no Sharpe ceiling is
decisively closed.** Doubles as a dataset warning: apparent edge in sub-$5
structures on cheap names is quote noise.

### Trial 172 — THE ACCOUNTING IDENTITY (invalidates Study 5's headline IC)

For premium structures held to expiry: **ret = credit_yield − loss/margin**,
and credit is KNOWN AT ENTRY. 76.3% of credit spreads expire worthless, so
ret *literally equals* credit_yield most of the time.

| measurement | value |
|---|---:|
| corr(credit_yield, ret) | **+0.53** ← what Study 5 reported as signal |
| corr(credit_yield, −loss) | **−0.13** ← real content, and NEGATIVE |
| among losing trades only | +0.05 |
| mean ret \| no loss | +0.1228 (= mean credit_yield exactly) |
| mean ret \| loss | −0.5099 |

Richer premium predicts BIGGER losses. This resolves pass 1's paradox: a
t=+8.4 signal that never produced a profitable sleeve was measuring an
identity. **Generalizes: any "signal" that is a component of the payoff
(premium, credit, IV level) shows large spurious IC in options backtests.**
Not caught by pre-registration, walk-forward, or DSR — caught by asking why a
strong signal made no money.

### Trials 173-176 — Study 9: TRUE ceiling on clean signals

IC vs the only uncertain term (−loss), signals not part of the payoff:

| signal | IC | t |
|---|---:|---:|
| **vol-of-vol** | **−0.0330** | **−5.30** |
| **skew25** | **+0.0231** | **+3.81** |
| ivrv | −0.0078 | −1.32 |
| mom8 | −0.0084 | −1.34 |
| iv_rank | −0.0006 | −0.09 |
| idio_inv | +0.0029 | +0.34 |

11.7 effective names x 12 rounds = 141 bets/yr → **IR ceiling 0.39** (vs 0.48
contaminated). IR=5 needs IC=0.421 = **13x** best clean signal.

### Trials 177-200 — Study 8: composite x tenor x overlay — NO IMPROVEMENT

- Composite (7 rank-signals, walk-forward OLS weights, fit on train only):
  pooled OOS IC **+0.40** — but sleeve Sharpe ≈ 0. Same tautology: weights load
  on z_cy. High IC, no money, again.
- Signal cross-correlation mean |off-diag| 0.157 (genuinely diverse) — did not
  help.
- Shortest tenor (8-14 DTE) to double rounds/yr: OOS scrSharpe +0.05 to +0.10.
- Stress overlays (SPY IV z-score, drawdown gates): −0.54 to +0.02, no gate
  improved the base.

### Trials 201-210 — risk-avoidance using the CLEAN signals (best honest idea)

Since vov/skew genuinely predict losses, use them to avoid blowups rather than
pick winners. Credit put spreads, full panel:

| filter | mean ret | loss rate |
|---|---:|---:|
| unfiltered | −0.0145 | 20.2% |
| low vov | −0.0094 | 18.4% |
| high skew | −0.0060 | 20.2% |
| **low vov + high skew** | **−0.0024** | **17.0%** |

Removes 83% of the deficit and cuts loss rate by a fifth — **still negative.**
Credit-yield quintiles (cheap vs rich premium) all negative for every
structure, no monotonic exploitable pattern in either direction.

**Pass 2 conclusion: the ceiling is lower than pass 1 said (0.39), the one
unbounded-Sharpe family is closed, and the best genuinely-clean signals
improve but do not rescue the economics.**

---

## THIRD PASS — bringing the STOCK in (trials 211-240)

Premise: every failure so far paid the OPTION spread (100-500 bps). Nothing
forces the trade to be expressed in options. Three ways to involve stock:
(11) option signals timing an index position, (12) stock and options held
jointly, (10) option signals expressed purely in stock.

### Trials 211-222 — Study 11: option-implied MARKET TIMING of SPY — DEAD

| strategy | Sharpe | CAGR | maxDD |
|---|---:|---:|---:|
| **buy & hold SPY (benchmark)** | **+0.85** | +14.6% | −34.2% |
| vol-target on option-implied IV | +0.84 | +12.1% | **−24.5%** |
| vol-target on trailing realized | +0.79 | +11.8% | −21.2% |
| long only when contango | +0.67 | +6.4% | −15.2% |
| long only when IV < 80th pct | +0.64 | +8.2% | −28.5% |
| long only when VRP > 0 | +0.40 | +4.7% | −31.2% |
| vol-target x VRP>0 | +0.38 | +3.6% | −19.0% |

**No option-implied timing rule beats buy-and-hold.** Every gate that sits out
of the market gives up more return than risk. Vol-targeting on IMPLIED vol
matches buy-and-hold's Sharpe while cutting drawdown ~10pp, and is the single
most STABLE result in the project (dev 0.93 / holdout 0.85, essentially no
decay) — but it is risk scaling, not alpha.

### Trials 223-229 — Study 12: STOCK + OPTION combinations — ALL WORSE

Held to expiry, benchmarked against buy-and-hold of the same stock over
identical dates. Liquid tier, 46,037 observations each:

| structure | Sharpe | vs stock | hit rate |
|---|---:|---:|---:|
| **stock only** | **+0.77** | — | 55.2% |
| prot_put 5% | +0.65 | −0.11 | 48.1% |
| buy_write 5% | +0.56 | −0.21 | 61.0% |
| buy_write 2% | +0.48 | −0.29 | 65.7% |
| **put_write 5%** | +0.46 | −0.30 | **83.2%** |
| buy_write ATM | +0.44 | −0.33 | 70.0% |
| collar 5% | +0.25 | −0.51 | 53.7% |

**Every overlay reduces risk-adjusted return.** SPY-only: stock 0.74 vs
buy_write_5pct 0.73 — the closest anything came, still not an improvement.

**The hit-rate column runs OPPOSITE to the Sharpe column.** Put writes win
83% of the time (92.6% on SPY) with the second-worst Sharpe in the table.
Options reshape the payoff into many small wins and rare large losses without
creating edge. A "wins 9 months out of 10" claim describes payoff shape, not
profitability — and this table is the cleanest demonstration of it in the
whole project.

### Trials 230-260 — Study 10: option signals expressed in STOCK

Premise: stock spreads are 1-5bp vs 100-500bp in options. IC vs FORWARD STOCK
returns (signal at t, entry t+1 so the same quote never both signals and
prices the entry):

| signal | h=1 | h=3 | h=5 | h=8 |
|---|---:|---:|---:|---:|
| **skew25** | +0.005 | +0.019(t3.5) | +0.025(t4.7) | **+0.028(t+5.6)** |
| d_iv | +0.015(t2.9) | +0.017(t3.4) | +0.014 | +0.011 |
| ivrv | +0.017(t2.9) | +0.015 | +0.009 | +0.005 |
| iv_spread (parity dev) | −0.001 | +0.002 | +0.006 | +0.003 |
| mom / rev | ~0 | ~0 | ~0 | ~0 |

**Skew is a REAL signal** — monotone in horizon, t=+5.6. **Parity deviation
(Cremers-Weinbaum) is DEAD** (t 0.3-1.5), arbitraged away post-publication.

Long-short stock book, skew25 h=8, cost sensitivity:

| cost | gross | net | dev | holdout |
|---|---:|---:|---:|---:|
| 0bp | +0.25 | +0.25 | +0.18 | +0.47 |
| **2bp (realistic)** | +0.25 | **+0.22** | +0.15 | **+0.43** |
| 10bp (punitive) | +0.25 | +0.07 | +0.02 | +0.24 |

**This is the ONLY strategy in the project that is POSITIVE out of sample**
(+0.43 holdout vs options ensemble −1.12). Robust but small.

**Breadth, the punchline:** 70 names → **8.7 effective independent (12%)**.
Even a market-neutral stock book hits the same wall. IR = 0.0283 x sqrt(182)
= 0.38 — the same ~0.4 ceiling found in options, reached by a completely
different route.

### Trials 261-290 — Study 13: stock effects x option state — NOISE

No monotonic gradient across conditioning buckets (the pre-stated bar). Best
cell momentum x low-IV h=5 = +0.32, but that is 1 of 30 cells and the rest
straddle zero. Reported as noise, not a finding.

### Study 14 — the leverage question: why CAGR is capped by Sharpe^2

g(L) = L*mu − (L*sigma)^2/2, maximised at Kelly L* = mu/sigma^2, giving
**g_max = S^2/2 regardless of leverage, vol, or notional.**

| strategy | Sharpe | max CAGR at optimal leverage |
|---|---:|---:|
| skew→stock (best OOS) | 0.22 | **2.4%** |
| measured market ceiling | 0.39 | 7.9% |
| best options sleeve (dev only) | 0.63 | 22.0% |
| SPY buy & hold | 0.85 | 43.5% |
| options ensemble (full period) | −0.24 | no growth at any leverage |

1000% CAGR requires **S=2.19 at full Kelly**, 2.53 at half Kelly, 3.31 at
quarter Kelly. Against a measured ceiling of 0.39 for this market.

### Study 15 — IS THE MARKET'S BREACH FORECAST BEATABLE? (the core question)

For a vertical held to expiry, return = credit/width − loss/width, and
credit/width IS the market's risk-neutral expected loss ratio. So the market
publishes its own forecast and we can grade ours against it.

**TEST 1 — realized vs breakeven win rate (put credit spreads):**

| credit/width | breakeven win% | realized win% | edge |
|---|---:|---:|---:|
| Q0 ~0.06 | 94.0% | 84.6% | −9.4% |
| Q2 ~0.15 | 85.3% | 74.6% | −10.7% |
| Q4 ~0.27 | 73.4% | 66.3% | −7.2% |

Every bucket falls short of the accuracy its own price demands. Call spreads
miss by 18-20% (sample artifact: SPY ~tripled over the window).

**TEST 2 — VRP exists; costs are larger (n=5,974):**

| | loss ratio |
|---|---:|
| risk-neutral at MID | 0.1853 |
| REALIZED | 0.1599 |
| **gross premium (VRP)** | **+0.0254 EXISTS** |
| half-spread cost | 0.0346 |
| **net at worst-side** | **−0.0092** |
| **cost / gross premium** | **1.36x** |

Every earlier study in this log conflated "no edge" with "costs eat the edge."
The edge is real. The spread is 136% of it.

**TEST 3 — the market out-predicts any model I can build:**
market baseline (credit/width) AUC **0.599**; best single feature 0.586;
walk-forward logistic model mean AUC **0.028 BELOW market**, losing 5 of 6
years. You cannot out-forecast the quoted odds on breach probability.

**Fill-quality sensitivity — the actionable result:**

| f (half-spread captured) | win rate | ann. return |
|---|---:|---:|
| 0.00 worst-side | 75.8% | −19.0% |
| 0.50 | 77.2% | −0.2% |
| 1.00 mid | 78.1% | **+22.5%** |

**BREAKEVEN f* = 0.504.** Win rate moves 2.3pp across the whole range while
annual return moves 41pp. **Accuracy is not the lever; execution is.**
Caveats: mid fills carry adverse selection (filled preferentially when the
market moves against you), and returns are on width-as-margin (levered).

### Study 16 — WHERE is the edge largest relative to the toll? (best result)

The toll is roughly FIXED per leg in dollars; premium scales ~sqrt(tenor) and
risk scales with width. Both are levers. f* = breakeven fraction of the
half-spread that must be captured.

| tenor x width | f* | gross | net (worst-side) | toll | n |
|---|---:|---:|---:|---:|---:|
| **75d / 20%** | **−0.585** | +0.0266 | **+0.0098** | 0.0168 | 52,233 |
| **75d / 10%** | **−0.428** | +0.0341 | **+0.0102** | 0.0239 | 52,293 |
| 75d / 5% | +0.147 | +0.0380 | −0.0065 | 0.0446 | 51,352 |
| 30d / 5% (what pass 1-2 tested) | +0.740 | +0.0119 | −0.0339 | 0.0458 | 92,122 |

Both structural predictions confirmed: gross premium ~TRIPLES from 30d to 75d
for the same toll, and the toll falls 4.6x from 3%-wide to 20%-wide. **The
entire project had been sitting in the 30d/5% corner — structurally the worst
cell in the space, and the one retail defaults to because it is most liquid.**

**BUG FOUND on inspection:** worst trade showed −4.66 of width, impossible for
a defined-risk spread (max loss = width). Cause: filtered on mid-credit>0 but
not worst-side-credit>0, keeping 0.21% of rows where the requested strike
spacing did not exist and the long leg cost more than the short collected —
not credit spreads at all. Fixed by requiring cw>0.

**CLEAN 75d/wide (n=101,343):**

| | |
|---|---:|
| net per trade, worst-side | **+0.0170 of width** |
| dev 2019-2024 | **+0.0215** |
| holdout 2025-2026 | **+0.0087** |
| worst trade | −0.998 (properly capped) |
| loss rate | 24.4% |
| annualized on width | +8.6% |
| **t-stat on independent rounds** | **+0.51** |

Positive in dev AND holdout, positive in 7 of 8 years (only 2022 negative at
−0.053). **First result in the project positive at worst-side fills out of
sample.** BUT a 75-day hold gives ~5 independent rounds/yr, so 7 years is
~35 independent observations: the 101k "trades" are overwhelmingly overlapping
and correlated. **t=+0.51 is not significant.** Suggestive, not established.

### FINAL — 75d/20% put spreads through the event-loop engine

| | Sharpe | CAGR | maxDD | CI95 | DSR |
|---|---:|---:|---:|---|---:|
| dev 2019-2024 | +0.55 | +16.6% | −50.4% | (−0.04, 1.29) | 0.00 |
| **full incl. holdout** | **+0.51** | **+15.3%** | **−50.4%** | (−0.04, 1.11) | 0.00 |
| LD_wide10 full | +0.51 | +16.4% | −52.3% | (−0.03, 1.07) | 0.00 |

Yearly: 2019 0.98, 2020 0.75, 2021 1.61, 2022 −0.94, 2023 1.97, 2024 0.20,
**2025 0.24, 2026 0.85** — positive in 7 of 8 including both holdout years
and COVID. Best result in the project by a wide margin (previous best
full-period was −0.24).

**BUT vs the benchmark: SPY buy & hold = Sharpe 0.85, CAGR 14.6%, maxDD
−34.2%.** The best options strategy delivers the same return with a worse
Sharpe and a 16pp deeper drawdown. CI touches zero; DSR ~0 at ~300 trials.

**PROJECT CONCLUSION: no honest options strategy here beats holding the index,
let alone reaches Sharpe 5.** The one genuine structural discovery is that
tenor and width dominate signal selection — the toll is fixed per leg while
premium grows with sqrt(T) and risk grows with width — and that the standard
30d/5% retail structure is the worst cell in the design space.

### Study 17 — ACTIVE management (revisiting a premise held for ~250 trials)

"Hold to expiry" came from trials 18-71 on 30d ATM straddles held 2-3 days —
the worst possible case — then went unexamined through ~250 configurations.
Re-tested on the 75d/20% structure. Sharpe* annualizes on REALIZED holding
period so faster rules get credit for the breadth they create.

| rule | ret/trade | hold | rounds/yr | Sharpe* | ann | lossRate | worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| hold to expiry | +0.0150 | 57.5d | 6.4 | +0.21 | +9.9% | 22.6% | −0.99 |
| profit 25% | +0.0052 | 34.2d | 10.7 | +0.18 | +5.7% | 15.0% | −0.99 |
| **profit 50%** | +0.0115 | 43.5d | 8.4 | **+0.26** | +10.0% | 18.8% | −0.99 |
| **profit 75%** | +0.0138 | 51.3d | 7.1 | **+0.26** | +10.2% | 21.5% | −0.99 |
| stop 2x credit | −0.0039 | 45.7d | 8.0 | **−0.17** | −3.1% | **37.8%** | **−2.02** |
| stop 3x credit | +0.0021 | 52.4d | 7.0 | −0.12 | +1.5% | 28.3% | −2.02 |

**Profit-taking helps modestly** (Sharpe 0.21 → 0.26, loss rate −4pp). Taking
at 25% over-trades: pays the spread too often for too little, the same failure
mode as the original exit tests.

**STOPS ARE ACTIVELY HARMFUL, and the worst-trade column says why.** Hold-to-
expiry is bounded at −0.99 of width; stop rules reach **−2.02**. A defined-risk
spread cannot lose more than its width AT EXPIRY, but closing early at
worst-side prices can cost MORE than that: buy back the short leg at a wide
ask, sell the long leg at a beaten-down bid, and the round trip on a deep-ITM
spread exceeds the max loss you would have taken by doing nothing. Stops also
realize temporary losses that recover (loss rate 22.6% → 37.8%).
**On defined-risk structures the stop-loss is itself the risk** — the opposite
of standard retail practice.
