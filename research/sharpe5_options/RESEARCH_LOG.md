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
