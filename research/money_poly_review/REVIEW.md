# zDTE and Ratchet: final review and options-data decision (2026-09-25)

## 1. Bottom line

**zDTE** sells pivot-confirmed 0DTE SPX credit spreads. Credits come from Black-Scholes at 1.2× trailing 1-min volatility, times a haircut of 0.70 for puts and 0.45 for calls (`v19-btc-predictor/lib/zdteLiveCore.ts:23-24`).
- Its own research has already killed it. At honest prices the Sharpe falls from 5.66 to 0.87 (`hl-quant/ZDTE_FINDINGS.md:784-797`), 0 of 892 credit structures survive (`:811-813`), and the final line is "nothing should be traded until κ_ATM is measured on 30+ real sessions" (`:1009-1012`).
- A second independent real-quote check today confirms it. The model priced the signalled spreads at $115/$196/$221 against CBOE mids of $35/$82/$105, which implies κ of about 0.55–0.66.
- The live site does not match the research. /zdte and /api/zdte still issue SELL cards and show 98.4% win rate, Sharpe 6.19 and "+89%/yr at 5% (suggested)".
- The proposed replacement, a long debit vertical, rests on one κ snapshot. Its VIX1D corroboration disappears once a day-count error is fixed.

**Ratchet** sells 3%-OTM, 5%-wide, ~21-session SPY/SPX put spreads in uptrends, gated by VRP, and harvests at +1%.
- Its headline (13.1% CAGR / −23.6% max DD / Sharpe 1.04) reproduces exactly from free data, but it is a calibrated model, not a replay of real quotes.
- On real quotes the model over-credits by about 7–10% since 2023, and the forward CBOE log is at 0.83–0.87. A realistic figure is about 8–11% CAGR with excess Sharpe around 0.45–0.7.
- The /ratchet page shows the right rule with stale evidence. The home page tells users to trade a retired structure using superseded numbers.

**Data:** don't buy anything until the free fixes are done. Then buy one month of ThetaData Options Standard ($80), or a $276 Cboe DataShop pilot, to settle zDTE's κ question. Buy ORATS near-EOD 2007+ ($599) for Ratchet only if a free 2010–2026 EOD replay leaves it viable.

## 2. zDTE

### Mechanics
The strategy is what the live engine runs today.
- **Signal.** SPY 1-min bars, scanned every 5 minutes from 10:00 to 15:30 ET after 40 bars. A put side arms when the day low is at least 15 bars old, price has reclaimed at least 35% of the range, and price is above VWAP; calls mirror this (`zdteLiveCore.ts:107,167,178-181`).
- **Strike.** Short strike = the defended extreme ± z·σ1m·√minsLeft, clamped to 0.15%–20%. There are 3 tiers, with z multiplied by 1.00, 0.70 or 0.55. Wings are 20 points wide on SPX via the SPX/SPY ratio.
- **Credit and exits.** Credit = BS(IV = σ1m·√(390·252)·1.20) × REAL (0.70 put, 0.45 call). A trade is skipped below 1.2bp. There is no stop. Take-profit fires after 12:30 at ≤10% of credit; otherwise the spread cash-settles.
- **Pricing basis.** Every price is modelled. The REAL multipliers came from a single weekend snapshot of 2-DTE SPY options, never from 0DTE (`ZDTE_FINDINGS.md:751-754`).

### Research arc (30 iterations)
| Stage | Claim | Basis |
|---|---|---|
| Iter 1–10 | QQQ z=1.25: 99.5% WR, Sharpe 7.45; "survives half the credit" (Sharpe 3.5); pivot beats random 2–3×; 8-name book Sharpe 8.8 | Model × weekend multipliers |
| Iter 14–22 | Take-profit, tiers, 40-bar warm-up; site book n=1,339, 98.4% WR, Sharpe 6.19 | Model; take-profit is EV-neutral (CI [−0.12, +0.30], `:455-467`) |
| Iter 27 (09-01) | Model credit 1.59 pts vs broker ~0.30 (0.19×) | **Real** (one screenshot, 3 prints) |
| Iter 27b | At κ=0.66: Sharpe 0.87, "not tradeable"; break-even κ≈0.6 | Calibrated model |
| Iter 28 | 0/892 credit structures positive in all 3 windows | Diurnal+HAR+smile model fit on one day |
| Iter 29 | Resting limit orders lose to adverse selection | Model |
| Iter 30 (09-02) | Pivots, pinning and direction are all dead (AUC 0.46–0.51); flip to a **long** call debit vertical (+0.10σ/+1.00σ); κ_ATM≈0.79; break-even κ 1.34/1.41/1.25; trade nothing until 30+ κ sessions | One live SPXW chain + VIX1D test + model |

### Live state vs research
Total mismatch. The latest commits to the page and engine are dated 2026-08-31 and 09-01; iteration 30 is aa8f211 (2026-09-02). Checks on the live site:
- `/api/zdte` on 2026-09-25 returned a SELL put at 10:55: SAFE 7635/7615 at $114, BAL at $194, AGG at $218. It later showed TAKE_PROFIT and "SETTLE win".
- The page HTML contains 0 mentions of κ, "debit" or "not tradeable".
- Three page claims are false or contradicted:
  - "Credits … measured from live CBOE SPX chains … re-measured every session" (`page.tsx:442`). Only 2 sessions were ever measured, both multi-day tenors.
  - "Paper probe (now) … calibration recorded daily" (`:447`). Recording stopped on 09-02 (`zdte-recorder.yml:3-6`).
  - "Survives half the credit" (`:441`). Contradicted by 27b.
- The paper track ("10 signals, 3/3 wins") is three model-priced end-of-day replays after the fact, padded with 6 duplicate records.

### Verified findings (ranked)
| # | Finding | Evidence | Effect |
|---|---|---|---|
| 1 | Site/API still publish the retired credit engine as current, with sizing advice (ZDTE-01/06/08) | Live fetch; `zdteLiveCore.ts:23-24,192`; `page.tsx:358-383,436-447` | At measured credits (×0.31) the "suggested" 5% risk gives CAGR −10.5% and max DD −44.6%, not +89%/−7.9% (rebuild reproduces the published hold basis: n 1368, Sharpe 3.82 vs 3.88). Kelly fraction = 0 |
| 2 | Second real-quote session (09-25): model credits 2.1–3.8× CBOE quotes; SAFE real credit ≈0.45bp is below the engine's own 1.2bp floor (ZDTE-02) | `scratchpad/review/spx_chain_20260925.json` vs API debug | κ ≈0.55–0.66 (depends on the spot convention), at or below 27b's break-even. It also contradicts iteration 30's put smile, which predicts κ≈1.0–1.1 at these wings |
| 3 | κ recorder cannot work, and its failure is hidden (ZDTE-03) | No pip install (`zdte-recorder.yml:38-40`); `|| echo` (`:53`); `Z.load_m1` needs an absent /tmp path (reproduced `ValueError`); 16 "kappa record" commits touched only ratchet files; `zdte_kappa.jsonl` = 1 line; runs land 13:35–15:25 ET | The research's go/no-go gate never progresses; ~16 free sessions lost |
| 4 | VIX1D "model-free" test uses √(1/365); CBOE's methodology is 252 × 405 business minutes (ZDTE-05) | `zdte_vix1d.py:9,69,82,96`; CBOE VIX1D methodology PDF | At VS/ATM 1.108: +13.09bp (t 6.48) → **+0.67bp (t 0.33)**. At 1.00: +6.50 → **−7.26bp (t −3.62)**. Realised/implied variance 1.265 → 0.873, i.e. a normal VRP. Implied κ ≈0.87 → ~1.05. 39% of the old P&L also came from back-calculated, pre-dissemination VIX1D (ZDTE-18) |
| 5 | The only κ_ATM (0.74–0.80) reflects a high model σ_rem, not cheap options (ZDTE-04) | Joint forward fit IV 0.0589 vs realised 10:26→close 0.0538 (market/realised ≈1.09) vs model 0.079 (1.46× realised) | Removes the only real-quote support for "near-money 0DTE is cheap". Whether the prior was stale is unproven |
| 6 | Debit edge is conditional on n=1 κ; t-stats hold κ fixed (ZDTE-07, mostly acknowledged) | `zdte_debit.py:42-43,126`; κ scan reproduces the doc table | At κ≈1.05: +1.62/+1.95/+0.73bp; trimmed HO −1.21. HO was not a clean holdout: the winner was picked from 30 structures with HO visible (ZDTE-14) |
| 7 | 0% of the backtest priced on real quotes; all real quotes post-date the evaluation window (ZDTE-10) | `zdte_export_history.py:97,139-140` | Every edge number is BS × multiplier |
| 8 | Page presentation: Sharpe/CAGR annualised on traded days only (ZDTE-25); "OOS" means selected on 2022-06+ (ZDTE-13); take-profit timing is model-marked (ZDTE-20); window excludes 2018/2020/2022-H1, where the put family loses (ZDTE-21); strikes re-derived with request-time SPX (ZDTE-12: the label moves; scoring is effectively unchanged) | cited files | Each inflates the page by ~10–40% even on the model basis; secondary to pricing |
| 9 | Hygiene: XSP small-account advice ignores per-contract fees (ZDTE-24); debit PRE window includes days with no SPXW expiry (ZDTE-29: result unchanged, MWF-only +6.5bp); the κ P&L table and the break-even table use different κ axes (ZDTE-31: the table *does* reproduce; the orchestrator's "not reproducible" is corrected) | | Minor |

Refuted: ZDTE-11. The 0.70/0.45 multipliers *do* reproduce from the 08-29 SPY 2-DTE rows (0.72/0.41). The doc mislabels them as SPX.

### Established vs open
- **Established.** Model credits overstate real 0DTE credits 2–5× (two sessions, real quotes). Credit spreads are not tradeable at honest prices. There is no directional or pivot signal (AUC 0.46–0.51, lookahead-corrected). The site is wrong.
- **Open.** The level and dispersion of κ_ATM at 10:00 across regimes, and whether the long debit vertical earns anything at real bid/ask. The two independent real-data checks (ZDTE-04, ZDTE-05) both point to κ≈1, where the debit is marginal or negative.
- **What settles it.** Historical SPXW 0DTE 1-min NBBO for 2022-05-11 onward (about 1,100 sessions), ideally with Mon/Wed/Fri 2016+ added. Then: κ(t, z) with IV inverted from mids using the engine's own τ; and a replay of the debit at ask/bid to PM settlement.

## 3. Ratchet

### Mechanics
Deployed rule: Addendum 26 (`LINE_IN_SAND_FINDINGS.md:1851-1872`; `python/ratchet_engine.py`).
- **Entry.** Every day, when SPY is above its 200-day MA (or VIX ≥ 30) and the VRP rank is ≥ 0.20. VRP = VIX − 21-day realised vol, using an expanding rank (min 500 observations).
- **Structure.** Short put 3% OTM, long put 5% below it, ~21 sessions.
- **Sizing.** 1% of equity at risk per unit, doubled at VIX ≥ 20.
- **Exit.** Harvest at the first close ≥ line × 1.01 (median hold 7 sessions); otherwise hold to expiry. There is no stop. About 158 entries a year, with overlapping positions.
- **Backtest pricing.** ^GSPC closes, r=q=0 Black-Scholes on a surface linear in VIX, fitted to 1,210 free DoltHub SPY EOD chains (2019-02..2026-08). Only entry credits are pinned to real quotes; exit marks are extrapolated.

### Research arc (5 self-audits)
Headline CAGR moved 141% → 19.2% → 16.8% → 11.5% → 6.5% → **13.1%** (`page.tsx:640`). The steps:
1. Assumed payoffs were replaced with Black-Scholes pricing.
2. Mark-to-market accounting was added.
3. Real/model credit measured at 0.726 on real quotes.
4. Direct repricing on a calibrated surface.
5. A skew-tenor defect was fixed. This revised the result **upward** to 13.1% / Sharpe 1.04, and "the rule does not change".

The docs admit the following: pre-2019 prices are synthetic; 2008 is untestable; plan on about half the historical edge; distrust audit #5 until the fill gate closes.

### Live state vs research
- **/ratchet** shows the right rule, but it is a stale 09-02 build. It says "0.98 … the model and the tape agree" and "median 0.994". Repo HEAD has the 09-24 check at 0.86. The "agree" text is also hard-coded (`page.tsx:351,361,662`; `ratchet.ts:431-432`).
- **Home page ("/")** tells users today to "sell SPX put spread at strike ≈ the current price, ~3 months out". That is ATM, 63 sessions, with a fixed 2-point VRP gate (`app/page.tsx:181,405`; `ratchet.ts:58,72`). It backs this with the superseded 16.8% / Sharpe 1.26 and a trend sleeve labelled "current best 19.1%" (`page.tsx:206,287`).
- **"Live: 6 open"** is frozen. The cron is off (`vercel.json` crons [] ; v19-strategy.yml paused 09-01). By closes, 4 of the 6 positions would already have harvested.
- **No paper or live order has ever been placed.** The published cron `--pricecheck --paper` never reaches `place()`.

### Verified findings (ranked)
| # | Finding | Evidence | Effect |
|---|---|---|---|
| 1 | Home page recommends a retired, never-calibrated structure with superseded numbers (R-01) | Live HTML; `app/page.tsx` last changed 2026-08-22 | Overstates CAGR +3.7pp and Sharpe +0.22, and for a *different* trade |
| 2 | Model over-credits the traded spread since 2023 (R-02). "≈1.00 on pinned fit" does not reproduce; incumbent by-year ratios were never saved | Three independent DoltHub samples: traded-day ratio 0.899–0.914 (2023–26) vs 1.03 (2020–22); 2024 about 0.83–0.86 | ×0.907 → 9.9% / Sharpe 0.81 / excess 0.60. As a surface shift, 11.9–12.3%. **Real quotes** |
| 3 | Forward CBOE ledger has turned against the model (R-04, R-05). The "9 of 9 agree" rows are in-sample: the rows that triggered the fix and were restated, all pre-open or overnight, with fix commit 9ee8800 11 minutes later | `data/ratchet_probe.jsonl`: otm3 n=17, median 0.931, last five 0.871/0.827/0.84/0.85/0.86; mid-based 0.937, so this is not spread cost | Live confirmation of the upward audit is essentially nil. Engine uses calendar/365 vs backtest /252, flattering the ratio 3–6% (R-12). Consistent-convention median ≈0.90, last five 0.78–0.83 |
| 4 | Exit/harvest marks never validated on real quotes (R-03) | exits3: 45 exits, real/model 1.145 (but 44 of 45 interpolated); pe_marks: n=317 at 9–12 sessions left, ≈+0.08% of risk bias; 13–20 sessions left, 1.06–1.08 | Range 0 to −1.7pp CAGR; worst case combined with entry gives 8.7% / Sharpe 0.72. Largest open lever |
| 5 | Fill gate (≥60 records, median ≥0.60) sits below break-even (≈0.65) and its denominator can be restated (R-11) | `ratchet_engine.py:84,220-238`; entry credit ×0.60 → 0.7% CAGR, excess Sharpe −0.16 | A pass would not confirm the headline. Today's ~0.86 implies ~8% CAGR |
| 6 | Engine sizing: `qty = max(1, …)` with no open-risk or buying-power check (R-07) | `ratchet_engine.py:401`; $3,611 risk per spread | Accounts under ~$361k are over-sized (a $50k account runs ~7.2% per entry). The page says "rounded down" |
| 7 | Settlement: backtest cash-settles; engine trades American SPY and simply drops expired positions (R-10) | `ratchet_engine.py:387-389,429-433` vs `page.tsx:654-656` | Assignment risk: a between-strikes expiry leaves ~$74k of stock per spread. Fix: trade XSP/SPX, or close at T-1 |
| 8 | "Holdout" and "real-quote era" are date slices of one modelled book whose surface was fit on 2019–26; "sealed OOS +1.55" belongs to a different hl-quant strategy (R-06) | `lis_ratchet_site_curves.py:135-136`; `STRATEGY.md:93` | Removes the claimed out-of-sample support; point estimate barely affected (13.1% is ranked 36th of 60, not a grid maximum) |
| 9 | Accounting and benchmark (R-18, R-19): Sharpe includes T-bill with no risk-free subtraction; benchmark is price-only | Excess Sharpe 0.83. S&P total return 11.07% (full period); holdout 11.8% vs TR 15.1%; real era 15.8% vs 16.8% | Still has in-model alpha of 6.9%/yr (t 5.45, beta 0.38); a ~15% credit error would erase most of it |
| 10 | Crisis rules chosen in-sample and priced purely by the model (R-16, R-17). Max DD is sample-bound (R-09): 1987 is not covered; worst one-day gap on the historical open book is −31% | VIX≥30 re-entry and 2× sizing add ~4.7pp CAGR but also cause the 2009 max-DD episode. VRP gate helps in 2 episodes (2008, 2022) and costs return in 29 of 36 years | Tail rests on ~28–46 clustered loss episodes (R-20) |

Lower severity: R-08 aggregate open risk (median 6%, peak ~40%; the docs already state ~37%); R-13 exit timing (next-close exit: DD −25.2%); R-15 frozen ledger; R-22 flat costs (−0.3 to −1.8pp); R-23 3% OTM is mid-pack (4–5% OTM match it); R-24 to R-27 ledger hygiene. Nothing was refuted.

### Established vs open
- **Established.**
  - The in-model result is statistically solid: monthly excess t ≈5, stable across sub-periods, no lookahead, headline reproduces from Yahoo in ~4 s.
  - Entry credit is roughly right on average across 2019–26 (0.976 real), but about 7–10% rich in the 2023–26 regime the rule actually trades in (**real quotes**).
  - The MA200 filter dominates.
- **Open.**
  - Real exit marks (harvest buy-back at 7–20 sessions left).
  - Behaviour in 2008, 2011, 2015 and 2018 on real quotes.
  - SPX vs SPY execution.
  - Whether the 2023+ credit shortfall persists.
- **What settles it.** Full-strike daily EOD chains with bid/ask for SPY and SPX from 2007, used to replay each trade entry to exit on the exact strikes. For 2010–2023 this is free (OptionsDX SPY EOD). 2008 needs paid data. Also a 60-*session* 15:45 ET forward log with exit quotes.

## 4. How the two relate
- Both are short index downside. Ratchet's daily correlation with ^GSPC is 0.657 (beta 0.43, −1.35% on S&P days below −2%).
- The zDTE book shown on the site is 28 of 40 trades on the put side.
- Measured monthly correlation over 2022–26 is low (0.02 model, 0.12 at real-credit scale), because zDTE is flat overnight and trades both sides. The overlap sample, however, contains no 2008/2020-type crash, and both lost in 2025-04.
- Run together, they stack same-day crash exposure onto Ratchet's multi-day open put ladder (up to ~40% of equity at risk).
- The research's replacement, a long call debit, does not hedge this. Moot anyway, since zDTE should not be traded.

## 5. Data-provider decision

Prices below were verified 2026-09-25 unless marked otherwise.

| Provider | SPXW intraday history | EOD history | Bid/ask | Delivery | Price | zDTE fit | Ratchet fit |
|---|---|---|---|---|---|---|---|
| **ThetaData** Options Std/Pro | Tick NBBO from 2016-01-01 / 2012-06-01 (Value: 1-min from 2020). Pricing card says "8/12 yrs"; ask them | Included on paid tiers | Yes, every OPRA NBBO | Local Java terminal + REST, 4–8 concurrent | $40/$80/$160/mo; SPX index add-on $50 (from 2022) or $100 (from 2017) | **Best value**: one month covers the whole 0DTE era | Good for SPY 2016+; SPY stock only from 2020 |
| **Cboe DataShop** Option Quotes 1-min | Jan 2012+, exchange-sourced; free sample confirms SPXW rows | EOD Summary 2012+ (15:45 + close) | NBBO + size | Daily zipped CSV / SFTP | ^SPX 1-min: 30-day pilot $276; any 4+ years $2,200 ($3,385 with IV). SPY EOD 2019–26 $580. **No SPX spot unless CGIF (≥$1k/mo)**, so use parity | Excellent, authoritative | OK (EOD 2012+, misses 2008) |
| **Databento** OPRA.PILLAR | 2013-04+; 1-min only before 2023-03-28, tick after | Must assemble | Consolidated BBO | Python client, batch | cbbo-1m $2.00/GB; $125 signup credit; exact cost unverified | Good; possibly inside the credit | Adequate |
| **ORATS** | 1-min from Aug 2020 (SPXW not named explicitly) | **Near-EOD (15:46) from 2007** | NBBO; data is SMV-cleaned | S3 (14-day window) / API | Near-EOD $599 one-time; 1-min $1,500 + $1–2k S3; API $199–$899/mo | Partial | **Best value for 2008+ crises** |
| HistoricalOptionData | Go-forward only | SPX 1990+ ($849), all US 2002+ ($1,150–$1,495) | Close bid/ask; 4:00 vs 4:15 mismatch | Zipped CSV | as listed | None | Only pre-2007 route |
| OptionsDX | SPX 15-min $5/yr, 1-min $50/yr (2019–23); SPXW inclusion **unverified**; bundled IVs wrong for 0DTE | SPY/SPX EOD **free** 2010–2023 | Yes + size | CSV | $0–$50/yr | Cheap but risky | **Free Ratchet exit/crisis replay 2010–23** |
| DoltHub | None | SPY only, 2019-02+, sparse strikes, no quotes under ~11 DTE | Yes | Free SQL API | $0 | Unusable | Entry-credit baseline |
| Massive (Polygon) | Quotes from 2022-03-07 | 2–5 yrs | Advanced tier only | REST / flat files | $199/mo + $49 indices | Adequate | Weak |
| QuantConnect (AlgoSeek) | Minute quote bars 2012+ | Daily 2012+ | Quote bars | Cloud only | Free in cloud (org-licensed); local 500 GB bulk $30k/yr | Good if you port the code | Moderate |
| AlgoSeek / LiveVol / All Access / IVolatility / OptionMetrics / dxFeed / Intrinio / EODHD / Alpaca / Tradier / IBKR / Schwab | See data track | | | | ≥$18.6k/yr, $420–$4,599/mo, quote-only, or no expired history | Poor / unfit | Poor / unfit |

**Free first, in order:**
1. Fix the κ recorder: compute sigma_rem from free Alpaca/Yahoo 1-min bars at runtime, add pip install, remove `|| echo`, run at 10:00 ET off GitHub cron.
2. Correct /zdte (retract the credit engine) and "/" (the Ratchet card). Fix R-07, R-10 and R-25 in the engine before any paper trading.
3. Rebuild the DoltHub SPY cache (~30 minutes). Pull OptionsDX free SPY EOD 2010–2023 and replay Ratchet trades entry to exit on real strikes. Save the incumbent by-year ratios.
4. Get one extra κ session from the free Cboe DataShop 2022-02-02 sample.

**Recommended purchase, in stages:**
- **Stage 1 (zDTE, ~$80–$276).** One month of ThetaData Options Standard ($80). Imply spot by parity, or add Indices Std at $50. Alternatively, the Cboe DataShop 30-day ^SPX pilot ($276), scaling to $2,200 only if it is promising. Pre-register first: κ_ATM at 10:00 across all sessions, then the debit vertical at ask/bid to PM settlement, with the 2025-07+ window scored once. If κ comes out around 1, stop.
- **Stage 2 (Ratchet, $599).** ORATS near-EOD 2007+, bought only if the free 2010–2026 replay still shows positive excess return after real exit marks. It adds 2008 and a full-strike, cross-vendor check.
- Buy HistoricalOptionData ALLSPX ($849) only if pre-2007 matters.

**What NOT to buy:** anything with annual commitments or index-licence fees (AlgoSeek, LiveVol, All Access API, CGIF, $64k index tick files); IVolatility retail; Massive for Ratchet; vendors with no expired-option history (Tradier, IBKR, Schwab, Alpaca, EODHD, Intrinio); dxFeed; OptionMetrics without WRDS. Retail licences are personal-use only, so do not publish raw quotes on the public site.

## 6. On "a very profitable model"

The honest prior is low. The recorded negatives:
- zDTE's Sharpe 5–7 was 3–5× inflated credit.
- 0 of 892 credit structures survive at honest pricing.
- There is no directional signal.
- The one remaining lead rests on a single κ snapshot, and its supporting evidence reversed once the day-count was fixed.
- Ratchet is one risk factor, the equity put premium. Its research log itself concludes that much higher Sharpe is "not available from this structure" (`LINE_IN_SAND_FINDINGS.md:1755-1763`) and that Sharpe 5 is not reachable with EOD listed options (Addendum 20).
- Once honestly priced, Ratchet sits at the literature's put-writing Sharpe of about 0.5–1.

A realistic best case is a Sharpe ~0.6–1.0 short-premium book, plus possibly a small 0DTE long-convexity effect. P(Sharpe ≥ 2 with tolerable drawdown from retail data) < 5%. Data can confirm or kill these edges. It does not create one.

A disciplined plan with the purchased data:
1. **Write down before any data is loaded:** the hypothesis, the single metric (P&L at prices actually paid: ask to buy, bid to sell, with fees), the parameter grid, and the number of trials.
2. **Freeze a holdout.** Everything from 2025-07 onward, plus forward data from 2026-10, is untouched until a single final run. Use the pre-2022 and 2022–25 windows for development.
3. **Real-quote fills only.** No model credits or model exits. IV inverted from mids using the model's own clock. Spot and quotes taken from one timestamp.
4. **Account for selection.** Deflate the Sharpe for the full trial count. Use block bootstrap by loss episode. Report excess-of-T-bill Sharpe against the S&P total return.
5. **Forward test.** 60 *sessions* of paper orders with recorded fills, run on a reliable scheduler before any capital. The fill gate's threshold should be set where the edge actually survives (≥0.85–0.90 for Ratchet), not at 0.60.
6. **Keep the live site in sync** with the research state, and publish negatives.

Real-quote numbers vs model numbers, in brief:
- **Real quotes:** iteration-27 print; 09-02 κ ladder; 09-25 CBOE chain; VIX1D closes; DoltHub entry credits (0.976 pooled; 0.90–0.92 on traded days 2023–26); CBOE forward ledger (0.931 median, 0.83–0.87 recent).
- **Model:** every CAGR, Sharpe, win rate, drawdown and debit P&L figure quoted anywhere above.

---

## Known gaps in this review (read before acting)

A completeness critic found six gaps. One was investigated before the session usage limit stopped the run; five were not.

**Closed: the 09-25 real-quote comparison.** The CBOE feed is delayed about 15 minutes. All 652 SPXW 0DTE quotes are effectively from 10:43 ET, while the engine signal is from 10:55. Repriced at the chain's own time, with spot implied by put-call parity, the engine's credit is **2.6–4.3× the CBOE mid**. That is *worse* for the engine than the 2.1–3.8× quoted above. The conclusion stands, and the range above should read 2.6–4.3×.

**Open (not verified — treat the related claims as provisional):**
1. **κ definition consistency.** The engine's quoted credit already includes the REAL haircut (`zdteLiveCore.ts:187,210`). So "model vs real" ratios in this report are sometimes against the haircut credit and sometimes against raw Black-Scholes. The direction of every conclusion is unaffected; the κ point values are not on one definition.
2. **ThetaData feasibility.** It has not been confirmed that SPXW *index* options are in Options Standard, as opposed to needing the Indices add-on. Nor has it been confirmed whether history starts in 2016 (docs) or covers "8 years" (pricing card). **Email ThetaData before paying.**
3. **Unsupported quantitative claims.** "P(Sharpe ≥ 2 from retail data) < 5%" has no derivation behind it; it is an opinion. Ratchet's "8–11% CAGR, excess Sharpe 0.45–0.7" combines separate haircut scenarios (entry-only ×0.907 → 9.9%; entry and exit worst case → 8.7%). It is a range of scenarios, not a confidence interval.
4. **Ratchet forward ledger.** Among the recent 0.83–0.87 records, some chain timestamps fall overnight (e.g. 2026-09-23 03:54 UTC). n=17, median 0.931. No uncertainty band has been computed.
5. **Execution and account risks not covered.** For Ratchet: early exercise of deep-ITM American SPY short puts during drawdowns; margin and buying power for about 158 overlapping entries a year; per-contract commissions at that frequency on a small account.
