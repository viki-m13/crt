# Candidate 18 publication audit — 2026-09-07

**Verdict: not safe to publish as-is.** The frozen research account reconciles; the public reproduction account implements materially different rules. Publication also mixes account histories, drawdown resolutions and trade cohorts.

Paths below are relative to this directory, except `site/…` (under `/root/spy`) and the team-root inventories. Fresh = $100,000 started 2025-01-01; history = $100,000 started 2020-01-01. All windows end **2026-09-01 00:00 UTC exclusive**. Last 90 days starts **2026-06-03**. “Natural” excludes reason 4, the nine modeled cutoff liquidations.

I recomputed from both runs' trades, orders, minute equity and daily ledgers; used archived minute prices/funding to reconcile calendar-window coin P&L and funding; also checked the 12/30-bp and four neighborhood ledgers. `02` and `03` ran unchanged in `/tmp/c18_audit_ba_6m_2t/`, with read-only links to supplied data/reference files. I edited only this audit. External changes to `thread.md` (15:47 UTC), `make_charts.py` and `1_hero.png` (15:45 UTC) were detected and the updated claims rechecked before finalization. Independent calculations, rerun logs and targeted probes are in that scratch directory (`recompute.py`, `recomputed.json`, `per_coin.json`, `dsr.json`, `probes.py`).

## BLOCKER

1. **Next-open sizing uses information from later that day.** `package/02_peer_breadth_backtest.py:107`, `:141`, `:144`, `:153`: it processes today's entire high/low and marks surviving positions at today's close before admitting orders at today's open. A probe changing only a carried position's future close changes the next-open risk from **$250 to $251.25**; correct risk is **$250 in both cases**. An intraday stop also frees capacity for an order supposedly placed earlier that morning. **Fix:** process funding/known open exits, size/admit using open-time equity, then process intraday protection and closing marks.

2. **Admission is a different portfolio rule.** `package/02_peer_breadth_backtest.py:146–169` lacks the engine's occupied-symbol check and allocates alphabetically, first come first served. The saved package has **87 post-2025 entries overlapping an existing same-coin position**, including opposing sides. Research skips **687 occupied-slot signals**. On the fresh account's first seven-signal batch, research allocates **$214.285714 risk to each**; the package allocates **$250 to the first six**. Engine: `package/research_engine/engine.py:132–183`. **Fix:** enforce one open slot per coin and allocate each simultaneous eligible batch proportionally.

3. **Quantity/risk formula and minimum order rules differ.** `package/02_peer_breadth_backtest.py:159–162`: quantity uses `risk/(entry*stop_pct)` and omits the cost reserve, $100 minimum notional and 1e-8 quantity flooring. Research uses `floor((risk/(stop_pct+0.002)/entry)*1e8)/1e8`; see `package/research_engine/engine.py:177–192`. The saved package contains **131 trades below $100 notional**, including two natural exits below $1e-8 notional. It rejects a gross-cap breach rather than proportionally scaling it. **Fix:** port the engine's risk denominator, rounding, minimum order and gross-cap allocation rules, then regenerate statistics.

4. **Targets are disabled for an extra day; target execution also differs.** `package/02_peer_breadth_backtest.py:121`, `:125–126`, `:167`: `entry_bar=True` survives until the following day's management pass, so that entire day cannot hit a target. Probe: a target at **130**, penetrated on **Jan 3** after Jan 2 entry, instead exits at **100 on Jan 5**. Package also fills exact touches and favorable opening gaps, and can choose a later intraday stop despite an already-executable opening target. Research requires strict penetration, fills at target price and processes known opening exits first (`package/research_engine/engine.py:104–118`, `:230–234`). **Fix:** correct the flag lifecycle and apply the research target/open-exit rules; disclose any deliberately conservative daily ordering policy.

5. **Open-account fees/funding are booked at the wrong time.** `package/02_peer_breadth_backtest.py:131–143`, `:165–180`: entry fees and accrued funding are absent from cash/equity until exit; the comment claiming entry cash already excludes fees is false. Stop/target exits receive funding through **23:59 on the exit date**, potentially after exit; time exits exclude that day's **00:00** settlement, which research's 00:01 exit pays/receives. Entry-day stops receive zero funding regardless of settlement timing. **Fix:** debit entry cost immediately and settle funding at recorded events while the position is open; document any daily intraday-timing approximation separately. The funding **sign itself is correct** (see notes).

6. **The advertised minute reproduction route is not runnable from the download.** `package/README.md:76`; `site/peer-breadth.html:246`; `package/research_engine/run_campaign.py:22`, `:104`; `package/research_engine/prepare_data.py:63`, `:120`: copied runner fails with **missing `expanded_20260906/data/catalog.json`**. Its generator discovery expects `research/*/signals*.py`, but the supplied generator is flat `signals_trend.py`. Preparation requires an absent website snapshot and prior catalog, dynamically selects a universe, and defaults to 2024 onward. The engine/runner/generator files do match frozen source hashes, but do not form a standalone reproduction. **Fix:** supply a C18-specific driver, fixed universe/configuration, working data paths and minute-route dependencies/instructions, and test the extracted ZIP.

## FIX

1. **Comparison mixes account initializations.** `package/02_peer_breadth_backtest.py:42`; `package/03_compare_to_reference.py:29–35`; `package/README.md:54–58`; `build_page.py:139–142`; `site/peer-breadth.html:245`: package **+13.334%, 282 exits, PF 1.779** is a 2025-window slice of its continuous-2020 account; reference **+16.736%, 468 exits, PF 2.138** is fresh-2025. Matching continuous research is **+16.031%, 477 exits, PF 2.057**. Changing only the package start to 2025 gives **+14.312%, 272 exits, PF 1.875**, before fixing its bugs. **Fix:** run and label both account initializations and compare like with like after the blockers are repaired.

2. **“Minute closes” is the wrong fill description.** `package/README.md:51–52`; `package/03_compare_to_reference.py:5`; `site/peer-breadth.html:244`, `:253–254`; `build_page.py:278`, `:287–288`: research uses **minute OHLC, opening market fills and stop/target levels**, with equity marked at minute closes. `thread.md:50`, `:53`; `make_charts.py:134`, `:140`; `package/README.md:28–30`; page header `site/peer-breadth.html:156` omit the deliberate **one-minute delay**: all research entries are **signal midnight + 1 minute**, not the 00:00 daily open; scheduled exits are **20,160 minutes later**. Research also permits an entry-minute target; “entry bar can only stop out” is not its rule. **Fix:** publish the actual 00:01 UTC entry/time-exit and minute-OHLC policy, and label daily execution as an approximation.

3. **“1,313/1,313, including stop distance” is false; `03` checks only keys.** `package/README.md:49–50`; `site/peer-breadth.html:246`; `build_page.py:280`; `package/03_compare_to_reference.py:18–26`: all **1,313 date/coin/side keys** match, but only **1,310 stop distances** match within 1e-12. BCH signal-bar dates **2025-01-14, Jan 22, Jan 26** differ; maximum absolute stop-fraction error **0.000113480090** (0.011348 percentage points). Running the research generator on the package daily bars reproduces the package stops exactly: these three differences are input-bar differences, not a different ATR formula. Across history, only **3,479 keys** match, with **17 reference-only and 1 package-only**. `03` neither tests stop distances nor fails on disagreement. **Fix:** compare one-to-one keys and stop values with declared tolerances, return failure on mismatches and publish the actual agreement.

4. **Public rule summaries omit executable conditions.** `thread.md:40–44`; `make_charts.py:123–140`; `site/peer-breadth.html:156`: “other 19” must mean an eligible subset, requiring **max(8, ceil(0.8×19)) = 16 peers**, complete five-day windows at both endpoints and the same membership for both shares. The page header omits the coin's **today-close versus yesterday-close** condition. The rule-card footer omits the **0.5% stop floor**. Generic “ATR(14)” leaves its smoothing unspecified: this is the **14-bar simple mean of true range**, not Wilder smoothing. The purported complete rule in `package/README.md:27–36` also omits the one-open-position-per-coin restriction. Correct definitions are in `package/reference/config.json:29–46`, `package/research_engine/signals_trend.py:200–248`, `:394–398` and engine `:153–158`. **Fix:** add the eligibility/own-close conditions, SMA-ATR/floor definition and one-position-per-coin admission rule to the canonical rule and its summaries.

5. **Headline accounts differ across surfaces and lack precise labels.** `thread.md:20–22`, `:74`, `:78`; `make_charts.py:83–88`, `:117`; `site/peer-breadth.html:159`, `:269`: the updated hero leads with **51% wins / PF 1.66 / −4.6% DD / 1,403 exits**, all correct for continuous-2020. Its smaller **54% / 2.14 / −2.8% / 468** figures are fresh-2025, also correct; thread post 1 and the page lead with those fresh figures. “Since 2025” alone does not identify a fresh account: the historical account's 2025 slice instead has **53.9% wins / PF 2.06 / 477 exits**. Neither hero nor posts 1/6 identifies DD resolution. **Fix:** choose the same primary account across thread/hero/page and use the shared account/window/minute-mark labels below; identify the historical curve separately.

6. **Yearly drawdowns are daily closes labeled as minute marks.** `build_page.py:70`, `:251`; `site/peer-breadth.html:217`, `:267`: displayed **−2.41, −2.77, −4.37, −1.82, −2.62, −2.33, −1.84%** for 2020–2026 are daily-close DDs. Correct minute DDs are **−2.60, −3.06, −4.64, −2.39, −2.92, −2.83, −2.40%**. Block-table DDs (`build_page.py:83`; page `:241`) likewise use daily closes without saying so. **Fix:** calculate yearly/block DDs from minute equity, or explicitly label those separate columns “daily-close DD”; do not call the current values minute marks.

7. **Underwater curve uses the wrong denominator for percentage drawdown.** `build_page.py:328`; `site/peer-breadth.html:294`: `v - running_peak(v)` on cumulative returns gives loss as percentage points of initial $100K, minimum **−4.995170 pp**. Continuous-account daily peak-relative DD is **−4.372378%**. **Fix:** plot `100*(equity/running_peak_equity - 1)` and label it “continuous-2020, daily-close drawdown.”

8. **Sortino is mislabeled/calculated with negative-return standard deviation.** `quant_metrics.json:4`; `make_charts.py:209`; `build_page.py:165`; `site/peer-breadth.html:269`; `package/README.md:85`: **3.14** reproduces `mean(r)/std(r[r<0])*sqrt(365)`, not the stated downside deviation. With zero minimum acceptable return and the declared **607 close-to-close returns**, `mean(r)/sqrt(mean(min(r,0)^2))*sqrt(365)` is **3.425631 → 3.43**. **Fix:** use the latter denominator and regenerate the quant card/page/README, or explicitly rename the current nonstandard statistic.

9. **README drop-best table uses cutoff-inclusive P&L; graphics/page use natural exits.** `package/README.md:86` says **$16.0K/$13.9K/$12.2K** after deleting best 1/5/10. For the declared 468 natural exits, correct values are **$15,614.43/$13,551.82/$11,848.41 → $15.6K/$13.6K/$11.8K**, matching `make_charts.py:183–195` and `site/peer-breadth.html:208`. README values retain **$348.217226** of cutoff P&L. **Fix:** use the natural-exit amounts throughout and label the cohort.

10. **Largest-win denominator is misstated on the graphic.** `make_charts.py:194–195`: **“2.5% of all the profit”** is actually **2.512879% of the sum of positive natural-trade P&Ls**. Largest win **$773.625989** is **4.720669% of natural net profit**. Page `site/peer-breadth.html:198` correctly says gross profit. **Fix:** change the graphic footer to “2.5% of gross profit,” or report 4.7% of natural net profit.

11. **Signals/week and skip reasons are wrong on the page.** `build_page.py:155`, `:262`; `site/peer-breadth.html:228`, `:269`: **5.4 signals/week** is **5.420455 admitted entries/week** over 88 calendar-week bins; raw signals are **15.116776/week** over elapsed time. **836 skipped by the risk cap** is **687 occupied-slot skips + 149 capacity/rounding skips**. Another **439 filled entries were scaled**; they are not part of 836. **Fix:** label the activity “entries/week,” give raw-signal frequency separately, and print the actual skip/scale breakdown.

12. **Sizing prose confuses full-size opportunities with typical filled trades.** `thread.md:60–62`; `make_charts.py:116`; `package/README.md:33–36`; `site/peer-breadth.html:174`, `:224`: “each trade risks $250,” “typical position 1–2%,” and “later signals get shrunk” do not describe this ledger. Correct: **up to 0.25% of current equity**, pro-rata simultaneous allocation; median reserved risk **$103.98**, median filled notional **$776.61 = 0.7263% of entry equity**. **13.9358%** is median raw-signal stop; filled median is **14.1350%**. Full-size ~$1,800 on the initial account is a valid illustration (about **$1,769 including the cost reserve**), not the typical position. The **1.5% cap is enforced at admission**, not continuously: reserved risk/marked equity subsequently reaches **1.514824% fresh / 1.516231% history** without new orders. **Fix:** distinguish the unscaled illustration from actual medians, state the admission-time cap and 20-bp risk reserve, and label $777 as 0.73% of entry equity or 0.78% of initial capital.

13. **Median holding period is wrong.** `build_page.py:248`; `site/peer-breadth.html:214`: **“median hold a few days” → 14 days**; 318 of 468 natural exits are scheduled time exits. **Fix:** replace the holding-period phrase; the independently reconciled **$82.64 funding / $920.51 fill-cost totals** themselves are correct.

14. **25 is a Sharpe-available subset, not the library count or a declared “pass.”** `thread.md:106`; `make_charts.py:214`; `quant_metrics.json:51–57`; `package/README.md:101–103`; `site/peer-breadth.html:269`: `TALLY.json:3` and `LIBRARY.json` contain **31 candidates**. Exactly **C01–C25** expose the selected `2025_onward.sharpe_daily_365` field; C26–C31 use other result schemas. For that subset, population sigma **0.510839479**, hurdle **1.020223008**, DSR **0.866388528** reproduce. That is a conditional probability, not an established pass criterion. Merely using N=31 with the same incomplete sigma gives **0.851252**, not a complete 31-candidate recalculation. **Fix:** say “25 of 31 candidates with comparable Sharpe inputs: DSR 0.87,” remove “passes,” and rename the misleading `fresh2025_N31_library` key, or recompute a fully specified 31-candidate universe.

15. **Full-search DSR input provenance and quant reproduction are missing.** `quant_metrics.json:10–14`, `:30`, `:36`; `package/README.md:98–103`; `site/peer-breadth.html:251`: sigma **2.268074676 → 2.27** comes from **285 finite Sharpe values in 286 top-level campaign run files**, including **34 history replays**; it is not a one-to-one measurement of all 465 distinct expressions. With those inputs and N=465, hurdle is **6.873726867**, DSR **2.51e-14 ≈ 0**. The JSON's 6.87 versus variant 6.88 mixes unrounded and rounded inputs. Family sigma **1.119365961**, hurdle **1.334949130**, probabilities **0.740502 fresh / 0.436782 history** also reproduce. There is no quant generator/input manifest; the README's promised `quant_metrics.json` is neither linked by the page nor present in the ZIP. **Fix:** ship the calculation and exact source list, disclose the sampled universe, use unrounded inputs consistently, and expose the JSON in the download.

16. **Monte Carlo amounts cannot be reproduced exactly from the supplied artifacts.** `quant_metrics.json:22–24`; `make_charts.py:213`; `site/peer-breadth.html:269`: **$853 median / $1,306 95th / $2,169 worst** have no saved generator, RNG seed, ordering or initial-peak convention. An explicit independent check (468 natural net-dollar trades, 5,000 `default_rng(0)` permutations, prepend $0) gives **$858.13 / $1,321.87 / $2,384.27**. Those are a reproducible comparison, not proof the unpublished draws are wrong. **Fix:** save the intended generator/seed and rerun; label this **closed-trade P&L order-shuffle DD in dollars**, not minute-mark account DD.

17. **Search-count wording overstates what the inventory records.** `thread.md:24`; `package/README.md:7–8`; `build_page.py:190`; `site/peer-breadth.html:156`: **465 distinct expression IDs** is correct, independently counted across **1,118 ledger rows** (`TESTED_LEDGER.md:3`), including earlier seed work and screens. It is not a count of 465 distinct full portfolio runs performed overnight. “The one expression that survived” conflicts with **31 registered candidates**. **Fix:** say “selected from 465 recorded crypto expressions; the library contains 31 candidates,” retaining “most promising” only as Pedro's judgment.

18. **The majority-failed concentration claim has no counted evidence.** `thread.md:96`, `:100`; `make_charts.py:181`: **“the other 464 mostly failed” / “most … went negative … best day”** is not established by a 464-expression result table or the inventory count. Verified here: C18 remains positive even after removing its best 50 natural trades. **Fix:** remove the majority claim or supply an expression-level numerator/denominator and the precise best-trade versus best-day test.

19. **Improvement/family claims exceed the actual windows shown.** `thread.md:92`; `make_charts.py:176`; `package/README.md:92`; `site/peer-breadth.html:232`: “getting better, not decaying” is not monotonic—history annual returns go **6.586% → 4.432%** in 2021–22 and **5.236% → 4.949%** in 2023–24. The tested 4h settings lose **−11.591%/−1.783% since 2025**, but latest-90 returns are **+0.570%/+0.727%**, and 4h/70% earns **+2.957% in 2026**. **Fix:** state “2026 YTD is the strongest calendar-period return in this run” and name the two tested 4h expressions/windows; remove the family-wide “mechanism is a daily one.”

20. **2025 onward is not exempt from retrospective universe selection.** `package/README.md:105–106`; `site/peer-breadth.html:252`: **“2025 → numbers are unaffected”** is unsupported. The same August-2026-selected universe is applied to 2025; `package/research_engine/prepare_data.py:92–108` records the selection period. Neither zero effect in recent years nor the direction of older-year bias was measured. **Fix:** say all windows use the fixed 2026 universe and historical membership was not reconstructed.

21. **“2026 only” uses a different trade cohort from 2026 headline statistics.** `build_page.py:363`; `site/peer-breadth.html:187`, `:329`: filter tests entry date, yielding **211 natural exits**, while YTD metrics select exit date and contain **225**. **Fix:** filter `eout >= '2026'`, or label it “entered in 2026.”

22. **README says later downloads extend a run already capped in code.** `package/README.md:114–115` says later runs include later data unless `END_DATE` is set. It is already fixed at **2026-09-01** in `package/02_peer_breadth_backtest.py:43`. **Fix:** state that downloads may extend the CSVs but reproductions remain capped until the user changes `END_DATE`.

## NOTE

1. **Daily approximation versus bug.** `package/02_peer_breadth_backtest.py:98–100`, `:170`: daily OHLC stop-before-target ordering, daily-close equity and a disclosed entry-day target ban can be accepted conservative approximations. They cannot explain missing symbol slots, sequential cap allocation, unavailable information at order time, the extra day's target ban or delayed cash charges. Those require the blocker fixes.

2. **Funding sign and 14-day count pass.** `package/02_peer_breadth_backtest.py:115–117`, `:132–136`: positive rates debit longs and credit shorts, matching research. Sign probe: **+$2.50 paid long / −$2.50 paid short**. Multiplying funding by entry notional rather than each settlement's marked notional is an acceptable *disclosed* daily approximation. The **14-day elapsed-time count itself is correct**; clock/funding differences are listed above. Every research trade's funding independently reconciled against archived event prices/rates, maximum error **7.2e-15 dollars**.

3. **Missing-data and cutoff edge cases remain in `02`.** `package/02_peer_breadth_backtest.py:66–68`, `:84–89`, `:127–128`, `:144`: endpoint-only five-day changes and a one-time 25-bar warmup do not enforce the research generator's complete consecutive windows after gaps (`package/research_engine/signals_trend.py:200–214`, `:397`). Final-day positions are liquidated before new admissions, so a final-day entrant can remain outside the exported trade ledger. No final-day entrant occurs at this frozen cutoff. **Fix:** use a complete UTC grid/rolling eligibility and finalize the account after all last-day actions.

4. **Quant sample/moment labels need precision.** `quant_metrics.json:2`, `:8–9`; `make_charts.py:205`, `:211`; `package/README.md:85`: **607** is the count of close-to-close returns from **608 daily marks/calendar days**, not 607 days of account coverage. History has **2,434 returns / 2,435 marks**. Skew **3.407583** and kurtosis **37.249770** reproduce as biased empirical moments with **Pearson**, not excess, kurtosis. CAGR **9.758903%** uses 365.25/607; Calmar **4.190393** divides by **daily** DD, not minute DD. **Fix:** name the sample, Pearson convention and annualization bases; do not imply Calmar is multiplied by √365.

## Drawdown convention and complete figure map

**Use one primary convention:** marked equity, **minute closes**, peak-relative percentage loss; name **account, window, cost and resolution**. Initialize each window's running peak with equity immediately before its start. End all labels at Aug 31, 2026. Daily-close and shuffled-trade diagnostics must carry their own explicit labels.

Suggested shared thread/hero wording: **“Fresh $100K account, Jan 2025–Aug 2026: max drawdown −2.83% (minute marks, 20 bp + funding).”** Historical comparison: **“Continuous $100K account, Jan 2020–Aug 2026: −4.64% (minute marks).”** On the hero, identify the upper statistics as fresh-2025 and the curve as continuous-2020. If space requires one decimal, use **−2.8% / −4.6% with exactly those labels**. Neither number replaces the other.

### Main research and package figures

All figures below are independently recomputed. Package figures validate saved arithmetic only; its simulator still has the blockers above.

| Account/model | Window | Minute DD | Daily-close DD | Where used / necessary label |
|---|---|---:|---:|---|
| Fresh-2025 research, 20 bp | Jan 2025–Aug 2026 | −2.831045% | −2.328875% | `thread.md:22,74`; hero secondary value `make_charts.py:88`; page `:159,269`; README `:56,83`. Quant JSON `:7` / quant graphic `:210` / page Calmar use the **daily −2.33/−2.3%** instead. |
| Fresh-2025 research, 20 bp | 2026 YTD | −2.404718% | −1.841220% | README `:57`; page `:245`; reference fresh-20 JSON `:37`: **−2.40%**, sometimes rounded **−2.4%**. |
| Fresh-2025 research, 20 bp | Latest 90 days | −1.891761% | −1.654288% | README `:58`; page `:245`; reference fresh-20 JSON `:70`: **−1.89% / −1.9%**. |
| Continuous-2020 research, 20 bp | Full history | −4.639332% | −4.372378% | `thread.md:78`; hero primary value `make_charts.py:88`; page `:269`; README `:89`; reference history JSON `:4`: **−4.64% / −4.6%**. Daily underwater curve should bottom at **−4.37%**, not −4.995170 pp. |
| Continuous-2020 research, 20 bp | Jan 2025–Aug 2026 | −2.831045% | −2.328875% | Reference history JSON `:37`; page block row `:241` is **−2.33% daily**, despite resembling the fresh-account number. |
| Continuous-2020 research, 20 bp | 2026 YTD | −2.401777% | −1.839884% | Reference history JSON `:70`; page year data `:267` is **−1.84% daily**. |
| Continuous-2020 research, 20 bp | Latest 90 days | −1.891761% | −1.654288% | Reference history JSON `:103`; account differs despite equal DD to displayed precision. |
| Continuous-2020 **package**, 20 bp | Full history | Not produced | −6.319% | `package/out/stats.json:65`. |
| Continuous-2020 **package**, 20 bp | Jan 2025–Aug 2026 | Not produced | −2.777% | `package/out/stats.json:11`; README `:56` / page `:245`: **−2.78%**, unrelated to the fresh research daily −2.33%. |
| Continuous-2020 **package**, 20 bp | 2026 YTD | Not produced | −2.006% | `package/out/stats.json:29`; README `:57` / page `:245`: **−2.01%**. |
| Continuous-2020 **package**, 20 bp | Latest 90 days | Not produced | −1.400% | `package/out/stats.json:47`; README `:58` / page `:245`: **−1.40%**. |

The fresh-2025 minute-DD peak/trough is **2025-10-10 21:19 → 2025-12-01 15:44 UTC**. Full-history minute-DD peak/trough is **2022-08-14 02:44 → 2022-11-09 06:50 UTC**. YTD uses **2026-02-06 00:19 → Apr 29 18:31**; latest 90 days uses **2026-06-17 02:23 → Aug 16 22:40**. These confirm which window produced each headline.

### Cost scenarios: fresh-2025 research, minute marks

`package/reference/metrics-fresh2025-{12,20,30}bp.json:4,37,70`; page cost table `site/peer-breadth.html:213` shows the first DD column below, without an explicit DD window/resolution. Add **“2025 onward, minute DD”** to that column.

| Round-trip cost + funding | Jan 2025–Aug 2026 | 2026 YTD | Latest 90 days |
|---|---:|---:|---:|
| 12 bp | −2.825947% → −2.83% | −2.371266% | −1.866723% |
| 20 bp | −2.831045% → −2.83% | −2.404718% | −1.891761% |
| 30 bp | −2.837358% → −2.84% | −2.445753% | −1.922198% |

### Neighborhood graphics/table: fresh-2025 research, Jan 2025–Aug 2026, 20 bp, minute marks

`build_page.py:126–134`; `site/peer-breadth.html:233`. These DDs are correct but the column should explicitly state the account window/resolution.

| Expression | Recomputed DD | Displayed |
|---|---:|---:|
| Daily, 65% | −2.083625% | −2.08% |
| Daily, 70% | −2.831045% | −2.83% |
| Daily, 75% | −2.216963% | −2.22% |
| 4h, 65% | −18.249167% | −18.25% |
| 4h, 70% | −10.620659% | −10.62% |

### Yearly and block figures: continuous-2020 research, 20 bp

Page year data `site/peer-breadth.html:267` / generator `build_page.py:70`; block table page `:241` / generator `:83`. Year values are **wrongly labeled minute marks**; block values lack resolution. Both currently equal the daily-close columns.

| Calendar window | Published daily-close DD | Correct minute DD |
|---|---:|---:|
| 2020 | −2.411098% → −2.41% | −2.595528% → −2.60% |
| 2021 | −2.768354% → −2.77% | −3.060662% → −3.06% |
| 2022 | −4.372378% → −4.37% | −4.639332% → −4.64% |
| 2023 | −1.819798% → −1.82% | −2.387170% → −2.39% |
| 2024 | −2.622680% → −2.62% | −2.916544% → −2.92% |
| 2025 | −2.328875% → −2.33% | −2.831045% → −2.83% |
| 2026 YTD | −1.839884% → −1.84% | −2.401777% → −2.40% |
| 2020–2021 block | −2.768354% → −2.77% | −3.060662% → −3.06% |
| 2022–2023 block | −4.372378% → −4.37% | −4.639332% → −4.64% |
| 2024 block | −2.622680% → −2.62% | −2.916544% → −2.92% |
| Jan 2025–Aug 2026 block | −2.328875% → −2.33% | −2.831045% → −2.83% |

**Monte Carlo is a separate resolution:** the JSON's **−$853 / −$1,306 / −$2,169** are proposed shuffled **natural-exit net-dollar P&L** drawdowns for the fresh-2025 trade cohort, not daily/minute account drawdowns. The card/page display positive loss magnitudes $853/$1,306. Exact draws remain unverified without the seed/generator; do not compare them directly with −2.83%.

## Verified correct

| Research account/window, 20 bp + funding | Natural exits | Win rate | PF | Marked return |
|---|---:|---:|---:|---:|
| Fresh-2025, Jan 2025–Aug 2026 | 468 | 54.4872% | 2.138187 | +16.736277% |
| Fresh-2025, 2026 YTD | 225 | 53.3333% | 2.667444 | +10.477295% |
| Fresh-2025, latest 90 days | 80 | 58.7500% | 3.450455 | +3.928256% |
| Continuous-2020, full history | 1,403 | 51.3899% | 1.663372 | +41.516660% |
| Continuous-2020, Jan 2025–Aug 2026 | 477 | 53.8784% | 2.056676 | +16.031313% |
| Continuous-2020, 2026 YTD | 225 | 52.8889% | 2.654695 | +10.341466% |
| Continuous-2020, latest 90 days | 79 | 58.2278% | 3.335006 | +3.843543% |

- **Account/export integrity:** all frozen manifest artifact hashes match; reference config/metrics/QC/signals/trades/daily equity/exposures match their sources. Fresh/history exports contain **477/1,412 trade rows** and **1,313/3,496 raw signals**. CSV trade differences are only floating serialization (<1.5e-11); website trade export matches. Account ending equity equals initial cash plus all ledger net P&L within $1e-9. ZIP files match the package files it contains.
- **Package rerun:** all four `out/` files reproduce **byte for byte**. `03` prints 1,313 matching keys and package return/PF/DD of **13.334%/1.779/−2.777%, 9.253%/2.388/−2.006%, 5.620%/5.022/−1.400%**. Its continuous full-history output is **30.353%, 924 natural exits, 49.03% wins, PF 1.411, DD −6.319%**. These are confirmed outputs, not engine parity.
- **Natural trade statistics:** 255 wins/213 losses; net **$16,388.059469**, average **$35.017221**, t **5.748260**, two-sided p **1.629927e-8**; exact Clopper–Pearson 95% win CI **49.852387–59.064927%**. Long/short **281/187**, net **$7,554.67/$8,833.39**, PF **1.894862/2.483079**; streaks **25 wins/22 losses** in recorded exit order. Top 10 contribute **14.7456% of positive-trade profit / 27.7010% of natural net**. Drop-best 0/1/5/10/25/50 = **$16,388/$15,614/$13,552/$11,848/$7,538/$1,325**. Nine cutoff rows add **$348.217226** to marked account P&L.
- **Annual/coin receipts:** all displayed year counts/wins/PFs/nets/returns and all 120 per-coin table cells reconcile. History returns 2020–2026 YTD: **−0.7906%, +6.5862%, +4.4324%, +5.2360%, +4.9492%, +5.1566%, +10.3415%**; **6 of 7 positive** and 2022 profitable. Fresh 2025 alone is **+5.6654%**, a different account. **18 of 20 coins** are profitable since 2025: NEAR **−$275.85**, TRX **−$62.86**; ZEC leads **+$1,936.77**. NEAR is the only negative coin in all three published windows. Calendar coin nets sum to **$16,736.276695 / $11,070.876205 / $4,412.370885**; they include window carry marks, whereas coin n/win/PF use natural exit cohorts.
- **Other validated figures:** fresh time in market **94.243193%** and max gross **21.801139%** are minute-based; full-history time in market is **80.992015%**, not 94%. Sharpe **1.77** fresh / **1.28** history, 105 consecutive underwater daily closes, CAGR/Calmar and empirical moments as qualified above reproduce. All page cost/neighborhood returns/counts/wins/PFs reconcile; daily 65%/75% returns are **+16.857795%/+17.471746%**. The 5-day/70% directional transition, 16-peer eligibility, SMA ATR(14), 2.5 multiplier, 0.5% floor/>20% skip, 3R price target, 14-day cap and configured 0.25%/1.5%/2x budgets are confirmed. Observed gross exposure stays below 1x in both research accounts.

**Publication verdict:** keep the research candidate, but do not publish these artifacts unchanged. Repair the package's chronology, admissions, sizing, targets and cash/funding bookkeeping; make the promised minute reproduction runnable; compare matching account starts and test stop-distance agreement. Correct the drawdown labels/curve, Sortino, README drop-best amounts, activity/sizing/holding-period descriptions and unsupported inventory/statistical claims, then regenerate the graphics, page, README outputs and ZIP. The favorable recent research results—**+16.736% since 2025 and +10.477% in 2026 YTD, with −2.831%/−2.405% minute drawdown**—are independently verified; this audit does not reject the strategy family.
