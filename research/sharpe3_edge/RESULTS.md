# IPD Sharpe-3 research — executed results

**Sharpe >=3 was NOT achieved. No tested full-period case reached Sharpe 1.** This is now implemented, executed portfolio research, not another proposal. Main and production are unchanged.

The run grid contains **185 share/cash simulations**: experiment A 42 S&P +43 Nasdaq; experiment B 50 per universe. Two SPY benchmark calculations are additional. Controls, cost/delay sensitivities and universes overlap: this is not 185 independent strategies or independent evidence. Frozen numerical source for clean reproduction: `1ca29167b786751a7685342885315a011ec5b4ab`.

## Main results

Base execution: signal at close t, enter close t+1, hold 30–252 NYSE sessions AFTER entry, immutable exit, daily mark-to-market, no long-portfolio borrowing. **25bp (0.25%) cost on EACH purchase and sale.** Cash earns zero. Sharpe uses annualized daily portfolio returns including all cash/no-trade days. Figures below use a zero hurdle; the machine results additionally subtract an explicitly assumed 3% cash hurdle, not historical T-bill returns.

| Universe / strategy | Net Sharpe | CAGR | Maximum drawdown | Average stock exposure |
|---|---:|---:|---:|---:|
| S&P adaptive IPD (primary) | **-0.447** | -0.29% | -4.05% | 0.43% |
| S&P fixed60 IPD | 0.549 | 6.07% | -16.83% | 66.75% |
| S&P ordinary residual momentum60 | 0.756 | 8.82% | -17.23% | 66.78% |
| S&P all-member control60 | 0.545 | 6.35% | -18.57% | 66.79% |
| SPY over S&P evaluation dates | 0.860 | 13.94% | -33.72% | 100% |
| Nasdaq adaptive IPD (primary) | **0.282** | 1.34% | -10.89% | 10.23% |
| Nasdaq fixed60 IPD | 0.761 | 10.53% | -16.33% | 62.46% |
| Nasdaq volume-confirmed fixed60 | **0.791** | 10.16% | -14.97% | 59.17% |
| Nasdaq ordinary residual momentum60 | 0.518 | 6.96% | -15.99% | 66.34% |
| Nasdaq all-member control60 | 0.648 | 9.65% | -20.59% | 66.35% |
| SPY over Nasdaq evaluation dates | 0.799 | 14.46% | -33.72% | 100% |

The best new ordinary base case was Nasdaq volume-confirmed IPD, Sharpe 0.791 versus SPY 0.799. Its Sharpe after the assumed 3% hurdle is **0.570**, not 3. Lower exposure and drawdown are not proof of information alpha.

The highest descriptive value anywhere in all 185 simulations was **0.919**, Nasdaq fixed60 with a five-session entry delay and 25bp one-way costs. This is a post-result maximum over predefined tests, not a chosen deployable winner. Zero-cost cases also failed to approach 3.

## Did the proposed edge exist?

The economic hypothesis was continuation after persistent information repricing versus reversal after temporary idiosyncratic selling. The operationalization is explicit price-state scoring plus learned payoff horizons, **not** observed informed trading or actual liquidity-flow labels. Its information and transient scores are complementary, not statistically independent classifiers. Score separation is a deterministic ambiguity proxy, not a separately supervised probability model. This rejects the tested implementation, not all possible versions of the economic hypothesis.

At 60 sessions, mean date-balanced candidate return minus the exact same-origin eligible-stock mean was:

| Universe | State | Candidate rows | Issue dates | Horizon buckets | Mean paired excess | Bucket SE |
|---|---|---:|---:|---:|---:|---:|
| S&P | information | 63,056 | 653 | 55 | -0.02% | 0.22% |
| S&P | transient | 609 | 290 | 53 | +0.86% | 0.65% |
| Nasdaq | information | 7,934 | 407 | 35 | +0.64% | 0.54% |
| Nasdaq | transient | 67 | 51 | 27 | +1.99% | 2.99% |

These are pre-cost mechanism diagnostics across ALL candidates, not a top-five trading portfolio. Buckets do not imply independence or eliminate overlapping outcomes. The continuation differential was negligible in S&P and small in Nasdaq; reversal estimates were sparse/noisy. No reliable new edge was established.

The primary horizon rule required positive, shrunk, past-only state excess after a 50bp roundtrip allowance. It mostly abstained. S&P selected no state in 2013–2025; transient/180 first passed in 2026, producing **19 lots, all still open** at the archive end. Its negative full-period Sharpe comes from one short active period amid years of cash, not 13 years of independent closed trades. Nasdaq selected information/90 in 2019 and transient/90 in 2026, 74 lots/70 closed. The strict Nasdaq variant never traded; its Sharpe is undefined.

A second finding was score-scale dominance: fixed60 IPD and information-only produced identical trades in BOTH universes, because the ranker crowded out rare transient setups. Experiment B reserved separate opportunities to test whether retaining both states helps.

## Experiment B — continued invention after A

B was registered after the first A Nasdaq results were observed, before calculating B outcomes. No claim of an untouched research holdout. New mechanisms: reserved information/transient clocks, asymmetric market-response absorption, breadth release, residual-rank rotation, residual convexity and alternating absorption/convexity. Exact definitions are in EXPERIMENT_B.md and extensions.py.

The B scheduler assigns the first free cash sleeve at each five-session signal date instead of waiting for an expired sleeve's next 30-session slot. All B strategies AND controls get this same scheduler. It never shortens holds or borrows cash; it is an implementation comparison, not a new information edge.

| Base 25bp method | S&P Sharpe | S&P CAGR | S&P max DD | Nasdaq Sharpe | Nasdaq CAGR | Nasdaq max DD |
|---|---:|---:|---:|---:|---:|---:|
| Balanced information/transient60 | 0.661 | 7.62% | -30.03% | 0.730 | 11.21% | -23.33% |
| Absorption60 | 0.638 | 6.59% | -20.67% | 0.552 | 5.63% | -15.33% |
| Breadth release30 | 0.140 | 0.42% | -12.47% | 0.265 | 1.04% | -11.51% |
| Rank rotation60 | 0.056 | 0.03% | -30.88% | 0.414 | 3.50% | -17.82% |
| Residual convexity60 | 0.345 | 4.31% | -41.79% | 0.675 | 7.84% | -19.71% |
| Absorption/convexity barbell60 | 0.581 | 7.71% | -34.06% | 0.736 | 9.74% | -19.61% |
| Adaptive IPD, same scheduler | -0.449 | -0.29% | -4.08% | 0.303 | 1.61% | -14.63% |
| Fixed60 IPD, same scheduler | 0.415 | 5.55% | -41.14% | 0.721 | 12.70% | -28.20% |
| Residual momentum60, same scheduler | 0.569 | 7.95% | -29.96% | 0.564 | 9.66% | -28.31% |
| All-member60, same scheduler | 0.481 | 6.77% | -38.88% | 0.613 | 11.11% | -32.25% |

Balanced states improved S&P versus the B fixed60 control but not SPY. The alternate capital clock increased exposure and sometimes drawdown; it did not deliver Sharpe3. No B base, zero-cost, high-cost, delayed or hedge case reached Sharpe1.

## Costs, delays, missing prices and hedges

Original fixed60 IPD Sharpe:

| Case | S&P | Nasdaq |
|---|---:|---:|
| Zero transaction costs | 0.665 | 0.852 |
| 10bp per side | 0.619 | 0.816 |
| 25bp per side / next-close | 0.549 | 0.761 |
| 50bp per side | 0.433 | 0.669 |
| Two-session entry delay | 0.550 | 0.864 |
| Five-session entry delay | 0.631 | 0.919 |
| Half-capital beta-hedge diagnostic | -0.430 | 0.022 |
| Five-session missing-price grace | 0.550 | 0.792 |

Primary missing holdings are permanently written to zero immediately and remain locked until their original endpoint. Grace5 is an openly different assumption, not silent deletion. The hedge diagnostic uses fixed entry beta clipped to [0,1], SPY shorts with the same deadline, 25bp costs on both legs, assumed 1% annual borrow and no short-proceeds reuse/rebate. Actual borrow availability/recalls, margin liquidation, financing and dividend-adjustment handling are not verified. It is not a live execution claim.

## Temporal robustness and controls

Original fixed60 Sharpe in 2013–2017 / 2018–2021 / 2022 onward / 2024 onward: S&P **0.626 / 0.764 / 0.269 / 0.297**; Nasdaq unavailable / **0.758 / 0.763 / 0.971**. The 2024 segment is reused history, not a fresh untouched holdout. Annual tables include partial 2026 and disclose the source cutoff.

Circular 126-session block bootstrap (500 draws) produced conditional fixed60 Sharpe intervals **[0.171,0.958] S&P** and **[0.397,1.137] Nasdaq**. Nasdaq volume-confirmed: **[0.344,1.156]**; paired Sharpe difference vs SPY **[-0.541,0.451]**. These diagnostics are not multiplicity-adjusted and do not cure missing data or repeated historical research. HAC-21 statistics and leave-era-out deletion sensitivities are also saved; deletion is not training on a genuinely unseen era.

Five deterministic random-stock controls, pure residual momentum, reversal, low-vol momentum, all-member60, an inverted-ranking/noncandidate long control, and within-origin state permutation are preserved. The inverted-ranking control is NOT a true sign-flipped short portfolio. S&P state-null never traded; Nasdaq state-null Sharpe 0.538 used only six lots, five closed, so it is not reliable evidence either.

## Validation

**40 new regression tests passed locally.** Tests cover future prices/labels/membership, immutable endpoints, >=30-session holdings, cash conservation, fees, writeoffs, sparse/no-trade cash, ranking ties, scheduler behavior and independent reconstruction, including deliberate quantity corruption.

All 185 simulations internally reconcile final NAV to closed plus marked-open lot P&L. Independent code additionally reconstructed every daily NAV from original prices, quantities, fees and borrow in **79 base/hedged runs / 490,005 lot records**: A S&P 19/117,190; A Nasdaq 20/23,349; B S&P 20/303,002; B Nasdaq 20/46,464. Maximum absolute NAV difference was below **3e-13**. These are overlapping strategy/control records, mostly all-member baseline lots, NOT independent trades.

Actual archive feature prefixes were unchanged by corrupting future stocks, benchmark and membership. Future-label mutations could not affect past horizon estimates; actual primary NAV prefixes were also invariant to future-price corruption. Independent GitHub Actions run **34429860149** reproduces both universes/experiments with pinned dependencies and downloaded/hash-verified inputs. See CI_VERIFICATION.md for completed status and programmatic local-vs-CI comparison, rather than assuming that scheduling CI means it finished.

## Data, scope and novelty limits

S&P: **732 eligible historical tickers, 484,772 feature rows, 2013+ through 2026-03-20**. Nasdaq: **163 tickers, 51,072 rows, 2018+ through 2026-05-07**. Historical constituent prices and monthly membership are incomplete; universes overlap. This is not the other agent's 10,991-stock panel. Adjusted/mixed prices and unverified terminal corporate actions prevent a survivorship-free or completely executable claim.

No actual SEC/fundamental event ingestion or verified information/liquidity-cause classifier was built in this batch. No separately learned ambiguity network, overnight/intraday model, synchronized CRT v3/v6 comparison or executable sign-reversed short control was run. Volume was available only in Nasdaq. These are disclosed scope gaps, not claimed completed checks. A pandas `asof` attribute bug was fixed before S&P calculation; adding B's optional scheduler with the default disabled did not change A logic. No numerical thresholds were changed after each experiment's outcomes.

The economic concept has prior literature; this is a new project implementation, not a verified world-first. Registered code on repeatedly used historical data is NOT a virgin holdout. No prospective market evidence, current buy list, >95% directional guarantee, or Sharpe3 success is established. Full ledgers, every variant, sensitivities and logs are in the downloadable CI artifacts and user-delivered validation ZIP. Run README.md commands to reproduce.
