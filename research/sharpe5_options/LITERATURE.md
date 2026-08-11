# What the literature says about 1000% CAGR

Research pass over the academic and practitioner literature to find whether a
1000%-CAGR strategy exists, and to extract anything testable that this project
had not already tried.

---

## 1. The decisive anchor: Medallion

Renaissance Technologies' Medallion fund is the best-documented high-return
fund in history. The published analysis:

| | |
|---|---|
| Gross annual return | **~66%** (1988–2018) |
| Net annual return | ~39% |
| Standard deviation | 31.7% |
| **Sharpe ratio** | **~2.0–2.72** |
| **Average leverage** | **~12.5×** |
| Unlevered returns | *"comparable to the S&P 500"* |

The last line matters most. **Medallion's raw edge, before leverage, is roughly
index-like.** The 66% comes from applying ~12.5× leverage to a Sharpe ~2.7
signal, and the fund is famously capacity-capped (~$10bn, employee-only)
precisely because that edge does not scale.

So the best quantitative fund ever documented, running arguably the strongest
signal research operation in finance, produces **66%/yr — not 1000%.** Any
claim of 1000% sustained is claiming to be an order of magnitude beyond
Medallion.

## 2. What Sharpe ratios actually exist

| Domain | Reported Sharpe | Source/context |
|---|---|---|
| Stat arb pre-2000 (MS BiSort/Spread) | 4–7 | compressed sharply after 2000 as anomalies were adopted |
| Medallion | ~2.0–2.72 | at 12.5× leverage |
| Quant fund internal hurdle | **> 2**, some **> 3** | strategies below are not considered |
| HFT vs non-HFT | ~2.5× higher than non-HFT | but capacity-bound |
| Market making in liquid names | ≈ market Sharpe | *"tight spreads"* prevent significant alpha |
| Published equity/option ML strategies | ~0.9–1.4 | typical recent academic results |
| Simulated HFT with perfect foresight | >5,000 | *"omniscient trader who is never at risk"* — meaningless |

The pattern: **Sharpe 4–7 existed, in stat arb, before 2000, and was
arbitraged away.** Current documented territory tops out near 2–3, and the
highest numbers in the literature come with explicit warnings that they
either assume omniscience or ignore costs.

## 3. What the literature says about costs — repeatedly

Three independent findings that match this project's own results exactly:

- *"Countless examples of trading strategies that have high Sharpes only to be
  reduced to low Sharpe strategies once realistic costs have been factored in."*
- **0DTE**: a positive variance risk premium exists, *"but at same-day horizons
  its economic magnitude is small and difficult to monetize after realistic
  frictions."*
- **Retail options traders are net losers in aggregate**, through *"bid-ask
  spread, time decay, and adverse selection by high-frequency market makers,
  with short-dated expirations showing the worst loss rates."*

This is the same wall measured here directly: the variance risk premium is
real (+2.54% of width) and the half-spread is larger (3.46%).

## 4. The one testable lead — and it failed

**Bali, Beckmeyer, Moerke & Weigert, "Option Return Predictability with Machine
Learning and Big Data", Review of Financial Studies 36(9), 2023.** 12M+
observations, 1996–2020. Headline claim: *allowing for nonlinearities*
significantly improves out-of-sample option-return prediction, and long-short
equity-option portfolios remain profitable after transaction costs.

This was worth testing because study 8 here fitted a **linear** model — exactly
the limitation that paper identifies. Study 26 replicates the recipe:
gradient-boosted trees on 19 characteristics, walk-forward, payoff-component
features excluded, market-implied benchmark scored alongside.

**Result — nonlinearity does not transfer:**

| model | pooled OOS IC | t | long-short Sharpe |
|---|---:|---:|---:|
| linear (ridge) | **+0.0443** | **+4.80** | −0.02 |
| nonlinear (GBM) | +0.0285 | +3.36 | −0.41 |
| market implied | +0.0252 | +2.27 | −0.99 |

Two honest observations:

1. **GBM underperforms ridge** (0.029 vs 0.044). Almost certainly sample size:
   the paper has 12M observations, this universe has 20,769. Gradient boosting
   needs their data density; on 20k rows it overfits.
2. **The linear IC is genuinely significant (t=+4.80) and beats the market's
   own implied measure** — and still nets to a Sharpe of −0.02, because the
   long leg must be bought at the ask. An earlier version of this test scored
   +0.66 using the negated short return as a long-leg proxy; that is the same
   error that made a sleeve here look like +2.32 before collapsing to −1.64.

## 5. Verdict on 1000% CAGR

Convex payoffs escape the `g = S²/2` bound, so 1000% *is* mathematically
reachable **with sufficient predictive power** — a 10:1 payoff hit 15% of the
time compounds to ~9,000%/yr at 250 independent bets. The arithmetic is not
the obstacle.

The obstacle is that **no such predictive power is documented anywhere in the
literature, at any horizon, in any liquid market.** What the literature
documents instead:

- the best fund ever runs **Sharpe ~2.7 at 12.5× leverage for 66%/yr**
- Sharpe 4–7 existed in stat arb **before 2000** and was competed away
- every high number is either pre-cost, pre-frictions, or assumes omniscience
- the specific families with the most retail enthusiasm (0DTE, short-dated
  options) are where the literature finds the *worst* realised outcomes

1000% CAGR requires Sharpe ≈ 2.19 at **full Kelly** — leverage so aggressive
that ~50% drawdowns are routine and a Sharpe overestimated by 2× turns optimal
sizing into ruin. Medallion, at Sharpe 2.7, deliberately runs ~0.068 of Kelly.
The gap between "the maths permits it" and "anyone has done it" is the entire
finding.

## 6. Where high Sharpe actually lives, per the literature

For completeness, the domains where >2 Sharpe is documented:

- **Capacity-constrained HFT / latency arbitrage** — real, but requires
  colocation, tick data and infrastructure, and the literature notes market
  making in *liquid* names earns roughly market Sharpe because spreads are
  tight. The money is in being the one who *earns* the spread.
- **Stat arb before crowding** — the MS BiSort 4–7 era, gone by 2000.
- **Multi-strategy diversification** — many uncorrelated ~1.0 sleeves. This is
  the only route available without new data, and it is what METHOD.md does:
  three sleeves, correlations −0.03 and 0.00, combining to ~1.0–1.13.

All three require either infrastructure this project does not have or a
signal-count this market does not offer.

---

**Sources**

- [Medallion Fund: The Ultimate Counterexample — Cornell Capital Group](https://www.cornell-capital.com/blog/2020/02/medallion-fund-the-ultimate-counterexample.html)
- [Famed Medallion Fund "Stretches Explanation to the Limit" — Institutional Investor](https://www.institutionalinvestor.com/article/2bswymr8cih3jeaslxc00/portfolio/famed-medallion-fund-stretches-explanation-to-the-limit-professor-claims)
- [Option Return Predictability with Machine Learning and Big Data — RFS 36(9) 2023](https://academic.oup.com/rfs/article/36/9/3548/7056660)
- [Bali, Beckmeyer, Moerke, Weigert — SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3895984)
- [Commodity Option Return Predictability — Journal of Futures Markets 2025](https://onlinelibrary.wiley.com/doi/10.1002/fut.22614)
- [0DTEs: Trading, Gamma Risk and Volatility Propagation — Dim, Eraker, Vilkov](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4692190)
- [0DTE Asset Pricing — Almeida, Freire, Hizmeri](https://www.fma.org/assets/docs/Derivatives2025/Almeida.pdf)
- [Sharpening Sharpe Ratios — Goetzmann et al., NBER w9116](https://www.nber.org/system/files/working_papers/w9116/w9116.pdf)
- [Empirical Limitations on High Frequency Trading Profitability — arXiv 1007.2593](https://arxiv.org/pdf/1007.2593)
- [Risk and Return in High Frequency Trading — CFTC](https://www.cftc.gov/sites/default/files/idc/groups/public/@economicanalysis/documents/file/oce_riskandreturn0414.pdf)
- [Sharpe Ratio for Algorithmic Trading — QuantStart](https://www.quantstart.com/articles/Sharpe-Ratio-for-Algorithmic-Trading-Performance-Measurement/)

---

## Appendix — live validation of the v3 recalibration

First nightly cron run under the v3 surface (2026-08-11), against live quotes:

| underlying | v2 booked/natural | **v3 booked/natural** |
|---|---:|---:|
| SPY | 1.413 | **1.024** |
| ^SPX | 1.582 | **1.123** |

The convex-skew fix holds on live market data, not only on the historical
chains it was fitted to. SPX remains ~12% optimistic and trips the 1.10 guard
added to the workflow.

Caveat worth recording: the model's individual leg IVs are still far above
market (0.225/0.281 vs 0.168/0.210) even though the *credit* now matches. For
a spread only the difference matters, so the strategy prices correctly, but
this surface should not be trusted for anything requiring absolute IV levels.

---

## Appendix 2 — documented 1000%+ EPISODES, and study 27

The target does appear in the published record — as *episodes*, never as a
compound annual rate. The two best-documented:

| case | return | window | mechanism |
|---|---|---|---|
| **Universa Investments** | **+4,144%** YTD (investor letter; +3,612% in March alone) | Q1 2020 | deep-OTM SPX puts into the COVID crash; ~$3bn gain on <$100M premium; monetised mid-crash |
| **Larry Williams, WCTC** | **+11,376%** ($10k → $2.1M peak → $1.14M net) | calendar 1987 | extreme futures leverage; **−46% drawdown from peak inside the winning year**; never repeated at scale |

Both are convexity events — the one mathematical route to 1000% (study 24).
The record then documents what the same instruments do BETWEEN events:

- **CBOE Eurekahedge Tail Risk HF Index: −3.23%/yr annualized 2008–2022**;
  cumulative **−60%** from 2011 to the eve of COVID. The professionals running
  the Universa trade, as an index, lost money over the full cycle.
- **AQR ("Chasing Your Own Tail (Risk)")**: insurance drag ~2.65%/yr for the
  tail-manager index; implied variance persistently exceeds realized — the same
  VRP this project measured at +2.54% of width. **The tail buyer is the
  counterparty paying the premium every seller here harvests.**

### Study 27 — the construction on real chains (2019–2026)

Monthly cohorts of far-OTM SPY puts at the real ask, window containing COVID,
2022 and the 2025 shock:

| depth | cost/spot | hold-to-expiry | 10×-trigger | omniscient exit* |
|---|---:|---:|---:|---:|
| 10% OTM | 0.672% | 0.31× | 0.31× | 0.82× |
| 15% OTM | 0.373% | 0.20× | 0.45× | 0.89× |
| 20% OTM | 0.299% | 0.00× | 0.00× | 0.57× |

*sell at the best bid ever printed — hindsight upper bound, not tradable.

**Even with a perfect-hindsight exit, every achievable depth returns <1× per
dollar of premium.** Best real cohort: 17.6× (Feb-2020). The pre-stated 10×
monetisation trigger rescues crash cohorts that expire worthless (Mar-2020
0×→11.8×; Apr-2025 0×→10.4×) and the average is still 0.45×. Portfolio level:
SPY alone 14.7%/yr; +3%/yr put budget → 12.4% with NO drawdown reduction; all
levered "Spitznagel" variants Sharpe-inferior to plain SPY. Universa's 100×+
multiples live in strikes/tenors/execution EOD retail-grade data cannot reach
(deepest reliably-quoted strikes here max out at 17.6×).

### Final synthesis

1000% **years** are documented and their anatomy is public: cheap convexity ×
tail event × monetisation at peak fear. 1000% **CAGR** is documented nowhere,
because the convexity that produces the episode costs a documented −3%/yr to
hold between episodes, and no documented method times the episodes better
than the price of the insurance. The two halves of the published record agree
with each other and with every measurement in this project.

**Additional sources**
- [Bloomberg — Taleb-advised Universa returned 3,600% in March 2020](https://www.bloomberg.com/news/articles/2020-04-08/taleb-advised-universa-tail-risk-fund-returned-3-600-in-march)
- [CNBC — Universa gains over 4,000% in Q1 2020](https://www.cnbc.com/video/2020/08/17/universa-hedge-fund-investments-gain-over-4000-percent-return.html)
- [Trading Greats — Larry Williams, 11,376% WCTC record](https://www.tradinggreats.com/traders/larry-williams)
- [World Cup Trading Championships](https://www.worldcupchampionships.com/)
- [AQR — Chasing Your Own Tail (Risk)](https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/AQR-Chasing-Your-Own-Tail-Risk.pdf)
- [Disciplined Systematic Global Macro — Eurekahedge Tail Risk index](http://mrzepczynski.blogspot.com/2020/10/cboe-eurekahedge-tail-risk-hedge-fund.html)
- [Forbes — Are tail risk funds worth the losses in good times?](https://www.forbes.com/sites/jacobwolinsky/2022/06/30/are-tail-risk-hedge-funds-worth-the-steep-losses-in-good-times-to-win-big-in-bad-times/)
