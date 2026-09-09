# Disclosure-clock extension — registered before fetching or scoring new facts

The fixed price-based edge batch and its independent-price replication did not reach net Sharpe 3. Those results stay in the ledger. This extension adds a genuinely different information source, not another relabeling of price direction.

## Hypothesis: self-funded improvement not yet reflected in price

A newly filed improvement in operating profitability is more credible when growth is accompanied by cash generation, rather than accruals. If price has not already outperformed since the comparable earlier filing, a continuing analyst/investor update process may create drift over 30–60 sessions. Conversely, accounting growth with deteriorating cash conversion may be a weak signal. This is a testable information-underreaction hypothesis, not proof of an edge or globally novel prior art.

Use ONLY SEC companyfacts rows with filing dates and accession identifiers, never the latest-value quarterly frames currently stored in Bonds. At every event, filter all facts to filed<=event date and choose among those records only. Do not backfill later restatements or use an old report-period end as the availability date. First observation of a newly ended quarterly revenue/operating-income period forms an event; later amendments update past state only at their own availability date, never create duplicate first-report trades. Use next exchange session after filing as the signal date and enter one further session later; this is conservative to unknown intraday dissemination. Date-only timestamps do not prove original real-time API availability.

Quarterly operating income/revenue require durations 70–105 days, matching start/end dates, USD units and 10-K/10-Q forms. Compare with the quarter roughly one year earlier (end-date separation 330–400 days), with only available facts. Cash-flow validation uses the corresponding year-to-date period and its prior-year same-duration equivalent; absent or incomparable cash facts are missing, never assigned zero. Cash flow is NOT quarterly unless its duration is quarterly. Tag changes, custom extensions, current-ticker-to-CIK mapping and lost delisted issuers remain material coverage limitations.

## Four fixed strategies, no outcome tuning

1. profitability: positive change in operating margin and positive quarterly revenue growth; rank by within-issue-day sum of the two features' standardized signs/magnitudes.
2. cash_confirmed: same improvement, require positive comparable cash flow and improving operating-cash-flow/revenue ratio.
3. disclosure_gap: cash-confirmed improvement, but only when the preceding 63-session market-relative return is <=0. The proposed increment is the conjunction of credible financing and lack of prior price confirmation.
4. accrual_warning: long cash-confirmed improvements and short positive revenue-growth issuers whose operating cash-flow margin deteriorates and whose operating margin is not improving; long/short gross halves balanced when both sides exist, otherwise abstain.

No filing event later than the archived price cutoff. At each eligible event date, use the historical S&P membership snapshot, at least 252 prior valid stock/benchmark prices, and no future-price availability filters. Up to 10 names per event basket, no minimum quota. Equal weights within each side, max per-symbol event notional 10% of the cohort. Raw long-only and dollar-neutral versions are separate, never call the long-only return alpha without benchmark controls. Initial candidate holds30/60 sessions; one-session delayed execution and fixed share-unit lots; same 5bp/3% borrow/actual RF accounting as the existing batch. Signals expire, pending lots remain marked. Different event dates can overlap; full daily account returns define Sharpe.

For event-driven cash planning, allow a cohort each trading session but divide new notional by the hold in sessions, so the sum of planned gross cohorts is <=1 before market drift. Also report a fixed five-session batching cadence, where events received since the previous batch are frozen and invested at the next close. This ablation is registered before results.

Development: facts known through2015; validation2016–2020; test2021–March2026 already used for price research, not a virgin holdout. Select using validation only, report all 16 method/hold/cadence combinations and same-day matched eligible-stock controls. Conditional block-bootstrap, subperiods, next-close delay stress, trading-cost stress, and future-fact mutation tests are required. Document unresolvable missing issuers/periods. The current provider Financial Datasets returned a zero-credit balance; no paid credits are purchased. Public SEC retrieval is attempted with an identifying user agent and <=2 requests/second; any access block stops retrieval rather than bypassing it.

The primary question is whether the as-filed information increment improves a real self-financing stock portfolio, not whether any small selected cohort happened to rise. No current trading authorization, no >95% or Sharpe3 claim without evidence.
