# 01 — Review of the 11 parallel agent submissions

All 11 agents merged into `main` on 2026-05-10. I read every report and
result file produced by each agent and reproduced the headline number for
every candidate I planned to take further.

## Agent inventory

| # | Agent (PR / branch) | Identifier | Claim |
|---|---|---|---|
| 1 | PR #155 `zc4cv` v4 | "v3 near-optimal" | No improvement found |
| 2 | PR #156 `zc4cv` v5/v6 novel | "v3 stays deployed" | Pattern-match, vertical-classifier, proprietary GBM all worse |
| 3 | PR #168 `zc4cv` v5 Chronos | **+5.04pp full CAGR / 10-10 beats SPY** | HF foundation-model filter (Option C) |
| 4 | PR #157 `G0nfM` v6 invvol | **Pareto Sharpe/MaxDD lift** | invvol weighting (Option A) |
| 5 | PR #157 `G0nfM` v7 hedges | MaxDD −29% vs −50% | Daily stops + SH hedge + TLT (different product) |
| 6 | PR #158 `uDXqh` v8 | **+3.28pp WF mean, 10-10 beats SPY** | kb=2 bull-regime concentration (Option B) |
| 7 | PR #160 `2qHxY` v8 k=1+TLT | WF mean 50.16% **but author says "don't deploy"** | k=1 fragile to delisting; -32% in 2025 |
| 8 | PR #159 `h2JKH` v8b leveraged | "≫2× v3 WF CAGR" | Leverage 1.5×–3×, MDD −76% — different product |
| 9 | PR #161 `FHtzX` Pre-Runner CRT | Full CAGR 42.30% | WF mean OOS only 27.55% — **lower than baseline** |
| 10 | PR #162 `43Agh` multi-pillar | All 4 pillars + composite | Author concludes "do not deploy" |
| 11 | PR #163-#167 `YLOka` 5 sessions | 80+ experiments | "v3 conclusively local optimum" |
|   | PR #165 `O0MtP` ETF universes | v3 on ETFs | "do not generalise to ETFs in production" |

## Skeptical first-pass verdict

**6 of 11 agents explicitly concluded v3 is near-optimal on its home universe.** That's the single strongest finding of the day. The price-only PIT S&P 500 search space has been explored ~600 variants across v4-v8 and the marginal alpha is largely exhausted.

Of the 5 that proposed lifts, three were dropped immediately:

- **v7 hedge stack** (`G0nfM`): explicitly a risk-product trade-off, not a CAGR improvement. Gives up 8.6pp CAGR for 17pp lower MaxDD. A separate product, not a v3 replacement.
- **v8 k=1 + TLT** (`2qHxY`): the agent's own report says *do not deploy* — the k=1 strategy collapses to −100% CAGR at α=4%/yr delisting in their own MC. The 2025 holdout was −32% vs SPY +16%. Don't take a lift that the agent that produced it disclaims.
- **v8b leveraged** (`h2JKH`): the "CAGR ×2.5" claim is just gross leverage. MaxDD goes to −76%. The agent also admits the leverage hyperparameters were chosen on the same WF splits used to evaluate them — explicit overfit.
- **FHtzX Pre-Runner**: WF mean OOS 27.55% < baseline 33.29%. The full-period 42.30% CAGR comes from a few outlier splits.
- **multi-pillar 43Agh**: every pillar reduces CAGR. The agent's own composite is worse than v3.

That leaves three real candidates for deep validation:

1. **Option A** = `G0nfM` v6 invvol + 3% cash yield
2. **Option B** = `uDXqh` v8 kb=2 bull-regime concentration + invvol
3. **Option C** = `zc4cv` v5 Chronos-bolt-tiny p70 quantile filter at q=0.4

These three each made a Pareto-improvement claim on v3 *with* a published mechanism and *without* leverage. They are the only candidates worth the full validation pipeline.

## What carries forward

- **Option A** was deeply tested by `G0nfM`'s agent across 8 universes and showed 8/8 Sharpe/MaxDD improvement vs v3 — strong cross-universe generalisation claim.
- **Option B** showed +3.28pp WF mean lift on home but the agent acknowledged the kb=2 was chosen from 45 (kn, kr, kb) combinations evaluated on the same data being reported. **Modest sweep-overfit risk** flagged by the agent.
- **Option C** showed +5.04pp full CAGR but the q=0.4 was picked from a 30+ chronos variant sweep on the same data. **Bigger sweep-overfit risk** + a temporal-leakage concern (Chronos was released in 2024 by Amazon, trained on a public corpus that *might* overlap part of the backtest window).

The next document defines the formal v3 baseline and the three options precisely.

The full validation methodology is in `03_methodology.md` and results in `04_*` through `09_*`.
