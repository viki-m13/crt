# Data-integrity finding & fix — 2026-05-16

## What was found

While running the meta-allocation experiment (novel-v8) I hit a
contradiction: two "v5" return streams disagreed about the recent era
(one said 2021–26 DCA lost at ~+12%/yr, another said +90%/yr).

Root cause: **`augmented/v5_winner_equity.csv` is stale/corrupted from
2023-04 onward.** The two streams are identical through 2023-03, then
that CSV's monthly returns inflate to an implausible ~85%/yr for a
2-stock S&P picker (e.g. 2023-04 +29% vs the real −7%, 2023-05 +40% vs
+5%). Its full-history CAGR still looks ~40% only because the inflation
is recent and full-history is dominated by the shared pre-2023 path.

## What was and wasn't affected

- **The live website is NOT affected.** `build_webapp_v5_pit.py`
  builds the deployed `data.json` (equity curve, dca_investor block,
  by-era table, etc.) from the canonical production sim
  (`run_full_sim` → `rets_log`), exported as
  `data.json:dca_investor.growth[*].r`. The site's honest finding —
  "2021–25 DCA underperformed S&P-DCA (+8.8% vs +17.7%)" — comes from
  this canonical stream and **stands, independently re-confirmed**.
- **Affected: standalone research scratch** that read the CSV via
  `dca_investor_eval.load_streams()` — novel-v6 baseline refs,
  novel-v7, the first novel-v8 run. Their *recent-era / short-window*
  numbers were inflated. Full-history and 10-year-rolling figures
  (≈295× money-in, 100% 10y win, ≈40% IRR) are essentially unchanged
  because the corruption is post-2023 and those metrics are dominated
  by the shared 2003–2022 path — verified after the fix.

## Fix

`dca_investor_eval.load_streams()` now reads the **canonical** v5/SPY
monthly returns from the deployed `data.json`
(`dca_investor.growth[*].r/s`) instead of the corrupted CSV, so all
research == what the website actually shows. Re-ran
`dca_investor_eval.py` (committed JSON/CSV now canonical) and novel-v8.

`v5_winner_equity.csv` is left in place but should be treated as
**deprecated/untrusted post-2023-04**; regenerating it correctly from
the production sim is a separate cleanup (not required for the site,
which never uses it).

## Honest experiment result (novel-v8, canonical data)

Meta-allocation over the three validated streams (v5 / market-neutral
sleeve / S&P), all weights walk-forward, thresholds a-priori:

| Variant | CAGR | Sharpe | MaxDD | eras beating S&P-DCA |
|---|--:|--:|--:|--:|
| v5 only | 40.4% | 0.87 | −81% | 3/4 |
| trend v5→MN | 39.2% | 1.05 | −43% | 3/4 |
| inv-vol v5/MN | 23.2% | 1.12 | −36% | 3/4 |
| min-var {v5,MN,S&P} | 17.5% | 0.96 | −29% | 2/4 |

S&P-DCA era IRRs: +1 / +13 / +17 / +17%. **Every variant still loses
the 2021–26 era** (~+10–15% vs S&P-DCA +17%). Meta-allocation improves
drawdown/Sharpe (consistent with the earlier MN-switch result) but
does **not** manufacture recent-era alpha. The one-alpha ceiling holds
on clean data; "beats every period" remains unachievable honestly.
