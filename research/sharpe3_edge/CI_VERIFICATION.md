# IPD clean-runtime verification — completed

GitHub Actions run **34429860149** succeeded for BOTH universes and BOTH experiments. Immutable executable source: **1ca29167b786751a7685342885315a011ec5b4ab**. S&P job102722860925; Nasdaq job102722860684.

Each clean Ubuntu24.04 / Python3.13 runner installed the pinned dependencies, passed **40 regression cases** (S&P16.82 seconds, Nasdaq15.50 seconds), downloaded and size/SHA-256-verified all eight public pinned inputs, ran the complete A and B portfolio grids and independently reconstructed their base/hedged daily ledgers. Thus all185 portfolio cases were rerun. The two extra SPY benchmark calculations were also reproduced.

## Actual artifact comparison

Both ZIP artifacts were downloaded and compared programmatically to the original local outputs. We did not infer equivalence just from a green workflow badge. Every CSV cell, every stock and locked entry/exit, every order/abstention record, every daily NAV and all summary/era/edge metrics agreed within1e-12 relative/absolute tolerance. Nested horizon maps, uncertainty and validation reports also agreed.

| Experiment/universe | Compared CSVs | Lot records | Order records | Daily NAV rows | Largest absolute numeric difference |
|---|---:|---:|---:|---:|---:|
| A / S&P | 132 | 129,609 | 140,365 | 139,608 | 2.00e-13 |
| A / Nasdaq | 135 | 31,408 | 37,907 | 90,214 | 5.60e-13 |
| B / S&P | 152 | 756,943 | 768,700 | 166,200 | 2.88e-13 |
| B / Nasdaq | 152 | 116,047 | 123,827 | 104,900 | 7.80e-13 |

These counts aggregate overlapping strategies, sensitivities and all-member controls. They are NOT independent investment observations. Total compared records:1,034,007 lots and500,922 daily-NAV rows. Most lot volume is in broad all-member controls.

A local metadata schema preceded the optional B scheduler: its absent `work_conserving` key was normalized to the existing False default (benchmark rows remain not-applicable). No price, order, holding, model score, NAV or measured performance changed. The final frozen source reproduced the earlier A results. Both formal experiment specifications preceded their respective outcome calculations; B is openly post-A research.

## Independent accounting versus repetition

Separately from local-vs-CI equality, validate.py rebuilt daily NAV directly from archived input prices, quantities, entry/exit dates, costs and borrow in79 base/hedged cases/490,005 lot records. Largest NAV discrepancy was below3e-13. All185 simulations also internally reconciled final NAV to closed and marked-open lot P&L. Future-price/label/benchmark/membership attacks and minimum30-session holding checks passed. Accounting correctness is not market predictability.

## Preserved artifacts

- S&P: artifact **10134311977**, `ipd-sharpe-sp500`, SHA-25677364a5e4307f4567a31a766d23721f0187b95dc309f965d11e5f269392a6d3b.
- Nasdaq: artifact **10134127358**, `ipd-sharpe-ndx`, SHA-2567d1798e8086e2716465befe575ced2da3bee594eccd8c92237d6f44d0637d60a.

Retention is30days. The user-delivered validation ZIP preserves the complete local ledgers, tests, source, all assumptions/results and numerical comparison reports without duplicating the already-matching CI CSVs or raw input price panels.

## Result unchanged

Primary adaptive IPD: S&P Sharpe-0.447, Nasdaq0.282. Best new ordinary25bp base case: Nasdaq volume-confirmed fixed60 at0.791, versus same-date SPY0.799. Highest descriptive full-period value anywhere in all185 cases:0.919, a delayed-entry sensitivity. **Sharpe3 and a reliable new edge were not established.**

Same-data runtime reproduction is not independent market evidence, an untouched holdout, globally unprecedented novelty or an executable/live performance claim. Incomplete historical constituents, adjusted/mixed data, terminal-action ambiguity and stale March/May archives remain material. Draft PR **#194** contains this research; main and production remain unchanged.
