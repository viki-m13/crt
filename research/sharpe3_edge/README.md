# IPD research: executable portfolio tests

Start with RESULTS.md. **No Sharpe3 result or new reliable edge has been established.** This directory implements and actually runs the information-persistence/temporary-pressure hypothesis, subsequent mechanisms, controls, cash/share ledgers, sensitivity grid and independent accounting checks. Research only; no live orders or production endpoint.

## Reproduce

Python 3.13, about 2–3GB peak RAM for one S&P process. Run universes sequentially on small machines. From repository root:

```bash
python -m pip install -r research/sharpe3_edge/requirements.txt
python -m pytest tests/test_ipd.py -q
python -m research.sharpe3_edge.download --output pinned-inputs

python -m research.sharpe3_edge.run --inputs pinned-inputs --output output/A/sp500 --universe sp500
python -m research.sharpe3_edge.run_extensions --inputs pinned-inputs --base output/A/sp500 --output output/B/sp500 --universe sp500
python -m research.sharpe3_edge.validate --inputs pinned-inputs --directory output/A/sp500 --universe sp500
python -m research.sharpe3_edge.validate --inputs pinned-inputs --directory output/B/sp500 --universe sp500

python -m research.sharpe3_edge.run --inputs pinned-inputs --output output/A/ndx --universe ndx
python -m research.sharpe3_edge.run_extensions --inputs pinned-inputs --base output/A/ndx --output output/B/ndx --universe ndx
python -m research.sharpe3_edge.validate --inputs pinned-inputs --directory output/A/ndx --universe ndx
python -m research.sharpe3_edge.validate --inputs pinned-inputs --directory output/B/ndx --universe ndx
```

The downloader uses immutable public CRT commit paths and verifies byte count and SHA-256. No API secret or private repository is needed. All eight files are listed in input_manifest.json. Existing exact-hash copies are reused. `--quick` omits A's sensitivity grid and must NOT be represented as the full experiment.

The workflow `.github/workflows/ipd-sharpe-validation.yml` runs both A and B plus independent daily-NAV audits and preserves all CSV/JSON/log outputs, excluding feature pickles. It runs only when its own workflow path is pushed on the research branch, or manually dispatched, avoiding gratuitous reruns for report edits. Actions artifacts expire after 30 days; the delivered user ZIP preserves the local ledger outputs and audit comparisons.

## File map

- PROTOCOL.md: original economic hypothesis and Sharpe3 objective.
- EXPERIMENT_A.md: numerical state, horizon, portfolio and control definitions committed before A outcomes.
- EXPERIMENT_B.md: continued invention registered after A Nasdaq results, before B calculations. This is openly sequential research, not an untouched holdout.
- data.py: input verification, positive/sorted data checks, exchange sessions, historical membership and feed cutoff.
- signals.py: lagged-beta residuals, deterministic information/transient scores, completed-past payoff clocks, rankings and fixed controls.
- portfolio.py: actual shares, cash, costs, borrowing assumptions for optional hedge, minimum30-session locked lots, writeoffs and cash/NAV reconciliation. No short proceeds are reused to buy stocks.
- extensions.py: six B mechanisms. The optional work-conserving scheduler uses idle cash but never prematurely sells a lot.
- metrics.py: all-day daily Sharpe, HAC, annual returns, CAGR, drawdown, exposure and conditional block uncertainty.
- run.py / run_extensions.py: full fixed experiment grids and future-data attacks.
- validate.py: independently reconstructs daily NAV from original price paths, quantities, fees and locked endpoints, plus membership/entry/Sharpe checks.
- tests/test_ipd.py: 40 invariant/accounting/regression cases, including intentional corruption.

## Reading outputs

Every run writes summary.csv, eras.csv, all_metrics.json, uncertainty.json and metadata.json. Each case has daily.csv, trades.csv and orders.csv. A additionally preserves mechanism_edge.csv, horizon_fits.csv, horizon_maps.json, data coverage and feature cache. Independent audits write independent_ledger_audit.json. Do not sum trades across strategy variants as independent evidence. All-member controls account for many records and intentionally repeat constituents in separate sleeves.

`sharpe_zero` is annualized mean daily portfolio return divided by daily standard deviation. `sharpe_cash_hurdle3` uses an ASSUMED constant3% opportunity hurdle while simulated cash earns zero. It is not historical risk-free Sharpe. Zero-variance results are undefined. All non-trading/cash days remain in the sample. Partial 2026 is identified by the input cutoff. Subperiods contain the existing continuing holdings, not newly reset books.

Base costs are25bp per side. Actual commission/spread/impact and terminal distributions are unavailable. Hedge rows assume1% annual SPY borrow, fixed entry beta and half cash deployment; they are diagnostic, not executable brokerage validation. At first missing held close, the primary writes the name to zero permanently while retaining its locked deadline; missing_grace5 is only a disclosed sensitivity. A halted/acquired stock is not automatically economically worthless, so this is a conservative scenario rather than a complete corporate-action accounting system.

## Honesty boundary

Historical member data are not complete or certified survivorship-free; raw prices are adjusted/mixed. This is not the 10,991-stock dataset in other research. No current prices or real SEC/informed-trading labels were ingested. Score complements are not independent models. No separately supervised ambiguity classifier, synchronized v3/v6 comparison or tradable sign-flipped short control was run. The implementation cannot infer that its economic labels represent actual market causes.

All tests use previously researched archives; a preregistered new implementation does NOT create a virgin historical holdout. Runtime reproducibility is not new market evidence. Main and production are unchanged. See CI_VERIFICATION.md for the actual clean-run completion and numerical comparisons.
