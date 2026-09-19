# Return-floor and failure-memory stock selector

An executable research tool, not a validated 95%-accurate trading system.

It ranks stocks, selects a future endpoint, records the reference price and immutable deadline, and returns `NO_PICK` when the recommendation checks fail. The present archived datasets cannot authorize a current buy: their prices are stale, adjustment bases are not verified as price-only, and none of the complete policies has demonstrated >95% precision.

## What is new in this experiment

The return-floor selector combines pooled-horizon prediction with a downside quantile, median-direction veto, and a correction based on the worse of global and high-score-tail errors from matured earlier forecasts. It selects the shortest qualifying horizon per stock and ranks eligible stocks. The model refits annually; correction and evidence histories update at each decision.

The second, **failure-memory selector**, tests a different premise: a useful forecast need not have a model score of 0.95. It ranks each forecast by how seldom an equally strong score appeared among previous failed forecasts, conditions that comparison on market state, and applies a batch selection threshold across all stock/horizon comparisons before choosing at most one stock. Four versions were tested, not silently selected and presented as one confirmed strategy.

These are project-specific experimental combinations, not claims of worldwide novelty. Failure-memory is inspired by Jin and Candes, *Selection by Prediction with Conformal p-values*, arXiv:2210.01408. The financial implementation does **not** inherit that paper's error-control guarantee: temporal dependence, model changes and top-one postselection matter.

## Results

See `RESULTS.md`. Both methods were executed, not just specified. The strongest non-tiny new selective result was 20 positive endpoints out of 26 matured S&P calls (76.9%, exploratory). It fails the requested >95% objective. Zero strictly qualified recommendations is **not** a successful forecasting result.

## Run

Python 3.13 on Linux was used for validation. Run commands from the repository root:

```bash
python -m pip install -r research/floor_selector/requirements.txt
python -m unittest discover -s tests -p 'test_floor_selector.py' -v
python -m research.floor_selector.fetch_inputs --out /tmp/floor-inputs

# Historical replay; several minutes of CPU, not a live scan.
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m research.floor_selector.core \
  --inputs /tmp/floor-inputs --universe sp500 --out /tmp/floor-sp500
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m research.floor_selector.core \
  --inputs /tmp/floor-inputs --universe ndx --out /tmp/floor-ndx

# All four failure-memory variants, including unsuccessful variants.
python -m research.floor_selector.failure_rank \
  --raw /tmp/floor-sp500/raw_forecasts.parquet --out /tmp/failure-sp500
python -m research.floor_selector.failure_rank \
  --raw /tmp/floor-ndx/raw_forecasts.parquet --out /tmp/failure-ndx

# Latest AVAILABLE archive session, explicitly NOT today's prices.
python -m research.floor_selector.latest --inputs /tmp/floor-inputs \
  --universe sp500 --replay /tmp/floor-sp500 --out /tmp/floor-sp500/latest_scan.json
python -m research.floor_selector.view --scan /tmp/floor-sp500/latest_scan.json \
  --report /tmp/floor-sp500/report.json --out /tmp/floor-sp500/index.html
```

`--today YYYY-MM-DD` is available for reproducible historical status checks. It does not create current market data. The pinned-input downloader uses only already-public CRT files. Private repositories, brokerage accounts, orders and production deployment are not touched.

## Output contract

`scan.json` separates `recommendations` from `research_candidates_not_recommendations`. Only the latter can contain an unvalidated historical ranking. It includes ticker, reference date/value, horizon, endpoint date, raw model score and unvalidated checkpoint bands. Never turn that field into a buy alert. `latest.py` explicitly blocks treating a monthly-tested strategy as a validated daily picker.

`shadow_picks.csv` records policy, reference, deadline, endpoint success, pending/unresolved status, observed return and interim close drawdown. `strict_picks.csv` records only the evidence-screened adaptive policy. `report.json` includes coverage, precision, sample counts, horizons, per-year results, benchmark control, source hashes and code hashes. `research_ledger.jsonl` refuses changes to a previously recorded forecast identity.

Horizons are 30/60/90/126/180/252/504/756 **trading sessions**, up to approximately three years. Deadlines never move after issuance. No pick quota. A stock touching a higher price before its deadline does not count as endpoint success. Flat endpoints fail.

## Important limitations

Historical features use membership known at the decision, and training/calibration exclude immature endpoints. That does not make the source data complete. Historical membership coverage, failed-stock coverage, security identity and terminal corporate-action proceeds remain imperfect. Dividend-adjusted appreciation is not automatically price-only appreciation. No claim of a complete survivorship-free equity universe is made.

These periods have been researched before. Chronological model-out-of-sample forecasts are **not** a new untouched research holdout. S&P and Nasdaq universes overlap. Non-overlapping time thinning and binomial lower bounds are diagnostics, not proof of independent market trials. The displayed checkpoint fan is not a validated continuous daily path. Prospective evaluation and verified fresh price-only/corporate-action data remain necessary for a validated live recommendation tool.
