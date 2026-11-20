# Return-floor selector experiment — 2026-09-09

Status: a new experimental combination, not a claim of worldwide novelty or achieved accuracy.

Question: can the earliest conservative positive return-floor identify individually positive stock endpoints, rather than relying on an ensemble's average direction?

Frozen initial run: pooled-horizon LightGBM classification and 5th/50th/95th quantile regressions; a horizon is a feature, not chosen with knowledge of its outcome. Horizons 30/60/90/126/180/252/504/756 **trading sessions**. Horizons over 252 explicitly permit approximately two/three years; they do not shorten the requirement by waiting until a stock happens to win. Training uses normalized log returns, equal issue-date weights and a five-year recency half-life. Fits annually on a trailing 12 years, strictly purged by endpoint maturity. 100 trees / 15 leaves / minimum 100 rows per leaf / seed 20260909; no hyperparameter sweep.

New combination: downside quantile + disagreement penalty + conservative correction from global AND upper-score-tail matured forecast errors. The correction may widen but never narrow the lower band. The youngest admissible horizon with a positive floor is eligible; candidates are ranked by floor divided by square-root time. At most one per monthly decision; no repeat ticker until its original horizon expires. All forecasts keep their original reference and deadline.

Run and disclose raw-probability policies (0.80/0.90/0.95/0.975/0.99), a positive-return-floor policy, a positive-floor + p>=0.95 policy, and a forced rank-one baseline. These are exploratory diagnostic policies, not eight independently confirmed strategies. A shadow-policy evidence gate must judge the complete ranking/horizon/cooldown policy using ONLY matured earlier selected calls. A cutoff can be selected from the fixed policy family only using that history. No quotas and no counting abstentions as wins.

Evaluation: prequential forecasts from 2013, post-2019 report plus earlier-period diagnostics. ALL of this archive was already researched by prior agents. This is model-out-of-sample, not an untouched research holdout. The NDX archive is a secondary universe, NOT statistically independent from S&P. Model development is not driven by a final-holdout label.

Report missing matured endpoints separately and pessimistically as failures, pending separately, sample sizes, issue-date coverage, unique names, overlapping-period block uncertainty, median return, lower tail, benchmark controls and trajectory checkpoint errors. The output is a checkpoint uncertainty fan, not a validated continuous daily path or a guaranteed floor.

The strict recommendation layer requires measured precision >95%, a conservative lower-bound diagnostic >95%, adequate separate periods, exact target basis and complete source provenance. Statistical diagnostics are not an exchangeability guarantee for markets. The current archives have unverified adjusted-price bases and incomplete terminal corporate actions. They CANNOT authorize a live buy. A live scan defaults to NO_PICK on stale data, absent evidence, or unverified target basis. Do not call an empty output a forecasting success.

## Exploratory extension after the initial quantile-policy results

The first quantile/probability policies did not reach 95%. Test a DIFFERENT selector on the same prequential scores: rank candidates by how rarely comparable scores occurred among matured past failures, rather than requiring a raw model score >=.95. Use the conformal-selection-inspired rank (1 + number of past negative outcomes with score >= current score)/(N+1), separately by horizon. Test p_model versus q05 ranking and pooled versus bull/nonbull calibration: four declared variants, no parameter sweep. Correct the batch threshold by the full count of stock/horizon comparisons before choosing at most one name, with a locked shortest qualifying horizon and cooldown.

This is exploratory reuse of already-observed data. The financial adaptation DOES NOT inherit the original paper's FDR guarantee: changing fitted models, correlated market observations, temporal distribution shifts and top-one postselection are material departures. No claim of mathematically guaranteed 95% stock precision. Record every variant and all missing/pending outcomes. Reference: Jin and Candes, Selection by Prediction with Conformal p-values, arXiv:2210.01408.
