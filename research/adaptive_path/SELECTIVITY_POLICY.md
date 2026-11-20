# Precision-first selection policy

Added 2026-09-09 following the user's explicit approval of confidence thresholds and periods with no picks. This is a prospective research requirement, not a backtest result or an implemented production guarantee. It supplements the original experiment registration without relabeling previously inspected data as untouched.

## Objective

Identify a subset of stock-and-horizon forecasts with strictly greater than 95% verified endpoint-direction precision. Coverage is secondary. There is no minimum daily, weekly, or monthly pick quota. The system may return no qualifying stock for an extended period.

Success means the specified stock's price at its recorded future endpoint is strictly above its issue-time reference price. Minimum horizon remains 30 trading sessions. Flat outcomes fail. Merely touching a higher price before the endpoint is not success. A deadline selected at issuance is immutable; later updates are separate forecasts, never replacements for losing calls.

## Selection requirements for subsequent experiments

1. Learn or select a confidence cutoff using only prior forecasts whose complete outcomes were available before the decision. Cutoffs may differ by horizon or regime only where historical support permits. Unsupported subgroups abstain rather than borrow an unvalidated high-confidence label.
2. A high model score alone never authorizes a validated-confidence claim. Record both the model probability and the measured reliability of the selection policy, including uncertainty and sample support.
3. Consider the stock, horizon, threshold, ranking rule, and any fallback as one complete decision policy. Test that complete policy chronologically. Separately impressive per-horizon buckets do not validate adaptive horizon selection.
4. Once thresholds are learned, choose the shortest qualifying horizon for a stock. Rank qualifying candidates using a rule fixed before evaluation. An optional maximum number of picks is a cap, not a quota. Never emit the best available stock when every candidate fails the gate.
5. Permit stricter thresholds and abstention under poor historical reliability, model disagreement, stale data, unresolved corporate actions, or unfamiliar market states. Do not lower thresholds because no recent picks have appeared.
6. Freeze each experimental policy before its evaluation; log all tested variants. Threshold searches, regime partitions, and horizon searches must be accounted for when interpreting results. Further invention remains allowed, with new experiments clearly separated from confirmatory evidence.

## Required reporting

For every threshold/policy, report issued forecasts, matured forecasts, successes, failures, unresolved outcomes, pending outcomes, eligible stock-date opportunities, selection coverage, active decision dates, longest no-pick interval, and horizon distribution. Report forecast-weighted and date/block-sensitive reliability, era results, return magnitudes and tail losses, and matched-date/horizon controls.

Missing matured outcomes remain visible and count as failures in a pessimistic sensitivity analysis. Pending forecasts and abstentions do not count as successes. Zero issued or zero matured forecasts means precision is undefined, not 100%. Very rare selection does not waive the need for adequate evidence across distinct periods.

Report retrospective model testing separately from prospective performance. Reused historical data remain reused historical data. No production claim of supported >95% precision until the existing preregistration's evidence and data-quality requirements are satisfied.

## Presentation

The user-facing result should be either a short qualifying-picks table (ticker, reference timestamp/price, fixed horizon/endpoint, probability estimate, evidence status) or: `No stock meets the confidence requirement.` Experimental candidates must be labeled separately and must never be presented as validated >95% picks.
