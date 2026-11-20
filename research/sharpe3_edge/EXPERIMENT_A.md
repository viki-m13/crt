# IPD experiment A: operational definitions before outcome calculation

This supplements PROTOCOL.md. No outcome from this new experiment has yet been inspected. Same historical archives as failure-first; a new executable test does NOT make reused data an untouched holdout. Original protocol's >=3 objective and >=30-session minimum remain. No live orders or production changes.

## Mechanism operationalization

Use trailing 126-session beta, lag it one session before computing daily residuals, and form 5/21/63-session residual sums. Define 21-session residual shock z using trailing 63-session residual volatility. Path efficiency is absolute residual displacement / residual absolute variation. Rank efficiency only within current eligible historical members. Direction agreement is the average agreement of 5- and 63-session residuals with the 21-session shock. Volatility expansion is clip(vol21/vol63 - 1,0,1).

Information score = .4*efficiency_rank + .3*direction_agreement + .3*(1-expansion). Transient score = .4*(1-efficiency_rank) + .3*(1-direction_agreement) + .3*expansion. Their difference is a deterministic ambiguity proxy, not an independently calibrated probability. Continuation candidates: shock z>.5, residual63>0, information-minus-transient>.2. Negative-pressure reversal candidates: z<-.5, residual5>0, score difference<-.2, and market21>0 OR current breadth>.5. Rank by absolute score difference times min(abs(z),3). These scores are NOT statistical independence or verified causal attribution.

At each yearly fit, learn each state's horizon from the preceding 2,520 sessions of already-completed next-close-entry returns, net of a 50bp roundtrip assumption and minus same-origin eligible-stock average return. Aggregate by origin date and horizon-sized time buckets; shrink the estimated annualized excess toward zero and use mean minus one block standard error as the selection score. At least eight time buckets and 24 origin dates are required. Choose the horizon with greatest positive score, tie to shortest. All future/missing outcomes remain invisible until their full scheduled horizon matures. Treat any held missing price as a full loss in this primary conservative simulation, not a vanished observation. Marked missing holdings stay locked until their original deadline.

## Frozen variants and controls

Primary: ipd_adaptive (both states, learned state horizons 30/60/90/126/180/252).
Ablations: ipd_fixed60; information-only60; transient-only60; ipd_selective (also positive paired mean in each of three equal past time sections and >=12 horizon buckets); ipd_volume60 (NDX only, same candidate gate with trailing 5/63 volume-ratio rank confirmation). Pure residual-momentum60, reversal60, low-vol-momentum60, equal-weight-eligible60, inverted-IPD-ranking60 and five deterministic random60 seeds are controls, not inventions. S&P volume is unavailable and its volume variant must be unsupported rather than imputed.

## Self-financing ledger, not endpoint hit-rate annualization

Six equal initial capital sleeves, decision grid every five NYSE sessions, rotating sleeve assignment. Each sleeve buys up to five currently eligible and not-already-held stocks; positive inverse-volatility weights capped at 30% of sleeve capital, and fewer than five names leave proportional cash. No portfolio leverage, daily rebalancing, or stops. Signals use close t, primary entry close t+1, exit t+1+H: actual holding >=30 sessions. All purchases/sales charge one-way 25bp. No future survival selection. A sleeve is reused only after all of its locked holdings mature. Prices marked daily; empty/no-trade days included. Open final holdings are marked, not counted as completed wins. Cash earns zero in the primary ledger. Report Sharpe versus zero and an explicit assumed 3% cash opportunity hurdle; neither is claimed to use historical T-bill data.

Full reruns at 0/10/25/50bp one-way costs and entry lags 1/2/5 sessions (H sessions AFTER entry in each run). Benchmark SPY and equal-weight historical members use same evaluation dates. Primary sample: 2013+ S&P; 2018+ Nasdaq. Report 2013-17 / 2018-21 / 2022+ and 2024+ separately, WITHOUT calling the latter untouched. No annual reoptimization on test-year data.

Optional beta-hedge diagnostic: half-capital stock sleeve and fixed entry-beta SPY short, beta clipped [0,1], one-way costs on both legs and assumed 1%/year short borrow, no short-proceeds reuse or interest/rebate. Hold hedge to same locked deadline. Report margin breaches and do not treat this assumed-borrow experiment as executable evidence.

## Required falsification and reporting

Mutation invariance (future prices, market and memberships); mark-to-market accounting; fee cash conservation; literal holding duration; missing-price writeoff; no future labels in horizon fitting; flat portfolio has undefined Sharpe; cumulative returns independent reconciliation. Conditional mechanism study: paired stock-versus-origin-universe excess at each horizon and across eras. Controls at comparable horizons; exact matched-origin/horizon stock-pool comparisons. Daily-return stationary/time-block bootstrap uncertainty for Sharpe and paired differences, HAC volatility diagnostics, yearly returns, CAGR/drawdown/exposure/trades/turnover. A nominal Sharpe3 from sparse zero-return days or one period is not certification. All frozen variants and sensitivity runs remain visible.

## Novelty boundary

Liquidity-versus-information return dynamics have prior literature (Campbell/Grossman/Wang 1993; Llorente/Michaely/Saar/Wang 2002, NBER w8312). The IPD state/horizon/portfolio implementation is new to this project; we cannot claim the economic concept never existed. No pretrained model with unknown training cutoff is used. A failed test rejects the implemented mechanism, not all possible future inventions.
