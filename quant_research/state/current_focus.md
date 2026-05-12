# Current Focus

## Active Experiment: exp_006 — Volatility Targeting

**Hypothesis:** Scaling portfolio exposure inversely with realized SPY volatility (capped at 1.0, no leverage) will reduce portfolio vol during market stress periods more than it reduces expected returns. Since vol and returns are negatively correlated in equities, this should improve Sharpe ratio.

**Math check:**
- Current K=50 inv_vol + regime: mean_m=4.13%, std_m=8.52%, Sharpe=1.68 (annualized)
- If SPY vol target = 15% and SPY vol averages 20% in invested months: scale ≈ 0.75
  - Naive: new mean = 0.75×4.13% = 3.10%, new std = 0.75×8.52% = 6.39%, Sharpe unchanged
  - But: vol is higher precisely in bad months → scaling down during bad months reduces std MORE than mean
  - Empirical estimate: 15-25% Sharpe improvement from vol timing (Moreira & Muir 2017)
- Target: Sharpe ≥ 2.0 with CAGR ≥ 50%

**Test matrix (17 configs):**
- K ∈ [40, 50, 60, 80]
- vol_target ∈ [0.12, 0.15, 0.18] annualized
- Signal: SPY vol_21d from daily prices (already computed in engine)
- Base: lgbm inv_vol + regime_200ma (best configs from exp_003/005)

**Implementation note:**
- `scale = min(target_ann_vol / spy_vol_21d, 1.0)` — no leverage
- `port_ret = scale × raw_port_ret - 2 × cost × scale`
- If regime gate fires (cash), scale = 0 regardless of vol signal

**Alternative in same experiment:**
- Sector-filtered: restrict to top-3 sectors by 3m relative strength before LGBM scoring

## Blocker
None currently. State files now written. Next: implement and run exp_006.
