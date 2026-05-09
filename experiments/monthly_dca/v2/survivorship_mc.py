"""
Survivorship-bias Monte Carlo overlay.

Approach: at each month, every pick has independent probability
`p_del = 1 - (1 - alpha)^(1/12)` of being synthetically delisted to -100%
in the next month (capturing the underlying delisting hazard).

We run N_iters MC iterations and report median / p10 / p90 of final CAGR.

Run: python3 -m experiments.monthly_dca.v2.survivorship_mc
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.monthly_dca.v2.ml_strategy import (
    OUT, build_strategy_outputs, simulate_strategy, cagr,
    _nearest_monthly_pos,
)


def simulate_with_delist(
    outs, monthly_returns, alpha_yr: float, seed: int, cost_bps: float = 10.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p_per_month = 1.0 - (1.0 - alpha_yr) ** (1.0 / 12.0)
    equity = 1.0
    rows = []
    cf = cost_bps / 10000.0

    for o in outs:
        if o.cash or len(o.picks) == 0:
            ret_m = 0.0
        else:
            pos1 = _nearest_monthly_pos(monthly_returns.index, o.asof)
            if pos1 is None or pos1 + 1 >= len(monthly_returns.index):
                ret_m = 0.0
            else:
                next_d = monthly_returns.index[pos1 + 1]
                pick_rets = []
                for tk in o.picks:
                    if rng.random() < p_per_month:
                        pick_rets.append(-1.0)
                        continue
                    if tk in monthly_returns.columns:
                        r = monthly_returns.at[next_d, tk]
                        pick_rets.append(-1.0 if pd.isna(r) else float(r))
                    else:
                        pick_rets.append(-1.0)
                pick_rets = np.array(pick_rets)
                ret_m = float((pick_rets * o.weights).sum())
        if not o.cash and len(o.picks) > 0:
            equity *= (1 + ret_m) * (1 - cf)
        rows.append({"date": o.asof, "equity": equity, "ret_m": ret_m,
                     "regime": o.regime})
    return pd.DataFrame(rows)


def main():
    big = pd.read_parquet(OUT / "panel_cross_section_v3.parquet")
    monthly_returns = pd.read_parquet(OUT / "monthly_returns_clean.parquet")
    preds = pd.read_parquet(OUT / "ml_preds_v2.parquet")
    preds["asof"] = pd.to_datetime(preds["asof"])

    # Restrict to honest range
    preds_eval = preds[(preds["asof"].dt.year >= 2003) & (preds["asof"].dt.year <= 2024)].copy()

    outs = build_strategy_outputs(
        preds_eval, big,
        top_k_normal=15, top_k_recovery=7, top_k_bull=7,
        use_conviction_weighting=False, cash_in_crash=True, regime_mode="tight",
    )

    rows = []
    for alpha in (0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20):
        cagrs = []
        for seed in range(30):
            eq = simulate_with_delist(outs, monthly_returns, alpha, seed)
            if not eq.empty:
                cagrs.append(cagr(eq) * 100)
        cagrs = np.array(cagrs)
        rows.append({
            "alpha_yr": alpha,
            "p10_CAGR": round(np.percentile(cagrs, 10), 2) if len(cagrs) else 0,
            "median_CAGR": round(np.median(cagrs), 2) if len(cagrs) else 0,
            "p90_CAGR": round(np.percentile(cagrs, 90), 2) if len(cagrs) else 0,
            "mean_CAGR": round(cagrs.mean(), 2) if len(cagrs) else 0,
            "n_iters": len(cagrs),
        })
        print(f"alpha={alpha*100:5.1f}%/yr: median CAGR={rows[-1]['median_CAGR']:.2f}% "
              f"(p10={rows[-1]['p10_CAGR']:.2f}%, p90={rows[-1]['p90_CAGR']:.2f}%)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v2_bias_sensitivity.csv", index=False)
    print("\nSaved bias sensitivity to v2_bias_sensitivity.csv")


if __name__ == "__main__":
    main()
