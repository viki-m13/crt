"""Synthetic-delisting bias overlay on the PIT S&P 500 filter backtest.

Even with a true PIT S&P 500 membership panel, residual survivorship bias can
remain in the underlying price panel: an S&P 500 member that was acquired or
went bankrupt mid-membership may have incomplete return data in our Yahoo-style
panel.  We stress-test by injecting synthetic per-pick delisting:

  At each (asof, picked_ticker) cell, with probability `1 - (1-alpha)**(1/12)`
  the pick is set to -100% return (instantaneous loss).  Repeat 30 MC iterations
  per alpha.

Reports p10 / median / p90 of full-window CAGR at each alpha.

Inputs:
  cache/v2/sp500_pit/sp500_pit_filter_equity.csv

Output:
  cache/v2/sp500_pit/sp500_pit_bias_sensitivity.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"

ALPHAS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20]
N_MC = 30
COST_BPS = 10.0
SEED = 42


def load_per_pick_returns():
    """Reconstruct per-pick monthly returns from the filter equity CSV."""
    eq = pd.read_csv(PIT / "sp500_pit_filter_equity.csv")
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    eq["date"] = pd.to_datetime(eq["date"])

    rows = []
    for _, r in eq.iterrows():
        d = pd.Timestamp(r["date"])
        if r["regime"] == "crash" or pd.isna(r.get("picks")) or r["picks"] == "":
            rows.append({"date": d, "regime": r["regime"], "picks": [], "pick_rets": []})
            continue
        picks = str(r["picks"]).split(",")
        # Find next month-end in monthly_returns
        pos = monthly_returns.index.searchsorted(d)
        cands = []
        for j in (pos - 1, pos):
            if 0 <= j < len(monthly_returns.index):
                cands.append((j, abs((monthly_returns.index[j] - d).days)))
        cands.sort(key=lambda x: x[1])
        if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(monthly_returns.index):
            rows.append({"date": d, "regime": r["regime"], "picks": [], "pick_rets": []})
            continue
        next_d = monthly_returns.index[cands[0][0] + 1]
        prets = []
        for tk in picks:
            if tk in monthly_returns.columns:
                rr = monthly_returns.at[next_d, tk]
                prets.append(-1.0 if pd.isna(rr) else float(rr))
            else:
                prets.append(-1.0)
        rows.append({"date": d, "regime": r["regime"], "picks": picks, "pick_rets": prets})
    return rows


def simulate_with_delist(rows, alpha: float, rng: np.random.Generator,
                         starting_cash: float = 1.0, cost_bps: float = COST_BPS):
    p_month = 1.0 - (1.0 - alpha) ** (1.0 / 12.0) if alpha > 0 else 0.0
    eq = starting_cash
    cf = cost_bps / 10000.0
    for r in rows:
        if not r["picks"]:
            continue
        prets = np.array(r["pick_rets"], dtype=float).copy()
        if p_month > 0:
            wipe = rng.random(len(prets)) < p_month
            prets[wipe] = -1.0
        ret_m = prets.mean()
        eq *= (1 + ret_m) * (1 - cf)
    return eq


def main():
    print("=== Loading per-pick returns from PIT filter equity curve ===")
    rows = load_per_pick_returns()
    n_active = sum(1 for r in rows if r["picks"])
    print(f"  {len(rows)} months, {n_active} with active positions")

    print(f"\n=== MC overlay: alpha in {ALPHAS}, n_iters={N_MC} ===")
    rng_master = np.random.default_rng(SEED)
    out = []
    for a in ALPHAS:
        finals = []
        for it in range(N_MC):
            rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
            f = simulate_with_delist(rows, a, rng)
            finals.append(f)
        finals = np.array(finals)
        years = sum(1 for r in rows) / 12.0
        cagrs = finals ** (1.0 / years) - 1.0
        rec = {
            "alpha_yr": a,
            "p10_CAGR": float(np.percentile(cagrs, 10) * 100),
            "median_CAGR": float(np.median(cagrs) * 100),
            "p90_CAGR": float(np.percentile(cagrs, 90) * 100),
            "mean_CAGR": float(np.mean(cagrs) * 100),
            "min_CAGR": float(cagrs.min() * 100),
            "max_CAGR": float(cagrs.max() * 100),
            "n_iters": N_MC,
        }
        out.append(rec)
        print(f"  alpha={a:.2f}: median CAGR={rec['median_CAGR']:.2f}%, "
              f"p10={rec['p10_CAGR']:.2f}%, p90={rec['p90_CAGR']:.2f}%")

    df = pd.DataFrame(out)
    df.to_csv(PIT / "sp500_pit_bias_sensitivity.csv", index=False)
    print(f"\nSaved -> {PIT / 'sp500_pit_bias_sensitivity.csv'}")


if __name__ == "__main__":
    main()
