"""Bias sensitivity / synthetic delisting Monte-Carlo for Options A and B.

Methodology mirrors v3/v6 bias overlay:
  - For each rebalance month and each pick, draw a delisting Bernoulli at
    p_month = 1 - (1 - alpha)^(1/12); if True, that pick wipes (-100%) for the month.
  - 30 iterations per alpha; report p10, median, p90, mean of resulting CAGR.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lib_engine import (
    V2, V6Config, build_spy_aligned, evaluate, load_score_panel,
    load_spy_features, simulate,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

ALPHAS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.20]
ITERS = 30


def make_synthetic_returns(monthly_returns: pd.DataFrame, alpha: float, seed: int) -> pd.DataFrame:
    """Inject random delistings. p_month = 1 - (1-alpha)^(1/12)."""
    if alpha <= 0:
        return monthly_returns
    rng = np.random.default_rng(seed)
    p = 1.0 - (1.0 - alpha) ** (1.0 / 12.0)
    mr = monthly_returns.copy()
    mask = rng.random(mr.shape) < p
    arr = mr.values
    arr = np.where(mask, -1.0, arr)
    return pd.DataFrame(arr, index=mr.index, columns=mr.columns)


def run_one(cfg, panel, mr, spy):
    eq = simulate(cfg, panel, mr, spy)
    spy_aln = build_spy_aligned(eq, mr)
    return evaluate(eq, spy_aln, cfg.name)["cagr_full"]


def main():
    print("[load]")
    panel = load_score_panel("ml_3plus6", "sp500_pit")
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_feats = load_spy_features()

    cfgs = {
        "v3": V6Config(name="v3", scorer="ml_3plus6", universe="sp500_pit",
                       regime_gate="tight", k_normal=3, k_recovery=3, k_bull=3,
                       weighting="ew", hold_months=6, cost_bps=10.0),
        "A":  V6Config(name="A", scorer="ml_3plus6", universe="sp500_pit",
                       regime_gate="tight", k_normal=3, k_recovery=3, k_bull=3,
                       weighting="invvol", hold_months=6, cost_bps=10.0,
                       cash_yield_yr=0.03),
        "B":  V6Config(name="B", scorer="ml_3plus6", universe="sp500_pit",
                       regime_gate="tight", k_normal=3, k_recovery=3, k_bull=2,
                       weighting="invvol", hold_months=6, cost_bps=10.0,
                       cash_yield_yr=0.03),
    }

    rows = []
    for label, cfg in cfgs.items():
        for alpha in ALPHAS:
            cagrs = []
            t0 = time.time()
            for it in range(ITERS):
                seed = hash((label, alpha, it)) & 0x7fffffff
                mr2 = make_synthetic_returns(monthly_returns, alpha, seed) if alpha > 0 else monthly_returns
                try:
                    c = run_one(cfg, panel, mr2, spy_feats)
                except Exception as e:
                    c = -1.0
                cagrs.append(c)
            arr = np.array(cagrs)
            rows.append({
                "strategy": label, "alpha_yr": alpha,
                "p10": float(np.percentile(arr, 10)),
                "median": float(np.median(arr)),
                "p90": float(np.percentile(arr, 90)),
                "mean": float(np.mean(arr)),
                "n_iters": ITERS,
            })
            print(f"[{label}|α={alpha*100:5.1f}%] {time.time()-t0:.1f}s "
                  f"p10={arr.min()*100:6.2f}% med={np.median(arr)*100:6.2f}% "
                  f"p90={np.percentile(arr, 90)*100:6.2f}% mean={arr.mean()*100:6.2f}%")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bias_options_summary.csv", index=False)
    print("\n=== Median CAGR by alpha & strategy ===")
    piv = df.pivot_table(index="alpha_yr", columns="strategy", values="median").round(3)
    print(piv.to_string())
    print("\n=== p10 (worst-case) CAGR by alpha & strategy ===")
    piv = df.pivot_table(index="alpha_yr", columns="strategy", values="p10").round(3)
    print(piv.to_string())


if __name__ == "__main__":
    main()
