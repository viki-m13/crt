"""Bias sensitivity per universe — MC delisting overlay for the most important
universes: sp500_pit (home), tech_broad (target), iyw_tech (target)."""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "monthly_dca" / "v6"))

from lib_engine import (
    V2, PIT, V6Config, build_spy_aligned, evaluate, load_score_panel,
    load_spy_features, simulate,
)
from universes import IYW, TECH_BROAD  # noqa: E402

OUT = ROOT / "experiments" / "monthly_dca" / "v6" / "results"

ALPHAS = [0.0, 0.04, 0.08, 0.12]
ITERS = 20  # per alpha


def filter_universe(panel, tickers):
    return panel[panel["ticker"].isin(set(tickers))].copy()


def make_synthetic(mr, alpha, seed):
    if alpha <= 0: return mr
    rng = np.random.default_rng(seed)
    p = 1.0 - (1.0 - alpha) ** (1.0 / 12.0)
    mask = rng.random(mr.shape) < p
    return pd.DataFrame(np.where(mask, -1.0, mr.values), index=mr.index, columns=mr.columns)


def run_one(cfg, panel, mr, spy):
    eq = simulate(cfg, panel, mr, spy)
    sa = build_spy_aligned(eq, mr)
    return evaluate(eq, sa, cfg.name)["cagr_full"]


def main():
    mr = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy = load_spy_features()
    panel_sp = load_score_panel("ml_3plus6", "sp500_pit")
    panel_broad = load_score_panel("ml_3plus6", "broader")
    panel_iyw = filter_universe(panel_broad, IYW)
    panel_tech = filter_universe(panel_broad, TECH_BROAD)

    universes = [
        ("sp500_pit", panel_sp),
        ("iyw_tech", panel_iyw),
        ("tech_broad", panel_tech),
    ]
    cfgs = {
        "v3": V6Config(name="v3", scorer="ml_3plus6", regime_gate="tight",
                       k_normal=3, k_recovery=3, k_bull=3, weighting="ew",
                       hold_months=6, cost_bps=10.0),
        "A": V6Config(name="A", scorer="ml_3plus6", regime_gate="tight",
                      k_normal=3, k_recovery=3, k_bull=3, weighting="invvol",
                      hold_months=6, cost_bps=10.0, cash_yield_yr=0.03),
        "B": V6Config(name="B", scorer="ml_3plus6", regime_gate="tight",
                      k_normal=3, k_recovery=3, k_bull=2, weighting="invvol",
                      hold_months=6, cost_bps=10.0, cash_yield_yr=0.03),
    }

    rows = []
    for u, panel in universes:
        for label, cfg in cfgs.items():
            for alpha in ALPHAS:
                t0 = time.time()
                cagrs = []
                for it in range(ITERS):
                    seed = hash((u, label, alpha, it)) & 0x7fffffff
                    mr2 = make_synthetic(mr, alpha, seed) if alpha > 0 else mr
                    try:
                        cagrs.append(run_one(cfg, panel, mr2, spy))
                    except Exception:
                        cagrs.append(-1.0)
                arr = np.array(cagrs)
                rows.append({
                    "universe": u, "strategy": label, "alpha_yr": alpha,
                    "p10": float(np.percentile(arr, 10)),
                    "median": float(np.median(arr)),
                    "p90": float(np.percentile(arr, 90)),
                    "mean": float(arr.mean()),
                    "n_iters": ITERS,
                })
                print(f"[{u:11s}|{label:3s}|α={alpha*100:5.1f}%] {time.time()-t0:.1f}s "
                      f"med={np.median(arr)*100:6.2f}% p10={np.percentile(arr, 10)*100:6.2f}%")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "universe_bias.csv", index=False)

    print("\n=== Median CAGR at α=4%/yr (historical delisting) ===")
    sub = df[df["alpha_yr"] == 0.04]
    print(sub.pivot_table(index="universe", columns="strategy", values="median").round(4).to_string())
    print("\n=== p10 (worst-case) CAGR at α=4% ===")
    print(sub.pivot_table(index="universe", columns="strategy", values="p10").round(4).to_string())


if __name__ == "__main__":
    main()
