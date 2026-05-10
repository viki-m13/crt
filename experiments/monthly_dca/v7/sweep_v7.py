"""V7 comprehensive sweep — combine DAILY stops + permanent sleeve + CDI hedge.

Daily stop is now correctly modelled (no monthly-resolution optimism). The
sweep evaluates many combinations to find the Pareto frontier for downside.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from daily_stop_validator import simulate_daily_stop
from lib_engine_v7 import (V7Config, load_panel_v7, load_spy_features,
                           build_spy_aligned, evaluate, V2)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    panel = load_panel_v7("ml_3plus6", "sp500_pit")
    mr = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy = load_spy_features()

    rows = []
    t0 = time.time()
    n = 0

    # Baseline (v6 winner)
    base = V7Config(weighting="invvol", cash_yield_yr=0.03, pick_stop_loss=0.0)
    eq = simulate_daily_stop(base, panel, mr, spy)
    m = evaluate(eq, build_spy_aligned(eq, mr), "v6_winner_baseline")
    m["cfg"] = json.dumps(base.__dict__, default=str)
    rows.append(m)

    # Sweep grids
    weightings = ["invvol", "ew"]
    sl_grid = [0.0, 0.20, 0.25, 0.30]
    tlt_grid = [0.0, 0.10, 0.20, 0.30]
    spy_grid = [0.0, 0.10, 0.20]
    cdi_grid = [(0.0, 0.10, 0.25), (0.20, 0.10, 0.25), (0.30, 0.10, 0.25)]

    for w in weightings:
        for sl in sl_grid:
            for tlt in tlt_grid:
                for spy_w in spy_grid:
                    if tlt > 0 and spy_w > 0:
                        continue  # mutually exclusive (use one defensive sleeve)
                    for (cdi_max, cdi_dd, cdi_vol) in cdi_grid:
                        cfg = V7Config(
                            weighting=w, cash_yield_yr=0.03,
                            pick_stop_loss=sl,
                            perm_sleeve_ticker="TLT" if tlt > 0 else ("SPY" if spy_w > 0 else ""),
                            perm_sleeve_weight=tlt if tlt > 0 else spy_w,
                            cdi_max_hedge=cdi_max,
                            cdi_dd_threshold=cdi_dd, cdi_vol_threshold=cdi_vol,
                            cdi_hedge_ticker="SH",
                        )
                        cfg.name = f"w={w}|sl{sl}|tlt{tlt}|spy{spy_w}|cdi{cdi_max}"
                        try:
                            eq = simulate_daily_stop(cfg, panel, mr, spy)
                            m = evaluate(eq, build_spy_aligned(eq, mr), cfg.name)
                            m["cfg"] = json.dumps(cfg.__dict__, default=str)
                            rows.append(m)
                        except Exception as e:
                            print(f"  ! {cfg.name}: {e}")
                        n += 1
                        if n % 20 == 0:
                            print(f"  {n} variants in {time.time()-t0:.0f}s")
    print(f"[done] {len(rows)} in {time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v7_sweep_results.csv", index=False)

    base_row = df[df["name"] == "v6_winner_baseline"].iloc[0]
    df["delta_cagr"] = df["cagr_full"] - base_row["cagr_full"]
    df["delta_sharpe"] = df["sharpe"] - base_row["sharpe"]
    df["delta_dd"] = df["max_dd"] - base_row["max_dd"]
    df["delta_wf"] = df["wf_mean_cagr"] - base_row["wf_mean_cagr"]

    print(f"\nv6 baseline: cagr={base_row.cagr_full:.3f} sh={base_row.sharpe:.3f} mdd={base_row.max_dd:.3f} wf={base_row.wf_mean_cagr:.3f}")

    # Pareto: downside-focused (much lower MaxDD)
    print("\n=== Top by ABS MaxDD reduction (with WF mean >= 30%) ===")
    cand = df[(df["wf_mean_cagr"] >= 0.30) & (df["wf_n_pos"] >= 9) & (df["wf_n_beats_spy"] >= 8)].copy()
    cand = cand.sort_values("max_dd", ascending=False).head(25)
    cols = ["name", "cagr_full", "sharpe", "max_dd",
            "wf_mean_cagr", "wf_min_cagr", "wf_n_beats_spy",
            "delta_cagr", "delta_sharpe", "delta_dd", "delta_wf"]
    print(cand[cols].round(4).to_string(index=False))

    print("\n=== Top by Sharpe (with WF mean >= 30%) ===")
    cand = df[(df["wf_mean_cagr"] >= 0.30) & (df["wf_n_pos"] >= 9) & (df["wf_n_beats_spy"] >= 8)].copy()
    cand = cand.sort_values("sharpe", ascending=False).head(25)
    print(cand[cols].round(4).to_string(index=False))

    print("\n=== Top by composite (CAGR + Sharpe + MaxDD bonus) ===")
    df["composite"] = df["wf_mean_cagr"] * 100 + df["sharpe"] * 12 + (df["max_dd"] - base_row["max_dd"]) * 100
    cand = df[(df["wf_n_pos"] >= 9) & (df["wf_n_beats_spy"] >= 8)
             & (df["wf_mean_cagr"] >= 0.30) & (df["max_dd"] > base_row["max_dd"])
             & (df["sharpe"] > base_row["sharpe"])].copy()
    cand = cand.sort_values("composite", ascending=False).head(20)
    print(cand[cols + ["composite"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
