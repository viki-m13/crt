"""
Fast sweep:
- Pre-build score panels once (cache in memory)
- Iterate over many configurations
- Run engine on cached panels
"""
import sys
from pathlib import Path
from dataclasses import asdict, dataclass, field
from itertools import product
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v8_engine import V8Config, simulate_v8, evaluate_v8
from score_factory import build_score_panel
ROOT = Path(__file__).resolve().parents[3]
V6 = ROOT / "experiments" / "monthly_dca" / "v6"
sys.path.insert(0, str(V6))
from lib_engine import load_spy_features


def load_data():
    mr = pd.read_parquet(ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "monthly_returns_clean.parquet")
    daily = pd.read_parquet(ROOT / "experiments" / "monthly_dca" / "cache" / "prices_extended.parquet")
    spy = load_spy_features()
    return mr, daily, spy


def build_panel_cache(strategies):
    cache = {}
    for s in strategies:
        cache[s] = build_score_panel(s)
        print(f"Built {s}: {cache[s].shape}")
    return cache


def run_sweep(grid: list[dict], strategy_panels: dict, mr, daily, spy):
    rows = []
    for spec in grid:
        strat = spec.pop("strategy")
        sp = strategy_panels[strat]
        cfg = V8Config(**spec)
        try:
            eq = simulate_v8(cfg, sp, mr, spy, daily_prices=daily)
            m = evaluate_v8(eq, mr, name=cfg.name)
            row = {**m}
            row.update({f"cfg_{k}": v for k, v in asdict(cfg).items()})
            row["strategy"] = strat
            rows.append(row)
            print(f"{cfg.name:60s}  WFmean={m['wf_mean_cagr']*100:7.2f}%  "
                  f"Full={m['cagr_full']*100:7.2f}%  Sh={m['sharpe']:.2f}  "
                  f"DD={m['max_dd']*100:7.2f}%  +SPY={m['wf_n_beats_spy']}")
        except Exception as e:
            print(f"{cfg.name}  FAILED: {e}")
            import traceback; traceback.print_exc()
    return pd.DataFrame(rows)


def main():
    mr, daily, spy = load_data()
    strategies = ["ml_3plus6"]
    cache = build_panel_cache(strategies)

    grid = []
    # K, hold, weighting, gross_target, dd_full_floor, gross_floor, spy_trend_only_lever
    for k in [2, 3]:
        for h in [6, 12]:
            for w in ["invvol"]:
                for gross in [1.0, 1.25, 1.5, 1.75, 2.0]:
                    for ddx in [0.0, 0.10, 0.15, 0.20, 0.25]:
                        for trend_only in [False, True]:
                            for sl in [0.0, 0.30]:
                                name = f"k{k}_h{h}_g{gross}_dd{ddx}_to{int(trend_only)}_sl{sl}"
                                if ddx == 0.0 and trend_only and gross == 1.0:
                                    continue  # skip trivial
                                grid.append({
                                    "name": name,
                                    "k_normal": k, "k_bull": k, "k_recovery": k,
                                    "weighting": w, "hold_months": h,
                                    "gross_target": gross,
                                    "dd_full_floor": ddx,
                                    "gross_floor": 0.0,
                                    "spy_trend_only_lever": trend_only,
                                    "pick_daily_stop": sl,
                                    "strategy": "ml_3plus6",
                                })
    print(f"Total configs: {len(grid)}")
    df = run_sweep(grid, cache, mr, daily, spy)
    out_dir = ROOT / "experiments" / "monthly_dca" / "v8b" / "results"
    out_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(out_dir / "fast_sweep.csv", index=False)
    print("\n*** Top 25 by WF mean (DD <= -55%): ***")
    df_safe = df[df["max_dd"] >= -0.55]
    print(df_safe.sort_values("wf_mean_cagr", ascending=False).head(25)[
        ["name", "wf_mean_cagr", "cagr_full", "sharpe", "max_dd", "wf_n_beats_spy"]].to_string())
    print("\n*** Top 25 overall: ***")
    print(df.sort_values("wf_mean_cagr", ascending=False).head(25)[
        ["name", "wf_mean_cagr", "cagr_full", "sharpe", "max_dd", "wf_n_beats_spy"]].to_string())


if __name__ == "__main__":
    main()
