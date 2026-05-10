"""Run v8_engine over various configurations."""
import sys
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v8_engine import V8Config, simulate_v8, evaluate_v8
from score_factory import build_score_panel
ROOT = Path(__file__).resolve().parents[3]
V6 = ROOT / "experiments" / "monthly_dca" / "v6"
sys.path.insert(0, str(V6))
from lib_engine import load_spy_features  # type: ignore


def load_data():
    mr = pd.read_parquet(ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "monthly_returns_clean.parquet")
    daily = pd.read_parquet(ROOT / "experiments" / "monthly_dca" / "cache" / "prices_extended.parquet")
    spy = load_spy_features()
    return mr, daily, spy


def run_one(cfg: V8Config, score_strategy: str, weights: dict | None = None):
    mr, daily, spy = load_data()
    sp = build_score_panel(score_strategy, weights=weights)
    eq = simulate_v8(cfg, sp, mr, spy, daily_prices=daily)
    m = evaluate_v8(eq, mr, name=cfg.name)
    return m, eq


def sweep(specs):
    rows = []
    for spec in specs:
        cfg, strat, weights = spec["cfg"], spec["strategy"], spec.get("weights")
        try:
            m, eq = run_one(cfg, strat, weights)
            print(f"{cfg.name:60s}  WFmean={m['wf_mean_cagr']*100:7.2f}%  "
                  f"Full={m['cagr_full']*100:7.2f}%  Sh={m['sharpe']:.2f}  "
                  f"DD={m['max_dd']*100:7.2f}%  +SPY={m['wf_n_beats_spy']}")
            row = {**m}
            row.update({f"cfg_{k}": v for k, v in asdict(cfg).items()})
            row["strategy"] = strat
            rows.append(row)
        except Exception as e:
            import traceback
            print(f"{cfg.name:60s}  FAILED: {e}")
            traceback.print_exc()
    return pd.DataFrame(rows)


if __name__ == "__main__":
    specs = []
    # Baseline (no leverage, no stop-loss) using ml score
    specs.append({"cfg": V8Config(name="v8_baseline", k_normal=3, weighting="invvol", hold_months=6), "strategy": "ml_3plus6"})

    # Try k=2 with leverage in normal/bull
    for gross in [1.0, 1.25, 1.5, 1.75, 2.0]:
        specs.append({
            "cfg": V8Config(
                name=f"v8_ml_k2_h12_invvol_g{gross}",
                k_normal=2, k_bull=2, k_recovery=2,
                weighting="invvol", hold_months=12,
                gross_target=gross,
            ),
            "strategy": "ml_3plus6",
        })

    # Add daily stop-loss to leveraged
    for gross in [1.0, 1.25, 1.5]:
        for stop in [0.0, 0.20, 0.30, 0.40]:
            specs.append({
                "cfg": V8Config(
                    name=f"v8_ml_k2_h12_g{gross}_sl{stop}",
                    k_normal=2, k_bull=2, k_recovery=2,
                    weighting="invvol", hold_months=12,
                    gross_target=gross,
                    pick_daily_stop=stop,
                ),
                "strategy": "ml_3plus6",
            })

    # K=3 with various leverage
    for k in [3]:
        for gross in [1.25, 1.5, 1.75]:
            for h in [6, 12]:
                specs.append({
                    "cfg": V8Config(
                        name=f"v8_ml_k{k}_h{h}_invvol_g{gross}",
                        k_normal=k, k_bull=k, k_recovery=k,
                        weighting="invvol", hold_months=h,
                        gross_target=gross,
                    ),
                    "strategy": "ml_3plus6",
                })

    df = sweep(specs)
    out_dir = ROOT / "experiments" / "monthly_dca" / "v8b" / "results"
    out_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(out_dir / "v8_run1.csv", index=False)
    print("\nTop 12 by WF mean:")
    print(df.sort_values("wf_mean_cagr", ascending=False).head(12)[
        ["name", "wf_mean_cagr", "cagr_full", "sharpe", "max_dd", "wf_n_beats_spy"]].to_string())
