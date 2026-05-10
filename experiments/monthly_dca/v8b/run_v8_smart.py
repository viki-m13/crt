"""V8 with smart dynamic leverage (de-lever on SPY DD, lever on bull trend)."""
import sys
from pathlib import Path
from dataclasses import asdict
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


def run_one(cfg, score_strategy, weights=None):
    mr, daily, spy = load_data()
    sp = build_score_panel(score_strategy, weights=weights)
    eq = simulate_v8(cfg, sp, mr, spy, daily_prices=daily)
    m = evaluate_v8(eq, mr, name=cfg.name)
    return m, eq


def main():
    rows = []
    eqs = {}
    # Smart leverage: scale gross by SPY DD-from-52w-high
    # DD=0% → full gross; DD=10% → 0; if SPY < 200dma → cash
    for k in [2, 3]:
        for h in [6, 12]:
            for w in ["invvol", "ew"]:
                for gmax in [1.5, 1.75, 2.0, 2.5, 3.0]:
                    for ddx in [0.06, 0.08, 0.10, 0.15]:
                        cfg = V8Config(
                            name=f"v8s_k{k}_h{h}_{w}_g{gmax}_dd{ddx}",
                            k_normal=k, k_bull=k, k_recovery=k,
                            weighting=w, hold_months=h,
                            gross_target=gmax,
                            dd_full_floor=ddx,
                            gross_floor=0.0,
                        )
                        m, eq = run_one(cfg, "ml_3plus6")
                        print(f"{cfg.name:54s}  WFmean={m['wf_mean_cagr']*100:7.2f}%  "
                              f"Full={m['cagr_full']*100:7.2f}%  Sh={m['sharpe']:.2f}  "
                              f"DD={m['max_dd']*100:7.2f}%  +SPY={m['wf_n_beats_spy']}")
                        row = {**m}
                        row.update({f"cfg_{kk}": vv for kk, vv in asdict(cfg).items()})
                        rows.append(row)
                        eqs[cfg.name] = eq
    df = pd.DataFrame(rows)
    out_dir = ROOT / "experiments" / "monthly_dca" / "v8b" / "results"
    df.to_csv(out_dir / "v8_smart_leverage.csv", index=False)
    # Filter for sane DD (worst < -70%)
    df_safe = df[df["max_dd"] >= -0.70].copy()
    print("\n*** Top 20 by WF mean (DD <= -70%): ***")
    print(df_safe.sort_values("wf_mean_cagr", ascending=False).head(20)[
        ["name", "wf_mean_cagr", "cagr_full", "sharpe", "max_dd", "wf_n_beats_spy"]].to_string())
    print("\n*** Top 10 overall: ***")
    print(df.sort_values("wf_mean_cagr", ascending=False).head(10)[
        ["name", "wf_mean_cagr", "cagr_full", "sharpe", "max_dd", "wf_n_beats_spy"]].to_string())


if __name__ == "__main__":
    main()
