"""
Focused sweep: ML score + invvol + variable K, hold, leverage, regime gates.
No per-pick stop-loss (faster). Uses crash regime gate for cash protection.
"""
import sys
import time
from pathlib import Path
from dataclasses import asdict
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v8_engine import V8Config, simulate_v8, evaluate_v8
from score_factory import build_score_panel
ROOT = Path(__file__).resolve().parents[3]
V6 = ROOT / "experiments" / "monthly_dca" / "v6"
sys.path.insert(0, str(V6))
from lib_engine import load_spy_features


def main():
    print("Loading data...", flush=True)
    mr = pd.read_parquet(ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "monthly_returns_clean.parquet")
    spy = load_spy_features()
    print("Building score panel...", flush=True)
    sp = build_score_panel("ml_3plus6")
    print(f"Score panel: {sp.shape}", flush=True)

    rows = []
    grid = []
    # Pure (no leverage)
    for k in [1, 2, 3, 4]:
        for h in [3, 6, 9, 12]:
            for w in ["ew", "invvol", "conv"]:
                grid.append((k, h, w, 1.0, 0.0, False))
    # With leverage gross > 1
    for k in [2, 3]:
        for h in [6, 12]:
            for w in ["invvol"]:
                for gross in [1.25, 1.5, 1.75, 2.0]:
                    # No DD scaling
                    grid.append((k, h, w, gross, 0.0, False))
                    # With DD scaling
                    for ddx in [0.10, 0.15, 0.20, 0.25, 0.30]:
                        grid.append((k, h, w, gross, ddx, False))
                    # With trend-only-lever
                    grid.append((k, h, w, gross, 0.0, True))
                    grid.append((k, h, w, gross, 0.10, True))
                    grid.append((k, h, w, gross, 0.20, True))

    print(f"Total configs: {len(grid)}", flush=True)
    t0 = time.time()
    for i, (k, h, w, gross, ddx, trend_only) in enumerate(grid):
        cfg = V8Config(
            name=f"k{k}_h{h}_{w}_g{gross}_dd{ddx}_to{int(trend_only)}",
            k_normal=k, k_bull=k, k_recovery=k,
            weighting=w, hold_months=h,
            gross_target=gross,
            dd_full_floor=ddx,
            gross_floor=0.0,
            spy_trend_only_lever=trend_only,
        )
        eq = simulate_v8(cfg, sp, mr, spy, daily_prices=None)
        m = evaluate_v8(eq, mr, name=cfg.name)
        row = {**m}
        row.update({f"cfg_{kk}": vv for kk, vv in asdict(cfg).items()})
        rows.append(row)
        if i % 20 == 0:
            elapsed = time.time() - t0
            print(f"[{i}/{len(grid)}] {cfg.name} WFmean={m['wf_mean_cagr']*100:6.2f}% "
                  f"DD={m['max_dd']*100:6.2f}% (elapsed={elapsed:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "experiments" / "monthly_dca" / "v8b" / "results" / "focused_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}", flush=True)

    # Top by WF mean overall
    print("\nTop 20 by WF mean (any DD):")
    print(df.sort_values("wf_mean_cagr", ascending=False).head(20)[
        ["name", "wf_mean_cagr", "cagr_full", "sharpe", "max_dd", "wf_n_pos", "wf_n_beats_spy"]].to_string())

    # Top with DD <= -55%
    df_safe = df[df["max_dd"] >= -0.55]
    print(f"\nTop 20 with DD <= -55% ({len(df_safe)} configs):")
    print(df_safe.sort_values("wf_mean_cagr", ascending=False).head(20)[
        ["name", "wf_mean_cagr", "cagr_full", "sharpe", "max_dd", "wf_n_pos", "wf_n_beats_spy"]].to_string())

    # Top with DD <= -50%
    df_safer = df[df["max_dd"] >= -0.50]
    print(f"\nTop 20 with DD <= -50% ({len(df_safer)} configs):")
    print(df_safer.sort_values("wf_mean_cagr", ascending=False).head(20)[
        ["name", "wf_mean_cagr", "cagr_full", "sharpe", "max_dd", "wf_n_pos", "wf_n_beats_spy"]].to_string())


if __name__ == "__main__":
    main()
