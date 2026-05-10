"""Fine-tune around the winner: k3_h12_invvol with trend_only + leverage."""
import sys, time
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
    mr = pd.read_parquet(ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "monthly_returns_clean.parquet")
    spy = load_spy_features()
    sp = build_score_panel("ml_3plus6")
    print(f"Score panel: {sp.shape}", flush=True)

    rows = []
    grid = []
    # Vary k, h, gross, DD scaling, regime gates
    for k in [2, 3]:
        for h in [6, 9, 12, 18]:
            for w in ["invvol"]:
                for gross in [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]:
                    for trend_only in [True]:
                        for ddx in [0.0, 0.05, 0.08, 0.12]:
                            grid.append((k, h, w, gross, ddx, trend_only, "tight"))
                            grid.append((k, h, w, gross, ddx, trend_only, "strict_dd"))
                            grid.append((k, h, w, gross, ddx, trend_only, "combo"))
    print(f"Total configs: {len(grid)}", flush=True)
    t0 = time.time()
    for i, (k, h, w, gross, ddx, trend_only, reg) in enumerate(grid):
        cfg = V8Config(
            name=f"k{k}_h{h}_{w}_g{gross}_dd{ddx}_to{int(trend_only)}_{reg}",
            k_normal=k, k_bull=k, k_recovery=k,
            weighting=w, hold_months=h,
            gross_target=gross,
            dd_full_floor=ddx,
            gross_floor=0.0,
            spy_trend_only_lever=trend_only,
            regime_gate=reg,
        )
        eq = simulate_v8(cfg, sp, mr, spy, daily_prices=None)
        m = evaluate_v8(eq, mr, name=cfg.name)
        row = {**m}
        row.update({f"cfg_{kk}": vv for kk, vv in asdict(cfg).items()})
        rows.append(row)
        if i % 30 == 0:
            print(f"[{i}/{len(grid)}] {cfg.name} WFmean={m['wf_mean_cagr']*100:6.2f}% "
                  f"DD={m['max_dd']*100:6.2f}%", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "experiments" / "monthly_dca" / "v8b" / "results" / "winner_finetune.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} to {out}", flush=True)

    # Top by CAGR (any DD)
    print("\nTop 25 by WF mean (any DD):")
    print(df.sort_values("wf_mean_cagr", ascending=False).head(25)[
        ["name", "wf_mean_cagr", "cagr_full", "sharpe", "max_dd", "wf_n_pos", "wf_n_beats_spy"]].to_string())
    # DD <= -55%
    df_safe = df[(df["max_dd"] >= -0.55) & (df["wf_n_pos"] == 10) & (df["wf_n_beats_spy"] >= 9)]
    print(f"\nTop 25 with DD<=-55%, 10/10 pos, 9+/10 beats SPY ({len(df_safe)} configs):")
    print(df_safe.sort_values("wf_mean_cagr", ascending=False).head(25)[
        ["name", "wf_mean_cagr", "cagr_full", "sharpe", "max_dd", "wf_n_pos", "wf_n_beats_spy"]].to_string())
    # DD <= -60%
    df_safer = df[(df["max_dd"] >= -0.60) & (df["wf_n_pos"] == 10) & (df["wf_n_beats_spy"] >= 9)]
    print(f"\nTop 25 with DD<=-60%, 10/10 pos, 9+/10 beats SPY ({len(df_safer)} configs):")
    print(df_safer.sort_values("wf_mean_cagr", ascending=False).head(25)[
        ["name", "wf_mean_cagr", "cagr_full", "sharpe", "max_dd", "wf_n_pos", "wf_n_beats_spy"]].to_string())


if __name__ == "__main__":
    main()
