"""Test ML score with various trend confirmation filters."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import StratSpec, benchmark, run_one
from score_factory import build_score_panel, load_panel, load_mlpreds

ROOT = Path(__file__).resolve().parents[3]


def build_filtered_panel(filter_kind: str, threshold: float = 0.0) -> pd.DataFrame:
    """Take the existing ML score, then ZERO out stocks failing a filter so they fall to bottom."""
    panel = load_panel()
    ml = load_mlpreds()
    p = panel.merge(ml[["asof", "ticker", "ml_score"]], on=["asof", "ticker"], how="inner")
    p["score"] = p.groupby("asof")["ml_score"].rank(pct=True)
    keep = pd.Series(True, index=p.index)
    if filter_kind == "above200":
        keep = p["d_sma200"] > threshold
    elif filter_kind == "trend_health":
        keep = p["trend_health_5y"].fillna(0) >= threshold
    elif filter_kind == "no_freefall":
        # mom_12_1 > -threshold (drop stocks down >threshold over 12m)
        keep = p["mom_12_1"].fillna(0) >= -threshold
    elif filter_kind == "near_high":
        keep = p.get("range_pos_1y", pd.Series(0, index=p.index)).fillna(0) >= threshold
    elif filter_kind == "rsi_ok":
        keep = (p["rsi_14"].fillna(50) >= threshold) & (p["rsi_14"].fillna(50) <= 100 - threshold)
    p["score"] = np.where(keep, p["score"], -1.0)  # filtered drop to bottom
    p["vol_rank"] = p.groupby("asof")["vol_1y"].rank(pct=True)
    out = p[["asof", "ticker", "score", "vol_1y", "vol_rank",
             "mom_12_1", "pullback_1y", "trend_health_5y", "d_sma200"]].copy()
    return out


# Hack the engine to accept our pre-built panel:
def main():
    from runner import load_monthly_returns
    sys.path.insert(0, str(ROOT / "experiments" / "monthly_dca" / "v6"))
    from lib_engine import V6Config, simulate, evaluate, build_spy_aligned, load_spy_features

    mr = load_monthly_returns()
    spy_feat = load_spy_features()

    rows = []
    cases = [
        ("ml_filt_above200_0", "above200", 0.0),
        ("ml_filt_above200_-5", "above200", -0.05),
        ("ml_filt_th_30", "trend_health", 0.30),
        ("ml_filt_th_50", "trend_health", 0.50),
        ("ml_filt_no_freefall_50", "no_freefall", 0.50),
        ("ml_filt_no_freefall_30", "no_freefall", 0.30),
        ("ml_filt_near_high_50", "near_high", 0.50),
        ("ml_filt_near_high_70", "near_high", 0.70),
        ("ml_filt_rsi_ok", "rsi_ok", 25.0),
    ]
    for ks in [2, 3]:
        for w in ["ew", "invvol"]:
            for hold in [3, 6, 12]:
                for name, fkind, thr in cases:
                    sp = build_filtered_panel(fkind, threshold=thr)
                    cfg = V6Config(
                        name=f"{name}_k{ks}_{w}_h{hold}",
                        scorer="custom",
                        regime_gate="tight",
                        k_normal=ks, k_recovery=ks, k_bull=ks,
                        weighting=w,
                        hold_months=hold,
                        cash_yield_yr=0.03,
                    )
                    eq = simulate(cfg, sp, mr, spy_feat)
                    aligned = build_spy_aligned(eq, mr)
                    m = evaluate(eq, aligned, name=cfg.name)
                    print(f"{cfg.name:48s}  WFmean={m['wf_mean_cagr']*100:6.2f}%  "
                          f"Full={m['cagr_full']*100:6.2f}%  Sharpe={m['sharpe']:.2f}  "
                          f"MaxDD={m['max_dd']*100:6.2f}%  PosSpl={m['wf_n_pos']}  Beat={m['wf_n_beats_spy']}")
                    rows.append({**m, "name": cfg.name, "filter_kind": fkind,
                                 "threshold": thr, "k": ks, "weighting": w, "hold": hold})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "experiments" / "monthly_dca" / "v8b" / "results" / "sweep_filters.csv", index=False)
    print("\nTop 10 by WF mean:")
    print(df.sort_values("wf_mean_cagr", ascending=False).head(10)[
        ["name", "wf_mean_cagr", "cagr_full", "sharpe", "max_dd", "wf_n_beats_spy"]].to_string())


if __name__ == "__main__":
    main()
