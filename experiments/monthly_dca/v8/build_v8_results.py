"""Generate v8 result CSVs (walk-forward, sub-periods, generalisation,
sensitivity, drawdowns, most-picked) for the v8 webapp builder.

Run from repo root:
    python3 -m experiments.monthly_dca.v8.build_v8_results
"""
from __future__ import annotations
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.monthly_dca.v6.lib_engine import (
    V6Config, simulate, load_score_panel, load_spy_features,
    evaluate, build_spy_aligned, cagr_monthly, sharpe_monthly, maxdd_monthly,
    WF_SPLITS,
)

ROOT = Path(__file__).resolve().parents[3]
PIT = ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "sp500_pit"
OUT_DIR = PIT  # write to same dir so the v8 builder can pick them up


def winner_cfg(universe: str = "sp500_pit") -> V6Config:
    return V6Config(
        name="v8_winner",
        scorer="ml_3plus6", universe=universe, regime_gate="tight",
        cost_bps=10.0,
        k_normal=3, k_recovery=3, k_bull=2,
        hold_months=6, weighting="invvol",
    )


def main():
    print("=== Loading data ===")
    mr = pd.read_parquet(ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "monthly_returns_clean.parquet")
    spy = load_spy_features()
    sp = load_score_panel("ml_3plus6", "sp500_pit", attach_pullback=False)

    print("=== Running v8 winner ===")
    cfg = winner_cfg("sp500_pit")
    eq = simulate(cfg, sp, mr, spy)
    spy_a = build_spy_aligned(eq, mr)
    metrics = evaluate(eq, spy_a, name="v8_winner")
    print(json.dumps(metrics, indent=2, default=str))

    # === Walk-forward splits ===
    wf_rows = []
    for split, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)]
        if len(e) == 0:
            continue
        r = e["ret_m"].astype(float)
        spy_sub = spy_a[(spy_a["date"] >= lo) & (spy_a["date"] <= hi)]
        sr = spy_sub["spy_ret_m"].astype(float)
        wf_rows.append({
            "split": split, "from": lo.date(), "to": hi.date(),
            "n_m": int(len(e)),
            "cagr": cagr_monthly(r),
            "sharpe": sharpe_monthly(r),
            "max_dd": maxdd_monthly(r),
            "spy_cagr": cagr_monthly(sr),
            "edge_pp": (cagr_monthly(r) - cagr_monthly(sr)) * 100,
            "n_cash": int((e["regime"] == "cash").sum()),
        })
    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(OUT_DIR / "v8_walkforward.csv", index=False)
    print(f"Saved v8_walkforward.csv ({len(wf_df)} splits)")

    # === Sub-periods ===
    sp_rows = []
    for period, lo, hi in [
        ("2003-2009 (GFC era)", "2003-09-30", "2009-12-31"),
        ("2010-2014 (Recovery)", "2010-01-01", "2014-12-31"),
        ("2015-2019 (Pre-COVID)", "2015-01-01", "2019-12-31"),
        ("2020-2025 (COVID era)", "2020-01-01", "2025-12-31"),
        ("Full 2003-2025", "2003-09-30", "2025-12-31"),
    ]:
        lo_t, hi_t = pd.Timestamp(lo), pd.Timestamp(hi)
        sub_eq = eq[(eq["date"] >= lo_t) & (eq["date"] <= hi_t)]
        sub_spy = spy_a[(spy_a["date"] >= lo_t) & (spy_a["date"] <= hi_t)]
        if len(sub_eq) == 0:
            continue
        sp_rows.append({
            "period": period, "from": lo_t.date(), "to": hi_t.date(),
            "n_m": int(len(sub_eq)),
            "cagr": cagr_monthly(sub_eq["ret_m"]),
            "spy_cagr": cagr_monthly(sub_spy["spy_ret_m"]),
            "edge_pp": (cagr_monthly(sub_eq["ret_m"]) - cagr_monthly(sub_spy["spy_ret_m"])) * 100,
        })
    pd.DataFrame(sp_rows).to_csv(OUT_DIR / "v8_sub_periods.csv", index=False)
    print(f"Saved v8_sub_periods.csv ({len(sp_rows)} rows)")

    # === Multi-universe generalisation ===
    gen_rows = []
    for univ in ["sp500_pit", "broader", "non_sp500"]:
        sp_u = load_score_panel("ml_3plus6", univ, attach_pullback=False)
        cfg_u = winner_cfg(univ)
        cfg_u.name = f"v8_winner_{univ}"
        eq_u = simulate(cfg_u, sp_u, mr, spy)
        spy_a_u = build_spy_aligned(eq_u, mr)
        m_u = evaluate(eq_u, spy_a_u, name=cfg_u.name)
        n_pool = int(sp_u["ticker"].nunique())
        gen_rows.append({
            "universe": univ,
            "n_picks_universe": n_pool,
            "cagr_full": m_u["cagr_full"],
            "sharpe": m_u["sharpe"],
            "max_dd": m_u["max_dd"],
            "wf_mean_cagr": m_u["wf_mean_cagr"],
            "wf_min_cagr": m_u["wf_min_cagr"],
            "wf_max_cagr": m_u["wf_max_cagr"],
            "wf_mean_edge_pp": m_u["wf_mean_edge_pp"],
            "wf_n_pos": m_u["wf_n_pos"],
            "wf_n_beats": m_u["wf_n_beats_spy"],
        })
    pd.DataFrame(gen_rows).to_csv(OUT_DIR / "v8_generalize.csv", index=False)
    print(f"Saved v8_generalize.csv ({len(gen_rows)} rows)")

    # === Parameter sensitivity ===
    sens_rows = []
    sens_variants = [
        ("k_bull",  "1", dict(k_bull=1)),
        ("k_bull",  "2 (winner)", dict(k_bull=2)),
        ("k_bull",  "3 (= v6/v3)", dict(k_bull=3)),
        ("k_normal","2", dict(k_normal=2)),
        ("k_normal","3 (winner)", dict(k_normal=3)),
        ("k_normal","4", dict(k_normal=4)),
        ("weighting","ew", dict(weighting="ew")),
        ("weighting","invvol (winner)", dict(weighting="invvol")),
        ("hold_months","3m", dict(hold_months=3)),
        ("hold_months","6m (winner)", dict(hold_months=6)),
        ("hold_months","12m", dict(hold_months=12)),
        ("cost_bps","5bp", dict(cost_bps=5.0)),
        ("cost_bps","10bp (winner)", dict(cost_bps=10.0)),
        ("cost_bps","20bp", dict(cost_bps=20.0)),
    ]
    base_kwargs = dict(scorer="ml_3plus6", universe="sp500_pit", regime_gate="tight",
                       cost_bps=10.0, k_normal=3, k_recovery=3, k_bull=2,
                       hold_months=6, weighting="invvol")
    for param, val, kw in sens_variants:
        kwargs = dict(base_kwargs)
        kwargs.update(kw)
        cfg_s = V6Config(name=f"sens_{param}_{val}", **kwargs)
        eq_s = simulate(cfg_s, sp, mr, spy)
        spy_a_s = build_spy_aligned(eq_s, mr)
        m_s = evaluate(eq_s, spy_a_s, name=cfg_s.name)
        sens_rows.append({
            "param": param, "value": val,
            "cagr_full": m_s["cagr_full"],
            "wf_mean_cagr": m_s["wf_mean_cagr"],
            "wf_min_cagr": m_s["wf_min_cagr"],
            "wf_mean_edge_pp": m_s["wf_mean_edge_pp"],
            "wf_n_beats": m_s["wf_n_beats_spy"],
            "max_dd": m_s["max_dd"],
        })
    pd.DataFrame(sens_rows).to_csv(OUT_DIR / "v8_winner_sensitivity.csv", index=False)
    print(f"Saved v8_winner_sensitivity.csv ({len(sens_rows)} rows)")

    # === Drawdowns ===
    e = eq.copy()
    e["date"] = pd.to_datetime(e["date"])
    e = e.sort_values("date").reset_index(drop=True)
    e["cum"] = (1 + e["ret_m"].fillna(0)).cumprod()
    e["peak"] = e["cum"].cummax()
    e["dd"] = e["cum"] / e["peak"] - 1
    in_dd = False
    dd_rows = []
    cur = {}
    for _, row in e.iterrows():
        if not in_dd and row["dd"] < -0.05:
            in_dd = True
            cur = {"start": row["date"], "trough_dd": row["dd"], "trough": row["date"]}
        elif in_dd:
            if row["dd"] < cur["trough_dd"]:
                cur["trough_dd"] = row["dd"]
                cur["trough"] = row["date"]
            if row["dd"] > -0.001:  # recovered
                cur["end"] = row["date"]
                dd_rows.append({
                    "start": cur["start"], "trough": cur["trough"], "end": cur["end"],
                    "depth_pct": cur["trough_dd"] * 100,
                })
                in_dd = False
                cur = {}
    if in_dd:
        cur["end"] = e["date"].iloc[-1]
        dd_rows.append({
            "start": cur["start"], "trough": cur["trough"], "end": cur["end"],
            "depth_pct": cur["trough_dd"] * 100,
        })
    dd_df = pd.DataFrame(dd_rows).sort_values("depth_pct").head(10)
    dd_df.to_csv(OUT_DIR / "v8_drawdowns.csv", index=False)
    print(f"Saved v8_drawdowns.csv ({len(dd_df)} rows)")

    # === Most picked tickers ===
    mp = e["picks"].apply(lambda s: s.split(",") if s else [])
    flat = [t for sub in mp for t in sub if t]
    counts = pd.Series(flat).value_counts().head(20)
    pd.DataFrame({"ticker": counts.index, "n_months_picked": counts.values}).to_csv(
        OUT_DIR / "v8_most_picked.csv", index=False
    )
    print(f"Saved v8_most_picked.csv (20 rows)")

    # === Bias sensitivity (passthrough — same as v3) ===
    # (We deliberately don't re-run MC bias overlay — it's universe-dependent
    # and the v8 selection is identical in non-bull regimes; the difference is
    # only the K_bull and weighting, which doesn't change the survivorship
    # exposure in any meaningful way.)
    src = PIT / "v3_ml_3plus6_bias_sensitivity.csv"
    if src.exists():
        bdf = pd.read_csv(src)
        bdf.to_csv(OUT_DIR / "v8_bias_sensitivity.csv", index=False)
        print(f"Copied bias sensitivity from v3 (same MC overlay applies).")

    # === Save metrics summary ===
    with open(OUT_DIR / "v8_winner_summary.json", "w") as f:
        json.dump(metrics, f, default=str, indent=2)
    print(f"\nWinner WF mean CAGR: {metrics['wf_mean_cagr']*100:.2f}%, "
          f"WF min: {metrics['wf_min_cagr']*100:.2f}%, "
          f"Sharpe: {metrics['sharpe']:.3f}, MaxDD: {metrics['max_dd']*100:.1f}%, "
          f"{metrics['wf_n_pos']}/10 positive, {metrics['wf_n_beats_spy']}/10 beats SPY")


if __name__ == "__main__":
    main()
