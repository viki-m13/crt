"""Full validation of the v7 winner: generalization + deep analysis + bias overlay."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v6"))

from daily_stop_validator import simulate_daily_stop
from lib_engine_v7 import (V7Config, load_panel_v7, load_spy_features,
                           build_spy_aligned, evaluate, V2)
from lib_engine import EXCLUDE_TICKERS, PIT, WF_SPLITS, load_score_panel

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


WINNER_CONFIG = dict(
    name="v7_safer",
    weighting="invvol", cash_yield_yr=0.03, hold_months=6, cost_bps=10.0,
    k_normal=3, k_recovery=3, k_bull=3,
    pick_stop_loss=0.30,
    cdi_max_hedge=0.20, cdi_dd_threshold=0.10, cdi_vol_threshold=0.25,
    cdi_hedge_ticker="SH",
    perm_sleeve_ticker="TLT", perm_sleeve_weight=0.10,
)

ALSO_RUN = [
    ("v6_baseline", dict(weighting="invvol", cash_yield_yr=0.03)),
    ("v7_safer", WINNER_CONFIG),
    ("v7_safest", dict(WINNER_CONFIG, name="v7_safest", cdi_max_hedge=0.30)),
]


def per_split(eq, spy_aligned):
    rows = []
    for split, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)]
        if len(e) == 0:
            continue
        r = e["ret_m"].astype(float)
        ec = (1 + r).cumprod()
        cagr = ec.iloc[-1] ** (12.0 / len(ec)) - 1
        sh = (r.mean() / max(r.std(), 1e-9)) * np.sqrt(12)
        peak = ec.cummax()
        mdd = float(((ec - peak) / peak).min())
        spy = spy_aligned[(spy_aligned["date"] >= lo) & (spy_aligned["date"] <= hi)]
        sr = spy["spy_ret_m"].astype(float)
        sc = (1 + sr).cumprod()
        scagr = sc.iloc[-1] ** (12.0 / len(sc)) - 1
        rows.append({
            "split": split, "from": lo.date(), "to": hi.date(), "n_m": len(e),
            "cagr_pct": cagr * 100, "spy_cagr_pct": scagr * 100,
            "edge_pp": (cagr - scagr) * 100, "sharpe": sh, "max_dd_pct": mdd * 100,
        })
    return pd.DataFrame(rows)


def yearly(eq, spy_aligned):
    eq = eq.copy()
    eq["year"] = pd.to_datetime(eq["date"]).dt.year
    yr = eq.groupby("year")["ret_m"].apply(lambda x: (1 + x).prod() - 1).rename("year_ret")
    sp = spy_aligned.copy()
    sp["year"] = pd.to_datetime(sp["date"]).dt.year
    sy = sp.groupby("year")["spy_ret_m"].apply(lambda x: (1 + x).prod() - 1).rename("spy_year_ret")
    out = yr.to_frame().join(sy.to_frame(), how="left")
    out["edge_pp"] = (out["year_ret"] - out["spy_year_ret"]) * 100
    return out


def drawdowns(eq, threshold=-0.05):
    s = pd.Series(eq["equity"].values, index=pd.DatetimeIndex(eq["date"]))
    peak = s.cummax()
    dd = (s - peak) / peak
    episodes, in_dd, start, depth, trough = [], False, None, 0, None
    for d, v in dd.items():
        if not in_dd and v < threshold:
            in_dd, start, depth, trough = True, d, v, d
        elif in_dd:
            if v < depth:
                depth, trough = v, d
            if v >= -0.001:
                episodes.append({"start": start, "trough": trough, "end": d, "depth_pct": depth*100})
                in_dd = False
    if in_dd:
        episodes.append({"start": start, "trough": trough, "end": s.index[-1], "depth_pct": depth*100})
    return pd.DataFrame(episodes).sort_values("depth_pct")


def random_subset(panel, k=500, seed=1):
    tickers = sorted(panel["ticker"].unique())
    rng = np.random.default_rng(seed)
    pick = set(rng.choice(tickers, size=min(k, len(tickers)), replace=False))
    return panel[panel["ticker"].isin(pick)].reset_index(drop=True)


def main():
    panel = load_panel_v7("ml_3plus6", "sp500_pit")
    mr = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy = load_spy_features()

    # 1. Run all 3 variants on home universe
    print("=" * 70)
    print("STAGE 1: Home universe (sp500_pit)")
    print("=" * 70)
    home_metrics = []
    for name, kw in ALSO_RUN:
        cfg = V7Config(**kw)
        cfg.name = name
        eq = simulate_daily_stop(cfg, panel, mr, spy)
        spy_aln = build_spy_aligned(eq, mr)
        m = evaluate(eq, spy_aln, name)
        m["variant"] = name
        home_metrics.append(m)
        eq.to_csv(OUT / f"{name}_equity.csv", index=False)
        (OUT / f"{name}_metrics.json").write_text(json.dumps(m, indent=2))
        sp = per_split(eq, spy_aln)
        sp.to_csv(OUT / f"{name}_per_split.csv", index=False)
        yr = yearly(eq, spy_aln)
        yr.to_csv(OUT / f"{name}_yearly.csv")
        dd = drawdowns(eq).head(10)
        dd.to_csv(OUT / f"{name}_drawdowns.csv", index=False)
        print(f"\n[{name}]")
        print(f"  CAGR={m['cagr_full']*100:.2f}% Sharpe={m['sharpe']:.3f} MaxDD={m['max_dd']*100:.2f}%")
        print(f"  WF mean={m['wf_mean_cagr']*100:.2f}% WF min={m['wf_min_cagr']*100:.2f}% WF n_pos={m['wf_n_pos']} beats_spy={m['wf_n_beats_spy']}")
        print(f"  Top drawdowns:")
        print(dd[["start", "trough", "end", "depth_pct"]].head(5).to_string(index=False))

    # 2. Generalization test
    print()
    print("=" * 70)
    print("STAGE 2: Generalization across 8 universes")
    print("=" * 70)
    panels = {"sp500_pit": panel}
    panels["broader_1811"] = load_panel_v7("ml_3plus6", "broader")
    panels["non_sp500"] = load_panel_v7("ml_3plus6", "non_sp500")
    for seed in [1, 2, 3, 4, 5]:
        panels[f"random_500_seed{seed}"] = random_subset(panels["broader_1811"], k=500, seed=seed)

    gen_rows = []
    for univ, p in panels.items():
        for name, kw in ALSO_RUN:
            cfg = V7Config(**kw)
            cfg.name = f"{univ}|{name}"
            try:
                eq = simulate_daily_stop(cfg, p, mr, spy)
                spy_aln = build_spy_aligned(eq, mr)
                m = evaluate(eq, spy_aln, f"{univ}|{name}")
                m["universe"] = univ
                m["variant"] = name
                gen_rows.append(m)
                print(f"  {univ:>20s} | {name:12s}: cagr={m['cagr_full']*100:.2f}% sh={m['sharpe']:.3f} mdd={m['max_dd']*100:.2f}% wf={m['wf_mean_cagr']*100:.2f}% beats={m['wf_n_beats_spy']}")
            except Exception as e:
                print(f"  {univ}|{name}: FAILED {e}")
    gen_df = pd.DataFrame(gen_rows)
    gen_df.to_csv(OUT / "v7_generalize_results.csv", index=False)

    print()
    print("=" * 70)
    print("STAGE 3: v7_safer vs v6_baseline summary")
    print("=" * 70)
    summary = []
    for univ in panels.keys():
        v6 = gen_df[(gen_df["universe"] == univ) & (gen_df["variant"] == "v6_baseline")].iloc[0]
        v7 = gen_df[(gen_df["universe"] == univ) & (gen_df["variant"] == "v7_safer")].iloc[0]
        summary.append({
            "universe": univ,
            "v6_cagr": v6["cagr_full"] * 100, "v7_cagr": v7["cagr_full"] * 100,
            "delta_cagr": (v7["cagr_full"] - v6["cagr_full"]) * 100,
            "v6_sharpe": v6["sharpe"], "v7_sharpe": v7["sharpe"],
            "delta_sharpe": v7["sharpe"] - v6["sharpe"],
            "v6_dd": v6["max_dd"] * 100, "v7_dd": v7["max_dd"] * 100,
            "delta_dd_pp": (v7["max_dd"] - v6["max_dd"]) * 100,
            "v6_wf": v6["wf_mean_cagr"] * 100, "v7_wf": v7["wf_mean_cagr"] * 100,
            "delta_wf": (v7["wf_mean_cagr"] - v6["wf_mean_cagr"]) * 100,
        })
    summ_df = pd.DataFrame(summary)
    summ_df.to_csv(OUT / "v7_vs_v6_summary.csv", index=False)
    print(summ_df.round(2).to_string(index=False))

    print()
    n_sharpe_better = int((summ_df["delta_sharpe"] > 0).sum())
    n_dd_better = int((summ_df["delta_dd_pp"] > 0).sum())
    print(f"v7 Sharpe better than v6 in {n_sharpe_better}/{len(summ_df)} universes")
    print(f"v7 MaxDD less negative than v6 in {n_dd_better}/{len(summ_df)} universes")


if __name__ == "__main__":
    main()
