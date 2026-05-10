"""
Final validation for v8 winner candidates.

Produces:
- Per-split details (CAGR, Sharpe, MaxDD, edge vs SPY)
- Full equity curve
- Yearly returns
- Bias sensitivity (synthetic delisting)
- Universe generalization (broader 1833, non_sp500_pit, 5x random_500 seeds)
- Most-picked stocks
- Top drawdowns ledger

Saves all artifacts to experiments/monthly_dca/v8b/results/<name>_*.csv|json
"""
from __future__ import annotations
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v8_engine import V8Config, simulate_v8, evaluate_v8
from score_factory import build_score_panel
ROOT = Path(__file__).resolve().parents[3]
V6 = ROOT / "experiments" / "monthly_dca" / "v6"
sys.path.insert(0, str(V6))
from lib_engine import load_spy_features, build_spy_aligned, evaluate, WF_SPLITS  # type: ignore

CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
PIT = CACHE / "v2" / "sp500_pit"
RESULTS = ROOT / "experiments" / "monthly_dca" / "v8b" / "results"
RESULTS.mkdir(exist_ok=True, parents=True)


def detailed_per_split(eq: pd.DataFrame, mr: pd.DataFrame) -> pd.DataFrame:
    spy_aligned = build_spy_aligned(eq, mr)
    rows = []
    for split, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)]
        if len(e) == 0:
            continue
        r = e["ret_m"].astype(float).fillna(0)
        spy = spy_aligned[(spy_aligned["date"] >= lo) & (spy_aligned["date"] <= hi)]
        sr = spy["spy_ret_m"].astype(float).fillna(0)
        eqcurve = (1 + r).cumprod()
        peak = eqcurve.cummax()
        dd = ((eqcurve - peak) / peak).min()
        rows.append({
            "split": split, "from": str(lo.date()), "to": str(hi.date()),
            "n_months": len(e),
            "cagr_pct": ((eqcurve.iloc[-1] ** (12 / len(e)) - 1) * 100) if len(e) > 1 else 0.0,
            "spy_cagr_pct": (((1 + sr).cumprod().iloc[-1] ** (12 / len(sr)) - 1) * 100) if len(sr) > 1 else 0.0,
            "sharpe": float((r.mean() / r.std()) * np.sqrt(12)) if r.std() > 0 else 0.0,
            "max_dd_pct": dd * 100,
        })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["edge_pp"] = df["cagr_pct"] - df["spy_cagr_pct"]
    return df


def yearly_breakdown(eq: pd.DataFrame, mr: pd.DataFrame) -> pd.DataFrame:
    spy_aligned = build_spy_aligned(eq, mr)
    eq = eq.copy()
    eq["year"] = eq["date"].dt.year
    spy_aligned = spy_aligned.copy()
    spy_aligned["year"] = spy_aligned["date"].dt.year
    rows = []
    for y, sub in eq.groupby("year"):
        r = sub["ret_m"].astype(float).fillna(0)
        if len(r) < 2:
            continue
        spy_y = spy_aligned[spy_aligned["year"] == y]["spy_ret_m"].astype(float).fillna(0)
        rows.append({
            "year": y,
            "ret_pct": (((1 + r).prod() - 1) * 100),
            "spy_ret_pct": (((1 + spy_y).prod() - 1) * 100),
            "edge_pp": (((1 + r).prod() - 1) * 100) - (((1 + spy_y).prod() - 1) * 100),
            "n_months": len(r),
        })
    return pd.DataFrame(rows)


def top_drawdowns(eq: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    eqcurve = eq["equity"].astype(float).reset_index(drop=True)
    dates = eq["date"].reset_index(drop=True)
    peaks = eqcurve.cummax()
    dd = (eqcurve / peaks) - 1
    drawdowns = []
    in_dd = False
    start_idx = trough_idx = end_idx = 0
    for i in range(len(eqcurve)):
        if not in_dd and dd.iloc[i] < 0:
            in_dd = True
            start_idx = i - 1 if i > 0 else 0
            trough_idx = i
        elif in_dd:
            if eqcurve.iloc[i] >= peaks.iloc[start_idx]:
                end_idx = i
                drawdowns.append({"start": dates.iloc[start_idx], "trough": dates.iloc[trough_idx],
                                  "end": dates.iloc[end_idx], "depth_pct": dd.iloc[trough_idx] * 100})
                in_dd = False
            elif dd.iloc[i] < dd.iloc[trough_idx]:
                trough_idx = i
    if in_dd:
        end_idx = len(eqcurve) - 1
        drawdowns.append({"start": dates.iloc[start_idx], "trough": dates.iloc[trough_idx],
                          "end": dates.iloc[end_idx], "depth_pct": dd.iloc[trough_idx] * 100})
    df = pd.DataFrame(drawdowns)
    if len(df) > 0:
        df = df.sort_values("depth_pct").head(n)
    return df


def most_picked(eq: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    counter = {}
    for picks in eq["picks"]:
        if isinstance(picks, str) and picks:
            for sleeve in picks.split(";"):
                for tk in sleeve.split("|"):
                    if tk:
                        counter[tk] = counter.get(tk, 0) + 1
    rows = sorted(counter.items(), key=lambda x: -x[1])[:top_n]
    return pd.DataFrame(rows, columns=["ticker", "count"])


def bias_sensitivity(name: str, cfg: V8Config, score_strategy: str, alphas=(0, 0.02, 0.04, 0.08), seeds=(1,2,3,4,5)) -> pd.DataFrame:
    """Synthetic survivorship-style: drop tickers from monthly_returns at rate α/yr."""
    mr = pd.read_parquet(CACHE / "v2" / "monthly_returns_clean.parquet")
    spy = load_spy_features()
    sp = build_score_panel(score_strategy)
    rows = []
    for alpha in alphas:
        for seed in seeds:
            rng = np.random.default_rng(seed * 1000 + int(alpha * 100))
            mr_perturbed = mr.copy().astype(float)
            if alpha > 0:
                p_drop = alpha / 12
                drop_mask = rng.random(mr_perturbed.shape) < p_drop
                arr = mr_perturbed.to_numpy().copy()
                arr[drop_mask] = np.nan
                mr_perturbed = pd.DataFrame(arr, index=mr_perturbed.index, columns=mr_perturbed.columns)
                if "SPY" in mr_perturbed.columns:
                    mr_perturbed["SPY"] = mr["SPY"]
            eq = simulate_v8(cfg, sp, mr_perturbed, spy, daily_prices=None)
            m = evaluate_v8(eq, mr_perturbed, name=f"{name}_alpha{alpha}_s{seed}")
            rows.append({
                "name": name, "alpha": alpha, "seed": seed,
                "wf_mean_cagr": m["wf_mean_cagr"],
                "cagr_full": m["cagr_full"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "wf_n_pos": m["wf_n_pos"],
                "wf_n_beats_spy": m["wf_n_beats_spy"],
            })
            if alpha == 0:
                break  # only one run for alpha=0
    return pd.DataFrame(rows)


def generalize_universe(name: str, cfg: V8Config, score_strategy: str) -> pd.DataFrame:
    """Run the same config on different universes."""
    rows = []
    mr = pd.read_parquet(CACHE / "v2" / "monthly_returns_clean.parquet")
    spy = load_spy_features()
    universes = {}
    # Home: SP500 PIT (already what build_score_panel does)
    universes["sp500_pit"] = build_score_panel(score_strategy)
    # broader: all tickers (no membership filter — use all preds)
    pred = pd.read_parquet(CACHE / "v2" / "ml_preds_v2.parquet")
    pred["asof"] = pd.to_datetime(pred["asof"])
    pred = pred.rename(columns={"pred": "score"})
    pred["score"] = (pred["pred_3m"] + pred["pred_6m"]) / 2
    EXC = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
           "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
           "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD"}
    pred = pred[~pred["ticker"].isin(EXC)]
    # vol_1y from features
    panel_root = CACHE / "features"
    feat_rows = []
    for f in sorted(panel_root.glob("*.parquet")):
        d = pd.Timestamp(f.stem)
        df = pd.read_parquet(f)
        if "vol_1y" not in df.columns:
            continue
        v = df[["vol_1y"]].copy()
        v["asof"] = d
        v["ticker"] = v.index
        feat_rows.append(v.reset_index(drop=True))
    vols = pd.concat(feat_rows, ignore_index=True) if feat_rows else pd.DataFrame()
    pred_b = pred.merge(vols, on=["asof", "ticker"], how="left") if len(vols) else pred
    pred_b["vol_rank"] = pred_b.groupby("asof")["vol_1y"].rank(pct=True)
    universes["broader"] = pred_b[["asof", "ticker", "score", "vol_1y", "vol_rank"]].copy()

    # non_sp500: not in PIT
    mem = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    mem["asof"] = pd.to_datetime(mem["asof"])
    mem["in_sp500"] = True
    pred_ns = pred_b.merge(mem[["asof", "ticker", "in_sp500"]], on=["asof", "ticker"], how="left")
    pred_ns = pred_ns[pred_ns["in_sp500"].isna()].drop(columns=["in_sp500"])
    universes["non_sp500"] = pred_ns

    # random_500 seeded (deterministic per seed)
    all_tickers = pred_b["ticker"].unique()
    rng_master = np.random.default_rng(42)
    for seed in [1, 2, 3, 4, 5]:
        rng = np.random.default_rng(seed)
        sample = rng.choice(all_tickers, size=min(500, len(all_tickers)), replace=False)
        universes[f"random_500_seed{seed}"] = pred_b[pred_b["ticker"].isin(sample)].copy()

    for u_name, sp in universes.items():
        try:
            eq = simulate_v8(cfg, sp, mr, spy, daily_prices=None)
            m = evaluate_v8(eq, mr, name=f"{name}_{u_name}")
            rows.append({
                "name": name, "universe": u_name,
                "n_rows": len(sp), "n_tickers": sp["ticker"].nunique(),
                "wf_mean_cagr": m["wf_mean_cagr"],
                "cagr_full": m["cagr_full"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "wf_n_pos": m["wf_n_pos"],
                "wf_n_beats_spy": m["wf_n_beats_spy"],
            })
        except Exception as e:
            print(f"FAIL {u_name}: {e}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
WINNERS = {
    "v8_moderate": V8Config(
        name="v8_moderate",
        k_normal=3, k_bull=3, k_recovery=3,
        weighting="invvol", hold_months=12,
        gross_target=2.0,
        spy_trend_only_lever=True,
        regime_gate="combo",
        cash_yield_yr=0.03,
    ),
    "v8_max_cagr": V8Config(
        name="v8_max_cagr",
        k_normal=2, k_bull=2, k_recovery=2,
        weighting="invvol", hold_months=18,
        gross_target=2.5,
        spy_trend_only_lever=True,
        regime_gate="combo",
        cash_yield_yr=0.03,
    ),
    "v8_aggressive": V8Config(
        name="v8_aggressive",
        k_normal=3, k_bull=3, k_recovery=3,
        weighting="invvol", hold_months=12,
        gross_target=3.0,
        spy_trend_only_lever=True,
        regime_gate="combo",
        cash_yield_yr=0.03,
    ),
    "v8_safe": V8Config(
        name="v8_safe",
        k_normal=3, k_bull=3, k_recovery=3,
        weighting="invvol", hold_months=12,
        gross_target=1.5,
        spy_trend_only_lever=True,
        regime_gate="combo",
        cash_yield_yr=0.03,
    ),
    "v3_baseline": V8Config(
        name="v3_baseline",
        k_normal=3, k_bull=3, k_recovery=3,
        weighting="ew", hold_months=6,
        gross_target=1.0,
        regime_gate="tight",
        cash_yield_yr=0.0,
    ),
}


def main():
    mr = pd.read_parquet(CACHE / "v2" / "monthly_returns_clean.parquet")
    spy = load_spy_features()
    sp = build_score_panel("ml_3plus6")
    print(f"Score panel: {sp.shape}", flush=True)

    summary_rows = []
    for name, cfg in WINNERS.items():
        print(f"\n=== {name} ===", flush=True)
        eq = simulate_v8(cfg, sp, mr, spy, daily_prices=None)
        eq.to_csv(RESULTS / f"{name}_equity.csv", index=False)
        m = evaluate_v8(eq, mr, name=name)
        with open(RESULTS / f"{name}_metrics.json", "w") as f:
            json.dump(m, f, indent=2, default=str)
        # Per-split
        ps = detailed_per_split(eq, mr)
        ps.to_csv(RESULTS / f"{name}_per_split.csv", index=False)
        # Yearly
        yb = yearly_breakdown(eq, mr)
        yb.to_csv(RESULTS / f"{name}_yearly.csv", index=False)
        # Drawdowns
        dd = top_drawdowns(eq, n=8)
        dd.to_csv(RESULTS / f"{name}_drawdowns.csv", index=False)
        # Most-picked
        mp = most_picked(eq, top_n=40)
        mp.to_csv(RESULTS / f"{name}_most_picked.csv", index=False)
        # Print summary
        print(f"  WF mean CAGR: {m['wf_mean_cagr']*100:.2f}%  Full: {m['cagr_full']*100:.2f}%  "
              f"Sharpe: {m['sharpe']:.2f}  MaxDD: {m['max_dd']*100:.2f}%  "
              f"+/SPY: {m['wf_n_pos']}/{m['wf_n_beats_spy']}", flush=True)
        summary_rows.append({**m, "config_name": name, **{f"cfg_{k}": v for k, v in asdict(cfg).items()}})

    pd.DataFrame(summary_rows).to_csv(RESULTS / "winners_summary.csv", index=False)

    # Universe generalization (only for the winners we care about)
    print("\n=== Universe generalization ===", flush=True)
    rows = []
    for name in ["v8_moderate", "v8_max_cagr", "v8_safe", "v3_baseline"]:
        cfg = WINNERS[name]
        gdf = generalize_universe(name, cfg, "ml_3plus6")
        rows.append(gdf)
        print(gdf.to_string())
    pd.concat(rows, ignore_index=True).to_csv(RESULTS / "winners_generalize.csv", index=False)

    # Bias sensitivity
    print("\n=== Bias sensitivity (synthetic delistings) ===", flush=True)
    bias_rows = []
    for name in ["v8_moderate", "v8_max_cagr"]:
        cfg = WINNERS[name]
        bdf = bias_sensitivity(name, cfg, "ml_3plus6")
        bias_rows.append(bdf)
        print(bdf.to_string())
    pd.concat(bias_rows, ignore_index=True).to_csv(RESULTS / "winners_bias.csv", index=False)


if __name__ == "__main__":
    main()
