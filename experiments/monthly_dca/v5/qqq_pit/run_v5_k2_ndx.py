"""K=2 v5 backtest on NDX PIT — cross-index generalization test.

Mirrors experiments/monthly_dca/v5/qqq_pit/run_v5_on_qqq_pit.py but
runs at K=2 (the new SP500 deployment) on NDX PIT data. Tests whether
the K=2 v5 picker also generalizes to the Nasdaq-100 cohort or
whether it's a S&P-500-specific optimum.

Comparison: K=2 NDX PIT vs K=3 NDX PIT (existing baseline) vs K=2
SP500 augmented PIT (deployed).

Inputs (NDX, already exists):
  experiments/monthly_dca/v5/qqq_pit/ndx_pit_membership_monthly.parquet
  experiments/monthly_dca/v5/qqq_pit/ml_preds_v2_ndx.parquet
  experiments/monthly_dca/v5/qqq_pit/ml_preds_chronos_ndx.parquet
  experiments/monthly_dca/v5/qqq_pit/ndx_monthly_returns.parquet
  experiments/monthly_dca/v5/qqq_pit/ndx_monthly_prices.parquet

Output:
  experiments/monthly_dca/v5/qqq_pit/v5_k2_ndx_summary.json
  experiments/monthly_dca/v5/qqq_pit/v5_k2_ndx_walkforward.csv
  experiments/monthly_dca/v5/qqq_pit/v5_k2_ndx_yearly.csv
  experiments/monthly_dca/v5/qqq_pit/v5_k2_ndx_equity.csv
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
NDX = ROOT / "experiments" / "monthly_dca" / "v5" / "qqq_pit"

# v5 K=2 config (matching deployed sp500 strategy)
CHR_Q = 0.45
CAP = 0.40
HOLD = 6
K = 2
COST_BPS = 10.0

EXCLUDE = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA"}

WF_SPLITS_NDX = [
    # NDX PIT data starts 2015-01, so we use later splits
    ("A1", "2017-01-01", "2022-12-31"),
    ("A2", "2018-01-01", "2024-12-31"),
    ("R1", "2018-01-01", "2020-12-31"),
    ("R2", "2019-01-01", "2021-12-31"),
    ("R3", "2020-01-01", "2022-12-31"),
    ("R4", "2021-01-01", "2023-12-31"),
    ("R5", "2022-01-01", "2024-12-31"),
    ("STRICT", "2023-01-01", "2024-12-31"),
]


def classify_regime_tight(s: dict) -> str:
    r21 = s.get("spy_ret_21d", 0.0); r6m = s.get("spy_mom_6_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    dsma = s.get("spy_dsma200", 0.0); mom12 = s.get("spy_mom_12_1", 0.0)
    if r21 <= -0.08 or (r6m <= -0.05 and r21 <= -0.03):
        return "crash"
    if streak >= 40 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


def load_spy_features():
    """Borrow SPY features from the v2 sp500 cache (regime gate uses SPY only)."""
    feat_dir = ROOT / "experiments" / "monthly_dca" / "cache" / "features"
    rows = []
    for f in sorted(feat_dir.glob("*.parquet")):
        d = pd.Timestamp(f.stem)
        df = pd.read_parquet(f)
        if "SPY" not in df.index:
            continue
        spy = df.loc["SPY"]
        rows.append({
            "asof": d,
            "spy_dsma200": float(spy.get("d_sma200", 0.0)),
            "spy_mom_12_1": float(spy.get("mom_12_1", 0.0)),
            "spy_mom_6_1": float(spy.get("mom_6_1", 0.0)),
            "spy_ret_21d": float(spy.get("ret_21d", 0.0)),
            "spy_below_200_streak": float(spy.get("max_below_200_streak", 0.0)),
        })
    return pd.DataFrame(rows).set_index("asof")


def calc_invvol_weights(tickers, mr, asof, cap):
    mr_idx = mr.index
    pos = mr_idx.searchsorted(asof, side="right") - 1
    if pos < 12:
        return np.ones(len(tickers)) / len(tickers)
    window = mr.iloc[max(0, pos - 11): pos + 1]
    vols = []
    for tk in tickers:
        if tk in window.columns:
            v = window[tk].dropna().std()
            vols.append(float(v) if v and not np.isnan(v) and v > 0 else np.nan)
        else:
            vols.append(np.nan)
    vols = np.array(vols)
    if np.all(np.isnan(vols)) or np.all(vols == 0):
        return np.ones(len(tickers)) / len(tickers)
    inv = np.where(np.isnan(vols) | (vols == 0), 1e-9, 1.0 / vols)
    w = inv / inv.sum()
    if cap < 1.0:
        for _ in range(8):
            over = w > cap
            if not over.any():
                break
            excess = (w[over] - cap).sum()
            w[over] = cap
            under = ~over
            if not under.any():
                break
            w[under] += excess * w[under] / w[under].sum()
        w = w / w.sum()
    return w


def run_v5_k(k: int, ml, chr_, spy, mr, members_g, months) -> tuple[pd.DataFrame, list]:
    """v5 sim at chosen K. Returns (eq_df, picks_log)."""
    cf = COST_BPS / 1e4
    ml_by = {a: g for a, g in ml.groupby("asof")}
    chr_by = {a: g for a, g in chr_.groupby("asof")}

    cur_picks = []; cur_weights = np.array([])
    cash = False; held_for = 0; equity = 1.0
    rows = []; picks_log = []

    for i, m in enumerate(months):
        regime = classify_regime_tight(spy.loc[m].to_dict() if m in spy.index else {})
        do_reb = (i == 0) or (held_for >= HOLD) or (cash != (regime == "crash"))
        ret_m = 0.0
        if not cash and cur_picks:
            mr_pos = mr.index.searchsorted(m)
            if mr_pos + 1 < len(mr.index):
                next_d = mr.index[mr_pos + 1]
                pick_rets = [0.0 if pd.isna(mr.at[next_d, tk]) else float(mr.at[next_d, tk])
                              for tk in cur_picks if tk in mr.columns]
                if len(pick_rets) == len(cur_weights):
                    ret_m = float((np.array(pick_rets) * cur_weights).sum())
                    equity *= (1 + ret_m)

        if do_reb:
            equity *= (1 - cf)
            if regime == "crash":
                cur_picks = []; cur_weights = np.array([]); cash = True
            else:
                sub_ml = ml_by.get(m); sub_chr = chr_by.get(m)
                if sub_ml is None:
                    cur_picks = []; cur_weights = np.array([])
                else:
                    pit_set = members_g.get(m, set())
                    sub = sub_ml[sub_ml["ticker"].isin(pit_set)].copy()
                    sub = sub[~sub["ticker"].isin(EXCLUDE)]
                    sub["ml_score"] = (sub["pred_3m"] + sub["pred_6m"]) / 2
                    sub = sub.dropna(subset=["ml_score"])
                    if CHR_Q > 0 and sub_chr is not None and not sub_chr.empty:
                        sub = sub.merge(sub_chr[["ticker", "chronos_p70_3m"]], on="ticker", how="left")
                        sub = sub.dropna(subset=["chronos_p70_3m"])
                        sub["chr_p70_rk"] = sub["chronos_p70_3m"].rank(pct=True)
                        sub = sub[sub["chr_p70_rk"] >= CHR_Q]
                    sub = sub.sort_values("ml_score", ascending=False)
                    top = sub.head(k)
                    if len(top) < k:
                        cur_picks = []; cur_weights = np.array([])
                    else:
                        cur_picks = top["ticker"].tolist()
                        cur_weights = calc_invvol_weights(cur_picks, mr, m, cap=CAP)
                        for tk, w in zip(cur_picks, cur_weights):
                            picks_log.append({"asof": m, "ticker": tk, "weight": float(w),
                                              "regime": regime})
                cash = False
            held_for = 0
        else:
            held_for += 1
        rows.append({"date": m, "regime": regime, "equity": equity, "ret_m": ret_m,
                     "cash": cash, "picks": ",".join(cur_picks)})
    return pd.DataFrame(rows), picks_log


def wf_table(eq, mr, splits, spy_series):
    spy_ret = spy_series
    next_months = pd.DatetimeIndex(eq["date"]) + pd.offsets.MonthEnd(1)
    spy_aligned = [float(spy_ret.loc[nxt]) if nxt in spy_ret.index else 0.0 for nxt in next_months]
    spy_df = pd.DataFrame({"date": eq["date"], "spy_ret_m": spy_aligned})
    rows = []
    for split, lo, hi in splits:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)].copy()
        if len(e) == 0:
            continue
        r = e["ret_m"].astype(float); ec = (1 + r).cumprod()
        cagr_v = (ec.iloc[-1]) ** (12.0 / len(ec)) - 1
        sh = (r.mean() / max(r.std(), 1e-9)) * np.sqrt(12)
        peak = ec.cummax(); mdd = float(((ec - peak) / peak).min())
        s = spy_df[(spy_df["date"] >= lo) & (spy_df["date"] <= hi)]
        sr = s["spy_ret_m"].astype(float); sc = (1 + sr).cumprod()
        scgr = (sc.iloc[-1]) ** (12.0 / len(sc)) - 1
        rows.append({
            "split": split, "from": str(lo.date()), "to": str(hi.date()),
            "n_m": int(len(e)), "cagr": float(cagr_v), "spy_cagr": float(scgr),
            "edge_pp": float((cagr_v - scgr) * 100), "sharpe": float(sh),
            "max_dd": float(mdd),
        })
    return pd.DataFrame(rows)


def yearly_table(eq, spy_series):
    eq = eq.copy()
    eq["year"] = pd.to_datetime(eq["date"]).dt.year
    yr = eq.groupby("year")["ret_m"].apply(lambda r: (1 + r).prod() - 1)
    spy_yr = spy_series.groupby(spy_series.index.year).apply(
        lambda r: (1 + r.dropna()).prod() - 1)
    rows = []
    for y in sorted(yr.index):
        rows.append({"year": int(y), "strategy_pct": float(yr[y] * 100),
                     "spy_pct": float(spy_yr.get(y, 0) * 100),
                     "qqq_pct": float(spy_yr.get(y, 0) * 100),
                     "edge_pp": float((yr[y] - spy_yr.get(y, 0)) * 100)})
    return pd.DataFrame(rows)


def main():
    print("=" * 64)
    print("K=2 v5 backtest on NDX PIT")
    print("=" * 64)

    mem = pd.read_parquet(NDX / "ndx_pit_membership_monthly.parquet")
    mem["asof"] = pd.to_datetime(mem["asof"])
    members_g = mem.groupby("asof")["ticker"].apply(set).to_dict()
    print(f"NDX PIT: {len(mem)} rows, {mem['ticker'].nunique()} unique tickers, "
          f"{mem['asof'].nunique()} months ({mem['asof'].min().date()}..{mem['asof'].max().date()})")

    ml = pd.read_parquet(NDX / "ml_preds_v2_ndx.parquet")
    ml["asof"] = pd.to_datetime(ml["asof"])
    chr_ = pd.read_parquet(NDX / "ml_preds_chronos_ndx.parquet")
    chr_["asof"] = pd.to_datetime(chr_["asof"])

    mr = pd.read_parquet(NDX / "ndx_monthly_returns.parquet").fillna(0.0)
    mp = pd.read_parquet(NDX / "ndx_monthly_prices.parquet")
    if not isinstance(mr.index, pd.DatetimeIndex):
        mr.index = pd.to_datetime(mr.index)
        mp.index = pd.to_datetime(mp.index)

    spy = load_spy_features()

    # Load QQQ as the natural NDX benchmark (SPY for comparison context).
    # Try the SP500 monthly returns cache for both.
    sp_mr = pd.read_parquet(ROOT / "experiments" / "monthly_dca" / "cache" / "v2"
                            / "monthly_returns_clean.parquet")
    if not isinstance(sp_mr.index, pd.DatetimeIndex):
        sp_mr.index = pd.to_datetime(sp_mr.index)
    spy_series = sp_mr["SPY"].dropna() if "SPY" in sp_mr.columns else pd.Series(dtype=float)
    qqq_series = sp_mr["QQQ"].dropna() if "QQQ" in sp_mr.columns else pd.Series(dtype=float)
    if qqq_series.empty and "QQQ" in mr.columns:
        qqq_series = mr["QQQ"].dropna()
    print(f"SPY benchmark: {len(spy_series)} months;  QQQ benchmark: {len(qqq_series)} months")

    # Months to simulate: intersection of NDX PIT and ml/Chronos data, NDX PIT
    # starts 2015-01. Use only months where we have all three.
    months = sorted(
        set(mem["asof"].unique())
        .intersection(set(pd.to_datetime(ml["asof"]).unique()))
        .intersection(set(spy.index))
    )
    months = [pd.Timestamp(m) for m in months]
    print(f"Simulation months: {len(months)} ({months[0].date()}..{months[-1].date()})")

    # Run K=2 and K=3 side-by-side, against both SPY and QQQ
    summary = {}
    for k in (2, 3):
        print(f"\n--- K={k} ---")
        eq, picks_log = run_v5_k(k, ml, chr_, spy, mr, members_g, months)
        n = len(eq); cagr = (eq["equity"].iloc[-1]) ** (12 / n) - 1
        r = eq["ret_m"].astype(float); sharpe = (r.mean() / max(r.std(), 1e-9)) * np.sqrt(12)
        peak = eq["equity"].cummax(); mdd = float(((eq["equity"] - peak) / peak).min())
        wf_spy = wf_table(eq, mr, WF_SPLITS_NDX, spy_series)
        wf_qqq = wf_table(eq, mr, WF_SPLITS_NDX, qqq_series)
        yr_spy = yearly_table(eq, spy_series)
        yr_qqq = yearly_table(eq, qqq_series)
        print(f"  Full CAGR {cagr*100:.2f}%   Sharpe {sharpe:.2f}   MaxDD {mdd*100:.2f}%")
        print(f"  vs SPY:  WF mean {wf_spy['cagr'].mean()*100:.2f}%, "
              f"beats {int((wf_spy['cagr'] > wf_spy['spy_cagr']).sum())}/{len(wf_spy)}")
        print(f"  vs QQQ:  WF mean {wf_qqq['cagr'].mean()*100:.2f}%, "
              f"beats {int((wf_qqq['cagr'] > wf_qqq['spy_cagr']).sum())}/{len(wf_qqq)}")
        suffix = "k2" if k == 2 else "k3"
        eq.to_csv(NDX / f"v5_{suffix}_ndx_equity.csv", index=False)
        wf_spy.to_csv(NDX / f"v5_{suffix}_ndx_walkforward_vs_spy.csv", index=False)
        wf_qqq.to_csv(NDX / f"v5_{suffix}_ndx_walkforward_vs_qqq.csv", index=False)
        yr_spy.to_csv(NDX / f"v5_{suffix}_ndx_yearly_vs_spy.csv", index=False)
        yr_qqq.to_csv(NDX / f"v5_{suffix}_ndx_yearly_vs_qqq.csv", index=False)
        wf = wf_qqq  # Use QQQ as primary NDX-natural benchmark for the summary
        summary[f"K{k}"] = {
            "n_months": int(n),
            "cagr_full": float(cagr),
            "sharpe": float(sharpe),
            "max_dd": float(mdd),
            "wf_mean_cagr": float(wf["cagr"].mean()),
            "wf_median_cagr": float(wf["cagr"].median()),
            "wf_min_cagr": float(wf["cagr"].min()),
            "wf_max_cagr": float(wf["cagr"].max()),
            "wf_mean_edge_pp_vs_qqq": float(wf_qqq["edge_pp"].mean()),
            "wf_mean_edge_pp_vs_spy": float(wf_spy["edge_pp"].mean()),
            "wf_n_beats_qqq": int((wf_qqq["cagr"] > wf_qqq["spy_cagr"]).sum()),
            "wf_n_beats_spy": int((wf_spy["cagr"] > wf_spy["spy_cagr"]).sum()),
            "wf_n_positive": int((wf["cagr"] > 0).sum()),
            "wf_n_splits": int(len(wf)),
            "n_picks_total": int(len(picks_log)),
        }

    (NDX / "v5_k2_ndx_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved -> {NDX / 'v5_k2_ndx_summary.json'}")

    print("\n=== K=2 vs K=3 on NDX PIT (vs QQQ benchmark) ===")
    print(f"{'Metric':<28}{'K=2':>12}{'K=3':>12}")
    for k in ("cagr_full", "sharpe", "max_dd", "wf_mean_cagr",
              "wf_mean_edge_pp_vs_qqq", "wf_mean_edge_pp_vs_spy",
              "wf_n_beats_qqq", "wf_n_beats_spy"):
        v2 = summary["K2"][k]; v3 = summary["K3"][k]
        if isinstance(v2, float):
            if k in ("cagr_full", "wf_mean_cagr", "max_dd"):
                print(f"{k:<28}{v2*100:>11.2f}%{v3*100:>11.2f}%")
            elif k.endswith("_pp_vs_qqq") or k.endswith("_pp_vs_spy"):
                print(f"{k:<28}{v2:>+11.2f}pp{v3:>+11.2f}pp")
            else:
                print(f"{k:<28}{v2:>12.2f}{v3:>12.2f}")
        else:
            print(f"{k:<28}{v2:>12}{v3:>12}")


if __name__ == "__main__":
    main()
