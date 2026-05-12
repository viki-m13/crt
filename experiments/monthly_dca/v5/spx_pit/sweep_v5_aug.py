"""Phase 7: parameter sweep for v5 on the augmented PIT panel.

Re-tunes the v5 hyperparameters on PIT-corrected data. The deployed v5
(k=3, Chronos q=0.45, h=6, cap=0.40) was tuned on the biased panel,
where the k=15 v3-baseline went UP under PIT correction while the
k=3 v3-winner went DOWN. The optimum may have shifted toward larger K.

Sweeps:
  1. K-sweep:        K ∈ {1, 2, 3, 5, 7, 10, 15}, defaults elsewhere
  2. Chronos sweep:  q ∈ {0.0 (off), 0.30, 0.40, 0.45, 0.50, 0.60}, K=3
  3. Hold sweep:     h ∈ {1, 3, 6, 9, 12}, K=3, q=0.45
  4. No-Chronos:     K ∈ {3, 5, 7, 10} with q=0.0 (Chronos disabled)

Output:
  augmented/v5_param_sweep_results.csv  one row per config with headline metrics
  augmented/v5_param_sweep_winner.json  best config (by WF mean CAGR)
"""
from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
PIT = CACHE / "v2" / "sp500_pit"
AUG = PIT / "augmented"

EXCLUDE = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
           "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
           "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD"}

COST_BPS = 10.0
WF_SPLITS = [
    ("A1",       "2011-01-01", "2018-12-31"),
    ("A2",       "2015-01-01", "2021-12-31"),
    ("A3",       "2018-01-01", "2024-12-31"),
    ("R1_GFC",   "2008-01-01", "2010-12-31"),
    ("R2",       "2011-01-01", "2013-12-31"),
    ("R3",       "2014-01-01", "2016-12-31"),
    ("R4",       "2017-01-01", "2019-12-31"),
    ("R5_COVID", "2020-01-01", "2022-12-31"),
    ("R6_AI",    "2023-01-01", "2024-12-31"),
    ("STRICT",   "2021-01-01", "2024-12-31"),
]


def classify_regime_tight(s: dict) -> str:
    r21 = s.get("spy_ret_21d", 0.0)
    r6m = s.get("spy_mom_6_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    dsma = s.get("spy_dsma200", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0)
    if r21 <= -0.08 or (r6m <= -0.05 and r21 <= -0.03):
        return "crash"
    if streak >= 40 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


def load_spy_features():
    rows = []
    for f in sorted((AUG / "features").glob("*.parquet")):
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


def calc_invvol_weights(tickers, monthly_returns, asof, cap):
    mr_idx = monthly_returns.index
    pos = mr_idx.searchsorted(asof)
    if pos == 0:
        return np.ones(len(tickers)) / len(tickers)
    window = monthly_returns.iloc[max(0, pos - 12): pos]
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


def run_one(cfg: dict, data: dict) -> dict:
    """Run one variant: cfg has k, chr_q, hold, cap, scorer (default ml_3plus6)."""
    k = cfg["k"]
    chr_q = cfg["chr_q"]
    hold = cfg["hold"]
    cap = cfg["cap"]
    scorer = cfg.get("scorer", "ml_3plus6")

    panel = data["panel"]; ml = data["ml"]; chr_ = data["chr_"]
    spy = data["spy"]; mr = data["mr"]; mp = data["mp"]; members_g = data["members_g"]

    panel_by_asof = data["panel_by_asof"]
    ml_by_asof = data["ml_by_asof"]
    chr_by_asof = data["chr_by_asof"]

    months = data["months"]
    cf = COST_BPS / 1e4

    cur_picks = []
    cur_weights = np.array([])
    cash = False
    held_for = 0
    equity = 1.0
    rows = []

    for i, m in enumerate(months):
        regime = classify_regime_tight(spy.loc[m].to_dict() if m in spy.index else {})
        do_reb = (i == 0) or (held_for >= hold) or (cash != (regime == "crash"))
        ret_m = 0.0
        if not cash and cur_picks:
            mr_pos = mr.index.searchsorted(m)
            if mr_pos + 1 < len(mr.index):
                next_d = mr.index[mr_pos + 1]
                pick_rets = []
                for tk in cur_picks:
                    if tk in mr.columns and next_d in mr.index:
                        r = mr.at[next_d, tk]
                        pick_rets.append(0.0 if pd.isna(r) else float(r))
                    else:
                        pick_rets.append(0.0)
                ret_m = float((np.array(pick_rets) * cur_weights).sum())
                equity *= (1 + ret_m)

        if do_reb:
            equity *= (1 - cf)
            if regime == "crash":
                cur_picks = []; cur_weights = np.array([]); cash = True
            else:
                sub_panel = panel_by_asof.get(m)
                sub_ml = ml_by_asof.get(m)
                sub_chr = chr_by_asof.get(m)
                if sub_panel is None or sub_ml is None:
                    cur_picks = []; cur_weights = np.array([])
                else:
                    sp_set = members_g.get(m, set())
                    sub = sub_panel[sub_panel["ticker"].isin(sp_set)]
                    sub = sub[~sub["ticker"].isin(EXCLUDE)]
                    # Pick score column based on scorer
                    score_col = "ml_score"
                    sub = sub.merge(sub_ml[["ticker", score_col]], on="ticker", how="left")
                    sub = sub.dropna(subset=[score_col])
                    if chr_q > 0 and sub_chr is not None and not sub_chr.empty:
                        sub = sub.merge(sub_chr[["ticker", "chronos_p70_3m"]],
                                        on="ticker", how="left")
                        sub = sub.dropna(subset=["chronos_p70_3m"])
                        sub["chr_p70_rk"] = sub["chronos_p70_3m"].rank(pct=True)
                        sub = sub[sub["chr_p70_rk"] >= chr_q]
                    sub = sub.sort_values(score_col, ascending=False)
                    top = sub.head(k)
                    if len(top) < k:
                        cur_picks = []; cur_weights = np.array([])
                    else:
                        cur_picks = top["ticker"].tolist()
                        cur_weights = calc_invvol_weights(cur_picks, mr, m, cap=cap)
                cash = False
            held_for = 0
        else:
            held_for += 1

        rows.append({"date": m, "regime": regime, "equity": equity, "ret_m": ret_m,
                     "cash": cash, "n_picks": len(cur_picks)})

    eq = pd.DataFrame(rows)
    n_months = len(eq)
    cagr_full = (eq["equity"].iloc[-1]) ** (12 / n_months) - 1
    r = eq["ret_m"].astype(float)
    sharpe = (r.mean() / max(r.std(), 1e-9)) * np.sqrt(12)
    peak = eq["equity"].cummax()
    mdd = float(((eq["equity"] - peak) / peak).min())

    # WF
    spy_ret = mr["SPY"].dropna()
    next_months = pd.DatetimeIndex(eq["date"]) + pd.offsets.MonthEnd(1)
    spy_aligned = [float(spy_ret.loc[nxt]) if nxt in spy_ret.index else 0.0 for nxt in next_months]
    spy_df = pd.DataFrame({"date": eq["date"], "spy_ret_m": spy_aligned})

    wf_rows = []
    for split, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)].copy()
        if len(e) == 0:
            continue
        rr = e["ret_m"].astype(float)
        ec = (1 + rr).cumprod()
        cagr_v = (ec.iloc[-1]) ** (12.0 / len(ec)) - 1
        s = spy_df[(spy_df["date"] >= lo) & (spy_df["date"] <= hi)]
        sr = s["spy_ret_m"].astype(float)
        sc = (1 + sr).cumprod()
        scgr = (sc.iloc[-1]) ** (12.0 / len(sc)) - 1
        wf_rows.append({"split": split, "cagr": cagr_v, "spy_cagr": scgr})
    wf = pd.DataFrame(wf_rows)

    spy_full = (1 + spy_ret.loc[eq["date"].iloc[0]:eq["date"].iloc[-1]]).prod() ** (12 / n_months) - 1

    return {
        "k": k, "chr_q": chr_q, "hold": hold, "cap": cap, "scorer": scorer,
        "n_months": int(n_months),
        "cagr_full": float(cagr_full),
        "spy_cagr_full": float(spy_full),
        "edge_full_pp": float((cagr_full - spy_full) * 100),
        "sharpe": float(sharpe),
        "max_dd": float(mdd),
        "wf_mean_cagr": float(wf["cagr"].mean()) if len(wf) else None,
        "wf_median_cagr": float(wf["cagr"].median()) if len(wf) else None,
        "wf_min_cagr": float(wf["cagr"].min()) if len(wf) else None,
        "wf_n_beats_spy": int((wf["cagr"] > wf["spy_cagr"]).sum()) if len(wf) else 0,
        "wf_n_positive": int((wf["cagr"] > 0).sum()) if len(wf) else 0,
        "wf_n_splits": int(len(wf)),
    }


def main():
    t0 = time.time()
    print("Loading augmented data ...")
    panel = pd.read_parquet(AUG / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    ml = pd.read_parquet(AUG / "ml_preds.parquet")[["asof", "ticker", "pred_3m", "pred_6m"]]
    ml["asof"] = pd.to_datetime(ml["asof"])
    ml["ml_score"] = (ml["pred_3m"] + ml["pred_6m"]) / 2
    chr_ = pd.read_parquet(AUG / "ml_preds_chronos.parquet")[["asof", "ticker", "chronos_p70_3m"]]
    chr_["asof"] = pd.to_datetime(chr_["asof"])
    spy = load_spy_features()
    mr = pd.read_parquet(AUG / "monthly_returns_clean.parquet").fillna(0.0)
    mp = pd.read_parquet(AUG / "monthly_prices_clean.parquet")
    if not isinstance(mr.index, pd.DatetimeIndex):
        mr.index = pd.to_datetime(mr.index)
        mp.index = pd.to_datetime(mp.index)
    members = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    members["asof"] = pd.to_datetime(members["asof"])
    members_g = members.groupby("asof")["ticker"].apply(set).to_dict()

    panel_by_asof = {a: g for a, g in panel.groupby("asof")}
    ml_by_asof = {a: g for a, g in ml.groupby("asof")}
    chr_by_asof = {a: g for a, g in chr_.groupby("asof")}
    months = sorted(set(panel["asof"]).intersection(set(spy.index)))
    months = [pd.Timestamp(m) for m in months]

    data = dict(panel=panel, ml=ml, chr_=chr_, spy=spy, mr=mr, mp=mp, members_g=members_g,
                panel_by_asof=panel_by_asof, ml_by_asof=ml_by_asof, chr_by_asof=chr_by_asof,
                months=months)

    # Build sweep grid
    DEFAULT = dict(k=3, chr_q=0.45, hold=6, cap=0.40)
    configs = []
    # 1. K sweep
    for k in [1, 2, 3, 5, 7, 10, 15]:
        configs.append({**DEFAULT, "k": k, "_sweep": "K"})
    # 2. Chronos q sweep
    for q in [0.0, 0.30, 0.40, 0.45, 0.50, 0.60]:
        configs.append({**DEFAULT, "chr_q": q, "_sweep": "chr_q"})
    # 3. Hold sweep
    for h in [1, 3, 6, 9, 12]:
        configs.append({**DEFAULT, "hold": h, "_sweep": "hold"})
    # 4. No-Chronos x K
    for k in [3, 5, 7, 10, 15]:
        configs.append({**DEFAULT, "k": k, "chr_q": 0.0, "_sweep": "no_chronos_K"})
    # 5. Cap sweep
    for cap in [0.34, 0.40, 0.50, 1.0]:
        configs.append({**DEFAULT, "cap": cap, "_sweep": "cap"})

    # Dedup
    seen = set()
    unique_configs = []
    for c in configs:
        key = (c["k"], c["chr_q"], c["hold"], c["cap"])
        if key in seen:
            continue
        seen.add(key)
        unique_configs.append(c)
    print(f"Total unique configs: {len(unique_configs)}")

    results = []
    for i, cfg in enumerate(unique_configs):
        sweep = cfg.pop("_sweep", "")
        r = run_one(cfg, data)
        r["sweep"] = sweep
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(unique_configs)}] {sweep:>12} "
              f"k={cfg['k']:>2} q={cfg['chr_q']:.2f} h={cfg['hold']:>2} cap={cfg['cap']:.2f}  "
              f"cagr={r['cagr_full']*100:>6.1f}%  "
              f"wf_mean={r['wf_mean_cagr']*100:>6.1f}%  "
              f"sharpe={r['sharpe']:.2f}  "
              f"dd={r['max_dd']*100:>6.1f}%  "
              f"beats={r['wf_n_beats_spy']}/{r['wf_n_splits']}  "
              f"({elapsed:.0f}s)")
        results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(AUG / "v5_param_sweep_results.csv", index=False)
    print(f"\nSaved -> {AUG / 'v5_param_sweep_results.csv'}")

    # Winner by WF mean CAGR
    winner = df.sort_values("wf_mean_cagr", ascending=False).iloc[0].to_dict()
    print(f"\nWinner by WF mean CAGR:")
    for k, v in winner.items():
        print(f"  {k}: {v}")
    (AUG / "v5_param_sweep_winner.json").write_text(json.dumps(winner, indent=2, default=str))


if __name__ == "__main__":
    main()
