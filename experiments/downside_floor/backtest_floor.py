"""Production backtest for the Floor picker.

Locks the deployed FloorScore and a max-safety variant, benchmarks them
against the universe average, naive low-vol, and "just buy SPY", and
checks temporal robustness (by era and by year) so the edge is not a
single-period artifact.

FloorScore (deployed) — higher = safer, a transparent z-score blend of
five leakage-free, economically-motivated ingredients:
    + gbm_maxdd_3m       learned forecast of max drawdown-from-entry (shallow)
    - gbm_uw_frac_3m     learned forecast of underwater fraction (low)
    - vol_3m_xs          realized 3-month volatility (low)
    + trend_health_5y_xs durable long-term uptrend (high) -> avoids value traps
    + chr_trough_q30_3m  Chronos-Bolt 30th-pct path trough cushion (shallow)

floor_maxsafe (variant) — pure downside rank-mix, no upside tilt.

Outputs: floor_backtest_results.json, floor_picks.csv, spy_underwater.csv
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from floor_lib import build, HERE, ROOT, HORIZONS

PIT = ROOT / "experiments" / "monthly_dca" / "cache" / "v2" / "sp500_pit"
K = 10
HZD = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}


def z(s):
    s = s.astype(float)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 1e-12 else s * 0.0


def floor_score(g):
    return (z(g["gbm_maxdd_3m"]) - z(g["gbm_uw_frac_3m"]) - z(g["vol_3m_xs"])
            + 0.5 * z(g["trend_health_5y_xs"]) + 0.5 * z(g["chr_trough_q30_3m"]))


def maxsafe_rank(g):
    return ((-g["gbm_maxdd_3m"]).rank() + g["gbm_uw_frac_3m"].rank()
            + g["vol_3m_xs"].rank() + (-g["trend_health_5y_xs"]).rank())


def spy_underwater():
    """Forward underwater stats for buying SPY at each asof."""
    spy = pd.read_parquet(PIT / "prices_extended_pit.parquet", columns=["SPY"])["SPY"].dropna()
    gdates = spy.index.values
    df = build()
    asofs = sorted(df["asof"].unique())
    rows = []
    for d in asofs:
        e = np.searchsorted(gdates, np.datetime64(d), side="right") - 1
        if e < 0:
            continue
        entry = spy.values[e]
        rec = {"asof": d}
        for h, H in HZD.items():
            if e + H >= len(gdates):
                for f in ("uw_frac", "ever_below", "end_below", "maxdd", "end_ret"):
                    rec[f"{f}_{h}"] = np.nan
                rec[f"censored_{h}"] = True
                continue
            win = spy.values[e + 1:e + H + 1]
            below = win < entry
            rec[f"uw_frac_{h}"] = below.mean()
            rec[f"ever_below_{h}"] = int(below.any())
            rec[f"end_below_{h}"] = int(win[-1] < entry)
            rec[f"maxdd_{h}"] = win.min() / entry - 1
            rec[f"end_ret_{h}"] = win[-1] / entry - 1
            rec[f"censored_{h}"] = False
        rows.append(rec)
    return pd.DataFrame(rows)


def stats(rows, h):
    m = rows[~rows[f"censored_{h}"]]
    if len(m) == 0:
        return None
    return dict(n=int(len(m)), uw=float(m[f"uw_frac_{h}"].mean()),
                ever=float(m[f"ever_below_{h}"].mean()),
                endb=float(m[f"end_below_{h}"].mean()),
                dd=float(m[f"maxdd_{h}"].mean()),
                ret=float(m[f"end_ret_{h}"].mean()),
                medret=float(m[f"end_ret_{h}"].median()),
                safe=float((m[f"uw_frac_{h}"] < 0.15).mean()))


def main():
    df = build()
    pick_frames, pick_records = {"floor": [], "maxsafe": [], "universe": [],
                                 "lowvol": []}, []
    for t, g in df.groupby("asof"):
        if len(g) < K * 2:
            continue
        pick_frames["universe"].append(g)
        fl = g.loc[floor_score(g).nlargest(K).index]
        ms = g.loc[maxsafe_rank(g).nsmallest(K).index]
        lv = g.loc[g["vol_3m_xs"].nsmallest(K).index]
        pick_frames["floor"].append(fl)
        pick_frames["maxsafe"].append(ms)
        pick_frames["lowvol"].append(lv)
        for tk in fl["ticker"]:
            pick_records.append({"asof": t, "ticker": tk})

    spy = spy_underwater()

    # ---------- pooled, all horizons ----------
    pooled = {}
    for name, frames in pick_frames.items():
        pooled[name] = {h: stats(pd.concat(frames), h) for h in HORIZONS}
    pooled["spy"] = {h: stats(spy, h) for h in HORIZONS}

    print(f"=== POOLED realized downside (K={K}, 2003-2026) ===")
    for h in HORIZONS:
        print(f"\n-- {h} --")
        print(f"{'strategy':<12}{'n':>7}{'uw_frac':>9}{'P(ever<)':>10}"
              f"{'P(end<)':>9}{'maxdd':>8}{'meanret':>9}{'medret':>8}{'safe%':>7}")
        for name in ["spy", "universe", "lowvol", "maxsafe", "floor"]:
            s = pooled[name][h]
            if s is None:
                continue
            print(f"{name:<12}{s['n']:>7}{s['uw']:>9.3f}{s['ever']:>10.3f}"
                  f"{s['endb']:>9.3f}{s['dd']:>8.3f}{s['ret']:>9.3f}"
                  f"{s['medret']:>8.3f}{s['safe']*100:>6.1f}%")

    # ---------- temporal robustness: by era + by year (3m end_below & uw) ----------
    def era(d):
        y = pd.Timestamp(d).year
        return "2003-2009" if y <= 2009 else "2010-2019" if y <= 2019 else "2020-2026"

    eras = {}
    fl_all = pd.concat(pick_frames["floor"]); fl_all["era"] = fl_all["asof"].map(era)
    uni_all = pd.concat(pick_frames["universe"]); uni_all["era"] = uni_all["asof"].map(era)
    print("\n=== robustness by era (3m): FloorScore vs universe ===")
    print(f"{'era':<12}{'flr uw':>8}{'uni uw':>8}{'flr end<':>10}{'uni end<':>10}"
          f"{'flr safe%':>11}{'uni safe%':>11}")
    for e in ["2003-2009", "2010-2019", "2020-2026"]:
        fs = stats(fl_all[fl_all["era"] == e], "3m")
        us = stats(uni_all[uni_all["era"] == e], "3m")
        eras[e] = {"floor": fs, "universe": us}
        print(f"{e:<12}{fs['uw']:>8.3f}{us['uw']:>8.3f}{fs['endb']:>10.3f}"
              f"{us['endb']:>10.3f}{fs['safe']*100:>10.1f}%{us['safe']*100:>10.1f}%")

    # by-year win rate (does FloorScore beat universe end_below most years?)
    fl_all["year"] = fl_all["asof"].dt.year
    uni_all["year"] = uni_all["asof"].dt.year
    yrs, wins = [], 0
    for y in sorted(fl_all["year"].unique()):
        fy = stats(fl_all[fl_all["year"] == y], "3m")
        uy = stats(uni_all[uni_all["year"] == y], "3m")
        if fy and uy:
            w = fy["endb"] <= uy["endb"]
            wins += int(w)
            yrs.append({"year": int(y), "floor_endb": fy["endb"],
                        "uni_endb": uy["endb"], "win": bool(w)})
    print(f"\nyears FloorScore <= universe on P(end below) at 3m: "
          f"{wins}/{len(yrs)}")

    out = {"pooled": pooled, "by_era": eras,
           "by_year_3m_endbelow": yrs, "year_win_rate": f"{wins}/{len(yrs)}", "K": K}
    (HERE / "floor_backtest_results.json").write_text(json.dumps(out, indent=2, default=str))
    pd.DataFrame(pick_records).to_csv(HERE / "floor_picks.csv", index=False)
    spy.to_csv(HERE / "spy_underwater.csv", index=False)
    print(f"\nwrote floor_backtest_results.json, floor_picks.csv, spy_underwater.csv")


if __name__ == "__main__":
    main()
