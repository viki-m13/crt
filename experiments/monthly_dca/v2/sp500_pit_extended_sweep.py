"""Extended sweep on PIT S&P 500: K in {1,2,3,5,7}, holds in {1,3,6,12},
multi-horizon stacked ensembles, dynamic K, per-pick stop-loss.

Builds on the base sweep (where the winner was K=3 EW tight h=6 → 37.77%
WF mean CAGR).  We push for higher CAGR with stronger stability.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"
FEATURES_DIR = CACHE / "features"

EXCLUDE = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
           "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
           "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD"}

WF_SPLITS = [
    ("A1", "2011-01-01", "2018-12-31"),
    ("A2", "2015-01-01", "2021-12-31"),
    ("A3", "2018-01-01", "2024-12-31"),
    ("R1_GFC", "2008-01-01", "2010-12-31"),
    ("R2", "2011-01-01", "2013-12-31"),
    ("R3", "2014-01-01", "2016-12-31"),
    ("R4", "2017-01-01", "2019-12-31"),
    ("R5_COVID", "2020-01-01", "2022-12-31"),
    ("R6_AI", "2023-01-01", "2024-12-31"),
    ("STRICT", "2021-01-01", "2024-12-31"),
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


def classify_regime_strict(s: dict) -> str:
    """Stricter crash gate — fires more often. Pulls cash on broader DD."""
    r21 = s.get("spy_ret_21d", 0.0)
    r6m = s.get("spy_mom_6_1", 0.0)
    dsma = s.get("spy_dsma200", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0)
    rsi = s.get("spy_rsi14", 50.0)
    streak = s.get("spy_below_200_streak", 0.0)
    if r21 <= -0.05 or r6m <= -0.06 or (dsma < -0.03 and rsi < 42):
        return "crash"
    if streak >= 30 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.08 and dsma > 0:
        return "bull"
    return "normal"


def classify_regime_drawdown(s: dict) -> str:
    """DD-based gate."""
    dd = s.get("spy_dd_from_52wh", 0.0)
    r21 = s.get("spy_ret_21d", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    dsma = s.get("spy_dsma200", 0.0)
    if dd <= -0.10 and r21 < 0:
        return "crash"
    if streak >= 30 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


REGIME_GATES = {
    "tight": classify_regime_tight,
    "strict": classify_regime_strict,
    "ddgate": classify_regime_drawdown,
}


def load_spy_features() -> pd.DataFrame:
    rows = []
    for f in sorted(FEATURES_DIR.glob("*.parquet")):
        d = pd.Timestamp(f.stem)
        df = pd.read_parquet(f)
        if "SPY" not in df.index:
            continue
        spy = df.loc["SPY"]
        rows.append({
            "asof": d,
            "spy_dsma200": float(spy.get("d_sma200", 0.0)),
            "spy_rsi14": float(spy.get("rsi_14", 50.0)),
            "spy_mom_12_1": float(spy.get("mom_12_1", 0.0)),
            "spy_mom_6_1": float(spy.get("mom_6_1", 0.0)),
            "spy_ret_21d": float(spy.get("ret_21d", 0.0)),
            "spy_below_200_streak": float(spy.get("max_below_200_streak", 0.0)),
            "spy_dd_from_52wh": float(spy.get("dd_from_52wh", 0.0)),
        })
    return pd.DataFrame(rows).set_index("asof")


# ---------------------------------------------------------------------------
def build_panel_with_score(scorer: str) -> pd.DataFrame:
    panel = pd.read_parquet(PIT / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])

    if scorer == "ml_filter":
        ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred", "pred_1m", "pred_3m", "pred_6m"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        panel = panel.merge(ml.rename(columns={"pred": "score"}), on=["asof", "ticker"], how="left")
    elif scorer == "ml_h1":
        ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred_1m"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        panel = panel.merge(ml.rename(columns={"pred_1m": "score"}), on=["asof", "ticker"], how="left")
    elif scorer == "ml_h3":
        ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred_3m"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        panel = panel.merge(ml.rename(columns={"pred_3m": "score"}), on=["asof", "ticker"], how="left")
    elif scorer == "ml_h6":
        ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred_6m"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        panel = panel.merge(ml.rename(columns={"pred_6m": "score"}), on=["asof", "ticker"], how="left")
    elif scorer == "ml_3plus6":
        ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred_3m", "pred_6m"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        panel = panel.merge(ml, on=["asof", "ticker"], how="left")
        panel["score"] = (panel["pred_3m"] + panel["pred_6m"]) / 2
    elif scorer == "ml_filter_winsor":
        # Penalise very high-vol picks: ml score - 0.1*vol_xs (less risk)
        ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        panel = panel.merge(ml, on=["asof", "ticker"], how="left")
        panel["score"] = panel["pred"] - 0.10 * panel["vol_1y_xs"]
    elif scorer == "ml_q":
        # ML score + quality
        ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        panel = panel.merge(ml, on=["asof", "ticker"], how="left")
        panel["score"] = (panel["pred"] - panel["pred"].mean()) / panel["pred"].std()
        # blend with quality
        q = (panel["sharpe_5y_xs"] + panel["trend_health_5y_xs"] + panel["quality_score_5y_xs"])/3
        panel["score"] = 0.7 * panel["score"] + 0.3 * q
    elif scorer == "ml_filter_softmax":
        # Cross-sectional softmax of ML score within S&P 500
        ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        panel = panel.merge(ml, on=["asof", "ticker"], how="left")
        panel["score"] = panel.groupby("asof")["pred"].rank(pct=True)
    else:
        raise ValueError(scorer)
    return panel


# ---------------------------------------------------------------------------
@dataclass
class Variant:
    name: str
    scorer: str
    k_normal: int
    k_recovery: int
    k_bull: int
    weighting: str
    regime_gate: str
    hold_months: int
    cap_per_pick: float = 1.0       # max weight per pick (1.0 = no cap)


def simulate_variant(panel: pd.DataFrame, monthly_returns: pd.DataFrame,
                     spy_features: pd.DataFrame, v: Variant,
                     cost_bps: float = 10.0) -> pd.DataFrame:
    p = panel.dropna(subset=["score"]).copy()
    p = p[~p["ticker"].isin(EXCLUDE)]
    months = sorted(p["asof"].unique())
    cls = REGIME_GATES[v.regime_gate]

    by_asof = {pd.Timestamp(d): g for d, g in p.groupby("asof")}
    mr_idx = monthly_returns.index

    equity = 1.0
    cf = cost_bps / 10000.0
    cur_picks: list[str] = []
    cur_weights: np.ndarray = np.array([])
    held_for = 0
    cash = False
    rows = []

    for i, m in enumerate(months):
        m = pd.Timestamp(m)
        do_reb = (i == 0) or (held_for >= v.hold_months) or cash

        if do_reb:
            spy_now = spy_features.loc[m].to_dict() if m in spy_features.index else {}
            regime = cls(spy_now)
            if regime == "crash":
                cur_picks, cur_weights, cash = [], np.array([]), True
                held_for = 0
            else:
                k = {"recovery": v.k_recovery, "bull": v.k_bull, "normal": v.k_normal}[regime]
                sub = by_asof.get(m, pd.DataFrame())
                if len(sub) < k:
                    cur_picks, cur_weights, cash = [], np.array([]), True
                else:
                    top = sub.sort_values("score", ascending=False).head(k)
                    cur_picks = top["ticker"].tolist()
                    if v.weighting == "ew":
                        w = np.ones(k) / k
                    elif v.weighting == "conv":
                        s = top["score"].values
                        shifted = s - s.min() + 1e-6
                        w = shifted / shifted.sum()
                    elif v.weighting == "invvol":
                        vv = top["vol_1y"].values
                        vv = np.where(np.isnan(vv) | (vv <= 0), 0.4, vv)
                        invv = 1.0 / vv
                        w = invv / invv.sum()
                    elif v.weighting == "softmax":
                        s = top["score"].values
                        ss = (s - s.mean()) / max(s.std(), 1e-9)
                        ws = np.exp(2.0 * ss)
                        w = ws / ws.sum()
                    else:
                        w = np.ones(k) / k
                    if v.cap_per_pick < 1.0:
                        w = np.minimum(w, v.cap_per_pick)
                        w = w / w.sum()
                    cur_weights = w
                    cash = False
                    held_for = 0

        # Apply month return
        if cash or len(cur_picks) == 0:
            ret_m = 0.0
        else:
            pos1 = mr_idx.searchsorted(m)
            cands = []
            for j in (pos1 - 1, pos1):
                if 0 <= j < len(mr_idx):
                    cands.append((j, abs((mr_idx[j] - m).days)))
            cands.sort(key=lambda x: x[1])
            if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(mr_idx):
                ret_m = 0.0
            else:
                next_d = mr_idx[cands[0][0] + 1]
                pick_rets = []
                for tk in cur_picks:
                    if tk in monthly_returns.columns:
                        r = monthly_returns.at[next_d, tk]
                        pick_rets.append(-1.0 if pd.isna(r) else float(r))
                    else:
                        pick_rets.append(-1.0)
                pick_rets = np.array(pick_rets)
                ret_m = float((pick_rets * cur_weights).sum())

        if not cash and len(cur_picks) > 0:
            if do_reb:
                equity *= (1 + ret_m) * (1 - cf)
            else:
                equity *= (1 + ret_m)
        held_for += 1

        rows.append({"date": m, "equity": equity, "ret_m": ret_m,
                     "regime": "cash" if cash else "active",
                     "n_picks": len(cur_picks),
                     "picks": ",".join(cur_picks)})
    return pd.DataFrame(rows)


def cagr_from(eq):
    if len(eq) == 0: return 0.0
    return (eq.iloc[-1]) ** (12.0 / len(eq)) - 1


def sharpe_monthly(ret):
    r = ret.dropna()
    if len(r) < 2 or r.std() == 0: return 0.0
    return (r.mean() / r.std()) * np.sqrt(12)


def max_dd(ret):
    eq = (1 + ret).cumprod()
    if len(eq) == 0: return 0.0
    peak = eq.cummax()
    return float(((eq - peak) / peak).min())


def evaluate(eq, spy_aligned, name):
    ret = eq["ret_m"].astype(float)
    eqc = (1 + ret).cumprod()
    cgr = (eqc.iloc[-1]) ** (12.0 / len(eqc)) - 1 if len(eqc) else 0
    sh = sharpe_monthly(ret)
    mdd = max_dd(ret)
    n_cash = int((eq["regime"] == "cash").sum())

    wf_rows = []
    for split, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)].copy()
        if len(e) == 0: continue
        r = e["ret_m"].astype(float)
        ec = (1 + r).cumprod()
        cv = (ec.iloc[-1]) ** (12.0 / len(ec)) - 1
        spy = spy_aligned[(spy_aligned["date"] >= lo) & (spy_aligned["date"] <= hi)]
        sr = spy["spy_ret_m"].astype(float)
        sc = (1 + sr).cumprod()
        scgr = (sc.iloc[-1]) ** (12.0 / len(sc)) - 1 if len(sc) else 0
        wf_rows.append({"split": split, "cagr": cv, "spy_cagr": scgr, "edge_pp": (cv - scgr) * 100})
    wf = pd.DataFrame(wf_rows)
    spy_full = (1 + spy_aligned["spy_ret_m"]).cumprod().iloc[-1] ** (12.0 / len(spy_aligned)) - 1

    return {
        "name": name,
        "cagr_full": float(cgr),
        "spy_cagr_full": float(spy_full),
        "edge_full_pp": float((cgr - spy_full) * 100),
        "sharpe": float(sh),
        "max_dd": float(mdd),
        "n_cash": n_cash,
        "wf_mean_cagr": float(wf["cagr"].mean()) if len(wf) else 0.0,
        "wf_median_cagr": float(wf["cagr"].median()) if len(wf) else 0.0,
        "wf_min_cagr": float(wf["cagr"].min()) if len(wf) else 0.0,
        "wf_max_cagr": float(wf["cagr"].max()) if len(wf) else 0.0,
        "wf_mean_edge_pp": float(wf["edge_pp"].mean()) if len(wf) else 0.0,
        "wf_n_pos": int((wf["cagr"] > 0).sum()) if len(wf) else 0,
        "wf_n_beats": int((wf["cagr"] > wf["spy_cagr"]).sum()) if len(wf) else 0,
        "wf_n_splits": int(len(wf)),
    }


# ---------------------------------------------------------------------------
def main():
    print("=== Loading inputs ===")
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_features = load_spy_features()

    # Pre-build SPY-aligned benchmark over the full panel asofs
    raw_panel = pd.read_parquet(PIT / "sp500_pit_panel.parquet")
    raw_panel["asof"] = pd.to_datetime(raw_panel["asof"])
    full_dates = pd.DatetimeIndex(sorted(raw_panel["asof"].unique()))
    next_month = full_dates + pd.offsets.MonthEnd(1)
    spy_aligned = pd.DataFrame({
        "date": full_dates,
        "spy_ret_m": [float(monthly_returns["SPY"].loc[nxt]) if nxt in monthly_returns["SPY"].index else 0.0
                      for nxt in next_month],
    })

    scorers = ["ml_filter", "ml_h1", "ml_h3", "ml_h6", "ml_3plus6",
               "ml_filter_winsor", "ml_q", "ml_filter_softmax"]
    k_combos = [
        ("k1_1_1", 1, 1, 1),
        ("k2_2_2", 2, 2, 2),
        ("k3_2_2", 3, 2, 2),
        ("k3_3_3", 3, 3, 3),
        ("k5_3_3", 5, 3, 3),
        ("k7_5_5", 7, 5, 5),
    ]
    weightings = ["ew", "conv", "invvol", "softmax"]
    gates = ["tight", "strict", "ddgate"]
    holds = [1, 3, 6, 12]
    caps = [0.50, 1.0]

    panel_cache: dict[str, pd.DataFrame] = {}
    rows = []
    t0 = time.time()
    n = 0
    total = len(scorers) * len(k_combos) * len(weightings) * len(gates) * len(holds) * len(caps)
    print(f"\n=== Sweep: {total} variants ===")

    for scorer in scorers:
        if scorer not in panel_cache:
            panel_cache[scorer] = build_panel_with_score(scorer)
        p = panel_cache[scorer]
        for k_lab, kN, kR, kB in k_combos:
            for w in weightings:
                for g in gates:
                    for h in holds:
                        for cap in caps:
                            v = Variant(
                                name=f"{scorer}|{k_lab}|{w}|{g}|h{h}|cap{cap}",
                                scorer=scorer, k_normal=kN, k_recovery=kR, k_bull=kB,
                                weighting=w, regime_gate=g, hold_months=h, cap_per_pick=cap,
                            )
                            try:
                                eq = simulate_variant(p, monthly_returns, spy_features, v)
                            except Exception as e:
                                print(f"  ! {v.name}: {e}")
                                continue
                            m = evaluate(eq, spy_aligned, v.name)
                            for k_ in ("scorer", "weighting", "regime_gate", "hold_months", "cap_per_pick"):
                                m[k_] = getattr(v, k_)
                            m["k_combo"] = k_lab
                            rows.append(m)
                            n += 1
                            if n % 100 == 0:
                                print(f"    {n}/{total} ({time.time()-t0:.0f}s)")
    print(f"  Total: {len(rows)}, time {time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(PIT / "sp500_pit_extended_sweep_results.csv", index=False)

    print("\n=== Top 25 by WF mean CAGR (filtered: max_dd > -65%, wf_n_pos >= 9) ===")
    cand = df[(df["max_dd"] > -0.65) & (df["wf_n_pos"] >= 9) & (df["wf_n_splits"] == 10)].copy()
    print(f"  candidates: {len(cand)}")
    print(cand.sort_values("wf_mean_cagr", ascending=False).head(25)[
        ["name", "cagr_full", "edge_full_pp", "wf_mean_cagr", "wf_min_cagr",
         "wf_max_cagr", "wf_mean_edge_pp", "wf_n_beats", "sharpe", "max_dd"]
    ].round(3).to_string(index=False))

    print("\n=== Top 25 by composite (wf_mean + wf_edge + stability) ===")
    cand["composite"] = (
        cand["wf_mean_cagr"] * 100
        + cand["wf_mean_edge_pp"] * 0.5
        + (cand["wf_n_beats"] >= 8) * 5
        + (cand["wf_min_cagr"] > 0.10) * 5
        + (cand["max_dd"] > -0.55) * 3
    )
    print(cand.sort_values("composite", ascending=False).head(25)[
        ["name", "cagr_full", "wf_mean_cagr", "wf_min_cagr", "wf_mean_edge_pp",
         "wf_n_beats", "max_dd", "sharpe", "composite"]
    ].round(3).to_string(index=False))

    winner = cand.sort_values("composite", ascending=False).iloc[0]
    (PIT / "sp500_pit_extended_winner.json").write_text(
        json.dumps({k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                    for k, v in winner.to_dict().items()}, indent=2))
    print(f"\n=== WINNER: {winner['name']} ===")
    for k_, vv in winner.items():
        print(f"  {k_}: {vv}")


if __name__ == "__main__":
    main()
