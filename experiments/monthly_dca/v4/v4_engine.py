"""V4 strategy engine — extends v3 with rich position-level controls.

Knobs supported:
  - K_normal / K_recovery / K_bull / K_crash (cash if 0)
  - weighting: ew / conv / invvol / softmax / score_pow
  - regime_gate: tight / strict / ddgate / breadth / breadth_tight / multi
  - hold_months: 1..12 (capacity to extend if conviction holds)
  - score_threshold: min score for top-K to even enter
  - score_thresh_pct: alternative — top-K must rank above this percentile
  - stop_loss: per-position max drawdown from entry (e.g. -0.30)
  - take_profit: per-position max gain from entry (e.g. 1.0 = +100%)
  - stop_to_cash: True -> exited slot stays cash; False -> redistribute among survivors
  - cap_per_pick: max single weight
  - retain_winners: True -> on rebalance, keep winners that are still top-2K
  - daily price stops not used; we use monthly granularity for stop-loss
  - cost_bps: cost per round-trip rebalance leg (charged on every rebalance month)

Loads features panel + ML preds, simulates, returns equity curve.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# REGIME GATES
def regime_tight(s: dict) -> str:
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


def regime_breadth(s: dict) -> str:
    """Use market breadth (% of stocks above 200dma) + SPY trend."""
    breadth = s.get("breadth_above_200", 0.5)
    r21 = s.get("spy_ret_21d", 0.0)
    r6m = s.get("spy_mom_6_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    dsma = s.get("spy_dsma200", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0)
    # Crash: breadth collapses + SPY drop
    if breadth <= 0.30 or r21 <= -0.08 or (r6m <= -0.05 and r21 <= -0.03):
        return "crash"
    if streak >= 40 and dsma > 0 and r21 > 0 and breadth >= 0.40:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0 and breadth >= 0.55:
        return "bull"
    return "normal"


def regime_breadth_tight(s: dict) -> str:
    """Tighter breadth-aware crash; preserve recoveries."""
    breadth = s.get("breadth_above_200", 0.5)
    r21 = s.get("spy_ret_21d", 0.0)
    r6m = s.get("spy_mom_6_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    dsma = s.get("spy_dsma200", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0)
    if (breadth <= 0.25 and r21 < 0) or r21 <= -0.08 or (r6m <= -0.05 and r21 <= -0.03):
        return "crash"
    if streak >= 40 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


def regime_multi(s: dict) -> str:
    """Multi-condition crash gate: SPY+breadth+drawdown."""
    breadth = s.get("breadth_above_200", 0.5)
    r21 = s.get("spy_ret_21d", 0.0)
    r6m = s.get("spy_mom_6_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    dsma = s.get("spy_dsma200", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0)
    dd = s.get("spy_dd_from_52wh", 0.0)

    crash_cond = (
        r21 <= -0.08
        or (r6m <= -0.05 and r21 <= -0.03)
        or (breadth <= 0.30 and r21 < 0)
        or (dd <= -0.15 and r21 < 0)
    )
    if crash_cond:
        return "crash"
    if streak >= 40 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


REGIME_GATES = {
    "tight": regime_tight,
    "breadth": regime_breadth,
    "breadth_tight": regime_breadth_tight,
    "multi": regime_multi,
}


# ---------------------------------------------------------------------------
def load_spy_features() -> pd.DataFrame:
    rows = []
    for f in sorted(FEATURES_DIR.glob("*.parquet")):
        d = pd.Timestamp(f.stem)
        df = pd.read_parquet(f)
        if "SPY" not in df.index:
            continue
        spy = df.loc["SPY"]
        # Compute breadth from this asof's panel: % of stocks above 200dma.
        # Use d_sma200 > 0 as proxy.
        if "d_sma200" in df.columns:
            ds = df["d_sma200"].dropna()
            breadth = float((ds > 0).mean()) if len(ds) > 0 else 0.5
        else:
            breadth = 0.5
        rows.append({
            "asof": d,
            "spy_dsma200": float(spy.get("d_sma200", 0.0)),
            "spy_rsi14": float(spy.get("rsi_14", 50.0)),
            "spy_mom_12_1": float(spy.get("mom_12_1", 0.0)),
            "spy_mom_6_1": float(spy.get("mom_6_1", 0.0)),
            "spy_ret_21d": float(spy.get("ret_21d", 0.0)),
            "spy_below_200_streak": float(spy.get("max_below_200_streak", 0.0)),
            "spy_dd_from_52wh": float(spy.get("dd_from_52wh", 0.0)),
            "breadth_above_200": breadth,
        })
    return pd.DataFrame(rows).set_index("asof")


# ---------------------------------------------------------------------------
def build_panel_with_score(scorer: str) -> pd.DataFrame:
    panel = pd.read_parquet(PIT / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])

    if scorer == "ml_3plus6":
        ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred_3m", "pred_6m"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        panel = panel.merge(ml, on=["asof", "ticker"], how="left")
        panel["score"] = (panel["pred_3m"] + panel["pred_6m"]) / 2
    elif scorer == "ml_3":
        ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred_3m"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        panel = panel.merge(ml.rename(columns={"pred_3m": "score"}), on=["asof", "ticker"], how="left")
    elif scorer == "ml_6":
        ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred_6m"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        panel = panel.merge(ml.rename(columns={"pred_6m": "score"}), on=["asof", "ticker"], how="left")
    elif scorer == "ml_136":
        ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred_1m", "pred_3m", "pred_6m"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        panel = panel.merge(ml, on=["asof", "ticker"], how="left")
        panel["score"] = (panel["pred_1m"] + panel["pred_3m"] + panel["pred_6m"]) / 3
    elif scorer == "ml_3plus6_qm":
        # Blend ML with quality+momentum factor-cluster
        ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred_3m", "pred_6m"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        panel = panel.merge(ml, on=["asof", "ticker"], how="left")
        ml_score = (panel["pred_3m"] + panel["pred_6m"]) / 2
        # Quality factor = mean of 5y health metrics (xs)
        q = (panel["sharpe_5y_xs"].fillna(0) + panel["trend_health_5y_xs"].fillna(0) + panel["quality_score_5y_xs"].fillna(0)) / 3
        # Momentum factor
        m = (panel["mom_12_1_xs"].fillna(0) + panel["mom_per_unit_vol_12_xs"].fillna(0) + panel["idio_mom_12_1_xs"].fillna(0)) / 3
        panel["score"] = 0.7 * ml_score + 0.15 * q + 0.15 * m
    elif scorer == "v4_ml":
        ml = pd.read_parquet(PIT / "v4_ml_preds.parquet")
        ml["asof"] = pd.to_datetime(ml["asof"])
        panel = panel.merge(ml[["asof", "ticker", "pred_v4"]].rename(columns={"pred_v4": "score"}), on=["asof", "ticker"], how="left")
    elif scorer == "v4_ml_blend":
        # Blend v3 ml_3plus6 with v4 ml
        ml_v2 = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred_3m", "pred_6m"]]
        ml_v2["asof"] = pd.to_datetime(ml_v2["asof"])
        ml_v4 = pd.read_parquet(PIT / "v4_ml_preds.parquet")
        ml_v4["asof"] = pd.to_datetime(ml_v4["asof"])
        panel = panel.merge(ml_v2, on=["asof", "ticker"], how="left")
        panel = panel.merge(ml_v4[["asof", "ticker", "pred_v4"]], on=["asof", "ticker"], how="left")
        ml3p6 = (panel["pred_3m"] + panel["pred_6m"]) / 2
        panel["score"] = 0.5 * ml3p6 + 0.5 * panel["pred_v4"]
    else:
        raise ValueError(f"unknown scorer: {scorer}")
    return panel


def get_monthly_returns_panel(monthly_returns: pd.DataFrame, asofs: list[pd.Timestamp]) -> dict[pd.Timestamp, pd.Timestamp]:
    """Map each panel asof -> next-month-end date in monthly_returns index."""
    mr_idx = monthly_returns.index
    out = {}
    for d in asofs:
        d = pd.Timestamp(d)
        pos = mr_idx.searchsorted(d)
        cands = []
        for j in (pos - 1, pos):
            if 0 <= j < len(mr_idx):
                cands.append((j, abs((mr_idx[j] - d).days)))
        cands.sort(key=lambda x: x[1])
        if cands and cands[0][1] <= 7:
            out[d] = cands[0][0]
    return out


# ---------------------------------------------------------------------------
@dataclass
class V4Variant:
    name: str
    scorer: str
    k_normal: int = 3
    k_recovery: int = 3
    k_bull: int = 3
    weighting: str = "ew"          # ew, conv, invvol, softmax, score_pow
    regime_gate: str = "tight"
    hold_months: int = 6
    cap_per_pick: float = 1.0
    score_threshold: float = -np.inf  # absolute score threshold (top-K average must exceed)
    stop_loss: float = -1.0           # cumulative position drawdown from entry triggers exit (-1 = disabled)
    take_profit: float = np.inf       # cumulative position gain from entry triggers exit
    stop_to_cash: bool = False        # True: stopped slot -> cash; False: redistribute to survivors
    extend_hold: bool = False         # True: extend hold beyond hold_months if conviction high
    cost_bps: float = 10.0


def simulate_v4(panel: pd.DataFrame, monthly_returns: pd.DataFrame,
                spy_features: pd.DataFrame, v: V4Variant) -> pd.DataFrame:
    p = panel.dropna(subset=["score"]).copy()
    p = p[~p["ticker"].isin(EXCLUDE)]
    months = sorted(p["asof"].unique())
    cls = REGIME_GATES[v.regime_gate]
    by_asof = {pd.Timestamp(d): g for d, g in p.groupby("asof")}
    asof_to_pos = get_monthly_returns_panel(monthly_returns, months)
    mr_idx = monthly_returns.index

    equity = 1.0
    cf = v.cost_bps / 10000.0

    # Per-position tracker:
    cur_picks: list[str] = []         # ticker symbols
    pos_weights: np.ndarray = np.array([])  # current weights (sum can be < 1 if some stopped to cash)
    pos_entry: np.ndarray = np.array([])    # cumulative pnl multiplier from entry (1.0 at entry)
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
                cur_picks, pos_weights, pos_entry, cash = [], np.array([]), np.array([]), True
                held_for = 0
            else:
                k = {"recovery": v.k_recovery, "bull": v.k_bull, "normal": v.k_normal}[regime]
                sub = by_asof.get(m, pd.DataFrame())
                if len(sub) < k:
                    cur_picks, pos_weights, pos_entry, cash = [], np.array([]), np.array([]), True
                else:
                    top = sub.sort_values("score", ascending=False).head(k)
                    top_avg_score = float(top["score"].mean())
                    if top_avg_score < v.score_threshold:
                        # Conviction too low -> hold cash
                        cur_picks, pos_weights, pos_entry, cash = [], np.array([]), np.array([]), True
                        held_for = 0
                    else:
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
                        elif v.weighting == "score_pow":
                            s = top["score"].values
                            shifted = np.clip(s, 0, None)
                            shifted = shifted - shifted.min() + 1e-6
                            ws = shifted ** 2
                            w = ws / ws.sum()
                        else:
                            w = np.ones(k) / k
                        if v.cap_per_pick < 1.0:
                            w = np.minimum(w, v.cap_per_pick)
                            w = w / w.sum()
                        pos_weights = w
                        pos_entry = np.ones(len(cur_picks))  # start at 1.0
                        cash = False
                        held_for = 0

        # Apply month return
        if cash or len(cur_picks) == 0:
            ret_m = 0.0
        else:
            pos = asof_to_pos.get(m)
            if pos is None or pos + 1 >= len(mr_idx):
                ret_m = 0.0
            else:
                next_d = mr_idx[pos + 1]
                pick_rets = []
                for tk in cur_picks:
                    if tk in monthly_returns.columns:
                        r = monthly_returns.at[next_d, tk]
                        pick_rets.append(-1.0 if pd.isna(r) else float(r))
                    else:
                        pick_rets.append(-1.0)
                pick_rets = np.array(pick_rets)

                # Update entry trackers
                pos_entry = pos_entry * (1.0 + pick_rets)

                # Apply stop-loss / take-profit AFTER this month
                # Compute returns delivered this month, then check triggers AFTER returns settle
                ret_m = float((pick_rets * pos_weights).sum())

                # Now mark stopped positions
                stop_hit = (pos_entry - 1.0) <= v.stop_loss
                tp_hit = (pos_entry - 1.0) >= v.take_profit
                exit_mask = stop_hit | tp_hit

                if exit_mask.any():
                    # Move stopped weight to cash (locked in at current pos_entry value)
                    # We've already applied this month's returns. Going forward, stopped positions
                    # are "removed" — their weight stays where it is but contributes 0% return.
                    if v.stop_to_cash:
                        # Just zero out their weight going forward (stays as cash within portfolio)
                        pos_weights = np.where(exit_mask, 0.0, pos_weights)
                    else:
                        # Redistribute exited weight to survivors
                        survive = ~exit_mask
                        if survive.any():
                            extra = float(pos_weights[exit_mask].sum())
                            pos_weights = np.where(exit_mask, 0.0, pos_weights)
                            denom = pos_weights[survive].sum()
                            if denom > 0:
                                pos_weights[survive] = pos_weights[survive] * (1.0 + extra / denom)
                        else:
                            pos_weights = np.zeros_like(pos_weights)

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


def cagr_from_returns(rets):
    if len(rets) == 0: return 0.0
    eq = (1 + rets).cumprod()
    return float((eq.iloc[-1]) ** (12.0 / len(eq)) - 1)


def sharpe_monthly(rets):
    r = pd.Series(rets).dropna()
    if len(r) < 2 or r.std() == 0: return 0.0
    return float((r.mean() / r.std()) * np.sqrt(12))


def max_dd_from_returns(rets):
    eq = (1 + pd.Series(rets)).cumprod()
    if len(eq) == 0: return 0.0
    peak = eq.cummax()
    return float(((eq - peak) / peak).min())


def evaluate_v4(eq: pd.DataFrame, spy_aligned: pd.DataFrame, name: str) -> dict:
    ret = eq["ret_m"].astype(float)
    cgr = cagr_from_returns(ret)
    sh = sharpe_monthly(ret)
    mdd = max_dd_from_returns(ret)
    n_cash = int((eq["regime"] == "cash").sum())

    wf_rows = []
    for split, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)].copy()
        if len(e) == 0: continue
        r = e["ret_m"].astype(float)
        cv = cagr_from_returns(r)
        spy = spy_aligned[(spy_aligned["date"] >= lo) & (spy_aligned["date"] <= hi)]
        sr = spy["spy_ret_m"].astype(float)
        scgr = cagr_from_returns(sr)
        wf_rows.append({"split": split, "cagr": cv, "spy_cagr": scgr, "edge_pp": (cv - scgr) * 100})
    wf = pd.DataFrame(wf_rows)
    spy_full = cagr_from_returns(spy_aligned["spy_ret_m"].astype(float))

    return {
        "name": name,
        "cagr_full": cgr,
        "spy_cagr_full": spy_full,
        "edge_full_pp": (cgr - spy_full) * 100,
        "sharpe": sh,
        "max_dd": mdd,
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


def build_spy_aligned(panel_path: Path, monthly_returns: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_parquet(panel_path)
    raw["asof"] = pd.to_datetime(raw["asof"])
    full_dates = pd.DatetimeIndex(sorted(raw["asof"].unique()))
    next_month = full_dates + pd.offsets.MonthEnd(1)
    spy_aligned = pd.DataFrame({
        "date": full_dates,
        "spy_ret_m": [float(monthly_returns["SPY"].loc[nxt]) if nxt in monthly_returns["SPY"].index else 0.0
                      for nxt in next_month],
    })
    return spy_aligned
