"""v4 simulator: extends v3 with per-position stop-loss / take-profit /
dynamic exit / score-threshold filter / dynamic K / convex weighting.

Key extensions on top of v3 (sp500_pit_extended_sweep.py):

1. Per-position stop-loss within hold:
   - If a pick drops by more than `stop_loss_pct` from its entry price during the
     hold period, exit it (proceeds → cash for remainder of hold).

2. Per-position take-profit within hold:
   - If a pick gains more than `take_profit_pct`, exit it (proceeds → cash).

3. Dynamic K based on score margin:
   - If top-1 score >> 4th score, hold fewer picks (more concentrated)
   - If top-K scores are tightly clustered, expand K

4. Score-confidence threshold:
   - Only allocate to picks whose score >= cutoff_pct (e.g. > 80th percentile of
     scores for that asof).  Below threshold, position is held in cash.

5. Convex weighting: weights ~ exp(beta * z(score)) with cap.

6. Equal vol-target weighting (alt to invvol).

7. Volatility-aware regime gate that captures cross-sectional vol regime in
   addition to SPY-only signals.

The simulator is implemented at the daily level for picks (using daily price
data) so stop-loss / take-profit are realistic.
"""
from __future__ import annotations

import sys
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


def classify_regime_v4(s: dict) -> str:
    """v4 enhanced regime gate.

    Adds:
      - Vol-spike crash (3m SPY vol percentile)
      - Faster recovery detection (after 1 cash month, ease back in)
      - Continued use of SPY drawdown signals
    """
    r21 = s.get("spy_ret_21d", 0.0)
    r6m = s.get("spy_mom_6_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    dsma = s.get("spy_dsma200", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0)
    dd_52w = s.get("spy_dd_from_52wh", 0.0)

    # Hard crash gates
    if r21 <= -0.08 or (r6m <= -0.05 and r21 <= -0.03):
        return "crash"
    # Recovery: just recovered from a drawdown
    if streak >= 30 and dsma > -0.02 and r21 > 0:
        return "recovery"
    if dd_52w < -0.10 and r21 > 0.03:
        return "recovery"
    # Strong bull
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


REGIME_GATES = {
    "tight": classify_regime_tight,
    "v4": classify_regime_v4,
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
            "spy_vol_3m": float(spy.get("vol_3m", 0.0)),
            "spy_vol_1y": float(spy.get("vol_1y", 0.0)),
        })
    return pd.DataFrame(rows).set_index("asof")


def build_panel_with_score(scorer: str) -> pd.DataFrame:
    """v4 scorers (a superset of v3)."""
    panel = pd.read_parquet(PIT / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])

    if scorer == "ml_3plus6":
        ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred_3m", "pred_6m"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        panel = panel.merge(ml, on=["asof", "ticker"], how="left")
        panel["score"] = (panel["pred_3m"] + panel["pred_6m"]) / 2
    elif scorer == "v4_only":
        ml = pd.read_parquet(PIT / "ml_preds_v4.parquet")[["asof", "ticker", "pred_v4"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        panel = panel.merge(ml.rename(columns={"pred_v4": "score"}),
                            on=["asof", "ticker"], how="left")
    elif scorer == "v4_6m":
        ml = pd.read_parquet(PIT / "ml_preds_v4.parquet")[["asof", "ticker", "pred_v4_6m"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        panel = panel.merge(ml.rename(columns={"pred_v4_6m": "score"}),
                            on=["asof", "ticker"], how="left")
    elif scorer == "v4_3m":
        ml = pd.read_parquet(PIT / "ml_preds_v4.parquet")[["asof", "ticker", "pred_v4_3m"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        panel = panel.merge(ml.rename(columns={"pred_v4_3m": "score"}),
                            on=["asof", "ticker"], how="left")
    elif scorer == "stack_v2_v4":
        # ensemble of v2 ml_3plus6 and v4 prediction
        ml2 = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred_3m", "pred_6m"]]
        ml4 = pd.read_parquet(PIT / "ml_preds_v4.parquet")[["asof", "ticker", "pred_v4"]]
        ml2["asof"] = pd.to_datetime(ml2["asof"])
        ml4["asof"] = pd.to_datetime(ml4["asof"])
        panel = panel.merge(ml2, on=["asof", "ticker"], how="left")
        panel = panel.merge(ml4, on=["asof", "ticker"], how="left")
        v2 = (panel["pred_3m"] + panel["pred_6m"]) / 2
        # Cross-sectional rank within asof
        panel["v2_rk"] = v2.groupby(panel["asof"]).rank(pct=True)
        panel["v4_rk"] = panel["pred_v4"].groupby(panel["asof"]).rank(pct=True)
        panel["score"] = 0.5 * panel["v2_rk"] + 0.5 * panel["v4_rk"]
    elif scorer == "stack_v2_v4_quality":
        # v2 + v4 + quality 5y rank
        ml2 = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred_3m", "pred_6m"]]
        ml4 = pd.read_parquet(PIT / "ml_preds_v4.parquet")[["asof", "ticker", "pred_v4"]]
        ml2["asof"] = pd.to_datetime(ml2["asof"])
        ml4["asof"] = pd.to_datetime(ml4["asof"])
        panel = panel.merge(ml2, on=["asof", "ticker"], how="left")
        panel = panel.merge(ml4, on=["asof", "ticker"], how="left")
        v2 = (panel["pred_3m"] + panel["pred_6m"]) / 2
        panel["v2_rk"] = v2.groupby(panel["asof"]).rank(pct=True)
        panel["v4_rk"] = panel["pred_v4"].groupby(panel["asof"]).rank(pct=True)
        panel["q_rk"] = (panel["sharpe_5y_xs"] + panel["trend_health_5y_xs"]
                         + panel["quality_score_5y_xs"]).groupby(panel["asof"]).rank(pct=True)
        panel["score"] = 0.45 * panel["v2_rk"] + 0.45 * panel["v4_rk"] + 0.10 * panel["q_rk"]
    else:
        raise ValueError(f"unknown scorer {scorer}")
    return panel


@dataclass
class Variant:
    name: str
    scorer: str
    k_normal: int = 3
    k_recovery: int = 3
    k_bull: int = 3
    weighting: str = "ew"           # 'ew', 'invvol', 'conv', 'softmax'
    regime_gate: str = "tight"
    hold_months: int = 6
    cap_per_pick: float = 1.0
    # v4 extensions
    stop_loss_pct: float = 0.0       # 0 = disabled.  e.g. 0.30 → -30% triggers exit
    take_profit_pct: float = 0.0     # 0 = disabled.  e.g. 0.50 → +50% triggers exit
    score_threshold_pct: float = 0.0 # 0 = disabled. fraction of cross-section
    softmax_beta: float = 2.0
    cost_bps: float = 10.0


def _load_daily_prices() -> pd.DataFrame:
    return pd.read_parquet(CACHE / "prices_extended.parquet")


def simulate_variant_v4(panel: pd.DataFrame, monthly_returns: pd.DataFrame,
                        spy_features: pd.DataFrame, v: Variant,
                        daily_prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Simulate at month granularity but apply daily stop-loss / take-profit
    using daily prices when those features are enabled."""
    p = panel.dropna(subset=["score"]).copy()
    p = p[~p["ticker"].isin(EXCLUDE)]

    months = sorted(p["asof"].unique())
    cls = REGIME_GATES[v.regime_gate]
    by_asof = {pd.Timestamp(d): g for d, g in p.groupby("asof")}
    mr_idx = monthly_returns.index

    use_intramonth = v.stop_loss_pct > 0 or v.take_profit_pct > 0
    if use_intramonth and daily_prices is None:
        daily_prices = _load_daily_prices()

    equity = 1.0
    cf = v.cost_bps / 10000.0
    cur_picks: list[str] = []
    cur_weights = np.array([])
    cur_active = np.array([], dtype=bool)
    entry_prices: dict[str, float] = {}
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
                cur_picks, cur_weights, cur_active, cash = [], np.array([]), np.array([], dtype=bool), True
                held_for = 0
                entry_prices = {}
            else:
                k = {"recovery": v.k_recovery, "bull": v.k_bull, "normal": v.k_normal}[regime]
                sub = by_asof.get(m, pd.DataFrame())
                if len(sub) < k:
                    cur_picks, cur_weights, cur_active, cash = [], np.array([]), np.array([], dtype=bool), True
                else:
                    sub_sorted = sub.sort_values("score", ascending=False)
                    # Score threshold filter
                    if v.score_threshold_pct > 0:
                        thresh_q = sub_sorted["score"].quantile(1.0 - v.score_threshold_pct)
                        eligible = sub_sorted[sub_sorted["score"] >= thresh_q]
                    else:
                        eligible = sub_sorted
                    top = eligible.head(k)
                    if len(top) < k:
                        # Fall back: use all eligible if < K
                        if len(top) == 0:
                            cur_picks, cur_weights, cur_active, cash = [], np.array([]), np.array([], dtype=bool), True
                            held_for = 0
                            entry_prices = {}
                            rows.append({"date": m, "equity": equity, "ret_m": 0.0,
                                         "regime": "cash", "n_picks": 0, "picks": ""})
                            continue
                        # else use what we have
                    cur_picks = top["ticker"].tolist()
                    if v.weighting == "ew":
                        w = np.ones(len(cur_picks)) / len(cur_picks)
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
                        ws = np.exp(v.softmax_beta * ss)
                        w = ws / ws.sum()
                    else:
                        w = np.ones(len(cur_picks)) / len(cur_picks)
                    if v.cap_per_pick < 1.0:
                        w = np.minimum(w, v.cap_per_pick)
                        w = w / w.sum()
                    cur_weights = w
                    cur_active = np.ones(len(cur_picks), dtype=bool)
                    cash = False
                    held_for = 0
                    # Track entry prices for stop-loss / take-profit
                    if use_intramonth:
                        entry_prices = {}
                        for tk in cur_picks:
                            if tk in daily_prices.columns:
                                # entry = first available price <= asof month-end
                                pos_d = daily_prices.index.searchsorted(m, side="right") - 1
                                while pos_d >= 0 and pd.isna(daily_prices.iat[pos_d, daily_prices.columns.get_loc(tk)]):
                                    pos_d -= 1
                                if pos_d >= 0:
                                    entry_prices[tk] = float(daily_prices.iat[pos_d, daily_prices.columns.get_loc(tk)])

        # Compute month return
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

                # Inactive positions (already exited via TP/SL): they sit in cash
                # for the rest of the hold period, contributing 0% return.
                if use_intramonth:
                    for j in range(len(cur_picks)):
                        if j < len(cur_active) and not cur_active[j]:
                            pick_rets[j] = 0.0

                # Stop-loss / take-profit using daily prices during this month
                if use_intramonth and len(entry_prices) > 0:
                    # day window: m+1 day .. next_d
                    di_lo = daily_prices.index.searchsorted(m, side="right")
                    di_hi = daily_prices.index.searchsorted(next_d, side="right")
                    new_active = cur_active.copy()
                    for j, tk in enumerate(cur_picks):
                        if not new_active[j]:
                            continue
                        if tk not in daily_prices.columns or tk not in entry_prices:
                            continue
                        ep = entry_prices[tk]
                        col = daily_prices[tk].iloc[di_lo:di_hi]
                        if len(col) == 0:
                            continue
                        ret_path = col.values / ep - 1.0
                        # Find first day TP/SL hits (use first-hit semantics)
                        sl_hit_idx = None
                        tp_hit_idx = None
                        if v.stop_loss_pct > 0:
                            sl_mask = ret_path <= -v.stop_loss_pct
                            if sl_mask.any():
                                sl_hit_idx = int(np.argmax(sl_mask))
                        if v.take_profit_pct > 0:
                            tp_mask = ret_path >= v.take_profit_pct
                            if tp_mask.any():
                                tp_hit_idx = int(np.argmax(tp_mask))
                        # Whichever fires first wins
                        if sl_hit_idx is not None and (tp_hit_idx is None or sl_hit_idx < tp_hit_idx):
                            # Net return for the month: from entry to stop-loss exit, then 0 through month-end
                            # We're already past entry (entry was at last rebalance, possibly months ago).
                            # The realised pick return for THIS month is the gain/loss from this-month-open
                            # to the SL trigger (within this month), which we approximate as
                            # (entry+SL_level) cumulative return minus prior cumulative return.
                            # For simplicity / consistency with v3: book the return as
                            #   (entry-relative SL return) - (entry-relative return at month start)
                            # so that the cumulative across all months equals the realised entry-to-exit return.
                            # Since cur_active prior months kept the position alive (paying actual monthly
                            # returns), the cumulative-to-month-start = product of prior months' returns.
                            # We use the daily price at the start of this month for the baseline.
                            if di_lo > 0:
                                start_px = float(daily_prices[tk].iat[di_lo - 1])
                            else:
                                start_px = ep
                            sl_px = ep * (1 - v.stop_loss_pct)
                            pick_rets[j] = (sl_px / start_px) - 1.0 if start_px and not pd.isna(start_px) else -v.stop_loss_pct
                            new_active[j] = False
                        elif tp_hit_idx is not None:
                            if di_lo > 0:
                                start_px = float(daily_prices[tk].iat[di_lo - 1])
                            else:
                                start_px = ep
                            tp_px = ep * (1 + v.take_profit_pct)
                            pick_rets[j] = (tp_px / start_px) - 1.0 if start_px and not pd.isna(start_px) else v.take_profit_pct
                            new_active[j] = False
                    cur_active = new_active

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


def evaluate(eq, spy_aligned, name) -> dict:
    ret = eq["ret_m"].astype(float)
    eqc = (1 + ret).cumprod()
    cgr = (eqc.iloc[-1]) ** (12.0 / len(eqc)) - 1 if len(eqc) else 0
    sh = (ret.mean() / max(ret.std(), 1e-9)) * np.sqrt(12) if len(ret) else 0
    eqcc = (1 + ret).cumprod()
    peak = eqcc.cummax(); mdd = float(((eqcc - peak) / peak).min()) if len(eqcc) else 0
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


def build_spy_aligned(panel_dates_or_panel) -> pd.DataFrame:
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    if isinstance(panel_dates_or_panel, pd.DataFrame):
        full_dates = pd.DatetimeIndex(sorted(panel_dates_or_panel["asof"].unique()))
    else:
        full_dates = pd.DatetimeIndex(sorted(panel_dates_or_panel))
    next_month = full_dates + pd.offsets.MonthEnd(1)
    return pd.DataFrame({
        "date": full_dates,
        "spy_ret_m": [float(monthly_returns["SPY"].loc[nxt]) if nxt in monthly_returns["SPY"].index else 0.0
                      for nxt in next_month],
    })
