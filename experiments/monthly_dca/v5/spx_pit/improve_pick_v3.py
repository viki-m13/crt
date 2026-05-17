"""Phase: NEW stock-picking levers (selection-side, no overlays).

The repo has only ever swept *fixed* K and the {ml,consensus,blend} scorer.
This module adds genuinely new pick-level levers, all causal:

  adaptive_k : dynamic basket breadth driven by cross-sectional conviction
               (score gap top-vs-pool). High conviction -> concentrate K=2;
               low conviction (picker unsure -> the bad years) -> widen.
  decorr2    : pick #1 by score, pick #2 = best-scored candidate whose
               trailing-12m monthly-return corr to #1 is below rho_max.
               Attacks the correlated-2-stock blowup that drives -66% DD.
  knife      : drop falling-knife candidates (trailing 1m return in the
               bottom q of the pool AND still below its own trailing 3m).
  trig/sel   : independent trigger_mode / select_mode (blend+blend combo).

Bit-exact to WIN1 when adaptive_k=False, decorr2=False, knife_q=0,
trigger_mode='blend', select_mode='ml_3plus6'.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import experiments.monthly_dca.v5.build_webapp_v5_pit as bw  # noqa
from improve_sim_v2 import _score_pool  # noqa


def _score_pool_w(df, mode, K, blend_w):
    """Mirror improve_sim_v2._score_pool but with an explicit blend weight
    for mode=='blend' (blend_w on consensus-rank, 1-blend_w on ml-rank).
    Falls back to the shared _score_pool for non-blend modes."""
    if mode != "blend":
        return _score_pool(df, mode, K)
    d = df.copy()
    ml = (d["pred_3m"] + d["pred_6m"]) / 2.0
    r1 = d["pred_1m"].rank(pct=True)
    r3 = d["pred_3m"].rank(pct=True)
    r6 = d["pred_6m"].rank(pct=True)
    cons = (r1 + r3 + r6) / 3.0
    mlr = ml.rank(pct=True)
    d["score"] = blend_w * cons + (1 - blend_w) * mlr
    return d

EXCLUDE = bw.EXCLUDE
CHRONOS_FILTER_Q = bw.CHRONOS_FILTER_Q
CAP_PER_PICK = bw.CAP_PER_PICK
MIN_HOLD_MONTHS = bw.MIN_HOLD_MONTHS
MAX_HOLD_MONTHS = bw.MAX_HOLD_MONTHS


def _trailing_ret(mr, asof, tk, nmonths):
    """Compound trailing nmonths monthly return of tk strictly before asof."""
    idx = mr.index
    pos = idx.searchsorted(asof, side="right") - 1
    if pos < nmonths or tk not in mr.columns:
        return np.nan
    w = mr[tk].iloc[pos - nmonths + 1: pos + 1]
    if w.isna().all():
        return np.nan
    return float((1 + w.fillna(0.0)).prod() - 1)


def _trailing_corr(mr, asof, a, b, nmonths=12):
    idx = mr.index
    pos = idx.searchsorted(asof, side="right") - 1
    if pos < nmonths or a not in mr.columns or b not in mr.columns:
        return 1.0
    wa = mr[a].iloc[pos - nmonths + 1: pos + 1]
    wb = mr[b].iloc[pos - nmonths + 1: pos + 1]
    m = (~wa.isna()) & (~wb.isna())
    if m.sum() < 6:
        return 1.0
    c = np.corrcoef(wa[m], wb[m])[0, 1]
    return 1.0 if np.isnan(c) else float(c)


def _pool_scored(df, mode, K, chronos_at_m, blend_w=0.5):
    """Chronos-filtered, score-sorted candidate pool (descending)."""
    d = _score_pool_w(df, mode, K, blend_w)
    if chronos_at_m is not None:
        d["chr_p70"] = d["ticker"].map(chronos_at_m)
        d["chr_p70_rk"] = d["chr_p70"].rank(pct=True)
        d = d[d["chr_p70_rk"] >= CHRONOS_FILTER_Q]
    return d.sort_values("score", ascending=False).reset_index(drop=True)


def _select(df, mode, K, chronos_at_m, mr, asof, *,
            adaptive_k=False, conv_hi=0.18, conv_lo=0.08, k_lo=2, k_mid=3,
            k_hi=4, decorr2=False, rho_max=0.6, knife_q=0.0,
            blend_w=0.5, regime=None, regime_w=None):
    """Return the chosen pick DataFrame (>=2 rows) or None."""
    bw_ = blend_w
    if regime_w is not None and regime is not None:
        bw_ = regime_w.get(regime, blend_w)
    d = _pool_scored(df, mode, K, chronos_at_m, blend_w=bw_)
    if len(d) < 2:
        return None

    # --- falling-knife screen (causal: uses returns strictly before asof) ---
    if knife_q > 0.0 and len(d) > 4:
        r1 = d["ticker"].map(lambda t: _trailing_ret(mr, asof, t, 1))
        r3 = d["ticker"].map(lambda t: _trailing_ret(mr, asof, t, 3))
        thr = r1.quantile(knife_q)
        bad = (r1 <= thr) & (r1 < r3)  # bottom-q AND still accelerating down
        keep = d[~bad.fillna(False)]
        if len(keep) >= 2:
            d = keep.reset_index(drop=True)

    # --- conviction-adaptive breadth ---
    Keff = K
    if adaptive_k and len(d) >= 5:
        s = d["score"].to_numpy()
        # gap between the top pick and the pool median, in score units
        conv = float(s[0] - np.median(s))
        if conv >= conv_hi:
            Keff = k_lo
        elif conv >= conv_lo:
            Keff = k_mid
        else:
            Keff = k_hi
        Keff = min(Keff, len(d))

    # --- decorrelated 2nd pick (only meaningful at Keff==2) ---
    if decorr2 and Keff == 2 and len(d) >= 3:
        first = d.iloc[0]
        for j in range(1, len(d)):
            cand = d.iloc[j]
            if _trailing_corr(mr, asof, first["ticker"],
                              cand["ticker"], 12) <= rho_max:
                return pd.concat([first.to_frame().T,
                                  cand.to_frame().T], ignore_index=True)
        return d.head(2)

    top = d.head(Keff)
    return top if len(top) >= 2 else None


def run_sim_v3(members_g, preds_wf, preds_live, spy_features,
               monthly_returns, monthly_prices, chronos_preds,
               cost_bps=10.0, K=2, trigger_mode="blend",
               select_mode="ml_3plus6", **selkw):
    wf_max_asof = preds_wf["asof"].max()
    months = sorted(set(pd.to_datetime(preds_wf["asof"].unique())).union(
        set(pd.to_datetime(preds_live["asof"].unique()))))
    months = [pd.Timestamp(m) for m in months]
    cf = cost_bps / 10000.0
    cur_picks, cur_weights = [], np.array([])
    cash, held_for, equity = False, 0, 1.0
    basket_id = 0
    rets_log = []
    mr_idx = monthly_returns.index

    def _sub_at(m_):
        s = (preds_wf if m_ <= wf_max_asof else preds_live)
        s = s[s["asof"] == m_].copy()
        s = s[~s["ticker"].isin(EXCLUDE)]
        return s[s["ticker"].isin(members_g.get(m_, set()))].copy()

    def _cand(m_, mode, regime_=None):
        sub = _sub_at(m_)
        if len(sub) == 0:
            return None
        ch = chronos_preds.get(m_) if chronos_preds else None
        return _select(sub, mode, K, ch, monthly_returns, m_,
                       regime=regime_, **selkw)

    for i, m in enumerate(months):
        spy_now = spy_features.loc[m].to_dict() if m in spy_features.index else {}
        regime = bw.classify_regime_tight(spy_now)
        do_reb = (i == 0) or (cash != (regime == "crash"))
        if held_for >= MAX_HOLD_MONTHS:
            do_reb = True
        elif held_for >= MIN_HOLD_MONTHS and cur_picks and regime != "crash":
            cand = _cand(m, trigger_mode, regime)
            if cand is not None and not (set(cur_picks)
                                         & set(cand["ticker"])):
                do_reb = True

        if do_reb:
            if regime == "crash":
                cur_picks, cur_weights, cash = [], np.array([]), True
                held_for = 0
            else:
                top = _cand(m, select_mode, regime)
                if top is None:
                    cur_picks, cur_weights, cash = [], np.array([]), True
                else:
                    cur_picks = top["ticker"].tolist()
                    cur_weights = bw._calc_invvol_weights(
                        top, monthly_returns, m, cap=CAP_PER_PICK)
                    cash = False
                    basket_id += 1
                    held_for = 0

        if cash or len(cur_picks) == 0:
            ret_m = 0.0
        else:
            pos1 = mr_idx.searchsorted(m)
            cands = [(j, abs((mr_idx[j] - m).days))
                     for j in (pos1 - 1, pos1) if 0 <= j < len(mr_idx)]
            cands.sort(key=lambda x: x[1])
            if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(mr_idx):
                ret_m = 0.0
            else:
                next_d = mr_idx[cands[0][0] + 1]
                pr, any_data = [], False
                for tk in cur_picks:
                    if tk in monthly_returns.columns:
                        rr = monthly_returns.at[next_d, tk]
                        if pd.isna(rr):
                            pr.append(0.0)
                        else:
                            pr.append(float(rr)); any_data = True
                    else:
                        pr.append(-1.0)
                ret_m = 0.0 if not any_data else float(
                    (np.array(pr) * cur_weights).sum())

        if not cash and len(cur_picks) > 0:
            equity *= (1 + ret_m) * (1 - cf) if do_reb else (1 + ret_m)
        held_for += 1
        rets_log.append({"date": str(m.date()), "regime": regime,
                         "ret_m": ret_m, "picks": list(cur_picks),
                         "basket_id": basket_id, "equity": float(equity),
                         "rebalanced": do_reb})
    return rets_log
