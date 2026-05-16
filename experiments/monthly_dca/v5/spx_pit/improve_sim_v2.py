"""Faithful parametrized copy of build_webapp_v5_pit.run_full_sim with two
independent scorer knobs:

  trigger_mode : scorer used by the score-drift REBALANCE trigger
  select_mode  : scorer used to FORM the basket at a rebalance

In production these are decoupled: trigger=SCORER_MODE (consensus),
select=ml_3plus6 (hard-coded). This module lets us test making selection
use the same / a blended scorer — a genuinely new pick-level change.

Validated to reproduce the production stream exactly when
trigger='consensus', select='ml_3plus6'.
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

EXCLUDE = bw.EXCLUDE
BLEND_W = 0.5  # weight on consensus rank in 'blend' (1-BLEND_W on ml rank)
CHRONOS_FILTER_Q = bw.CHRONOS_FILTER_Q
CAP_PER_PICK = bw.CAP_PER_PICK
MIN_HOLD_MONTHS = bw.MIN_HOLD_MONTHS
MAX_HOLD_MONTHS = bw.MAX_HOLD_MONTHS


def _score_pool(df: pd.DataFrame, mode: str, K: int) -> pd.DataFrame:
    """Attach a 'score' column (and apply the consensus dispersion filter
    when mode involves consensus). Mirrors build_webapp exactly for the
    'ml_3plus6' and 'consensus' branches."""
    d = df.copy()
    ml = (d["pred_3m"] + d["pred_6m"]) / 2.0

    def consensus_score(dd):
        r1 = dd["pred_1m"].rank(pct=True)
        r3 = dd["pred_3m"].rank(pct=True)
        r6 = dd["pred_6m"].rank(pct=True)
        disp = pd.concat([r1, r3, r6], axis=1).std(axis=1)
        keep = disp <= disp.median()
        if keep.sum() >= K:
            dd = dd[keep]
            r1, r3, r6 = r1[keep], r3[keep], r6[keep]
        sc = (r1 + r3 + r6) / 3.0
        return dd, sc

    if mode == "ml_3plus6":
        d["score"] = ml
        return d
    if mode == "consensus":
        if len(d) >= 4:
            d, sc = consensus_score(d)
            d["score"] = sc
        else:
            d["score"] = (d["pred_3m"] + d["pred_6m"]) / 2.0
        return d
    if mode == "blend":
        # rank-blend of ml_3plus6 and consensus, computed on the SAME pool
        # (no dispersion sub-selection — keeps breadth, blends signals).
        r1 = d["pred_1m"].rank(pct=True)
        r3 = d["pred_3m"].rank(pct=True)
        r6 = d["pred_6m"].rank(pct=True)
        cons = (r1 + r3 + r6) / 3.0
        mlr = ml.rank(pct=True)
        d["score"] = BLEND_W * cons + (1 - BLEND_W) * mlr
        return d
    raise ValueError(mode)


def _topK(df, mode, K, chronos_at_m):
    d = _score_pool(df, mode, K)
    if chronos_at_m is not None:
        d["chr_p70"] = d["ticker"].map(chronos_at_m)
        d["chr_p70_rk"] = d["chr_p70"].rank(pct=True)
        d = d[d["chr_p70_rk"] >= CHRONOS_FILTER_Q]
    top = d.sort_values("score", ascending=False).head(K)
    return top if len(top) >= K else None


def run_sim_v2(members_g, preds_wf, preds_live, spy_features,
               monthly_returns, monthly_prices, chronos_preds,
               cost_bps=10.0, K=2,
               trigger_mode="consensus", select_mode="ml_3plus6"):
    wf_max_asof = preds_wf["asof"].max()
    months = sorted(set(pd.to_datetime(preds_wf["asof"].unique())).union(
        set(pd.to_datetime(preds_live["asof"].unique()))))
    months = [pd.Timestamp(m) for m in months]
    cf = cost_bps / 10000.0
    cur_picks, cur_weights = [], np.array([])
    cash, held_for, equity = False, 0, 1.0
    last_rebalance, basket_id = None, 0
    rets_log = []
    mr_idx = monthly_returns.index

    def _sub_at(m_):
        s = (preds_wf if m_ <= wf_max_asof else preds_live)
        s = s[s["asof"] == m_].copy()
        s = s[~s["ticker"].isin(EXCLUDE)]
        return s[s["ticker"].isin(members_g.get(m_, set()))].copy()

    def _candidate_top(m_, mode):
        sub = _sub_at(m_)
        if len(sub) == 0:
            return None
        ch = chronos_preds.get(m_) if chronos_preds else None
        return _topK(sub, mode, K, ch)

    for i, m in enumerate(months):
        spy_now = spy_features.loc[m].to_dict() if m in spy_features.index else {}
        regime = bw.classify_regime_tight(spy_now)
        do_reb = (i == 0) or (cash != (regime == "crash"))
        if held_for >= MAX_HOLD_MONTHS:
            do_reb = True
        elif held_for >= MIN_HOLD_MONTHS and cur_picks and regime != "crash":
            cand = _candidate_top(m, trigger_mode)
            if cand is not None and not (set(cur_picks) & set(cand["ticker"])):
                do_reb = True

        if do_reb:
            if regime == "crash":
                cur_picks, cur_weights, cash = [], np.array([]), True
                held_for = 0
            else:
                sub_pit = _sub_at(m)
                ch = chronos_preds.get(m) if chronos_preds else None
                top = _topK(sub_pit, select_mode, K, ch) if len(sub_pit) else None
                if top is None:
                    cur_picks, cur_weights, cash = [], np.array([]), True
                else:
                    cur_picks = top["ticker"].tolist()
                    cur_weights = bw._calc_invvol_weights(
                        top, monthly_returns, m, cap=CAP_PER_PICK)
                    cash = False
                    last_rebalance = m
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
