"""Run baseline + all 5 improvement variants against the v5 walk-forward harness.

Variants tested:
  0. baseline       — v5 production unchanged (control)
  1. concentration_overlay — SPY sleeve sized by cross-sectional dispersion
  2. sector_diversification — ≤2 picks per GICS sector
  3. cap_loose_bull — per-pick cap raised to 0.55 in bull regime
  4. ensemble       — mean cross-sectional rank from v2 + v6 + pattern_sim + vertical
  5. anchor         — K=2 alpha + 1 mega-cap anchor (top 12-1 momentum in PIT pool)

Outputs:
  experiments/monthly_dca/v5/validations/results/<variant>.json
  experiments/monthly_dca/v5/validations/SUMMARY.csv
  experiments/monthly_dca/v5/validations/REPORT.md
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.monthly_dca.v5.validations.harness import (
    HarnessData, load_all, evaluate, invvol_weights, pick_v5_baseline,
    CHRONOS_FILTER_Q, CAP_PER_PICK, K_PICKS,
)

RESULTS = Path(__file__).resolve().parent / "results"


# ============================================================
#  Variant 1 — concentration overlay
# ============================================================
def make_dispersion_signal(data: HarnessData) -> dict:
    """Compute cross-sectional dispersion (std of monthly returns across PIT
    pool) for each month. Lower dispersion = mega-cap-led regime."""
    out = {}
    for asof, pool in data.members_g.items():
        if asof not in data.mret.index:
            continue
        rets = data.mret.loc[asof, [t for t in pool if t in data.mret.columns]]
        rets = rets.dropna()
        if len(rets) < 20:
            continue
        out[asof] = float(rets.std())
    return out


def make_concentration_overlay(data: HarnessData, sleeve: float = 0.25,
                                pctile_threshold: float = 0.30):
    """Overlay function: when dispersion is in the bottom `pctile_threshold`
    of trailing 36-month dispersion, blend `sleeve` of SPY return into the
    strategy return for that month.
    """
    disp = make_dispersion_signal(data)
    disp_series = pd.Series(disp).sort_index()

    def overlay(asof, regime, data):
        if asof not in disp_series.index:
            return 0.0, 0.0
        # Trailing 36-month percentile
        prior = disp_series.loc[:asof].iloc[:-1].tail(36)
        if len(prior) < 12:
            return 0.0, 0.0
        cur = disp_series.loc[asof]
        pct = (prior < cur).mean()
        if pct <= pctile_threshold:
            # narrow dispersion → add SPY sleeve
            spy_ret = (float(data.mret.at[asof, "SPY"])
                        if (asof in data.mret.index
                            and "SPY" in data.mret.columns
                            and pd.notna(data.mret.at[asof, "SPY"]))
                        else 0.0)
            return sleeve, spy_ret
        return 0.0, 0.0
    return overlay


# ============================================================
#  Variant 2 — sector diversification (≤2 per GICS sector)
# ============================================================
def pick_sector_diversified(asof, eligible, data: HarnessData, regime: str):
    sub = data.ml_v2[data.ml_v2["asof"] == asof].copy()
    sub = sub[sub["ticker"].isin(eligible)]
    if len(sub) == 0:
        return [], []
    sub["score"] = (sub["pred_3m"] + sub["pred_6m"]) / 2
    ch = data.chronos.get(asof, {})
    if ch:
        sub["chr"] = sub["ticker"].map(ch)
        sub["chr_rk"] = sub["chr"].rank(pct=True)
        sub = sub[sub["chr_rk"] >= CHRONOS_FILTER_Q]
    if len(sub) < K_PICKS:
        return [], []
    sub = sub.sort_values("score", ascending=False)
    # Greedy pick with ≤2-per-sector constraint
    picks = []
    sector_count: dict = {}
    for _, row in sub.iterrows():
        tk = row["ticker"]
        sec = data.sector_map.get(tk, "Unknown")
        if sector_count.get(sec, 0) >= 2:
            continue
        picks.append(tk)
        sector_count[sec] = sector_count.get(sec, 0) + 1
        if len(picks) >= K_PICKS:
            break
    if len(picks) < K_PICKS:
        return [], []
    weights = invvol_weights(picks, data.mret, asof, cap=CAP_PER_PICK)
    return picks, list(weights)


# ============================================================
#  Variant 3 — cap loosening in bull regime
# ============================================================
def pick_cap_loose_bull(asof, eligible, data: HarnessData, regime: str):
    sub = data.ml_v2[data.ml_v2["asof"] == asof].copy()
    sub = sub[sub["ticker"].isin(eligible)]
    if len(sub) == 0:
        return [], []
    sub["score"] = (sub["pred_3m"] + sub["pred_6m"]) / 2
    ch = data.chronos.get(asof, {})
    if ch:
        sub["chr"] = sub["ticker"].map(ch)
        sub["chr_rk"] = sub["chr"].rank(pct=True)
        sub = sub[sub["chr_rk"] >= CHRONOS_FILTER_Q]
    if len(sub) < K_PICKS:
        return [], []
    top = sub.sort_values("score", ascending=False).head(K_PICKS)
    picks = top["ticker"].tolist()
    cap = 0.55 if regime == "bull" else CAP_PER_PICK
    weights = invvol_weights(picks, data.mret, asof, cap=cap)
    return picks, list(weights)


# ============================================================
#  Variant 4 — multi-model ensemble
# ============================================================
def pick_ensemble(asof, eligible, data: HarnessData, regime: str):
    """Average the cross-sectional rank from v2 (3m+6m), v6, pattern_sim,
    vertical. Take top-K from the ensembled rank, then apply Chronos filter
    and inv-vol weighting.
    """
    # Build a cohort dataframe with all available signals
    sub = data.ml_v2[data.ml_v2["asof"] == asof].copy()
    sub = sub[sub["ticker"].isin(eligible)]
    if len(sub) == 0:
        return [], []
    sub["score_v2"] = ((sub["pred_3m"] + sub["pred_6m"]) / 2).rank(pct=True)

    # v6
    if len(data.ml_v6):
        v6 = data.ml_v6[data.ml_v6["asof"] == asof]
        v6_score = (v6.set_index("ticker")["pred_v6_3m"]
                     + v6.set_index("ticker")["pred_v6_6m"]) / 2
        sub["score_v6"] = sub["ticker"].map(v6_score.rank(pct=True))
    # pattern_sim
    if len(data.ml_pattern):
        p = data.ml_pattern[data.ml_pattern["asof"] == asof]
        ps = p.set_index("ticker")["pattern_sim"]
        sub["score_ps"] = sub["ticker"].map(ps.rank(pct=True))
    # vertical
    if len(data.ml_vertical):
        v = data.ml_vertical[data.ml_vertical["asof"] == asof]
        vs = v.set_index("ticker")["p_vertical"]
        sub["score_vert"] = sub["ticker"].map(vs.rank(pct=True))

    # Mean of available ranks
    rank_cols = [c for c in ["score_v2", "score_v6", "score_ps", "score_vert"]
                 if c in sub.columns]
    sub["score"] = sub[rank_cols].mean(axis=1)
    sub = sub.dropna(subset=["score"])

    # Chronos filter
    ch = data.chronos.get(asof, {})
    if ch:
        sub["chr"] = sub["ticker"].map(ch)
        sub["chr_rk"] = sub["chr"].rank(pct=True)
        sub = sub[sub["chr_rk"] >= CHRONOS_FILTER_Q]
    if len(sub) < K_PICKS:
        return [], []
    top = sub.sort_values("score", ascending=False).head(K_PICKS)
    picks = top["ticker"].tolist()
    weights = invvol_weights(picks, data.mret, asof, cap=CAP_PER_PICK)
    return picks, list(weights)


# ============================================================
#  Variant 5 — K=2 alpha + 1 mega-cap anchor
# ============================================================
def _mom_12_1(data: HarnessData, asof: pd.Timestamp, ticker: str) -> float:
    """12-1 momentum (12m return excluding most recent month) using monthly prices."""
    if ticker not in data.mp.columns or asof not in data.mp.index:
        return float("nan")
    idx = data.mp.index.searchsorted(asof)
    if idx < 13:
        return float("nan")
    px_now = data.mp.iloc[idx - 1][ticker]   # last month
    px_then = data.mp.iloc[idx - 13][ticker]  # 13 months ago
    if pd.isna(px_now) or pd.isna(px_then) or px_then == 0:
        return float("nan")
    return float(px_now / px_then - 1)


def pick_anchor(asof, eligible, data: HarnessData, regime: str):
    """K=2 from model (Chronos-filtered top score) + 1 mega-cap anchor
    (highest 12-1 momentum within eligible pool)."""
    sub = data.ml_v2[data.ml_v2["asof"] == asof].copy()
    sub = sub[sub["ticker"].isin(eligible)]
    if len(sub) == 0:
        return [], []
    sub["score"] = (sub["pred_3m"] + sub["pred_6m"]) / 2
    ch = data.chronos.get(asof, {})
    if ch:
        sub["chr"] = sub["ticker"].map(ch)
        sub["chr_rk"] = sub["chr"].rank(pct=True)
        sub = sub[sub["chr_rk"] >= CHRONOS_FILTER_Q]
    if len(sub) < 2:
        return [], []
    alpha_picks = sub.sort_values("score", ascending=False).head(2)["ticker"].tolist()

    # Anchor: top 12-1 momentum within the original eligible pool
    anchors = []
    for tk in eligible - set(alpha_picks):
        m = _mom_12_1(data, asof, tk)
        if pd.notna(m):
            anchors.append((tk, m))
    if not anchors:
        picks = alpha_picks
    else:
        anchors.sort(key=lambda x: x[1], reverse=True)
        picks = alpha_picks + [anchors[0][0]]
    weights = invvol_weights(picks, data.mret, asof, cap=CAP_PER_PICK)
    return picks, list(weights)


# ============================================================
#  Runner
# ============================================================
VARIANTS = [
    ("baseline",                pick_v5_baseline,        None),
    ("concentration_overlay",   pick_v5_baseline,        "concentration"),
    ("sector_diversification",  pick_sector_diversified, None),
    ("cap_loose_bull",          pick_cap_loose_bull,     None),
    ("ensemble",                pick_ensemble,           None),
    ("anchor",                  pick_anchor,             None),
]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = load_all()
    print(f"\nLoaded data. asofs range: {data.asofs[0].date()} → {data.asofs[-1].date()}")

    summary_rows = []
    for name, pick_fn, overlay_name in VARIANTS:
        print(f"\n{'=' * 60}\n  Running variant: {name}\n{'=' * 60}")
        overlay = None
        if overlay_name == "concentration":
            overlay = make_concentration_overlay(data, sleeve=0.25,
                                                  pctile_threshold=0.30)
        res = evaluate(data, pick_fn, name, cash_overlay_fn=overlay)
        # Strip the heavy "log" before saving
        log = res.pop("log")
        with open(RESULTS / f"{name}.json", "w") as f:
            json.dump(res, f, indent=2, default=str)
        # Equity curve for the variant
        pd.DataFrame(log).to_csv(RESULTS / f"{name}_equity.csv", index=False)

        print(f"  CAGR lump-sum: {res['cagr_lump_sum_pct']:7.2f}%  "
              f"DCA: {res['cagr_dca_pct']:7.2f}%")
        print(f"  vs SPY        lump-sum: {res['spy_lump_sum_pct']:7.2f}%  "
              f"DCA: {res['spy_dca_pct']:7.2f}%")
        print(f"  edge           lump-sum: {res['edge_lump_sum_pp']:+7.2f}pp  "
              f"DCA: {res['edge_dca_pp']:+7.2f}pp")
        print(f"  WF: mean={res['wf_mean_pct']:.2f}% min={res['wf_min_pct']:.2f}% "
              f"max={res['wf_max_pct']:.2f}% edge={res['wf_mean_edge_pp']:+.2f}pp  "
              f"beat_spy={res['wf_n_beat_spy']}/10 positive={res['wf_n_positive']}/10")
        print(f"  Sharpe: {res['sharpe']:.2f}  MaxDD: {res['max_dd_pct']:.1f}%")
        print(f"  Lagging years (2014/2018/2024/2025):")
        for yr in res["year_by_year"]:
            if yr["year"] in (2014, 2018, 2024, 2025):
                print(f"    {yr['year']}: strat={yr['strat_ret_pct']:+7.2f}% "
                      f"spy={yr['spy_ret_pct']:+7.2f}% edge={yr['edge_pp']:+7.2f}pp")

        summary_rows.append({
            "variant": name,
            "cagr_lump_sum_pct": res["cagr_lump_sum_pct"],
            "cagr_dca_pct": res["cagr_dca_pct"],
            "edge_lump_sum_pp": res["edge_lump_sum_pp"],
            "edge_dca_pp": res["edge_dca_pp"],
            "wf_mean_pct": res["wf_mean_pct"],
            "wf_min_pct": res["wf_min_pct"],
            "wf_max_pct": res["wf_max_pct"],
            "wf_mean_edge_pp": res["wf_mean_edge_pp"],
            "wf_n_beat_spy": res["wf_n_beat_spy"],
            "wf_n_positive": res["wf_n_positive"],
            "sharpe": res["sharpe"],
            "max_dd_pct": res["max_dd_pct"],
            "n_baskets": res["n_baskets"],
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / "SUMMARY.csv", index=False)
    print(f"\n\nSUMMARY:\n{summary.to_string(index=False)}")
    print(f"\nSaved to: {RESULTS}")


if __name__ == "__main__":
    main()
