"""Explore the anchor / momentum-screen family further to enhance CAGR
and protect downside. Every variant runs through the same 10-split
walk-forward on PIT SP500, then the top performers are re-tested
through the staggered monthly-tranche DCA simulator.

Variants:
  baseline           — v5 production (K=3 alpha, no anchor)
  anchor_baseline    — K=2 alpha + 1 top mom_12_1 (the previous winner)
  anchor_vol_adj     — K=2 alpha + 1 top mom_per_unit_vol_12 (Sharpe-like)
  anchor_idio        — K=2 alpha + 1 top idio_mom_12_1 (beta-stripped)
  anchor_uptrend     — K=2 alpha + 1 top mom_12_1 from stocks with
                        d_sma200 > 0 AND mom_3y > 0 (quality filter)
  anchor_multi_horiz — K=2 alpha + 1 top avg-rank(mom_12_1, mom_6_1, mom_3y)
  anchor_rs_spy      — K=2 alpha + 1 top rs_12m_spy (relative strength)
  anchor_chronos     — K=2 alpha + 1 top mom_12_1 from Chronos-passed
                        cohort (≥ 0.45 rank)
  anchor_low_cap     — K=2 alpha + 1 mom_12_1 anchor capped at 0.30
                        per pick (less single-name risk)
  dual_anchor        — K=1 alpha + 1 mom_12_1 + 1 mom_6_1 (two
                        momentum picks of different horizons)

Outputs (per variant):
  experiments/monthly_dca/v5/validations/results/<variant>.json
  experiments/monthly_dca/v5/validations/results/<variant>_equity.csv

Plus an updated SUMMARY.csv that includes the 6 original variants + the
new ones, sorted by wf_mean.

Top-3 by wf_mean are then re-run through the staggered monthly-tranche
DCA simulator (run_staggered_dca harness).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.monthly_dca.v5.validations.harness import (
    HarnessData, load_all, evaluate, invvol_weights,
    pick_v5_baseline,
    CHRONOS_FILTER_Q, CAP_PER_PICK, K_PICKS, COST_BPS,
)

RES = Path(__file__).resolve().parent / "results"


# ============================================================
#  Anchor variants — all share the same K=2 alpha core, differ
#  only in how the third "anchor" pick is selected.
# ============================================================
def _alpha_picks(asof, eligible, data: HarnessData, k_alpha: int = 2):
    """The v5 alpha core: GBM 3m+6m score, Chronos filter, top-k_alpha."""
    sub = data.ml_v2[data.ml_v2["asof"] == asof].copy()
    sub = sub[sub["ticker"].isin(eligible)]
    if len(sub) == 0:
        return [], None
    sub["score"] = (sub["pred_3m"] + sub["pred_6m"]) / 2
    ch = data.chronos.get(asof, {})
    if ch:
        sub["chr"] = sub["ticker"].map(ch)
        sub["chr_rk"] = sub["chr"].rank(pct=True)
        sub = sub[sub["chr_rk"] >= CHRONOS_FILTER_Q]
    if len(sub) < k_alpha:
        return [], None
    top = sub.sort_values("score", ascending=False).head(k_alpha)
    return top["ticker"].tolist(), sub


def _pick_anchor_by(feat: pd.DataFrame, ranking_col: str, exclude: set,
                     filter_fn=None) -> str | None:
    """Pick the top ranked ticker by `ranking_col`, excluding `exclude`,
    optionally pre-filtered by `filter_fn(row) -> bool`."""
    if feat is None or ranking_col not in feat.columns:
        return None
    candidates = feat.drop(index=[t for t in exclude if t in feat.index],
                            errors="ignore")
    if filter_fn is not None:
        mask = candidates.apply(filter_fn, axis=1)
        candidates = candidates[mask]
    candidates = candidates[ranking_col].dropna().sort_values(ascending=False)
    if len(candidates) == 0:
        return None
    return candidates.index[0]


def make_anchor_picker(feature_col: str, *,
                        cap: float = CAP_PER_PICK,
                        filter_fn=None,
                        gate_by_chronos: bool = False):
    """Factory: anchor variant that picks K=2 alpha + 1 top-feature."""
    def picker(asof, eligible, data: HarnessData, regime):
        alpha, sub_filtered = _alpha_picks(asof, eligible, data, k_alpha=2)
        if not alpha:
            return [], []
        feat = data.features(asof)
        if feat is None:
            picks = alpha
            return picks, list(invvol_weights(picks, data.mret, asof, cap=cap))
        # Restrict to eligible (PIT pool) and not-already-picked
        eligible_in_feat = feat.index.intersection(list(eligible))
        cand = feat.loc[eligible_in_feat]
        if gate_by_chronos and sub_filtered is not None:
            # Restrict to Chronos-passed tickers only
            chronos_passed = set(sub_filtered["ticker"].tolist())
            cand = cand.loc[cand.index.intersection(list(chronos_passed))]
        anchor = _pick_anchor_by(cand, feature_col, exclude=set(alpha),
                                   filter_fn=filter_fn)
        if anchor is None:
            picks = alpha
        else:
            picks = alpha + [anchor]
        weights = invvol_weights(picks, data.mret, asof, cap=cap)
        return picks, list(weights)
    return picker


def make_dual_anchor_picker():
    """K=1 alpha + 1 mom_12_1 + 1 mom_6_1 (different-horizon momentum)."""
    def picker(asof, eligible, data: HarnessData, regime):
        alpha, sub_filtered = _alpha_picks(asof, eligible, data, k_alpha=1)
        if not alpha:
            return [], []
        feat = data.features(asof)
        if feat is None:
            return alpha, list(invvol_weights(alpha, data.mret, asof))
        eligible_in_feat = feat.index.intersection(list(eligible))
        cand = feat.loc[eligible_in_feat]
        a12 = _pick_anchor_by(cand, "mom_12_1", exclude=set(alpha))
        excl = set(alpha) | ({a12} if a12 else set())
        a6 = _pick_anchor_by(cand, "mom_6_1", exclude=excl)
        picks = alpha + [t for t in (a12, a6) if t]
        weights = invvol_weights(picks, data.mret, asof, cap=CAP_PER_PICK)
        return picks, list(weights)
    return picker


def make_multi_horizon_anchor():
    """K=2 alpha + 1 anchor by avg rank of mom_12_1, mom_6_1, mom_3y."""
    def picker(asof, eligible, data: HarnessData, regime):
        alpha, _ = _alpha_picks(asof, eligible, data, k_alpha=2)
        if not alpha:
            return [], []
        feat = data.features(asof)
        if feat is None:
            return alpha, list(invvol_weights(alpha, data.mret, asof))
        eligible_in_feat = feat.index.intersection(list(eligible))
        cand = feat.loc[eligible_in_feat]
        cand = cand.drop(index=[t for t in alpha if t in cand.index],
                         errors="ignore")
        # Multi-horizon momentum: avg pct rank
        cols = ["mom_12_1", "mom_6_1", "mom_3y"]
        cand_clean = cand[cols].dropna()
        if len(cand_clean) == 0:
            return alpha, list(invvol_weights(alpha, data.mret, asof))
        ranks = cand_clean.rank(pct=True).mean(axis=1)
        anchor = ranks.sort_values(ascending=False).index[0]
        picks = alpha + [anchor]
        weights = invvol_weights(picks, data.mret, asof, cap=CAP_PER_PICK)
        return picks, list(weights)
    return picker


# ============================================================
#  Variant registry
# ============================================================
VARIANTS = [
    ("baseline_v5",
     pick_v5_baseline,
     "v5 production: K=3 alpha (GBM + Chronos), inv-vol cap 0.40"),

    ("anchor_mom12",
     make_anchor_picker("mom_12_1"),
     "K=2 alpha + 1 top mom_12_1 anchor (previous run's winner)"),

    ("anchor_vol_adj",
     make_anchor_picker("mom_per_unit_vol_12"),
     "K=2 alpha + 1 top mom_per_unit_vol_12 (Sharpe-like momentum)"),

    ("anchor_idio",
     make_anchor_picker("idio_mom_12_1"),
     "K=2 alpha + 1 top idio_mom_12_1 (beta-stripped momentum)"),

    ("anchor_uptrend",
     make_anchor_picker("mom_12_1",
                         filter_fn=lambda r: r.get("d_sma200", -1) > 0
                                              and r.get("mom_3y", -1) > 0),
     "K=2 alpha + 1 top mom_12_1 filtered to d_sma200>0 AND mom_3y>0"),

    ("anchor_multi_horizon",
     make_multi_horizon_anchor(),
     "K=2 alpha + 1 anchor by avg rank(mom_12_1, mom_6_1, mom_3y)"),

    ("anchor_rs_spy",
     make_anchor_picker("rs_12m_spy"),
     "K=2 alpha + 1 top rs_12m_spy (relative strength vs SPY)"),

    ("anchor_chronos_gated",
     make_anchor_picker("mom_12_1", gate_by_chronos=True),
     "K=2 alpha + 1 top mom_12_1 from Chronos-passed cohort"),

    ("anchor_low_cap_0.30",
     make_anchor_picker("mom_12_1", cap=0.30),
     "K=2 alpha + 1 top mom_12_1 anchor; inv-vol cap 0.30 (vs 0.40)"),

    ("dual_anchor_12_6",
     make_dual_anchor_picker(),
     "K=1 alpha + 1 mom_12_1 anchor + 1 mom_6_1 anchor"),

    ("anchor_sharpe_5y",
     make_anchor_picker("sharpe_5y"),
     "K=2 alpha + 1 top sharpe_5y (long-term risk-adjusted)"),

    ("anchor_quality_5y",
     make_anchor_picker("quality_score_5y"),
     "K=2 alpha + 1 top quality_score_5y (long-term quality)"),
]


def main():
    RES.mkdir(parents=True, exist_ok=True)
    data = load_all()
    print(f"\nLoaded. asofs {data.asofs[0].date()} -> {data.asofs[-1].date()}")

    summary_rows = []
    for name, pick_fn, desc in VARIANTS:
        print(f"\n{'='*60}\n  {name}\n  {desc}\n{'='*60}")
        try:
            res = evaluate(data, pick_fn, name)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        log = res.pop("log")
        with open(RES / f"{name}.json", "w") as f:
            json.dump(res, f, indent=2, default=str)
        pd.DataFrame(log).to_csv(RES / f"{name}_equity.csv", index=False)
        print(f"  Lump-sum CAGR: {res['cagr_lump_sum_pct']:7.2f}%  "
              f"DCA: {res['cagr_dca_pct']:7.2f}%")
        print(f"  WF: mean={res['wf_mean_pct']:.2f}% "
              f"min={res['wf_min_pct']:.2f}% max={res['wf_max_pct']:.2f}% "
              f"edge={res['wf_mean_edge_pp']:+.2f}pp  "
              f"beat_spy={res['wf_n_beat_spy']}/10")
        print(f"  Sharpe {res['sharpe']:.2f}  MaxDD {res['max_dd_pct']:.1f}%")
        lag = {y["year"]: y["edge_pp"] for y in res["year_by_year"]
                if y["year"] in (2014, 2018, 2024, 2025)}
        print(f"  Lagging-year edges (pp):  "
              f"2014={lag.get(2014, 0):+5.1f}  "
              f"2018={lag.get(2018, 0):+5.1f}  "
              f"2024={lag.get(2024, 0):+5.1f}  "
              f"2025={lag.get(2025, 0):+5.1f}")

        summary_rows.append({
            "variant": name,
            "description": desc,
            "cagr_lump_sum_pct": res["cagr_lump_sum_pct"],
            "cagr_dca_pct": res["cagr_dca_pct"],
            "edge_lump_sum_pp": res["edge_lump_sum_pp"],
            "wf_mean_pct": res["wf_mean_pct"],
            "wf_min_pct": res["wf_min_pct"],
            "wf_max_pct": res["wf_max_pct"],
            "wf_mean_edge_pp": res["wf_mean_edge_pp"],
            "wf_n_beat_spy": res["wf_n_beat_spy"],
            "wf_n_positive": res["wf_n_positive"],
            "sharpe": res["sharpe"],
            "max_dd_pct": res["max_dd_pct"],
            "y2014_edge_pp": lag.get(2014),
            "y2018_edge_pp": lag.get(2018),
            "y2024_edge_pp": lag.get(2024),
            "y2025_edge_pp": lag.get(2025),
        })

    summary = pd.DataFrame(summary_rows).sort_values("wf_mean_pct", ascending=False)
    summary.to_csv(RES / "SUMMARY_momentum.csv", index=False)
    print(f"\n\nFULL SUMMARY (sorted by wf_mean):")
    print(summary.to_string(index=False))
    print(f"\nSaved → {RES / 'SUMMARY_momentum.csv'}")


if __name__ == "__main__":
    main()
