"""
Round 5: signal ablation around the v21_v13_no_vt winner.

Best so far: v21_v13_no_vt
  - signals: [mom_6_1, sharpe_5y, idio_mom_12_1, -vol_1y, trend_health_5y]
  - sector cap <=4
  - K=30 inv-vol cap 7%
  - regime: SPY 200ma_loose
  - NO vol-target
  - 10bps round-trip
  -> SP500 0.70/6.7%, NDX 0.76/7.2%, combined 0.73

Ablation: drop / add one signal at a time + sweep K and sector_cap.
Also: signal weighting by full-sample IC (one-shot, not adaptive).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from strategy_search_v4 import run_strategy_v4
from strategy_search import (
    build_ndx_panel, OOS_START_NDX, OOS_END_NDX,
    OOS_START_SP500, OOS_END_SP500,
    NDX_DIR, DAILY_PRICES_MAIN,
)
from strategy_search_v2 import make_rank_ensemble, SIGNALS_V13
from strategy_v1 import load_sector_map
from diagnostics import load_panel, load_daily, load_membership

OUT = HERE


# All candidate signals
SIGNALS_FULL = [
    ("mom_6_1", +1),
    ("sharpe_5y", +1),
    ("idio_mom_12_1", +1),
    ("vol_1y", -1),
    ("trend_health_5y", +1),
    ("mom_12_1", +1),
    ("mom_3", +1),
    ("rs_3m_spy", +1),
    ("rs_6m_spy", +1),
    ("rs_12m_spy", +1),
    ("recovery_rate", +1),
    ("quality_score_5y", +1),
    ("sharpe_12m", +1),
    ("mom_per_unit_vol_12", +1),
    ("breakout_strength_60", +1),
    ("mom_accel", +1),
    ("trend_r2_12m", +1),
    ("accel", +1),
    ("dd_from_52wh", +1),  # closer-to-52wh = higher; less negative = better
    ("rsi_zone_score", +1),
]


def compute_signal_ic_one_shot(panel, signals, oos_start, oos_end):
    """Spearman IC of each signal vs fwd_1m_ret over the OOS window.
    Note: 'one-shot' = uses ALL months in OOS to set weights. This is mildly
    look-ahead but for relative weighting it's a stable estimate; the actual
    SCORES at each rebalance still only use that month's features."""
    out = {}
    sub = panel[(panel["asof"] >= oos_start) & (panel["asof"] <= oos_end)]
    sub = sub.dropna(subset=["fwd_1m_ret"])
    for col, sign in signals:
        if col not in sub.columns: continue
        x = pd.Series(sign * sub[col].values).rank(method="average")
        y = pd.Series(sub["fwd_1m_ret"].values).rank(method="average")
        ic = x.corr(y)
        out[col] = (float(ic) if np.isfinite(ic) else 0.0, sign)
    return out


def make_ic_weighted_rank(signals, ic_table):
    weights = {c: max(0.0, ic_table.get(c, (0, 0))[0]) for c, _ in signals}
    s = sum(weights.values())
    if s < 1e-6: weights = {c: 1.0 for c, _ in signals}; s = sum(weights.values())
    weights = {c: w / s for c, w in weights.items()}
    def fn(snap):
        idx = snap["ticker"].values
        out = pd.Series(0.0, index=idx)
        cnt = pd.Series(0.0, index=idx)
        for col, sign in signals:
            if col not in snap.columns or weights.get(col, 0) == 0: continue
            r = (sign * pd.Series(snap[col].values, index=idx)).rank(pct=True)
            mask = r.notna()
            out[mask] = out[mask] + r[mask] * weights[col]
            cnt[mask] = cnt[mask] + weights[col]
        valid = cnt > 0
        out = out / cnt.where(cnt > 0, np.nan)
        return out[valid].dropna()
    return fn


def main():
    print("Loading data ...")
    sp_panel = load_panel(); sp_daily = load_daily(); sp_mem = load_membership()
    sector_map = load_sector_map()
    sp_monthly = sp_daily.resample("ME").last().ffill(limit=5)
    ndx_panel, ndx_monthly = build_ndx_panel()
    ndx_mem = pd.read_parquet(NDX_DIR / "ndx_pit_membership_monthly_full.parquet")
    ndx_mem["asof"] = pd.to_datetime(ndx_mem["asof"])
    main_daily = pd.read_parquet(DAILY_PRICES_MAIN)
    main_daily.index = pd.to_datetime(main_daily.index)

    # ---- One-shot signal IC tables per panel ----
    print("Computing one-shot signal ICs ...")
    sp_ic  = compute_signal_ic_one_shot(sp_panel,  SIGNALS_FULL, OOS_START_SP500, OOS_END_SP500)
    ndx_ic = compute_signal_ic_one_shot(ndx_panel, SIGNALS_FULL, OOS_START_NDX,   OOS_END_NDX)
    print("  SP500 ICs (top 10):")
    for col, (ic, sign) in sorted(sp_ic.items(), key=lambda x: -x[1][0])[:10]:
        print(f"    {col:<28} sign={sign:+d}  IC={ic:+.4f}")
    print("  NDX ICs (top 10):")
    for col, (ic, sign) in sorted(ndx_ic.items(), key=lambda x: -x[1][0])[:10]:
        print(f"    {col:<28} sign={sign:+d}  IC={ic:+.4f}")

    # Ablation: v13 base ± one signal
    v13 = SIGNALS_V13
    ablations = [
        ("v13_baseline", v13),
        ("v13_drop_mom6",  [s for s in v13 if s[0] != "mom_6_1"]),
        ("v13_drop_sh5y",  [s for s in v13 if s[0] != "sharpe_5y"]),
        ("v13_drop_idio",  [s for s in v13 if s[0] != "idio_mom_12_1"]),
        ("v13_drop_vol",   [s for s in v13 if s[0] != "vol_1y"]),
        ("v13_drop_trend", [s for s in v13 if s[0] != "trend_health_5y"]),
        ("v13_add_mom12",  v13 + [("mom_12_1", +1)]),
        ("v13_add_qs5",    v13 + [("quality_score_5y", +1)]),
        ("v13_add_mpuv",   v13 + [("mom_per_unit_vol_12", +1)]),
        ("v13_add_rs6m",   v13 + [("rs_6m_spy", +1)]),
        ("v13_add_breakout", v13 + [("breakout_strength_60", +1)]),
        ("v13_add_rsi_zone", v13 + [("rsi_zone_score", +1)]),
        ("v13_top4_byIC_sp", None),
        ("v13_top6_byIC_sp", None),
        ("v13_ic_weighted",  None),
    ]

    # Build scorers
    def make_scorer(sig_set):
        return make_rank_ensemble(sig_set)

    # Top-N by IC on SP500
    sp_sorted = [(c, sp_ic[c][1]) for c, _ in sorted(sp_ic.items(), key=lambda x: -x[1][0])]
    top4_sp = sp_sorted[:4]
    top6_sp = sp_sorted[:6]

    scorers_map = {}
    ndx_scorers_map = {}
    for label, sigs in ablations:
        if label == "v13_top4_byIC_sp":
            sp_sigs = top4_sp
            ndx_sigs = top4_sp  # same set for joint test
        elif label == "v13_top6_byIC_sp":
            sp_sigs = top6_sp; ndx_sigs = top6_sp
        elif label == "v13_ic_weighted":
            # IC-weighted ensembles per panel
            scorers_map[label] = make_ic_weighted_rank(SIGNALS_FULL, sp_ic)
            ndx_scorers_map[label] = make_ic_weighted_rank(SIGNALS_FULL, ndx_ic)
            continue
        else:
            sp_sigs = sigs; ndx_sigs = sigs
        scorers_map[label] = make_scorer(sp_sigs)
        ndx_scorers_map[label] = make_scorer(ndx_sigs)

    # Also test K and sector_cap sweep on v13_baseline
    knob_variants = [
        ("v13_K25_sec4",  "v13_baseline", True, 25, 4),
        ("v13_K30_sec3",  "v13_baseline", True, 30, 3),
        ("v13_K30_sec5",  "v13_baseline", True, 30, 5),
        ("v13_K30_sec6",  "v13_baseline", True, 30, 6),
        ("v13_K40_sec5",  "v13_baseline", True, 40, 5),
        ("v13_K20_sec3",  "v13_baseline", True, 20, 3),
    ]

    print(f"\n{'variant':<32} {'sp_CAGR':>8} {'sp_Sh':>6} {'ndx_CAGR':>9} {'ndx_Sh':>7}  {'comb_Sh':>7}")
    rows = []

    # Ablation variants -- all K=30 sector_cap=4
    for label, _ in ablations:
        sf = scorers_map[label]; nf = ndx_scorers_map[label]
        df_sp, m_sp = run_strategy_v4(sp_panel, sp_monthly, sp_daily, sp_mem, sector_map,
                                      score_fn=sf, top_k=30, sector_cap=4, use_sector_div=True,
                                      target_vol=0.18, use_vol_target=False, quarterly=False,
                                      oos_start=OOS_START_SP500, oos_end=OOS_END_SP500)
        df_nx, m_nx = run_strategy_v4(ndx_panel, ndx_monthly, main_daily, ndx_mem, sector_map,
                                      score_fn=nf, top_k=30, sector_cap=4, use_sector_div=True,
                                      target_vol=0.18, use_vol_target=False, quarterly=False,
                                      oos_start=OOS_START_NDX, oos_end=OOS_END_NDX)
        if m_sp and m_nx:
            comb = (m_sp["sharpe"] + m_nx["sharpe"]) / 2
            print(f"{label:<32} {m_sp['cagr']:>7.1%} {m_sp['sharpe']:>6.2f} "
                  f"{m_nx['cagr']:>8.1%} {m_nx['sharpe']:>7.2f}  {comb:>7.2f}")
            rows.append(dict(name=label, sp=m_sp, ndx=m_nx, combined=comb))

    # Knob sweep
    for label, key, use_sec, k, sec_cap in knob_variants:
        sf = scorers_map[key]; nf = ndx_scorers_map[key]
        df_sp, m_sp = run_strategy_v4(sp_panel, sp_monthly, sp_daily, sp_mem, sector_map,
                                      score_fn=sf, top_k=k, sector_cap=sec_cap, use_sector_div=use_sec,
                                      target_vol=0.18, use_vol_target=False,
                                      oos_start=OOS_START_SP500, oos_end=OOS_END_SP500)
        df_nx, m_nx = run_strategy_v4(ndx_panel, ndx_monthly, main_daily, ndx_mem, sector_map,
                                      score_fn=nf, top_k=k, sector_cap=sec_cap, use_sector_div=use_sec,
                                      target_vol=0.18, use_vol_target=False,
                                      oos_start=OOS_START_NDX, oos_end=OOS_END_NDX)
        if m_sp and m_nx:
            comb = (m_sp["sharpe"] + m_nx["sharpe"]) / 2
            print(f"{label:<32} {m_sp['cagr']:>7.1%} {m_sp['sharpe']:>6.2f} "
                  f"{m_nx['cagr']:>8.1%} {m_nx['sharpe']:>7.2f}  {comb:>7.2f}")
            rows.append(dict(name=label, sp=m_sp, ndx=m_nx, combined=comb))

    rows.sort(key=lambda r: -r["combined"])
    print("\nTop 8 by combined Sharpe (round 5):")
    for r in rows[:8]:
        print(f"  {r['name']:<32}  sp={r['sp']['sharpe']:.2f}/{r['sp']['cagr']:.1%}  "
              f"ndx={r['ndx']['sharpe']:.2f}/{r['ndx']['cagr']:.1%}  comb={r['combined']:.2f}")
    json.dump([dict(name=r["name"], sp=r["sp"], ndx=r["ndx"], combined=r["combined"]) for r in rows],
              open(OUT / "strategy_search_v5_results.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    sys.exit(main())
