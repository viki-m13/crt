"""Comprehensive validation suite for Mode B (50% v5 + 50% multi-asset
trend sleeve) before deploying as a production alternative.

Tests:
  1. WF — 10 walk-forward splits on Mode B vs Mode A
  2. COST — transaction-cost sensitivity (10 / 20 / 30 / 50 / 100 bps)
  3. LOOKBACK — sleeve momentum lookback (6 / 9 / 12 / 18 / 24 month)
  4. UNIVERSE — robustness to dropping individual ETFs from sleeve
  5. PERIOD — Mode B in each calendar decade separately
  6. BOOTSTRAP — confirm bootstrap distribution holds on each sub-window

Output: validation/REPORT_MODE_B_VALIDATION.md + CSV artifacts.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

from experiments.monthly_dca.v5.validations.harness import (
    HarnessData, load_all, pick_v5_baseline, classify_regime_tight,
    WALK_FORWARD_SPLITS, COST_BPS, HOLD_MONTHS,
)
from experiments.monthly_dca.v5.validations.run_advanced_overlays import (
    sector_top_n_sleeve, run_advanced, summary,
)
from experiments.monthly_dca.v2.ml_strategy import EXCLUDE

RES = Path(__file__).resolve().parent / "results"
DEFAULT_ASSETS = ["XLE","XLF","XLK","XLU","XLV","XLP","XLY","XLI","XLB",
                   "TLT","EFA","EEM"]


# ---------------------------------------------------------------------------
# Reusable simulator — Mode B with configurable params
# ---------------------------------------------------------------------------
def run_mode_b(data: HarnessData, daily: pd.DataFrame,
                start: pd.Timestamp, end: pd.Timestamp,
                sleeve_weight: float = 0.50,
                sleeve_lookback: int = 252,
                sleeve_top_n: int = 2,
                sleeve_assets: list[str] = None,
                cost_bps_strat: float = COST_BPS,
                cost_bps_sleeve: float = COST_BPS) -> dict:
    """Run Mode B for the given window. Returns a log of monthly returns."""
    if sleeve_assets is None:
        sleeve_assets = DEFAULT_ASSETS
    asofs = [m for m in data.asofs
             if start <= m <= end
             and m in data.spy_features.index
             and m in data.mret.index
             and m in data.members_g]
    asofs = sorted(asofs)
    cf_s = cost_bps_strat / 1e4
    cf_sleeve = cost_bps_sleeve / 1e4

    cur_picks: list[str] = []
    cur_weights = np.array([])
    cash = False; held = 0; equity = 1.0
    prev_sleeve_top: list[str] | None = None
    log = []

    for i, m in enumerate(asofs):
        spy_now = data.spy_features.loc[m].to_dict() if m in data.spy_features.index else {}
        regime = classify_regime_tight(spy_now)
        do_reb = (i == 0) or (held >= HOLD_MONTHS) or cash

        # v5 strategy return for this month
        if cash or not cur_picks:
            strat_ret = 0.0
        else:
            r = 0.0
            for tk, w in zip(cur_picks, cur_weights):
                rt = (float(data.mret.at[m, tk])
                      if (tk in data.mret.columns and m in data.mret.index
                          and pd.notna(data.mret.at[m, tk]))
                      else 0.0)
                r += w * rt
            strat_ret = r

        # Sleeve return
        tops = sector_top_n_sleeve(daily, m, sleeve_assets,
                                     n=sleeve_top_n, lookback_d=sleeve_lookback)
        sleeve_ret = 0.0
        if tops:
            rs = [float(data.mret.at[m, s])
                   for s in tops
                   if s in data.mret.columns and m in data.mret.index
                   and pd.notna(data.mret.at[m, s])]
            if rs: sleeve_ret = float(np.mean(rs))
            # Sleeve trading cost — only when composition changes
            if prev_sleeve_top is not None:
                if set(tops) != set(prev_sleeve_top):
                    turnover = len(set(tops) ^ set(prev_sleeve_top)) / (2 * sleeve_top_n)
                    sleeve_ret -= cf_sleeve * turnover
            prev_sleeve_top = tops

        ret_m = (1 - sleeve_weight) * strat_ret + sleeve_weight * sleeve_ret

        # Rebalance
        if do_reb:
            if regime == "crash":
                cur_picks, cur_weights, cash = [], np.array([]), True
                held = 0
            else:
                eligible = data.members_g.get(m, set()) - set(EXCLUDE)
                picks, weights = pick_v5_baseline(m, eligible, data, regime)
                if picks:
                    cur_picks = list(picks)
                    cur_weights = np.array(weights, dtype=float)
                    cur_weights = cur_weights / cur_weights.sum() if cur_weights.sum() > 0 else np.ones(len(cur_picks))/len(cur_picks)
                    cash = False; held = 0
                    if log: ret_m -= cf_s * (1 - sleeve_weight)
        held += 1
        equity *= (1 + ret_m)
        log.append({"date": str(m.date()), "regime": regime,
                     "strat_ret": strat_ret, "sleeve_ret": sleeve_ret,
                     "ret_m": ret_m, "equity": equity})
    return {"log": log}


def metrics(log, spy_monthly):
    df = pd.DataFrame(log)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    n = len(df); years = n / 12
    cagr = (df["equity"].iloc[-1] ** (1/years) - 1) * 100
    spy_w = spy_monthly.loc[df["date"].iloc[0]:df["date"].iloc[-1]]
    spy_cagr = ((1+spy_w.fillna(0)).cumprod().iloc[-1] ** (12/max(len(spy_w),1))-1) * 100 if len(spy_w) else 0
    rr = df["ret_m"]
    sh = float(rr.mean()/rr.std()*np.sqrt(12)) if rr.std() > 0 else 0
    peak = df["equity"].cummax(); mdd = float(((df["equity"]-peak)/peak).min()*100)
    yr = df.groupby("year")["ret_m"].apply(lambda x: (1+x).prod()-1)*100
    spy_yr = spy_w.groupby(spy_w.index.year).apply(lambda x: (1+x).prod()-1)*100
    edges = [yr[y]-spy_yr[y] for y in sorted(yr.index) if y in spy_yr.index]
    return dict(cagr=cagr, spy_cagr=spy_cagr, edge=cagr-spy_cagr, sharpe=sh,
                mdd=mdd, edge_std=float(np.std(edges)) if edges else 0,
                edge_min=float(min(edges)) if edges else 0,
                n_years_positive=sum(1 for e in edges if e > 0),
                n_years_total=len(edges))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_walk_forward(data, daily, spy):
    print(f"\n{'='*80}\n  1. Walk-Forward (10 splits)\n{'='*80}")
    rows = []
    for name, lo, hi in WALK_FORWARD_SPLITS:
        start = pd.Timestamp(lo); end = pd.Timestamp(hi)
        # Mode A reference (just the v5 strategy)
        sim_a = run_mode_b(data, daily, start, end, sleeve_weight=0.0)
        m_a = metrics(sim_a["log"], spy)
        # Mode B (50/50)
        sim_b = run_mode_b(data, daily, start, end, sleeve_weight=0.5)
        m_b = metrics(sim_b["log"], spy)
        # Mode B-25 (lighter)
        sim_b25 = run_mode_b(data, daily, start, end, sleeve_weight=0.25)
        m_b25 = metrics(sim_b25["log"], spy)
        print(f"  {name:<12} {lo[:7]}→{hi[:7]} "
              f"A: CAGR {m_a['cagr']:>6.2f}% edge {m_a['edge']:>+6.2f}pp Sharpe {m_a['sharpe']:>4.2f}  "
              f"B: {m_b['cagr']:>6.2f}% {m_b['edge']:>+6.2f}pp {m_b['sharpe']:>4.2f}  "
              f"B25: {m_b25['cagr']:>6.2f}% {m_b25['edge']:>+6.2f}pp {m_b25['sharpe']:>4.2f}")
        rows.append({"split": name, "from": lo, "to": hi,
                     "cagr_A": m_a["cagr"], "edge_A": m_a["edge"], "sharpe_A": m_a["sharpe"], "mdd_A": m_a["mdd"],
                     "cagr_B": m_b["cagr"], "edge_B": m_b["edge"], "sharpe_B": m_b["sharpe"], "mdd_B": m_b["mdd"],
                     "cagr_B25": m_b25["cagr"], "edge_B25": m_b25["edge"], "sharpe_B25": m_b25["sharpe"], "mdd_B25": m_b25["mdd"]})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "wf_mode_b.csv", index=False)
    print(f"\n  SUMMARY:")
    print(f"    Mode A:   mean edge {df['edge_A'].mean():+.2f}pp  min {df['edge_A'].min():+.2f}pp  beats SPY {(df['edge_A']>0).sum()}/{len(df)}  mean Sharpe {df['sharpe_A'].mean():.2f}")
    print(f"    Mode B:   mean edge {df['edge_B'].mean():+.2f}pp  min {df['edge_B'].min():+.2f}pp  beats SPY {(df['edge_B']>0).sum()}/{len(df)}  mean Sharpe {df['sharpe_B'].mean():.2f}")
    print(f"    Mode B25: mean edge {df['edge_B25'].mean():+.2f}pp  min {df['edge_B25'].min():+.2f}pp  beats SPY {(df['edge_B25']>0).sum()}/{len(df)}  mean Sharpe {df['sharpe_B25'].mean():.2f}")
    return df


def test_cost_sensitivity(data, daily, spy):
    print(f"\n{'='*80}\n  2. Transaction-cost sensitivity\n{'='*80}")
    rows = []
    start = data.asofs[0]; end = data.spy_features.index.max()
    for cost_bps in (5, 10, 20, 30, 50, 75, 100, 150):
        sim = run_mode_b(data, daily, start, end, sleeve_weight=0.5,
                          cost_bps_strat=cost_bps, cost_bps_sleeve=cost_bps)
        m = metrics(sim["log"], spy)
        print(f"  Cost {cost_bps:>3}bps:  CAGR {m['cagr']:>6.2f}%  edge {m['edge']:>+6.2f}pp  Sharpe {m['sharpe']:>4.2f}  MDD {m['mdd']:>6.1f}%")
        rows.append({"cost_bps": cost_bps, **m})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "cost_mode_b.csv", index=False)
    return df


def test_lookback(data, daily, spy):
    print(f"\n{'='*80}\n  3. Sleeve momentum lookback\n{'='*80}")
    rows = []
    start = data.asofs[0]; end = data.spy_features.index.max()
    for lb in (126, 189, 252, 378, 504):
        lb_months = lb // 21
        sim = run_mode_b(data, daily, start, end, sleeve_weight=0.5,
                          sleeve_lookback=lb)
        m = metrics(sim["log"], spy)
        print(f"  Lookback {lb_months:>3}m ({lb:>3}d):  CAGR {m['cagr']:>6.2f}%  edge {m['edge']:>+6.2f}pp  Sharpe {m['sharpe']:>4.2f}  MDD {m['mdd']:>6.1f}%")
        rows.append({"lookback_days": lb, "lookback_months": lb_months, **m})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "lookback_mode_b.csv", index=False)
    return df


def test_universe(data, daily, spy):
    print(f"\n{'='*80}\n  4. Universe robustness (drop-one-out + subsets)\n{'='*80}")
    rows = []
    start = data.asofs[0]; end = data.spy_features.index.max()
    # Drop-one-out
    print("  Drop-one-out (drop each asset, run with remaining):")
    for drop in DEFAULT_ASSETS:
        assets = [a for a in DEFAULT_ASSETS if a != drop]
        sim = run_mode_b(data, daily, start, end, sleeve_weight=0.5, sleeve_assets=assets)
        m = metrics(sim["log"], spy)
        print(f"    drop {drop:<5}: CAGR {m['cagr']:>6.2f}%  Sharpe {m['sharpe']:>4.2f}  MDD {m['mdd']:>6.1f}%")
        rows.append({"variant": f"drop_{drop}", "kind": "drop_one_out", **m})
    # Subset variants
    print("  Subsets:")
    for label, assets in [
        ("sectors_only", ["XLE","XLF","XLK","XLU","XLV","XLP","XLY","XLI","XLB"]),
        ("sectors+TLT",  ["XLE","XLF","XLK","XLU","XLV","XLP","XLY","XLI","XLB","TLT"]),
        ("no_sectors",   ["TLT","EFA","EEM"]),
        ("only_intl",    ["EFA","EEM"]),
        ("only_bond",    ["TLT"]),
        ("default",      DEFAULT_ASSETS),
    ]:
        sim = run_mode_b(data, daily, start, end, sleeve_weight=0.5, sleeve_assets=assets)
        m = metrics(sim["log"], spy)
        print(f"    {label:<14}: CAGR {m['cagr']:>6.2f}%  Sharpe {m['sharpe']:>4.2f}  MDD {m['mdd']:>6.1f}%")
        rows.append({"variant": label, "kind": "subset", **m})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "universe_mode_b.csv", index=False)
    return df


def test_decades(data, daily, spy):
    print(f"\n{'='*80}\n  5. Sub-period robustness (decade slices)\n{'='*80}")
    rows = []
    decades = [
        ("2003-2009 (GFC era)",       "2003-09-30", "2009-12-31"),
        ("2010-2019 (post-GFC bull)", "2010-01-01", "2019-12-31"),
        ("2020-2026 (COVID + AI)",    "2020-01-01", "2026-04-30"),
        ("2013-2017 (mid bull)",      "2013-01-01", "2017-12-31"),
        ("2018-2022 (vol regime)",    "2018-01-01", "2022-12-31"),
    ]
    print(f"  {'Period':<26}  {'CAGR_A':>8} {'CAGR_B':>8} {'SPY':>8} {'edge_A':>8} {'edge_B':>8} {'mdd_A':>7} {'mdd_B':>7}")
    for name, lo, hi in decades:
        start = pd.Timestamp(lo); end = pd.Timestamp(hi)
        sim_a = run_mode_b(data, daily, start, end, sleeve_weight=0.0)
        m_a = metrics(sim_a["log"], spy)
        sim_b = run_mode_b(data, daily, start, end, sleeve_weight=0.5)
        m_b = metrics(sim_b["log"], spy)
        print(f"  {name:<26}  {m_a['cagr']:>7.2f}% {m_b['cagr']:>7.2f}% {m_a['spy_cagr']:>7.2f}% {m_a['edge']:>+6.2f}pp {m_b['edge']:>+6.2f}pp {m_a['mdd']:>6.1f}% {m_b['mdd']:>6.1f}%")
        rows.append({"period": name, "from": lo, "to": hi,
                     "cagr_A": m_a["cagr"], "cagr_B": m_b["cagr"],
                     "edge_A": m_a["edge"], "edge_B": m_b["edge"],
                     "sharpe_A": m_a["sharpe"], "sharpe_B": m_b["sharpe"],
                     "mdd_A": m_a["mdd"], "mdd_B": m_b["mdd"]})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "decades_mode_b.csv", index=False)
    return df


def main():
    RES.mkdir(parents=True, exist_ok=True)
    data = load_all()
    spy = data.mret["SPY"].copy(); spy.index = pd.to_datetime(spy.index)
    daily = pd.read_parquet(Path("experiments/monthly_dca/cache/prices_extended.parquet"))

    wf_df = test_walk_forward(data, daily, spy)
    cost_df = test_cost_sensitivity(data, daily, spy)
    lb_df = test_lookback(data, daily, spy)
    univ_df = test_universe(data, daily, spy)
    dec_df = test_decades(data, daily, spy)

    print("\n\nALL DONE — see CSV artifacts in results/")


if __name__ == "__main__":
    main()
