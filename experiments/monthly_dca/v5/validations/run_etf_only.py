"""Standalone ETF-only momentum strategy — does the trend sleeve work
without the v5 picker?

Tests:
  1. WF — 10 walk-forward splits, ETF sleeve standalone
  2. TOP_N — top-1, top-2, top-3, top-4 by 12m momentum
  3. LOOKBACK — 3, 6, 9, 12, 18, 24 month momentum
  4. COST — 5/10/20/30/50/100 bps per rotation
  5. UNIVERSE — different ETF baskets (sectors only, broad, asset classes)
  6. DECADES — sub-period robustness
  7. CRISIS — explicit GFC and COVID windows
  8. BOOTSTRAP — distribution of 12-month outcomes
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

from experiments.monthly_dca.v5.validations.harness import (
    load_all, WALK_FORWARD_SPLITS,
)

RES = Path(__file__).resolve().parent / "results"
DEFAULT_ASSETS = ["XLE","XLF","XLK","XLU","XLV","XLP","XLY","XLI","XLB",
                   "TLT","EFA","EEM"]


def run_etf_only(daily_prices: pd.DataFrame,
                   monthly_returns: pd.DataFrame,
                   asofs: list[pd.Timestamp],
                   assets: list[str] = DEFAULT_ASSETS,
                   lookback_d: int = 252,
                   top_n: int = 2,
                   cost_bps: float = 10.0) -> dict:
    """Simulate a pure ETF momentum strategy: each month-end rotate to
    top-N by trailing momentum, equal-weight. Returns log of monthly
    returns + equity curve.

    IMPORTANT: monthly_returns is indexed on TRADING-day month-ends (which
    differ from calendar month-ends in 2020-Feb, May, Oct etc when the
    last day falls on a weekend). The strategy must use the trading-day
    month-end's monthly return for the next-month-in-trace. We do a
    nearest-prior-trading-day lookup to handle this.
    """
    cf = cost_bps / 1e4
    available = [a for a in assets if a in daily_prices.columns]
    equity = 1.0
    log = []
    rotations = 0
    cash_months = 0
    mret_idx = monthly_returns.index

    # NO-LOOK-AHEAD SEMANTICS:
    # At iteration i with asof m, we first apply month m's return to the
    # basket carried from iteration i-1 (decided at month m-1). Then we
    # compute picks using prices up to m, which become the basket for
    # month m+1's return at iteration i+1. The basket formed at m never
    # receives credit for m's own return.
    carried_picks: list[str] = []
    carried_basket_was_rotation = False  # cost only if this carried basket replaced a prior one

    for i, m in enumerate(asofs):
        # 1) Apply month m's realised return to the basket carried from m-1
        pos = mret_idx.searchsorted(m, side="right") - 1
        m_idx = mret_idx[pos] if pos >= 0 else None
        if m_idx is None or abs((m_idx - m).days) > 7:
            m_idx = None
        if carried_picks and m_idx is not None:
            rs = []
            for tk in carried_picks:
                if (tk in monthly_returns.columns
                        and pd.notna(monthly_returns.at[m_idx, tk])):
                    rs.append(float(monthly_returns.at[m_idx, tk]))
            ret_m = float(np.mean(rs)) if rs else 0.0
            if carried_basket_was_rotation:
                # Pay transaction cost on the rotation that created this basket
                ret_m -= cf * carried_basket_was_rotation  # turnover * cf
                carried_basket_was_rotation = 0
        else:
            ret_m = 0.0
            if not carried_picks:
                cash_months += 1

        equity *= (1 + ret_m)

        # 2) Now decide NEW picks at month-end m for next iteration's return
        px = daily_prices.loc[:m, available].dropna(how="all")
        new_picks = []
        if len(px) >= lookback_d:
            ret_lookback = px.iloc[-1] / px.iloc[-lookback_d] - 1
            ret_lookback = ret_lookback[ret_lookback > 0]
            if len(ret_lookback) > 0:
                new_picks = ret_lookback.sort_values(ascending=False).head(top_n).index.tolist()

        # Log THIS month-end with the basket that produced m's return
        log.append({"date": str(m.date()), "ret_m": ret_m, "equity": equity,
                     "picks": ",".join(carried_picks) if carried_picks else "",
                     "next_picks": ",".join(new_picks) if new_picks else "",
                     "n_rotations": rotations})

        # 3) Carry the new picks for the next iteration
        if new_picks and carried_picks and set(new_picks) != set(carried_picks):
            turnover = len(set(new_picks) ^ set(carried_picks)) / (2 * top_n)
            carried_basket_was_rotation = turnover
            rotations += 1
        elif new_picks and not carried_picks:
            # Initial entry — also a rotation
            carried_basket_was_rotation = 1.0
            rotations += 1
        else:
            carried_basket_was_rotation = 0
        carried_picks = new_picks

    return {"log": log, "n_rotations": rotations, "cash_months": cash_months}


def metrics(log: list, spy_monthly: pd.Series):
    df = pd.DataFrame(log); df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    n = len(df); years = n / 12
    cagr = (df["equity"].iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0
    spy_w = spy_monthly.loc[df["date"].iloc[0]:df["date"].iloc[-1]]
    spy_eq = (1 + spy_w.fillna(0)).cumprod()
    spy_cagr = (spy_eq.iloc[-1] ** (12/len(spy_w)) - 1) * 100 if len(spy_w) else 0
    rr = df["ret_m"]
    sh = float(rr.mean() / rr.std() * np.sqrt(12)) if rr.std() > 0 else 0
    peak = df["equity"].cummax()
    mdd = float(((df["equity"] - peak) / peak).min() * 100)
    yr = df.groupby("year")["ret_m"].apply(lambda x: (1+x).prod()-1) * 100
    spy_yr = spy_w.groupby(spy_w.index.year).apply(lambda x: (1+x).prod()-1) * 100
    edges = [yr[y] - spy_yr[y] for y in sorted(yr.index) if y in spy_yr.index]
    return dict(cagr=cagr, spy_cagr=spy_cagr, edge=cagr-spy_cagr, sharpe=sh,
                mdd=mdd, edge_std=float(np.std(edges)) if edges else 0,
                edge_min=float(min(edges)) if edges else 0,
                n_years_positive=sum(1 for e in edges if e > 0),
                n_years_total=len(edges))


def test_walk_forward(daily, mret, spy):
    print(f"\n{'='*70}\n  1. Walk-forward — ETF-only on 10 splits\n{'='*70}")
    rows = []
    for name, lo, hi in WALK_FORWARD_SPLITS:
        # Build list of month-ends in window
        asofs = pd.date_range(start=lo, end=hi, freq="ME")
        sim = run_etf_only(daily, mret, list(asofs))
        m = metrics(sim["log"], spy)
        print(f"  {name:<12} {lo[:7]}→{hi[:7]}  CAGR {m['cagr']:>6.2f}%  edge {m['edge']:>+6.2f}pp  Sharpe {m['sharpe']:>4.2f}  MDD {m['mdd']:>6.1f}%  rotations {sim['n_rotations']}")
        rows.append({"split": name, "from": lo, "to": hi, **m, "rotations": sim["n_rotations"]})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "etf_only_wf.csv", index=False)
    print(f"\n  WF SUMMARY: mean edge {df['edge'].mean():+.2f}pp · min {df['edge'].min():+.2f}pp · "
          f"beats SPY {(df['edge']>0).sum()}/{len(df)} · mean Sharpe {df['sharpe'].mean():.2f}")
    return df


def test_top_n(daily, mret, spy):
    print(f"\n{'='*70}\n  2. Top-N sensitivity\n{'='*70}")
    rows = []
    asofs = pd.date_range(start="2003-01-31", end="2026-04-30", freq="ME")
    for n in (1, 2, 3, 4):
        sim = run_etf_only(daily, mret, list(asofs), top_n=n)
        m = metrics(sim["log"], spy)
        print(f"  top-{n}: CAGR {m['cagr']:>6.2f}%  edge {m['edge']:>+6.2f}pp  Sharpe {m['sharpe']:>4.2f}  MDD {m['mdd']:>6.1f}%  rotations {sim['n_rotations']}")
        rows.append({"top_n": n, "rotations": sim["n_rotations"], **m})
    pd.DataFrame(rows).to_csv(RES / "etf_only_topn.csv", index=False)
    return rows


def test_lookback(daily, mret, spy):
    print(f"\n{'='*70}\n  3. Lookback sensitivity (12-month default)\n{'='*70}")
    rows = []
    asofs = pd.date_range(start="2003-01-31", end="2026-04-30", freq="ME")
    for lb in (63, 126, 189, 252, 378, 504):
        sim = run_etf_only(daily, mret, list(asofs), lookback_d=lb)
        m = metrics(sim["log"], spy)
        lbm = lb // 21
        print(f"  {lbm:>3}m ({lb:>3}d): CAGR {m['cagr']:>6.2f}%  edge {m['edge']:>+6.2f}pp  Sharpe {m['sharpe']:>4.2f}  MDD {m['mdd']:>6.1f}%")
        rows.append({"lookback_days": lb, "lookback_months": lbm, **m})
    pd.DataFrame(rows).to_csv(RES / "etf_only_lookback.csv", index=False)
    return rows


def test_cost(daily, mret, spy):
    print(f"\n{'='*70}\n  4. Transaction cost sensitivity\n{'='*70}")
    rows = []
    asofs = pd.date_range(start="2003-01-31", end="2026-04-30", freq="ME")
    for cb in (0, 5, 10, 20, 30, 50, 75, 100, 150):
        sim = run_etf_only(daily, mret, list(asofs), cost_bps=cb)
        m = metrics(sim["log"], spy)
        print(f"  {cb:>3}bps: CAGR {m['cagr']:>6.2f}%  edge {m['edge']:>+6.2f}pp  Sharpe {m['sharpe']:>4.2f}  MDD {m['mdd']:>6.1f}%")
        rows.append({"cost_bps": cb, **m})
    pd.DataFrame(rows).to_csv(RES / "etf_only_cost.csv", index=False)
    return rows


def test_universes(daily, mret, spy):
    print(f"\n{'='*70}\n  5. Universe generalisation\n{'='*70}")
    rows = []
    asofs = pd.date_range(start="2003-01-31", end="2026-04-30", freq="ME")
    universes = {
        "default_12 (9 sectors + TLT + EFA + EEM)": DEFAULT_ASSETS,
        "9_sectors_only": ["XLE","XLF","XLK","XLU","XLV","XLP","XLY","XLI","XLB"],
        "8_sectors (no XLF)": ["XLE","XLK","XLU","XLV","XLP","XLY","XLI","XLB"],
        "3_broad (SPY/QQQ/IWM)": ["SPY","QQQ","IWM"],
        "5_broad (SPY/QQQ/IWM/TLT/EFA)": ["SPY","QQQ","IWM","TLT","EFA"],
        "asset_classes (TLT/EFA/EEM/USO/SLV)": ["TLT","EFA","EEM","USO","SLV"],
        "intl_only (EFA/EEM)": ["EFA","EEM"],
        "stock_etfs (SPY/QQQ/IWM/EFA/EEM)": ["SPY","QQQ","IWM","EFA","EEM"],
        "sectors_plus_bonds (9sectors + TLT)": ["XLE","XLF","XLK","XLU","XLV","XLP","XLY","XLI","XLB","TLT"],
        "minimal (SPY/TLT)": ["SPY","TLT"],
    }
    for label, assets in universes.items():
        # adjust top_n if universe is too small
        avail = [a for a in assets if a in daily.columns]
        tn = min(2, len(avail))
        sim = run_etf_only(daily, mret, list(asofs), assets=assets, top_n=tn)
        m = metrics(sim["log"], spy)
        print(f"  {label:<46}  n={len(avail):>2}  top-{tn}  CAGR {m['cagr']:>6.2f}%  edge {m['edge']:>+6.2f}pp  Sharpe {m['sharpe']:>4.2f}  MDD {m['mdd']:>6.1f}%")
        rows.append({"universe": label, "n_assets": len(avail), "top_n": tn, **m})
    pd.DataFrame(rows).to_csv(RES / "etf_only_universes.csv", index=False)
    return rows


def test_decades(daily, mret, spy):
    print(f"\n{'='*70}\n  6. Decade-by-decade\n{'='*70}")
    rows = []
    decades = [
        ("2003-2009 (GFC era)",       "2003-01-31", "2009-12-31"),
        ("2010-2019 (post-GFC bull)", "2010-01-31", "2019-12-31"),
        ("2020-2026 (COVID+AI)",      "2020-01-31", "2026-04-30"),
        ("2013-2017 (mid bull)",      "2013-01-31", "2017-12-31"),
        ("2018-2022 (vol regime)",    "2018-01-31", "2022-12-31"),
    ]
    print(f"  {'Period':<28}  {'CAGR':>7}  {'edge':>9}  {'Sharpe':>7}  {'MDD':>7}")
    for name, lo, hi in decades:
        asofs = pd.date_range(start=lo, end=hi, freq="ME")
        sim = run_etf_only(daily, mret, list(asofs))
        m = metrics(sim["log"], spy)
        print(f"  {name:<28}  {m['cagr']:>6.2f}%  {m['edge']:>+7.2f}pp  {m['sharpe']:>7.2f}  {m['mdd']:>6.1f}%")
        rows.append({"period": name, "from": lo, "to": hi, **m})
    pd.DataFrame(rows).to_csv(RES / "etf_only_decades.csv", index=False)
    return rows


def test_bootstrap(daily, mret, spy):
    print(f"\n{'='*70}\n  7. Bootstrap distribution\n{'='*70}")
    asofs = pd.date_range(start="2003-01-31", end="2026-04-30", freq="ME")
    sim = run_etf_only(daily, mret, list(asofs))
    df = pd.DataFrame(sim["log"]); df["date"] = pd.to_datetime(df["date"])
    rets = df["ret_m"].values
    spy_w = spy.reindex(df["date"]).fillna(0).values
    edge = rets - spy_w
    rng = np.random.RandomState(42)
    n = len(rets); n_iter = 5000; block = 3
    def bs(s):
        out = []
        for _ in range(n_iter):
            idx = []
            while len(idx) < 12:
                st = rng.randint(0, n - block)
                idx.extend(range(st, min(st+block, n)))
            idx = idx[:12]
            out.append((1 + s[idx]).prod() - 1)
        return np.array(out) * 100
    strat_sims = bs(rets); edge_sims = bs(edge)
    pcts = [5, 10, 25, 50, 75, 90, 95]
    print(f"  P(strat > 0):     {(strat_sims > 0).mean()*100:>5.1f}%")
    print(f"  P(beat SPY):       {(edge_sims > 0).mean()*100:>5.1f}%")
    print(f"  P(beat SPY +5pp):  {(edge_sims > 5).mean()*100:>5.1f}%")
    print(f"  P(lag SPY -10pp):  {(edge_sims < -10).mean()*100:>5.1f}%")
    print(f"  Edge percentiles: " + " | ".join([f"p{p}={np.percentile(edge_sims, p):+5.1f}pp" for p in pcts]))
    print(f"  Mean edge {edge_sims.mean():+.2f}pp, std {edge_sims.std():.2f}pp")


def main():
    RES.mkdir(parents=True, exist_ok=True)
    data = load_all()
    spy = data.mret["SPY"].copy(); spy.index = pd.to_datetime(spy.index)
    mret = data.mret
    daily = pd.read_parquet(Path("experiments/monthly_dca/cache/prices_extended.parquet"))

    # Full-window standalone reference
    print(f"\n{'='*70}\n  REFERENCE: ETF-only full window 2003-2026\n{'='*70}")
    asofs = pd.date_range(start="2003-01-31", end="2026-04-30", freq="ME")
    sim = run_etf_only(daily, mret, list(asofs))
    m = metrics(sim["log"], spy)
    print(f"  CAGR {m['cagr']:.2f}% (SPY {m['spy_cagr']:.2f}% → edge {m['edge']:+.2f}pp)")
    print(f"  Sharpe {m['sharpe']:.2f}  MaxDD {m['mdd']:.1f}%  year-edge std {m['edge_std']:.1f}pp")
    print(f"  Years positive vs SPY: {m['n_years_positive']}/{m['n_years_total']}")
    print(f"  Rotations: {sim['n_rotations']} (avg {sim['n_rotations']/len(asofs)*12:.1f}/yr)")
    pd.DataFrame(sim["log"]).to_csv(RES / "etf_only_full_window_equity.csv", index=False)

    test_walk_forward(daily, mret, spy)
    test_top_n(daily, mret, spy)
    test_lookback(daily, mret, spy)
    test_cost(daily, mret, spy)
    test_universes(daily, mret, spy)
    test_decades(daily, mret, spy)
    test_bootstrap(daily, mret, spy)


if __name__ == "__main__":
    main()
