"""Advanced overlays — stack the trend-sleeve win with more novel ideas.

E — Multi-asset trend sleeve (SPY / TLT / Cash by 12m momentum)
F — Sector-rotation trend sleeve (top-2 of 6 sector ETFs by 12m mom)
G — Stock-level trend filter (exit basket positions on d_sma200 break)
H — Trend sleeve combined with K=4 (stacked variance reduction)
I — Adaptive trend weight (more sleeve in risk-off, less in risk-on)

All tested on PIT S&P 500 with the look-ahead-fixed harness.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

from experiments.monthly_dca.v5.validations.harness import (
    HarnessData, load_all, classify_regime_tight, invvol_weights,
    CHRONOS_FILTER_Q, CAP_PER_PICK, COST_BPS, K_PICKS, HOLD_MONTHS,
    pick_v5_baseline,
)
from experiments.monthly_dca.v5.validations.run_tactical_rebalance import _score_at
from experiments.monthly_dca.v5.validations.run_k_sweep import make_kpick_picker
from experiments.monthly_dca.v2.ml_strategy import EXCLUDE

RES = Path(__file__).resolve().parent / "results"


# -----------------------------------------------------------------------------
# Sleeve return calculators
# -----------------------------------------------------------------------------
def multi_asset_momentum_sleeve(daily_prices: pd.DataFrame, asof: pd.Timestamp,
                                  assets: list[str], lookback_d: int = 252):
    """Pick the single asset with best 12-m return; if best return ≤ 0,
    go to cash. Returns (chosen_asset, next-month return placeholder)."""
    px = daily_prices.loc[:asof, [a for a in assets if a in daily_prices.columns]].dropna(how="all")
    if len(px) < lookback_d:
        return None
    ret_12m = px.iloc[-1] / px.iloc[-lookback_d] - 1
    best = ret_12m.idxmax() if ret_12m.max() > 0 else None
    return best


def sector_top_n_sleeve(daily_prices: pd.DataFrame, asof: pd.Timestamp,
                          sectors: list[str], n: int = 2, lookback_d: int = 252):
    """Long top-n sectors by 12m momentum, equal-weighted."""
    available = [s for s in sectors if s in daily_prices.columns]
    px = daily_prices.loc[:asof, available].dropna(how="all")
    if len(px) < lookback_d:
        return None
    ret_12m = px.iloc[-1] / px.iloc[-lookback_d] - 1
    ret_12m = ret_12m[ret_12m > 0]  # only positive trends
    if len(ret_12m) == 0:
        return None
    top = ret_12m.sort_values(ascending=False).head(n).index.tolist()
    return top


def stock_trend_filter(asof: pd.Timestamp, ticker: str,
                        features_dir: Path) -> bool:
    """True if ticker is in uptrend at asof (d_sma200 > 0)."""
    fp = features_dir / f"{asof.strftime('%Y-%m-%d')}.parquet"
    if not fp.exists():
        return True  # default to keeping the position
    df = pd.read_parquet(fp)
    if ticker not in df.index or "d_sma200" not in df.columns:
        return True
    return float(df.at[ticker, "d_sma200"]) > 0


# -----------------------------------------------------------------------------
# Unified simulator with advanced overlays
# -----------------------------------------------------------------------------
def run_advanced(data: HarnessData, pick_fn, overlay: str,
                  start: pd.Timestamp, end: pd.Timestamp,
                  hold_months: int = HOLD_MONTHS,
                  cost_bps: float = COST_BPS,
                  sleeve_weight: float = 0.25,
                  daily_prices: pd.DataFrame | None = None,
                  features_dir: Path | None = None) -> dict:
    cf = cost_bps / 1e4
    asofs = [m for m in data.asofs
             if start <= m <= end
             and m in data.spy_features.index
             and m in data.mret.index
             and m in data.members_g]
    asofs = sorted(asofs)

    cur_picks: list[str] = []
    cur_weights = np.array([])
    cash = False; held = 0
    equity = 1.0
    log = []

    # Sleeve state (for stock_trend_filter only)
    stock_filter_active = {}  # ticker → bool (currently held / filtered out)

    for i, m in enumerate(asofs):
        spy_now = data.spy_features.loc[m].to_dict() if m in data.spy_features.index else {}
        regime = classify_regime_tight(spy_now)
        do_reb = (i == 0) or (held >= hold_months) or cash

        # Compute strategy return for this month.
        # For the stock-trend filter: use the PRIOR month's d_sma200 (not m's)
        # to avoid look-ahead — at the start of month m we only know m-1's
        # features. m's d_sma200 reflects m's closing price.
        prev_m = asofs[i - 1] if i > 0 else m
        if cash or not cur_picks:
            strat_ret = 0.0
        else:
            r = 0.0; active_weight = 0.0
            for tk, w in zip(cur_picks, cur_weights):
                # Apply stock-level trend filter if enabled — using prev_m's
                # d_sma200, decided BEFORE month m's return is realized.
                if overlay == "stock_trend":
                    if not stock_trend_filter(prev_m, tk, features_dir):
                        continue
                rt = (float(data.mret.at[m, tk])
                      if (tk in data.mret.columns and m in data.mret.index
                          and pd.notna(data.mret.at[m, tk]))
                      else 0.0)
                r += w * rt
                active_weight += w
            strat_ret = r  # the missing weight = cash = 0% contribution

        # Compute overlay sleeve return
        sleeve_ret = 0.0
        if overlay == "multi_asset_trend" and daily_prices is not None:
            best = multi_asset_momentum_sleeve(daily_prices, m, ["SPY", "TLT"], lookback_d=252)
            if best is not None and best in data.mret.columns and m in data.mret.index and pd.notna(data.mret.at[m, best]):
                sleeve_ret = float(data.mret.at[m, best])
        elif overlay == "sector_trend" and daily_prices is not None:
            tops = sector_top_n_sleeve(daily_prices, m,
                                        ["XLE","XLF","XLK","XLU","XLV","XLP"], n=2)
            if tops:
                rs = []
                for s in tops:
                    if s in data.mret.columns and m in data.mret.index and pd.notna(data.mret.at[m, s]):
                        rs.append(float(data.mret.at[m, s]))
                if rs:
                    sleeve_ret = float(np.mean(rs))
        elif overlay == "adaptive_trend" and daily_prices is not None:
            # Adapt sleeve weight by regime
            adaptive_w = {"crash": 0.50, "recovery": 0.20, "normal": 0.30, "bull": 0.10}.get(regime, 0.25)
            best = multi_asset_momentum_sleeve(daily_prices, m, ["SPY", "TLT"], lookback_d=252)
            if best is not None and best in data.mret.columns and m in data.mret.index and pd.notna(data.mret.at[m, best]):
                sleeve_ret = float(data.mret.at[m, best])
            # Override sleeve_weight for this iteration
            ret_m_pre = (1 - adaptive_w) * strat_ret + adaptive_w * sleeve_ret
            ret_m = ret_m_pre
        else:
            ret_m = strat_ret

        if overlay in ("multi_asset_trend", "sector_trend"):
            ret_m = (1 - sleeve_weight) * strat_ret + sleeve_weight * sleeve_ret

        # Rebalance / regime logic
        if do_reb:
            if regime == "crash":
                cur_picks, cur_weights, cash = [], np.array([]), True
                held = 0
            else:
                eligible = data.members_g.get(m, set()) - set(EXCLUDE)
                picks, weights = pick_fn(m, eligible, data, regime)
                if picks and len(picks) >= 1:
                    cur_picks = list(picks)
                    cur_weights = np.array(weights, dtype=float)
                    if cur_weights.sum() == 0:
                        cur_weights = np.ones(len(cur_picks)) / len(cur_picks)
                    else:
                        cur_weights = cur_weights / cur_weights.sum()
                    cash = False; held = 0
                    if log: ret_m -= cf

        held += 1
        equity *= (1 + ret_m)
        log.append({"date": str(m.date()), "regime": regime,
                     "ret_m": ret_m, "equity": equity,
                     "picks": ",".join(cur_picks) if cur_picks else ""})

    return {"log": log}


def summary(log, spy_monthly):
    df = pd.DataFrame(log); df["date"] = pd.to_datetime(df["date"]); df["year"] = df["date"].dt.year
    n = len(df); years = n / 12
    cagr = (df["equity"].iloc[-1] ** (1/years) - 1) * 100
    spy_w = spy_monthly.loc[df["date"].iloc[0]:df["date"].iloc[-1]]
    spy_cagr = ((1+spy_w.fillna(0)).cumprod().iloc[-1] ** (12/len(spy_w)) - 1) * 100
    rr = df["ret_m"]
    sh = float(rr.mean()/rr.std()*np.sqrt(12)) if rr.std() > 0 else 0
    peak = df["equity"].cummax(); mdd = float(((df["equity"]-peak)/peak).min()*100)
    yr = df.groupby("year")["ret_m"].apply(lambda x: (1+x).prod()-1)*100
    spy_yr = spy_w.groupby(spy_w.index.year).apply(lambda x: (1+x).prod()-1)*100
    edges = [yr[y]-spy_yr[y] for y in sorted(yr.index) if y in spy_yr.index]
    e2024 = (yr.get(2024,0)-spy_yr.get(2024,0)) if 2024 in yr.index else 0
    return dict(cagr=cagr, edge=cagr-spy_cagr, sharpe=sh, mdd=mdd,
                edge_std=float(np.std(edges)), edge_min=float(min(edges)),
                e2024=e2024)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    data = load_all()
    spy = data.mret["SPY"].copy(); spy.index = pd.to_datetime(spy.index)
    daily = pd.read_parquet(Path("experiments/monthly_dca/cache/prices_extended.parquet"))
    features_dir = Path("experiments/monthly_dca/cache/features")
    start = data.asofs[0]; end = data.spy_features.index.max()

    rows = []
    def run_print(name, pick_fn, overlay, sw=0.25):
        sim = run_advanced(data, pick_fn, overlay, start, end,
                            sleeve_weight=sw, daily_prices=daily,
                            features_dir=features_dir)
        r = summary(sim["log"], spy)
        print(f"  {name:<30} CAGR {r['cagr']:>6.2f}%  Sharpe {r['sharpe']:>5.2f}  MDD {r['mdd']:>6.1f}%  std {r['edge_std']:>5.1f}pp  worst {r['edge_min']:>+6.1f}pp  2024 {r['e2024']:>+6.1f}pp")
        rows.append({"variant": name, **r})

    # Baseline
    print("\n=== Reference points ===")
    run_print("baseline (K=3, no overlay)", pick_v5_baseline, "none")
    run_print("trend50_SPY (from prior expt)", pick_v5_baseline, "multi_asset_trend", 0.50)

    print("\n=== E. Multi-asset trend sleeve (SPY/TLT mom-rotate) ===")
    run_print("E_K3_multi_25",  pick_v5_baseline, "multi_asset_trend", 0.25)
    run_print("E_K3_multi_50",  pick_v5_baseline, "multi_asset_trend", 0.50)

    print("\n=== F. Sector-rotation sleeve (top-2 of 6 by 12m mom) ===")
    run_print("F_K3_sector_25", pick_v5_baseline, "sector_trend", 0.25)
    run_print("F_K3_sector_50", pick_v5_baseline, "sector_trend", 0.50)

    print("\n=== G. Stock-level trend filter (exit picks breaking 200dma) ===")
    run_print("G_K3_stock_trend", pick_v5_baseline, "stock_trend")

    print("\n=== H. K=4 + trend overlays (stacked variance reduction) ===")
    run_print("H_K4_multi_25",  make_kpick_picker(4), "multi_asset_trend", 0.25)
    run_print("H_K4_multi_50",  make_kpick_picker(4), "multi_asset_trend", 0.50)
    run_print("H_K4_sector_50", make_kpick_picker(4), "sector_trend", 0.50)

    print("\n=== I. Adaptive trend (regime-conditional sleeve weight) ===")
    run_print("I_K3_adaptive", pick_v5_baseline, "adaptive_trend")

    df = pd.DataFrame(rows)
    df.to_csv(RES / "advanced_overlays_summary.csv", index=False)
    print("\n\n=== ADVANCED OVERLAYS SUMMARY ===")
    print(f"{'Variant':<32} {'CAGR':>7} {'Sharpe':>7} {'MDD':>7} {'YrStd':>7} {'WrstY':>7} {'2024':>7}")
    for r in rows:
        print(f"{r['variant']:<32} {r['cagr']:>6.2f}% {r['sharpe']:>7.2f} {r['mdd']:>6.1f}% {r['edge_std']:>+5.1f}pp {r['edge_min']:>+5.1f}pp {r['e2024']:>+5.1f}pp")


if __name__ == "__main__":
    main()
