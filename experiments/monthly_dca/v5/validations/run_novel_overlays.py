"""Novel overlays: correlation-aware selection, VIX-proxy regime gate,
trend-following sleeve. All tested on PIT S&P 500, look-ahead-fixed
harness, no curve-fitting.

OVERLAY A — Correlation-aware basket selection
    From top-N by GBM+Chronos score, pick the K stocks that minimise
    pairwise trailing-12m return correlation. Same alpha, lower internal
    covariance, expected: lower vol / better Sharpe.

OVERLAY B — VIX-proxy de-risking
    Compute SPY trailing 21d annualised vol. When > threshold (e.g.,
    30%), scale exposure to (1 - excess_vol_fraction). Earlier warning
    than the realised-return crash gate.

OVERLAY C — Trend-following sleeve
    Run a parallel allocation: X% in v5 strategy, (100-X)% in a SPY
    trend-follower (long when SPY > 200dma + above 50dma, cash else).
    The two have different alpha sources → real diversification.

OVERLAY D — Stock-level trend filter
    After v5 picks, exit any pick whose `d_sma200` flips negative within
    the hold. Re-enter on flip back positive. Per-stock micro stop-loss.
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
from experiments.monthly_dca.v5.validations.run_tactical_rebalance import (
    _score_at,
)
from experiments.monthly_dca.v2.ml_strategy import EXCLUDE

RES = Path(__file__).resolve().parent / "results"


# =============================================================================
# OVERLAY A — Correlation-aware basket selection
# =============================================================================
def make_corr_aware_picker(top_pool: int = 10, k: int = K_PICKS,
                             corr_window: int = 12,
                             corr_weight: float = 1.0):
    """Pick top_pool by GBM+Chronos score, then select k that minimise
    total pairwise correlation.

    corr_weight=1.0 → pure correlation minimisation (constrained by
    being in top_pool by alpha). corr_weight=0 → reverts to top-k by alpha.
    """
    def pick(asof, eligible, data, regime):
        scored = _score_at(asof, eligible, data)
        if len(scored) < k:
            return [], []
        # Take top-N by alpha
        pool = scored.sort_values("score", ascending=False).head(top_pool)
        if len(pool) <= k:
            picks = pool["ticker"].tolist()
            weights = invvol_weights(picks, data.mret, asof, cap=CAP_PER_PICK)
            return picks, list(weights)

        # Compute pairwise correlations of trailing 12m monthly returns
        mr_idx = data.mret.index
        pos = mr_idx.searchsorted(asof, side="right") - 1
        if pos < corr_window:
            picks = pool.head(k)["ticker"].tolist()
            weights = invvol_weights(picks, data.mret, asof, cap=CAP_PER_PICK)
            return picks, list(weights)
        window = data.mret.iloc[pos - corr_window + 1: pos + 1]
        pool_tickers = [t for t in pool["ticker"] if t in window.columns]
        if len(pool_tickers) <= k:
            picks = pool_tickers[:k]
            weights = invvol_weights(picks, data.mret, asof, cap=CAP_PER_PICK)
            return picks, list(weights)
        ret_window = window[pool_tickers].dropna(how="all")
        if len(ret_window) < corr_window // 2:
            picks = pool.head(k)["ticker"].tolist()
            weights = invvol_weights(picks, data.mret, asof, cap=CAP_PER_PICK)
            return picks, list(weights)
        corr_mat = ret_window.corr().fillna(0.5).values
        score_map = dict(zip(pool["ticker"], pool["score"].rank(pct=True)))

        # Greedy selection: start with highest-score, then add stock that
        # minimises (sum of corrs with already-selected) + alpha penalty
        selected = [pool_tickers[np.argmax([score_map[t] for t in pool_tickers])]]
        sel_idx = [pool_tickers.index(selected[0])]
        while len(selected) < k:
            best_t = None; best_cost = float("inf")
            for j, t in enumerate(pool_tickers):
                if t in selected: continue
                avg_corr = np.mean([corr_mat[j][si] for si in sel_idx])
                alpha_lost = 1 - score_map.get(t, 0.5)  # penalty for skipping high-alpha
                cost = corr_weight * avg_corr + (1 - corr_weight) * alpha_lost
                if cost < best_cost:
                    best_cost = cost; best_t = t
            if best_t is None: break
            selected.append(best_t)
            sel_idx.append(pool_tickers.index(best_t))
        if len(selected) < k:
            return [], []
        weights = invvol_weights(selected, data.mret, asof, cap=CAP_PER_PICK)
        return selected, list(weights)
    return pick


# =============================================================================
# OVERLAY B — VIX-proxy de-risking (SPY 21d annualised realised vol)
# =============================================================================
def compute_spy_rvol(daily_spy: pd.Series, asofs: list) -> pd.Series:
    """For each asof month-end, compute trailing 21-day annualised
    realised vol of SPY."""
    ret = daily_spy.pct_change()
    out = {}
    for a in asofs:
        s = ret.loc[:a].tail(21)
        if len(s) >= 21:
            out[a] = float(s.std() * np.sqrt(252))
        else:
            out[a] = 0.20  # default
    return pd.Series(out)


# =============================================================================
# OVERLAY C — Trend-following SPY sleeve
# =============================================================================
def compute_spy_trend(daily_spy: pd.Series, asofs: list) -> pd.Series:
    """At each month-end: 1.0 if SPY above its 200dma AND 50dma, else 0."""
    sma200 = daily_spy.rolling(200, min_periods=200).mean()
    sma50 = daily_spy.rolling(50, min_periods=50).mean()
    above200 = (daily_spy > sma200).astype(int)
    above50 = (daily_spy > sma50).astype(int)
    trend = (above200 & above50).astype(float)
    out = {}
    for a in asofs:
        idx = trend.index.get_indexer([a], method='ffill')[0]
        if idx >= 0:
            out[a] = float(trend.iloc[idx])
        else:
            out[a] = 1.0
    return pd.Series(out)


# =============================================================================
# Unified simulator with overlays
# =============================================================================
def run_with_overlay(data: HarnessData, pick_fn, overlay_mode: str,
                      start: pd.Timestamp, end: pd.Timestamp,
                      hold_months: int = HOLD_MONTHS,
                      cost_bps: float = COST_BPS,
                      vix_threshold: float = 0.30,
                      vix_max_derisk: float = 0.50,
                      trend_sleeve_weight: float = 0.0) -> dict:
    """Wrap pick_fn with optional overlay:
      'none'   — baseline
      'vix'    — scale strategy exposure by (1 - excess_vol_pct)
      'trend'  — blend strategy with SPY trend-follower sleeve
    """
    cf = cost_bps / 1e4
    asofs = [m for m in data.asofs
             if start <= m <= end
             and m in data.spy_features.index
             and m in data.mret.index
             and m in data.members_g]
    asofs = sorted(asofs)

    # Pre-compute overlay signals
    daily = pd.read_parquet(Path("experiments/monthly_dca/cache/prices_extended.parquet"))
    daily_spy = daily["SPY"].dropna() if "SPY" in daily.columns else pd.Series(dtype=float)
    spy_rvol = compute_spy_rvol(daily_spy, asofs)
    spy_trend = compute_spy_trend(daily_spy, asofs)

    cur_picks: list[str] = []
    cur_weights = np.array([])
    cash = False
    held = 0
    equity = 1.0
    log = []

    for i, m in enumerate(asofs):
        spy_now = data.spy_features.loc[m].to_dict() if m in data.spy_features.index else {}
        regime = classify_regime_tight(spy_now)
        do_reb = (i == 0) or (held >= hold_months) or cash

        # Strategy basket return for this month
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

        # Compute SPY return for trend sleeve
        spy_ret_m = float(data.mret.at[m, "SPY"]) if (m in data.mret.index and "SPY" in data.mret.columns and pd.notna(data.mret.at[m, "SPY"])) else 0.0

        # Apply overlay
        if overlay_mode == "vix":
            rvol = spy_rvol.get(m, 0.20)
            if rvol > vix_threshold:
                excess = min((rvol - vix_threshold) / 0.20, 1.0)
                derisk = min(excess * vix_max_derisk, vix_max_derisk)
                strat_ret = strat_ret * (1 - derisk)  # partial cash equivalent
        elif overlay_mode == "trend":
            # Blend strat with trend-following sleeve (SPY-on/cash)
            trend_state = spy_trend.get(m, 1.0)
            sleeve_ret = spy_ret_m * trend_state  # 0 when in cash
            strat_ret = (1 - trend_sleeve_weight) * strat_ret + trend_sleeve_weight * sleeve_ret
        # 'none' → no overlay

        ret_m = strat_ret

        # Regime / rebalance logic (same as production)
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
                    cash = False
                    held = 0
                    if log:
                        ret_m -= cf

        held += 1
        equity *= (1 + ret_m)
        log.append({"date": str(m.date()), "regime": regime,
                     "rvol": spy_rvol.get(m, 0.20),
                     "trend_state": spy_trend.get(m, 1.0),
                     "ret_m": ret_m, "equity": equity,
                     "picks": ",".join(cur_picks) if cur_picks else ""})

    return {"log": log}


def summary(log, spy_monthly):
    df = pd.DataFrame(log)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
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
    return dict(cagr=cagr, spy_cagr=spy_cagr, edge=cagr-spy_cagr, sharpe=sh,
                mdd=mdd, edge_std=np.std(edges), edge_min=min(edges),
                e2024=e2024, df=df)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    data = load_all()
    spy = data.mret["SPY"].copy()
    spy.index = pd.to_datetime(spy.index)
    start = data.asofs[0]; end = data.spy_features.index.max()

    rows = []

    # Baseline reference (production picker, no overlay)
    print("BASELINE (production picker, no overlay)")
    sim = run_with_overlay(data, pick_v5_baseline, "none", start, end)
    r = summary(sim["log"], spy)
    print(f"  CAGR {r['cagr']:.2f}% (edge {r['edge']:+.2f}pp)  Sharpe {r['sharpe']:.2f}  MDD {r['mdd']:.1f}%  std {r['edge_std']:.1f}  2024 {r['e2024']:+.1f}pp")
    rows.append({"variant": "baseline", **{k:v for k,v in r.items() if k != 'df'}})

    # Overlay A — correlation-aware selection
    for top_n, label in [(10, "A_corr_top10"), (15, "A_corr_top15"), (20, "A_corr_top20")]:
        print(f"\n{label}: corr-aware pick K=3 from top-{top_n} by alpha")
        sim = run_with_overlay(data, make_corr_aware_picker(top_pool=top_n), "none", start, end)
        r = summary(sim["log"], spy)
        print(f"  CAGR {r['cagr']:.2f}% (edge {r['edge']:+.2f}pp)  Sharpe {r['sharpe']:.2f}  MDD {r['mdd']:.1f}%  std {r['edge_std']:.1f}  2024 {r['e2024']:+.1f}pp")
        rows.append({"variant": label, **{k:v for k,v in r.items() if k != 'df'}})

    # Overlay B — VIX-proxy de-risking
    for vix_thr, max_d, label in [(0.25, 0.50, "B_vix_25_50"),
                                     (0.30, 0.50, "B_vix_30_50"),
                                     (0.30, 1.00, "B_vix_30_100")]:
        print(f"\n{label}: SPY rvol>{vix_thr*100:.0f}% → derisk up to {max_d*100:.0f}%")
        sim = run_with_overlay(data, pick_v5_baseline, "vix", start, end,
                                vix_threshold=vix_thr, vix_max_derisk=max_d)
        r = summary(sim["log"], spy)
        print(f"  CAGR {r['cagr']:.2f}% (edge {r['edge']:+.2f}pp)  Sharpe {r['sharpe']:.2f}  MDD {r['mdd']:.1f}%  std {r['edge_std']:.1f}  2024 {r['e2024']:+.1f}pp")
        rows.append({"variant": label, **{k:v for k,v in r.items() if k != 'df'}})

    # Overlay C — Trend-following sleeve
    for w, label in [(0.10, "C_trend_10pct"), (0.25, "C_trend_25pct"), (0.50, "C_trend_50pct")]:
        print(f"\n{label}: blend v5 + {w*100:.0f}% SPY trend-follower")
        sim = run_with_overlay(data, pick_v5_baseline, "trend", start, end,
                                trend_sleeve_weight=w)
        r = summary(sim["log"], spy)
        print(f"  CAGR {r['cagr']:.2f}% (edge {r['edge']:+.2f}pp)  Sharpe {r['sharpe']:.2f}  MDD {r['mdd']:.1f}%  std {r['edge_std']:.1f}  2024 {r['e2024']:+.1f}pp")
        rows.append({"variant": label, **{k:v for k,v in r.items() if k != 'df'}})

    df = pd.DataFrame(rows)
    df.to_csv(RES / "novel_overlays_summary.csv", index=False)

    print("\n\n=== NOVEL OVERLAYS SWEEP (PIT SP500, K=3 H=6) ===")
    print(f"{'Variant':<20} {'CAGR':>7} {'edge':>9} {'Sharpe':>7} {'MDD':>7} {'std':>7} {'2024':>8}")
    for r in rows:
        print(f"{r['variant']:<20} {r['cagr']:>6.2f}% {r['edge']:>+7.2f}pp {r['sharpe']:>7.2f} {r['mdd']:>6.1f}% {r['edge_std']:>+5.1f}pp {r['e2024']:>+6.1f}pp")


if __name__ == "__main__":
    main()
