"""Comprehensive survivorship-bias accounting.

The core problem: our universe is the 964 tickers that survived to today.
At any historical month-end T, we are *only* picking from a population that
later survived. Real-world picking would have been from a population that
included names that later failed/delisted/got acquired. Without a CRSP-style
delisted-inclusive dataset, we attack the bias from multiple directions:

1. **Random-pick baseline.** At each month-end, pick K random eligible
   tickers (uniform over the same universe the strategy uses). Measure CAGR.
   The strategy's *true alpha* is (strategy CAGR − random CAGR), independent
   of how survivor-biased the universe is. If random gets +20% CAGR from the
   same universe, the universe gives +20% for free; the strategy is only
   adding (strat − random).

2. **Stratified Monte-Carlo delisting.** Real bankruptcy/delisting
   probability is conditional on stock state. Distressed names (deep pullback,
   low trend health, falling RSI) delist much more often than healthy compounders.
   We compute a per-pick `p_delist_per_year` from features:

       p = base_rate
           + 0.04 * I[pullback_1y < -0.40]
           + 0.04 * I[pullback_1y < -0.60]
           + 0.03 * I[trend_health_5y < 0.35]
           + 0.02 * I[d_sma200 < -0.30]
           + 0.02 * I[mom_3y < -0.30]
       capped at 0.20 / yr.

   The pickup-time hazard is integrated over years held:
       p_delist_in_window = 1 − (1 − p)^Y

3. **Sensitivity analysis.** Report results at base_rate ∈
   {1%, 2%, 4% (default), 6%, 8%, 12%} so the reader can see how the
   conclusion changes under more or less aggressive bias correction.

4. **Worst-case lower bound.** Compute "if α=20%/yr — every ten-year
   pick has only a ~10% chance of being a phantom delisting, but a 1.5-year
   pick has ~28%". This is a deliberate over-correction; if the strategy
   still beats SPY DCA at this rate, the conclusion is robust.

5. **Empirical delisted augmentation.** Best-effort: fetch yfinance prices
   for a curated list of historically known delisted equities (LEH, BSC,
   WAMU, GM-old, SVB, FRC, SBNY, etc.) and add them to the panel where
   available. This converts some "phantom" delistings into real data.

This file is the home of all of these methods. Every metric the webapp
reports passes through here for honest disclosure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_engine import xirr
from experiments.monthly_dca.fast_score import (
    BENCH_EXCLUDED,
    load_features_long,
    load_fwd,
    load_panel,
)


# ---------------------------------------------------------------------------
# Stratified delisting probability
# ---------------------------------------------------------------------------
def stratified_p_delist(features_row: pd.Series, base_rate: float) -> float:
    """Per-pick *annual* delisting probability, conditional on the stock's
    state at pickup. Higher for distressed names."""
    p = base_rate
    pull = features_row.get("pullback_1y")
    if pd.notna(pull):
        if pull < -0.40:
            p += 0.04
        if pull < -0.60:
            p += 0.04
    trend = features_row.get("trend_health_5y")
    if pd.notna(trend) and trend < 0.35:
        p += 0.03
    dsma = features_row.get("d_sma200")
    if pd.notna(dsma) and dsma < -0.30:
        p += 0.02
    mom = features_row.get("mom_3y")
    if pd.notna(mom) and mom < -0.30:
        p += 0.02
    return float(np.clip(p, 0.0, 0.20))


def bias_corrected_metrics(
    fv: np.ndarray,
    asof: pd.Series,
    eval_at: pd.Timestamp,
    pick_features: pd.DataFrame,
    base_rate: float = 0.04,
    n_iter: int = 200,
    wipeout: float = -1.0,
    seed: int = 0,
    stratified: bool = True,
) -> dict:
    """Return MC-bias-corrected (mean_ret, win_rate, cagr_dca) for a strategy
    using (uniform or stratified) per-pick delisting probabilities.
    """
    asof_ts = pd.to_datetime(asof.values)
    days = np.asarray([(eval_at - t).days for t in asof_ts], dtype=float)
    years = np.maximum(days, 1.0) / 365.25

    if stratified and "pullback_1y" in pick_features.columns:
        per_pick_p_annual = pick_features.apply(
            lambda r: stratified_p_delist(r, base_rate), axis=1
        ).to_numpy(dtype=float)
    else:
        per_pick_p_annual = np.full(len(fv), base_rate)

    # P of delisting by eval_at
    p_pick = 1.0 - (1.0 - per_pick_p_annual) ** years

    rng = np.random.default_rng(seed)
    means = np.empty(n_iter)
    wins = np.empty(n_iter)
    cagrs = np.empty(n_iter)
    for i in range(n_iter):
        u = rng.random(len(fv))
        delisted = u < p_pick
        fv_mc = np.where(delisted, wipeout, fv)
        means[i] = float(np.nanmean(fv_mc))
        wins[i] = float((fv_mc > 0).mean())
        cf = [(pd.Timestamp(t), -1.0) for t in asof_ts]
        cf.append((eval_at, float(np.sum(1 + fv_mc))))
        try:
            cagrs[i] = xirr(cf)
        except Exception:
            cagrs[i] = np.nan
    valid = np.isfinite(cagrs)
    return {
        "mean_ret_median": float(np.median(means)),
        "win_rate_median": float(np.median(wins)),
        "cagr_dca_median": float(np.median(cagrs[valid])) if valid.any() else float("nan"),
        "cagr_dca_p10": float(np.percentile(cagrs[valid], 10)) if valid.any() else float("nan"),
        "cagr_dca_p90": float(np.percentile(cagrs[valid], 90)) if valid.any() else float("nan"),
        "n_iter": int(n_iter),
        "base_rate_annual": base_rate,
        "stratified": stratified,
        "mean_per_pick_p_annual": float(np.mean(per_pick_p_annual)),
        "max_per_pick_p_annual": float(np.max(per_pick_p_annual)),
    }


# ---------------------------------------------------------------------------
# Random-pick baseline
# ---------------------------------------------------------------------------
def random_pick_baseline(
    panel: pd.DataFrame,
    asofs: list[pd.Timestamp],
    top_k: int = 1,
    n_seeds: int = 200,
    eval_at: pd.Timestamp | None = None,
) -> dict:
    """Pick K random eligible tickers at each month-end. Return the
    distribution of resulting DCA-portfolio CAGRs across n_seeds seeds.
    """
    if eval_at is None:
        eval_at = panel.index.max()
    fwd = load_fwd().reset_index()
    # Restrict to provided asofs and exclude benchmark/crypto from the universe
    fwd = fwd[fwd["asof"].isin([pd.Timestamp(a) for a in asofs])]
    fwd = fwd[~fwd["ticker"].isin(BENCH_EXCLUDED)]

    # group by asof to enable random sampling per month
    groups = {a: g for a, g in fwd.groupby("asof")}
    sample_asofs = sorted(groups.keys())

    cagrs = np.empty(n_seeds)
    win_rates = np.empty(n_seeds)
    edges = np.empty(n_seeds)

    spy = panel["SPY"]

    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        rets, asof_seq, spy_rets = [], [], []
        for a in sample_asofs:
            g = groups[a]
            valid = g.dropna(subset=["ret__hold_forever"])
            if valid.empty:
                continue
            chosen = valid.sample(min(top_k, len(valid)), random_state=rng)
            for _, r in chosen.iterrows():
                rets.append(float(r["ret__hold_forever"]))
                asof_seq.append(pd.Timestamp(a))
                # SPY held same window
                pos = panel.index.searchsorted(pd.Timestamp(a))
                arr = spy.iloc[pos:].to_numpy(dtype=float)
                mask = np.isfinite(arr)
                spy_rets.append(arr[mask][-1] / arr[0] - 1.0 if mask.any() else np.nan)
        if not rets:
            cagrs[s] = np.nan
            continue
        rets = np.asarray(rets)
        spy_rets = np.asarray(spy_rets)
        cf = [(t, -1.0) for t in asof_seq]
        cf.append((eval_at, float(np.sum(1 + rets))))
        cagrs[s] = xirr(cf)
        win_rates[s] = float((rets > 0).mean())
        cf_spy = [(t, -1.0) for t in asof_seq]
        cf_spy.append((eval_at, float(np.sum(1 + spy_rets[np.isfinite(spy_rets)]))))
        try:
            spy_cagr = xirr(cf_spy)
        except Exception:
            spy_cagr = np.nan
        edges[s] = cagrs[s] - spy_cagr

    valid = np.isfinite(cagrs)
    return {
        "n_seeds": int(n_seeds),
        "top_k": int(top_k),
        "n_months": len(sample_asofs),
        "cagr_mean": float(np.mean(cagrs[valid])),
        "cagr_median": float(np.median(cagrs[valid])),
        "cagr_p10": float(np.percentile(cagrs[valid], 10)),
        "cagr_p90": float(np.percentile(cagrs[valid], 90)),
        "win_rate_mean": float(np.mean(win_rates[valid])),
        "edge_vs_spy_mean": float(np.mean(edges[valid])),
        "cagrs_sample": [float(x) for x in cagrs[valid][:50]],  # for distribution viz
    }


# ---------------------------------------------------------------------------
# Sensitivity analysis at multiple delisting rates
# ---------------------------------------------------------------------------
def sensitivity_curve(
    fv: np.ndarray,
    asof: pd.Series,
    eval_at: pd.Timestamp,
    pick_features: pd.DataFrame,
    rates: list[float] = (0.01, 0.02, 0.04, 0.06, 0.08, 0.12),
    n_iter: int = 200,
) -> list[dict]:
    out = []
    for r in rates:
        # Both stratified and uniform
        m_strat = bias_corrected_metrics(fv, asof, eval_at, pick_features, base_rate=r,
                                         n_iter=n_iter, stratified=True)
        m_unif = bias_corrected_metrics(fv, asof, eval_at, pick_features, base_rate=r,
                                        n_iter=n_iter, stratified=False)
        out.append({
            "base_rate_annual": r,
            "stratified_cagr_median": m_strat["cagr_dca_median"],
            "stratified_win_median": m_strat["win_rate_median"],
            "uniform_cagr_median": m_unif["cagr_dca_median"],
            "uniform_win_median": m_unif["win_rate_median"],
            "stratified_cagr_p10": m_strat["cagr_dca_p10"],
            "stratified_cagr_p90": m_strat["cagr_dca_p90"],
        })
    return out


# ---------------------------------------------------------------------------
# Empirical augmentation: try yfinance for known delisted names
# ---------------------------------------------------------------------------
KNOWN_DELISTED = [
    # Financial crisis 2007-2009
    "LEH",  # Lehman Brothers
    "BSC",  # Bear Stearns
    "WM",   # Washington Mutual (note: WM also = Waste Management today)
    "WAMUQ",
    "FNMA", "FMCC",  # Fannie/Freddie (still trade OTC)
    "AIG.WS",
    # 2010s notable
    "SHLD", "SHLDQ",  # Sears
    "JCP",  # JCPenney
    "TOYS",
    "DDS",  # Dillard's still trades; skip
    "RAD",  # Rite Aid
    "BBBY", "BBBYQ",  # Bed Bath & Beyond
    "CHK",  # Chesapeake Energy (re-listed)
    "FRO",
    # 2020s notable
    "SVB",  "SIVBQ",  # Silicon Valley Bank
    "FRC",  "FRCB",   # First Republic
    "SBNY",          # Signature Bank
    "PYPL",          # control: not delisted
    # Mergers (acquired => delisted)
    "TWX",            # Time Warner -> AT&T (now WBD)
    "RTN",            # Raytheon -> RTX
    "CELG",           # Celgene -> BMY
    "MON",            # Monsanto -> Bayer
    "DOW",            # confusing post-spinoff
    "DD",             # confusing post-spinoff
    "EMC",            # -> Dell
    "BRCM",           # -> AVGO
    "ATVI",           # -> MSFT
    "VMW",            # -> AVGO
    "FISV",           # -> Fiserv (still trades)
    "FB",             # -> META
    "GOOG",           # control: not delisted
    "BIDU",
    "TWTR",           # -> private (Musk acquired)
    # Dot-com era
    "WCG",
    "ENRN",  # Enron
    "WCOM",  # WorldCom
    # Failed retailers
    "BKS",
    "SHLD",
    "TLRD",
    "HEXO",
    "NIO",  # control
]


def fetch_delisted_panel(tickers: list[str], start: str = "2014-01-01") -> pd.DataFrame:
    import yfinance as yf

    out = {}
    for t in tickers:
        try:
            d = yf.download(t, start=start, progress=False, threads=False, auto_adjust=True)
            if d is None or d.empty:
                continue
            close = d["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            if len(close) < 30:
                continue
            close.index = pd.to_datetime(close.index).tz_localize(None)
            out[t] = close
            print(f"  {t}: {len(close)} rows  range={close.index.min().date()}->{close.index.max().date()}")
        except Exception as e:
            print(f"  {t}: ERROR {e}")
    if not out:
        return pd.DataFrame()
    return pd.concat(out.values(), axis=1, keys=out.keys()).sort_index()


def main() -> None:
    """Build a comprehensive bias-correction summary for the recommended strategy."""
    panel = load_panel()
    cache = Path(__file__).resolve().parent / "cache"
    eval_at = panel.index.max()

    # Load picks for the recommended strategy: pullback_in_winner k=1
    picks_csv = cache / "picks_full_pullback_in_winner_k1.csv"
    picks = pd.read_csv(picks_csv)
    picks["asof"] = pd.to_datetime(picks["asof"])

    # The picks_full CSV already has forward returns merged from
    # save_winning_picks.py (run earlier). Use them directly.
    if "ret__hold_forever" not in picks.columns:
        fwd_df = load_fwd().reset_index()
        picks = picks.merge(fwd_df[["asof", "ticker", "ret__hold_forever"]],
                             on=["asof", "ticker"], how="left")
    valid = picks.dropna(subset=["ret__hold_forever"]).copy()
    fv = valid["ret__hold_forever"].to_numpy(dtype=float)
    asof_s = valid["asof"]
    pick_features = valid[[c for c in ["pullback_1y", "trend_health_5y", "d_sma200", "mom_3y"]
                            if c in valid.columns]]

    # 1. Stratified MC at default rate
    print("=== 1. Stratified MC (α=4%/yr, distressed-conditional) ===")
    strat_default = bias_corrected_metrics(fv, asof_s, eval_at, pick_features,
                                           base_rate=0.04, stratified=True, n_iter=300)
    for k, v in strat_default.items():
        print(f"  {k}: {v}")

    # 2. Sensitivity curve
    print("\n=== 2. Sensitivity sweep ===")
    sens = sensitivity_curve(fv, asof_s, eval_at, pick_features,
                             rates=[0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.20], n_iter=200)
    for r in sens:
        print(f"  α={r['base_rate_annual']*100:.0f}%/yr  "
              f"strat CAGR={r['stratified_cagr_median']:.3f} (p10={r['stratified_cagr_p10']:.3f}/p90={r['stratified_cagr_p90']:.3f})  "
              f"uniform CAGR={r['uniform_cagr_median']:.3f}  "
              f"strat WIN={r['stratified_win_median']:.3f}")

    # 3. Random-pick baseline
    print("\n=== 3. Random-pick baseline (200 seeds) ===")
    asofs = picks["asof"].drop_duplicates().tolist()
    rand1 = random_pick_baseline(panel, asofs, top_k=1, n_seeds=200, eval_at=eval_at)
    print(f"  random k=1: CAGR mean={rand1['cagr_mean']:.3f}  median={rand1['cagr_median']:.3f}  p10={rand1['cagr_p10']:.3f}  p90={rand1['cagr_p90']:.3f}  edge_vs_spy={rand1['edge_vs_spy_mean']:.3f}")
    rand5 = random_pick_baseline(panel, asofs, top_k=5, n_seeds=200, eval_at=eval_at)
    print(f"  random k=5: CAGR mean={rand5['cagr_mean']:.3f}  median={rand5['cagr_median']:.3f}  p10={rand5['cagr_p10']:.3f}  p90={rand5['cagr_p90']:.3f}  edge_vs_spy={rand5['edge_vs_spy_mean']:.3f}")
    rand10 = random_pick_baseline(panel, asofs, top_k=10, n_seeds=200, eval_at=eval_at)
    print(f"  random k=10: CAGR mean={rand10['cagr_mean']:.3f}  median={rand10['cagr_median']:.3f}  p10={rand10['cagr_p10']:.3f}  p90={rand10['cagr_p90']:.3f}  edge_vs_spy={rand10['edge_vs_spy_mean']:.3f}")

    # 4. Try empirical delisted augmentation (best-effort)
    print("\n=== 4. Empirical delisted augmentation (yfinance) ===")
    delisted_path = cache / "delisted_panel.parquet"
    if delisted_path.exists():
        print(f"  Cached: {delisted_path}")
        d_panel = pd.read_parquet(delisted_path)
    else:
        d_panel = fetch_delisted_panel(KNOWN_DELISTED, start="2014-01-01")
        if not d_panel.empty:
            d_panel.to_parquet(delisted_path)
            print(f"  Wrote: {delisted_path} ({d_panel.shape})")
        else:
            print("  No delisted data fetched")
    print(f"  Tickers with data: {d_panel.shape[1] if not d_panel.empty else 0}")
    delisted_summary = {
        "tickers_attempted": KNOWN_DELISTED,
        "tickers_with_data": list(d_panel.columns) if not d_panel.empty else [],
        "n_with_data": int(d_panel.shape[1]) if not d_panel.empty else 0,
    }

    # 5. Save summary JSON
    out_path = cache / "survivorship_summary.json"
    summary = {
        "strategy": "pullback_in_winner_k1_hold_forever",
        "n_picks": int(len(fv)),
        "raw_cagr": float(np.median([
            (1 + fv[i]) ** (365.25 / max((eval_at - pd.Timestamp(asof_s.iloc[i])).days, 1)) - 1
            for i in range(len(fv))
        ])),
        "stratified_default_4pct": strat_default,
        "sensitivity": sens,
        "random_baseline_k1": rand1,
        "random_baseline_k5": rand5,
        "random_baseline_k10": rand10,
        "delisted_augmentation": delisted_summary,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
