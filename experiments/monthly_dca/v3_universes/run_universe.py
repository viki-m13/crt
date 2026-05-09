"""
Universe runner: takes a price panel + (optional) point-in-time membership table,
builds the v2-compatible cross-section + features, runs the ML walk-forward, and
reports headline stats.

Saves everything to `cache/v3_universes/<universe_name>/`:
  - monthly_prices_clean.parquet
  - monthly_returns_clean.parquet
  - panel_cross_section_v3.parquet  (features × month × ticker, with PIT filter applied)
  - ml_preds.parquet                (walk-forward predictions)
  - equity_curve.csv                (full-window backtest curve)
  - year_by_year.csv                (calendar year returns)
  - summary.json                    (CAGR, Sharpe, MaxDD, N_picks, etc)

The PIT membership table is a parquet with columns (date, ticker). At each
feature month-end T, only rows with ticker in the membership for T are kept.

Usage (from repo root):
    python3 -m experiments.monthly_dca.v3_universes.run_universe \\
        --name sp500_pit \\
        --prices experiments/monthly_dca/cache/v3_universes/sp500_pit/prices.parquet \\
        --membership experiments/monthly_dca/v3_universes/data/sp500_pit_membership.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.monthly_dca.backtester import compute_features
from experiments.monthly_dca.extra_features import compute_extras
from experiments.monthly_dca.alpha_features import compute_alpha_features
from experiments.monthly_dca.alpha2_features import compute_alpha2
from experiments.monthly_dca.v2.build_dataset import _detect_bad_months
from experiments.monthly_dca.v2.ml_strategy import (
    build_strategy_outputs, simulate_strategy, cagr,
)

UV_CACHE_BASE = ROOT / "experiments" / "monthly_dca" / "cache" / "v3_universes"


def make_clean_monthly(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Resample to month-end, detect data-error months, cap returns."""
    monthly = panel.resample("ME").last()
    mask_bad = _detect_bad_months(monthly)
    monthly_clean = monthly.where(~mask_bad)
    mret_clean = monthly_clean.pct_change().clip(lower=-1.0, upper=2.0)
    bad_list = sorted(mask_bad.any(axis=0)[mask_bad.any(axis=0)].index.tolist())
    return monthly_clean, mret_clean, bad_list


def build_features_per_month(panel: pd.DataFrame, months: list[pd.Timestamp],
                              min_history: int = 504) -> dict[pd.Timestamp, pd.DataFrame]:
    """Compute the 67-column feature pack per month-end."""
    feats: dict[pd.Timestamp, pd.DataFrame] = {}
    n = len(months)
    for i, m in enumerate(months):
        try:
            pack = compute_features(panel, m, min_history=min_history)
        except Exception as e:
            continue
        df = pack.df()
        # extras (mom_3y, sharpe_12m, beta_2y, max_below_200_streak, ...)
        try:
            ex = compute_extras(panel, m)
            df = df.join(ex, how="left")
        except Exception as e:
            print(f"    extras failed at {m.date()}: {e}", flush=True)
        # alpha_features (rs_*_spy, bb_width, trend_slope_252, mom_accel, ...)
        try:
            af = compute_alpha_features(panel, m)
            for c in af.columns:
                if c not in df.columns:
                    df[c] = af[c]
        except Exception as e:
            print(f"    alpha_features failed at {m.date()}: {e}", flush=True)
        # alpha2 (fip_score, idio_mom_12_1, sharpe_5y, ...)
        try:
            a2 = compute_alpha2(panel, m)
            for c in a2.columns:
                if c not in df.columns:
                    df[c] = a2[c]
        except Exception as e:
            print(f"    alpha2 failed at {m.date()}: {e}", flush=True)
        df.index.name = "ticker"
        feats[m] = df
        if (i + 1) % 12 == 0 or i == n - 1:
            print(f"  features [{i+1}/{n}] {m.date()}: {df.shape}", flush=True)
    return feats


def build_cross_section(monthly_clean: pd.DataFrame,
                         feats: dict[pd.Timestamp, pd.DataFrame],
                         membership: Optional[pd.DataFrame] = None,
                         ) -> pd.DataFrame:
    """Build cross-section with multi-horizon fwd returns + PIT membership filter."""
    months = sorted(feats.keys())
    rows = []
    # Build membership index for fast lookup
    mem_by_date = {}
    if membership is not None:
        membership = membership.copy()
        membership["date"] = pd.to_datetime(membership["date"])
        for d, gd in membership.groupby("date"):
            mem_by_date[pd.Timestamp(d)] = set(gd["ticker"].unique())
        mem_dates = sorted(mem_by_date.keys())

    for k, m in enumerate(months[:-1]):
        # PIT membership filter
        members_today = None
        if mem_by_date:
            # Find the membership snapshot at or before m
            pos = np.searchsorted(mem_dates, m, side="right") - 1
            if pos >= 0:
                members_today = mem_by_date[mem_dates[pos]]
        # Find panel position closest to feature month (within ±7d)
        pos = monthly_clean.index.searchsorted(m)
        candidates = []
        for j in (pos - 1, pos):
            if 0 <= j < len(monthly_clean.index):
                d = monthly_clean.index[j]
                candidates.append((j, abs((d - m).days)))
        candidates.sort(key=lambda x: x[1])
        if not candidates or candidates[0][1] > 7:
            continue
        pos1 = candidates[0][0]
        if pos1 + 1 >= len(monthly_clean.index):
            continue
        p1 = monthly_clean.iloc[pos1]
        targets = {}
        for horizon in (1, 3, 6, 12):
            pos_h = pos1 + horizon
            if pos_h >= len(monthly_clean.index):
                break
            ph = monthly_clean.iloc[pos_h]
            cap = 2.0 * horizon
            fwd = (ph / p1 - 1).clip(lower=-1.0, upper=cap)
            end_pos = min(pos1 + horizon + 6, len(monthly_clean.index) - 1)
            future_window = monthly_clean.iloc[pos1 + horizon: end_pos + 1]
            any_future = future_window.notna().any()
            p1_valid = p1.notna()
            ph_nan = ph.isna()
            delist = p1_valid & ph_nan & ~any_future.reindex(monthly_clean.columns, fill_value=False)
            fwd[delist] = -1.0
            targets[f"fwd_{horizon}m_ret"] = fwd
        feats_m = feats[m]
        out = feats_m.copy()
        for tname, tval in targets.items():
            out = out.join(tval.rename(tname), how="left")
        # Apply PIT membership filter (keep SPY for regime gate even if not in membership)
        if members_today is not None:
            keep = list(members_today | {"SPY"})
            out = out.loc[out.index.intersection(keep)]
        out["asof"] = m
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    big = pd.concat(rows, axis=0, ignore_index=False)
    big.index.name = "ticker"
    big = big.reset_index().set_index(["asof", "ticker"])
    return big


def fit_walkforward(big: pd.DataFrame,
                     train_start="2003-01-01",
                     train_end="2024-12-31",
                     embargo_months=7,
                     target_horizons=(1, 3, 6),
                     min_train_rows=10000,
                     min_train_per_target=2500,
                     ) -> pd.DataFrame:
    """Walk-forward fit + predict. Returns df with asof,ticker,fwd_1m_ret,pred,pred_*m."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    big = big.reset_index()
    big["asof"] = pd.to_datetime(big["asof"])
    EXCLUDE = {"SPY","QQQ","IWM","VTI","RSP","DIA","BTC-USD","ETH-USD",
               "TQQQ","SQQQ","UPRO","SPXL","SPXS","TZA","TNA","SOXL","SOXS",
               "FAS","FAZ","TMF","TMV","UGL","GLL","BOIL","KOLD"}
    big = big[~big["ticker"].isin(EXCLUDE)].copy()

    target_cols = [f"rank_target_{h}m" for h in target_horizons]
    fwd_cols = [f"fwd_{h}m_ret" for h in target_horizons]
    for h, tc, fc in zip(target_horizons, target_cols, fwd_cols):
        if fc not in big.columns:
            continue
        big[tc] = big.groupby("asof")[fc].rank(pct=True)

    feature_cols_raw = [c for c in big.columns
                        if c not in ("asof","ticker") and not c.startswith("fwd_")
                        and not c.startswith("rank_target_")]
    print(f"  Cross-sectional ranking {len(feature_cols_raw)} features...")
    t0 = time.time()
    for c in feature_cols_raw:
        big[c + "_xs"] = big.groupby("asof")[c].transform(
            lambda x: (x.rank(pct=True) - 0.5) * 2)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)
    xs_features = [c + "_xs" for c in feature_cols_raw]

    months = sorted(big["asof"].unique())
    last_trained = None
    models = {}
    all_preds = []
    for tm_raw in months:
        tm = pd.Timestamp(tm_raw)
        if tm < pd.Timestamp(train_start) or tm > pd.Timestamp(train_end):
            continue
        # Retrain at start of each calendar year
        do_retrain = (not models or
                      (tm.month == 1 and (last_trained is None or last_trained.year < tm.year)))
        if do_retrain:
            cutoff = tm - pd.DateOffset(months=embargo_months)
            train = big[big["asof"] < cutoff]
            if len(train) < min_train_rows:
                continue
            for h, tc in zip(target_horizons, target_cols):
                if tc not in train.columns:
                    continue
                m = train[tc].notna()
                if m.sum() < min_train_per_target:
                    continue
                Xt = train.loc[m, xs_features].values
                yt = train.loc[m, tc].values
                mdl = HistGradientBoostingRegressor(
                    max_iter=300, learning_rate=0.04, max_depth=6,
                    min_samples_leaf=300, l2_regularization=1.0,
                )
                mdl.fit(Xt, yt)
                models[h] = mdl
            last_trained = tm
            print(f"  Retrained at {tm.date()} (train rows={len(train)})", flush=True)
        if not models:
            continue
        test = big[big["asof"] == tm_raw]
        if len(test) == 0:
            continue
        Xtest = test[xs_features].values
        per_horizon = {h: models[h].predict(Xtest) for h in target_horizons if h in models}
        if not per_horizon:
            continue
        pred_avg = np.mean(list(per_horizon.values()), axis=0)
        rows = test[["asof","ticker","fwd_1m_ret"]].assign(pred=pred_avg)
        for h, p in per_horizon.items():
            rows[f"pred_{h}m"] = p
        all_preds.append(rows)
    if not all_preds:
        return pd.DataFrame()
    return pd.concat(all_preds, axis=0, ignore_index=True)


def simulate_universe(preds: pd.DataFrame, big: pd.DataFrame,
                       monthly_returns: pd.DataFrame,
                       K_normal=15, K_recovery=7, K_bull=7,
                       use_conviction=False, cash_in_crash=True,
                       regime_mode="tight", cost_bps=10.0,
                       year_min=2003, year_max=2024,
                       ) -> tuple[pd.DataFrame, dict]:
    """Apply regime gate + simulate equity curve."""
    if preds.empty:
        return pd.DataFrame(), {}
    preds = preds[(preds["asof"].dt.year >= year_min) & (preds["asof"].dt.year <= year_max)].copy()
    outs = build_strategy_outputs(
        preds, big,
        top_k_normal=K_normal, top_k_recovery=K_recovery, top_k_bull=K_bull,
        use_conviction_weighting=use_conviction, cash_in_crash=cash_in_crash,
        regime_mode=regime_mode,
    )
    eq = simulate_strategy(outs, monthly_returns, cost_bps=cost_bps, starting_cash=1.0)
    if eq.empty:
        return eq, {}
    c = cagr(eq) * 100
    sh = float(eq["ret_m"].mean() / eq["ret_m"].std() * np.sqrt(12)) if eq["ret_m"].std() > 0 else 0
    roll_max = eq["equity"].cummax()
    dd = float((eq["equity"] / roll_max - 1).min() * 100)
    eq["year"] = eq["date"].dt.year
    yr = eq.groupby("year")["ret_m"].apply(lambda x: float(((1 + x).prod() - 1) * 100)).round(1)
    summary = {
        "n_months": len(eq),
        "CAGR_pct": round(c, 2),
        "Sharpe": round(sh, 3),
        "MaxDD_pct": round(dd, 2),
        "Final_equity": round(float(eq["equity"].iloc[-1]), 2),
        "win_rate_months_pct": round(float((eq["ret_m"] > 0).mean() * 100), 2),
        "n_positive_years": int((yr > 0).sum()),
        "n_total_years": int(len(yr)),
        "worst_year_pct": float(yr.min()),
        "best_year_pct": float(yr.max()),
        "year_by_year": {int(y): float(v) for y, v in yr.items()},
    }
    return eq, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Universe name (subdir of cache/v3_universes/)")
    ap.add_argument("--prices", required=True, help="Path to daily price panel parquet")
    ap.add_argument("--membership", default=None,
                    help="Optional PIT membership parquet with cols (date, ticker)")
    ap.add_argument("--min-history", type=int, default=504,
                    help="Min trading days of history for ticker to be eligible (default 504 = 2y)")
    ap.add_argument("--K-normal", type=int, default=15)
    ap.add_argument("--K-recovery", type=int, default=7)
    ap.add_argument("--K-bull", type=int, default=7)
    ap.add_argument("--year-min", type=int, default=2003)
    ap.add_argument("--year-max", type=int, default=2024)
    ap.add_argument("--regime-mode", default="tight")
    ap.add_argument("--min-train-rows", type=int, default=10000,
                    help="Min total train rows to retrain (default 10000; lower for small universes)")
    ap.add_argument("--min-train-per-target", type=int, default=2500,
                    help="Min train rows with non-NaN target (default 2500)")
    args = ap.parse_args()

    out_dir = UV_CACHE_BASE / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Universe: {args.name} ===")
    print(f"  Output: {out_dir}")

    panel = pd.read_parquet(args.prices)
    print(f"  Panel: {panel.shape}, range {panel.index.min().date()} - {panel.index.max().date()}")

    membership = None
    if args.membership:
        membership = pd.read_parquet(args.membership)
        print(f"  Membership: {len(membership)} (date, ticker) rows, "
              f"{membership['ticker'].nunique()} unique tickers")

    print("[1/4] Building clean monthly panel...")
    monthly_clean, mret_clean, bad_list = make_clean_monthly(panel)
    monthly_clean.to_parquet(out_dir / "monthly_prices_clean.parquet")
    mret_clean.to_parquet(out_dir / "monthly_returns_clean.parquet")
    print(f"  Bad-data tickers: {len(bad_list)}")

    print("[2/4] Computing features per month-end...")
    months = pd.date_range(f"{max(args.year_min - 6, 1996)}-01-31",
                            f"{args.year_max + 2}-12-31", freq="ME")
    months = [m for m in months if m >= panel.index.min() + pd.Timedelta(days=args.min_history * 1.5)]
    months = [m for m in months if m <= panel.index.max()]
    feats = build_features_per_month(panel, months, min_history=args.min_history)
    print(f"  Built features for {len(feats)} months")

    print("[3/4] Building cross-section + applying PIT membership filter...")
    big = build_cross_section(monthly_clean, feats, membership=membership)
    if big.empty:
        print("ERROR: empty cross-section")
        return
    print(f"  Cross-section: {big.shape}")
    big.to_parquet(out_dir / "panel_cross_section_v3.parquet")

    print("[4/4] Walk-forward fit + simulate...")
    preds = fit_walkforward(big, train_start=f"{args.year_min}-01-01",
                              train_end=f"{args.year_max}-12-31",
                              min_train_rows=args.min_train_rows,
                              min_train_per_target=args.min_train_per_target)
    if preds.empty:
        print("ERROR: empty predictions")
        return
    preds.to_parquet(out_dir / "ml_preds.parquet")
    print(f"  Predictions: {len(preds)}")

    eq, summary = simulate_universe(
        preds, big, mret_clean,
        K_normal=args.K_normal, K_recovery=args.K_recovery, K_bull=args.K_bull,
        regime_mode=args.regime_mode, year_min=args.year_min, year_max=args.year_max,
    )
    if eq.empty:
        print("ERROR: empty equity curve")
        return
    eq.to_csv(out_dir / "equity_curve.csv", index=False)
    yr_df = pd.DataFrame(list(summary["year_by_year"].items()), columns=["year", "ret_pct"])
    yr_df.to_csv(out_dir / "year_by_year.csv", index=False)
    summary["universe"] = args.name
    summary["K_normal"] = args.K_normal
    summary["K_recovery"] = args.K_recovery
    summary["K_bull"] = args.K_bull
    summary["regime_mode"] = args.regime_mode
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary ({args.name}) ===")
    print(f"  CAGR: {summary['CAGR_pct']}%")
    print(f"  Sharpe: {summary['Sharpe']}")
    print(f"  MaxDD: {summary['MaxDD_pct']}%")
    print(f"  Final equity: {summary['Final_equity']}")
    print(f"  Win rate (months): {summary['win_rate_months_pct']}%")
    print(f"  Positive years: {summary['n_positive_years']}/{summary['n_total_years']}")
    print(f"  Worst year: {summary['worst_year_pct']}%, Best: {summary['best_year_pct']}%")


if __name__ == "__main__":
    main()
