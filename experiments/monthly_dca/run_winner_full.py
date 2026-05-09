"""Evaluate the winning strategy on the FULL 1997-2024 history with bias-correction
across multiple delisting probabilities. Saves everything for reproducibility.

Outputs:
  cache/winner_full_window.csv     — per-strategy CAGR on 1997-2024 vs 2002-2024
  cache/winner_bias_sensitivity.csv — CAGR at α = {0%, 4%, 8%, 12%, 16%, 20%} delisting
  cache/winner_picks_full.csv      — full picks log (1997-2024) for the winner
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd

from experiments.monthly_dca.fast_score import (
    BENCH_EXCLUDED, load_features_long, load_fwd, load_panel,
)
from experiments.monthly_dca.fast_engine import xirr
from experiments.monthly_dca.strategies_ensemble import grand_ensemble, strategy_rotation, diamond_ensemble


CACHE = Path("experiments/monthly_dca/cache")


def picks_for(score_fn, top_k: int, start: str = "1997-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    feats = load_features_long()
    feats = feats.loc[(feats.index.get_level_values("asof") >= pd.Timestamp(start)) &
                      (feats.index.get_level_values("asof") <= pd.Timestamp(end))]
    chunks = []
    for asof, sub in feats.groupby(level="asof"):
        df_asof = sub.copy()
        df_asof.index = df_asof.index.get_level_values("ticker")
        try:
            s = score_fn(df_asof)
        except Exception:
            continue
        s = s.dropna()
        bad = [t for t in BENCH_EXCLUDED if t in s.index]
        s = s.drop(bad, errors="ignore")
        if s.empty:
            continue
        s = s.reset_index()
        s.columns = ["ticker", "score"]
        s["asof"] = asof
        chunks.append(s)
    if not chunks:
        return pd.DataFrame(columns=["asof", "ticker", "score"])
    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values(["asof", "score"], ascending=[True, False])
    return df.groupby("asof", group_keys=False).head(top_k).reset_index(drop=True)


def cagr_with_bias(picks_with_ret: pd.DataFrame, ret_col: str, eval_at, panel,
                    delist_rates: list[float], delist_iters: int = 200) -> pd.DataFrame:
    """Compute CAGR at multiple bias-correction strengths."""
    df = picks_with_ret.dropna(subset=[ret_col])
    if df.empty:
        return pd.DataFrame()
    fv = df[ret_col].to_numpy(dtype=float)
    ah = pd.to_datetime(df["asof"].to_numpy())

    spy = panel["SPY"].dropna()
    bv = []
    for asof_t in ah:
        pos = spy.index.searchsorted(asof_t)
        if pos >= len(spy):
            bv.append(np.nan); continue
        arr = spy.iloc[pos:].to_numpy(dtype=float)
        mask = np.isfinite(arr)
        if mask.any():
            bv.append(arr[mask][-1] / arr[0] - 1.0)
        else:
            bv.append(np.nan)
    bv = np.asarray(bv, dtype=float)

    cashflows_spy = [(pd.Timestamp(t), -1.0) for t in ah]
    cashflows_spy.append((eval_at, float(np.sum(1 + bv[np.isfinite(bv)]))))
    cagr_spy = xirr(cashflows_spy)

    days_to_eval = np.asarray([(eval_at - t).days for t in ah], dtype=float)
    years_to_eval = np.maximum(days_to_eval, 1.0) / 365.25

    rng = np.random.default_rng(0)
    rows = []
    for alpha in delist_rates:
        if alpha == 0.0:
            cashflows = [(pd.Timestamp(t), -1.0) for t in ah]
            cashflows.append((eval_at, float(np.sum(1 + fv))))
            cagr = xirr(cashflows)
            win = float((fv > 0).mean())
            rows.append({
                "delist_alpha": 0.0,
                "cagr_dca": float(cagr),
                "cagr_dca_p10": float(cagr),
                "cagr_dca_p90": float(cagr),
                "win_rate": win,
                "win_rate_median": win,
            })
        else:
            p_del = 1.0 - (1.0 - alpha) ** years_to_eval
            cagrs = []
            wins = []
            for it in range(delist_iters):
                u = rng.random(len(fv))
                fv_mc = np.where(u < p_del, -1.0, fv)
                cf = [(pd.Timestamp(t), -1.0) for t in ah]
                cf.append((eval_at, float(np.sum(1 + fv_mc))))
                try:
                    c = xirr(cf)
                    if np.isfinite(c):
                        cagrs.append(c)
                except Exception:
                    pass
                wins.append(float((fv_mc > 0).mean()))
            if cagrs:
                rows.append({
                    "delist_alpha": float(alpha),
                    "cagr_dca": float(np.median(cagrs)),
                    "cagr_dca_p10": float(np.percentile(cagrs, 10)),
                    "cagr_dca_p90": float(np.percentile(cagrs, 90)),
                    "win_rate": float(np.median(wins)),
                    "win_rate_median": float(np.median(wins)),
                })
    df_out = pd.DataFrame(rows)
    df_out["cagr_spy_dca"] = cagr_spy
    df_out["edge"] = df_out["cagr_dca"] - cagr_spy
    return df_out


def main():
    print("Loading data...")
    feats = load_features_long()
    fwd = load_fwd()
    panel = load_panel()
    eval_at = panel.index.max()

    # Run on TWO windows for comparison
    windows = {
        "1997-2024 (FULL)": ("1997-01-01", "2024-12-31"),
        "2002-2024 (POST-DOTCOM)": ("2002-01-01", "2024-12-31"),
        "2018-2024 (RECENT)": ("2018-01-01", "2024-12-31"),
    }

    candidates = [
        ("grand_ensemble", grand_ensemble, 1),
        ("grand_ensemble", grand_ensemble, 2),
        ("grand_ensemble", grand_ensemble, 3),
        ("strategy_rotation", strategy_rotation, 5),
        ("strategy_rotation", strategy_rotation, 3),
        ("diamond_ensemble", diamond_ensemble, 1),
    ]

    rules = ["hold_forever", "fixed_3y"]
    fwd_reset = fwd.reset_index()

    results = []
    for win_name, (start, end) in windows.items():
        for name, fn, k in candidates:
            picks = picks_for(fn, k, start=start, end=end)
            if picks.empty: continue
            merged = picks.merge(fwd_reset, on=["asof", "ticker"], how="left")
            for rule in rules:
                rc = f"ret__{rule}"
                if rc not in merged.columns: continue
                f = merged[rc].to_numpy(dtype=float)
                valid = np.isfinite(f)
                if not valid.any(): continue
                fv = f[valid]
                ah = pd.to_datetime(merged["asof"].to_numpy()[valid])
                # CAGR portfolio
                cashflows = [(pd.Timestamp(t), -1.0) for t in ah]
                cashflows.append((eval_at, float(np.sum(1 + fv))))
                cagr = xirr(cashflows)
                spy = panel["SPY"].dropna()
                bv = []
                for asof_t in ah:
                    pos = spy.index.searchsorted(asof_t)
                    if pos >= len(spy):
                        bv.append(np.nan); continue
                    arr = spy.iloc[pos:].to_numpy(dtype=float)
                    mask = np.isfinite(arr)
                    if mask.any():
                        bv.append(arr[mask][-1] / arr[0] - 1.0)
                    else:
                        bv.append(np.nan)
                bv = np.asarray(bv, dtype=float)
                cashflows_spy = [(pd.Timestamp(t), -1.0) for t in ah]
                cashflows_spy.append((eval_at, float(np.sum(1 + bv[np.isfinite(bv)]))))
                cagr_spy = xirr(cashflows_spy)
                results.append({
                    "window": win_name,
                    "strategy": name, "top_k": k, "exit": rule,
                    "n_picks": int(valid.sum()),
                    "win_rate": float((fv > 0).mean()),
                    "median_ret": float(np.nanmedian(fv)),
                    "mean_ret": float(np.nanmean(fv)),
                    "cagr_dca": float(cagr),
                    "cagr_spy_dca": float(cagr_spy),
                    "edge": float(cagr - cagr_spy),
                })
                print(f"  {win_name}  {name} k={k} {rule}: CAGR={cagr*100:.2f}% vs SPY {cagr_spy*100:.2f}%", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(CACHE / "winner_full_window.csv", index=False)
    print(f"\nWrote {CACHE / 'winner_full_window.csv'}")
    print(df.to_string(index=False))

    # Bias sensitivity for the WINNERS on full 1997-2024 window
    print("\n=== Bias sensitivity (full 1997-2024) ===")
    all_bias = []
    bias_strategies = [
        ("grand_ensemble", grand_ensemble, 1),
        ("strategy_rotation", strategy_rotation, 5),
        ("strategy_rotation", strategy_rotation, 3),
    ]
    for name, fn, k in bias_strategies:
        picks = picks_for(fn, k, start="1997-01-01", end="2024-12-31")
        merged = picks.merge(fwd_reset, on=["asof", "ticker"], how="left")
        bias_df = cagr_with_bias(merged, "ret__hold_forever", eval_at, panel,
                                  delist_rates=[0.0, 0.04, 0.08, 0.12, 0.16, 0.20])
        bias_df["strategy"] = name
        bias_df["top_k"] = k
        all_bias.append(bias_df)
        print(f"  --- {name} k={k} ---")
        print(bias_df.to_string(index=False))
    out = pd.concat(all_bias, ignore_index=True)
    out.to_csv(CACHE / "winner_bias_sensitivity.csv", index=False)
    print(f"\nWrote {CACHE / 'winner_bias_sensitivity.csv'}")


if __name__ == "__main__":
    main()
