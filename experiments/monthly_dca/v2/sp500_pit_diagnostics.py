"""Diagnostics on the PIT S&P 500 backtests.

Computes:
  - Information coefficient (IC) of predictions vs forward returns within
    S&P 500 universe at each month-end (filter-only and retrain).
  - Most-frequently-picked tickers in each backtest.
  - Turnover statistics (overlap between consecutive months' picks).
  - SPY DCA (XIRR-style $1/month deposit) benchmark for parity with original report.
  - Per-decade rolling stats (drawdown, CAGR).

Outputs to cache/v2/sp500_pit/sp500_pit_diagnostics.json plus CSVs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"


def ic_within_sp500(preds: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    members = members.copy()
    members["asof"] = pd.to_datetime(members["asof"])
    preds = preds.copy()
    preds["asof"] = pd.to_datetime(preds["asof"])

    members_g = members.groupby("asof")["ticker"].apply(set)
    rows = []
    for d, sub in preds.groupby("asof"):
        sp = members_g.get(d, set())
        s = sub[sub["ticker"].isin(sp)]
        s = s.dropna(subset=["fwd_1m_ret", "pred"])
        if len(s) < 20:
            continue
        ic_pearson = float(s[["pred", "fwd_1m_ret"]].corr().iloc[0, 1])
        ic_spearman = float(s[["pred", "fwd_1m_ret"]].corr(method="spearman").iloc[0, 1])
        rows.append({"asof": d, "n": len(s), "ic_pearson": ic_pearson, "ic_spearman": ic_spearman})
    return pd.DataFrame(rows)


def most_picked(eq_csv: Path, top_n: int = 30) -> pd.DataFrame:
    eq = pd.read_csv(eq_csv)
    counts = {}
    for picks_str in eq["picks"].dropna():
        if not picks_str:
            continue
        for tk in picks_str.split(","):
            counts[tk] = counts.get(tk, 0) + 1
    df = pd.DataFrame(counts.items(), columns=["ticker", "n_months_picked"])
    return df.sort_values("n_months_picked", ascending=False).head(top_n)


def turnover(eq_csv: Path) -> dict:
    eq = pd.read_csv(eq_csv)
    last = []
    sames = []
    for _, r in eq.iterrows():
        cur = r.get("picks", "")
        cur_set = set(str(cur).split(",")) if isinstance(cur, str) and cur else set()
        if last and cur_set:
            common = len(set(last) & cur_set)
            sames.append(common / max(len(cur_set), 1))
        last = list(cur_set)
    sames = np.array(sames)
    return {
        "n_pairs": int(len(sames)),
        "mean_overlap_with_prev": float(sames.mean()) if len(sames) else 0.0,
        "median_overlap_with_prev": float(np.median(sames)) if len(sames) else 0.0,
        "approx_annualised_turnover": float((1 - sames.mean()) * 12) if len(sames) else 0.0,
    }


def spy_dca_xirr(monthly_returns: pd.DataFrame, eq_dates: pd.DatetimeIndex,
                 deposit: float = 1.0) -> dict:
    """Compute XIRR-style CAGR for SPY DCA over the backtest window."""
    spy = monthly_returns["SPY"]
    next_month = eq_dates + pd.offsets.MonthEnd(1)

    rets = []
    for nxt in next_month:
        if nxt in spy.index:
            rets.append(float(spy.loc[nxt]))
        else:
            rets.append(0.0)
    rets = np.array(rets)
    n = len(rets)

    # Each month deposit grows for n-i more months
    fwd = np.zeros(n)
    fv = 1.0
    for j in range(n - 1, -1, -1):
        fv *= (1 + rets[j])
        fwd[j] = fv  # forward growth from month j to end (after applying month j+1 .. end returns)
    # Actually: deposit at start of month i ends up at end with product of returns (1+r_{i+1})..(1+r_n).
    # Easier: simulate:
    eq = 0.0
    deposits = 0.0
    eq_path = []
    for r in rets:
        eq += deposit
        deposits += deposit
        eq *= (1 + r)
        eq_path.append(eq)
    final = eq

    # XIRR: solve final = sum(deposit * (1+x)^((n-i)/12)) for x
    def npv(x):
        return sum(deposit * (1 + x) ** ((n - i) / 12.0) for i in range(n)) - final

    # bisection
    lo, hi = -0.99, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if npv(mid) > 0:
            hi = mid
        else:
            lo = mid
    xirr = (lo + hi) / 2

    return {
        "n_months": n,
        "deposits_total": deposits,
        "final_equity": final,
        "xirr_cagr": xirr,
        "buyhold_cagr": (1 + rets).prod() ** (12 / n) - 1,
    }


def main():
    print("=== Filter-only IC ===")
    preds_filter = pd.read_parquet(V2 / "ml_preds_v2.parquet")
    members = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    ic_filter = ic_within_sp500(preds_filter, members)
    ic_filter.to_csv(PIT / "sp500_pit_filter_ic.csv", index=False)
    print(f"  Mean IC (Pearson): {ic_filter['ic_pearson'].mean():.4f}")
    print(f"  Mean IC (Spearman): {ic_filter['ic_spearman'].mean():.4f}")
    print(f"  IC IR (annualised): {(ic_filter['ic_pearson'].mean() / max(ic_filter['ic_pearson'].std(), 1e-9)) * np.sqrt(12):.2f}")

    print("\n=== Re-train IC ===")
    preds_retrain = pd.read_parquet(PIT / "sp500_pit_retrain_preds.parquet")
    # Only need members for the re-train predictions which are already SP500
    ic_retrain = preds_retrain.dropna(subset=["fwd_1m_ret", "pred"]).groupby("asof").apply(
        lambda s: pd.Series({
            "ic_pearson": s[["pred", "fwd_1m_ret"]].corr().iloc[0, 1],
            "ic_spearman": s[["pred", "fwd_1m_ret"]].corr(method="spearman").iloc[0, 1],
            "n": len(s),
        })
    ).reset_index()
    ic_retrain.to_csv(PIT / "sp500_pit_retrain_ic.csv", index=False)
    print(f"  Mean IC (Pearson): {ic_retrain['ic_pearson'].mean():.4f}")
    print(f"  Mean IC (Spearman): {ic_retrain['ic_spearman'].mean():.4f}")
    print(f"  IC IR (annualised): {(ic_retrain['ic_pearson'].mean() / max(ic_retrain['ic_pearson'].std(), 1e-9)) * np.sqrt(12):.2f}")

    print("\n=== Most-picked tickers (filter-only) ===")
    mp = most_picked(PIT / "sp500_pit_filter_equity.csv")
    mp.to_csv(PIT / "sp500_pit_filter_most_picked.csv", index=False)
    print(mp.head(20).to_string(index=False))

    print("\n=== Most-picked tickers (re-train) ===")
    # Need to reconstruct picks for retrain
    eq_re = pd.read_csv(PIT / "sp500_pit_retrain_equity.csv")
    counts_re = {}
    for picks_str in eq_re["picks"].dropna():
        if not picks_str:
            continue
        for tk in picks_str.split(","):
            counts_re[tk] = counts_re.get(tk, 0) + 1
    mp_re = pd.DataFrame(counts_re.items(), columns=["ticker", "n_months_picked"]).sort_values("n_months_picked", ascending=False)
    mp_re.head(30).to_csv(PIT / "sp500_pit_retrain_most_picked.csv", index=False)
    print(mp_re.head(20).to_string(index=False))

    print("\n=== Turnover ===")
    to_filter = turnover(PIT / "sp500_pit_filter_equity.csv")
    to_retrain = turnover(PIT / "sp500_pit_retrain_equity.csv")
    print(f"  Filter-only: {to_filter}")
    print(f"  Re-train   : {to_retrain}")

    print("\n=== SPY DCA benchmark (XIRR, $1/month) ===")
    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    eq_filter = pd.read_csv(PIT / "sp500_pit_filter_equity.csv")
    eq_filter["date"] = pd.to_datetime(eq_filter["date"])
    spy_dca_filter = spy_dca_xirr(monthly_returns, pd.DatetimeIndex(eq_filter["date"]))
    print(f"  Filter window: {spy_dca_filter}")

    eq_re["date"] = pd.to_datetime(eq_re["date"])
    spy_dca_re = spy_dca_xirr(monthly_returns, pd.DatetimeIndex(eq_re["date"]))
    print(f"  Retrain window: {spy_dca_re}")

    summary = {
        "ic_filter_pearson_mean": float(ic_filter["ic_pearson"].mean()),
        "ic_filter_spearman_mean": float(ic_filter["ic_spearman"].mean()),
        "ic_filter_ir_annualised": float(ic_filter["ic_pearson"].mean() / max(ic_filter["ic_pearson"].std(), 1e-9) * np.sqrt(12)),
        "ic_retrain_pearson_mean": float(ic_retrain["ic_pearson"].mean()),
        "ic_retrain_spearman_mean": float(ic_retrain["ic_spearman"].mean()),
        "turnover_filter": to_filter,
        "turnover_retrain": to_retrain,
        "spy_dca_filter_window": spy_dca_filter,
        "spy_dca_retrain_window": spy_dca_re,
    }
    (PIT / "sp500_pit_diagnostics.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\nSaved -> sp500_pit_diagnostics.json")


if __name__ == "__main__":
    main()
