"""Phase 5d: run the deployed v5-winner config on augmented inputs.

Deployed v5-winner: `v5_chr_p70_q0.45_k3_invvol_cap0.4_h6_tight`.
  - GBM ranking by ml_3plus6 (avg pred_3m + pred_6m)
  - Chronos-bolt-tiny p70 (3m) forecast filter:  keep stocks with
    cross-sectional p70-rank >= 0.45 (upper 55%)
  - Pick top-3 from the filtered pool by ml_3plus6 score
  - Inverse-vol weighting (1/vol_1y) capped at 40% per pick
  - Hold each basket for 6 months
  - Tight regime gate (crash -> cash)
  - 10 bps round-trip cost on turnover

This is apples-to-apples to experiments/monthly_dca/cache/v2/sp500_pit/v5_winner_summary.json
(WF mean CAGR 47.16%, Full CAGR 43.86% on the biased panel).

Inputs (augmented):
  augmented/sp500_pit_panel.parquet
  augmented/ml_preds.parquet
  augmented/ml_preds_chronos.parquet
  augmented/monthly_returns_clean.parquet
  augmented/monthly_prices_clean.parquet
  augmented/features/*.parquet (SPY regime features)

Outputs:
  augmented/v5_winner_summary.json
  augmented/v5_winner_walkforward.csv
  augmented/v5_winner_equity.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
PIT = CACHE / "v2" / "sp500_pit"
AUG = PIT / "augmented"

EXCLUDE = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
           "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
           "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD"}

CHRONOS_FILTER_Q = 0.45
CAP_PER_PICK = 0.40
HOLD_MONTHS = 6
K_PICKS = 3
COST_BPS = 10.0


WF_SPLITS = [
    ("A1", "2011-01-01", "2018-12-31"),
    ("A2", "2015-01-01", "2021-12-31"),
    ("A3", "2018-01-01", "2024-12-31"),
    ("R1_GFC", "2008-01-01", "2010-12-31"),
    ("R2", "2011-01-01", "2013-12-31"),
    ("R3", "2014-01-01", "2016-12-31"),
    ("R4", "2017-01-01", "2019-12-31"),
    ("R5_COVID", "2020-01-01", "2022-12-31"),
    ("R6_AI", "2023-01-01", "2024-12-31"),
    ("STRICT", "2021-01-01", "2024-12-31"),
]


def classify_regime_tight(s: dict) -> str:
    r21 = s.get("spy_ret_21d", 0.0)
    r6m = s.get("spy_mom_6_1", 0.0)
    streak = s.get("spy_below_200_streak", 0.0)
    dsma = s.get("spy_dsma200", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0)
    if r21 <= -0.08 or (r6m <= -0.05 and r21 <= -0.03):
        return "crash"
    if streak >= 40 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


def load_spy_features() -> pd.DataFrame:
    rows = []
    for f in sorted((AUG / "features").glob("*.parquet")):
        d = pd.Timestamp(f.stem)
        df = pd.read_parquet(f)
        if "SPY" not in df.index:
            continue
        spy = df.loc["SPY"]
        rows.append({
            "asof": d,
            "spy_dsma200": float(spy.get("d_sma200", 0.0)),
            "spy_mom_12_1": float(spy.get("mom_12_1", 0.0)),
            "spy_mom_6_1": float(spy.get("mom_6_1", 0.0)),
            "spy_ret_21d": float(spy.get("ret_21d", 0.0)),
            "spy_below_200_streak": float(spy.get("max_below_200_streak", 0.0)),
        })
    return pd.DataFrame(rows).set_index("asof")


def calc_invvol_weights(tickers: list, monthly_returns: pd.DataFrame,
                        asof: pd.Timestamp, cap: float = CAP_PER_PICK) -> np.ndarray:
    """1/vol_1y weights for the picks at asof, capped and renormalised."""
    mr_idx = monthly_returns.index
    pos = mr_idx.searchsorted(asof)
    if pos == 0:
        return np.ones(len(tickers)) / len(tickers)
    window = monthly_returns.iloc[max(0, pos - 12): pos]
    vols = []
    for tk in tickers:
        if tk in window.columns:
            v = window[tk].dropna().std()
            vols.append(float(v) if v and not np.isnan(v) and v > 0 else np.nan)
        else:
            vols.append(np.nan)
    vols = np.array(vols)
    if np.all(np.isnan(vols)) or np.all(vols == 0):
        return np.ones(len(tickers)) / len(tickers)
    inv = np.where(np.isnan(vols) | (vols == 0), 1e-9, 1.0 / vols)
    w = inv / inv.sum()
    # Cap and renormalize
    if cap < 1.0:
        for _ in range(8):  # iterate to converge
            over = w > cap
            if not over.any():
                break
            excess = (w[over] - cap).sum()
            w[over] = cap
            under = ~over
            if not under.any():
                break
            w[under] += excess * w[under] / w[under].sum()
        w = w / w.sum()
    return w


def main():
    print("=" * 64)
    print("Phase 5d: deployed v5-winner on AUGMENTED inputs")
    print("=" * 64)

    panel = pd.read_parquet(AUG / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    print(f"[1] panel: {panel.shape}")

    ml = pd.read_parquet(AUG / "ml_preds.parquet")[["asof", "ticker", "pred_3m", "pred_6m"]]
    ml["asof"] = pd.to_datetime(ml["asof"])
    ml["ml_score"] = (ml["pred_3m"] + ml["pred_6m"]) / 2
    print(f"[2] ml_preds: {ml.shape}")

    chr_ = pd.read_parquet(AUG / "ml_preds_chronos.parquet")[["asof", "ticker", "chronos_p70_3m"]]
    chr_["asof"] = pd.to_datetime(chr_["asof"])
    print(f"[3] chronos_preds: {chr_.shape}")

    spy = load_spy_features()
    print(f"[4] SPY features: {spy.shape}")

    monthly_returns = pd.read_parquet(AUG / "monthly_returns_clean.parquet")
    monthly_prices = pd.read_parquet(AUG / "monthly_prices_clean.parquet")
    if not isinstance(monthly_returns.index, pd.DatetimeIndex):
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_prices.index = pd.to_datetime(monthly_prices.index)
    print(f"[5] monthly_returns: {monthly_returns.shape}")

    members = pd.read_parquet(PIT / "sp500_membership_monthly.parquet")
    members["asof"] = pd.to_datetime(members["asof"])
    members_g = members.groupby("asof")["ticker"].apply(set).to_dict()
    print(f"[6] PIT members: {len(members)}")

    # Simulation: rebalance every 6 months (or on crash entry/exit)
    months = sorted(set(panel["asof"]).intersection(set(spy.index)))
    months = [pd.Timestamp(m) for m in months]
    print(f"[7] simulation months: {len(months)}")
    cf = COST_BPS / 10000.0

    cur_picks: list = []
    cur_weights = np.array([])
    cash = False
    held_for = 0
    equity = 1.0
    rows = []

    panel_by_asof = {a: g for a, g in panel.groupby("asof")}
    ml_by_asof = {a: g for a, g in ml.groupby("asof")}
    chr_by_asof = {a: g for a, g in chr_.groupby("asof")}

    for i, m in enumerate(months):
        regime = classify_regime_tight(spy.loc[m].to_dict() if m in spy.index else {})
        do_reb = (i == 0) or (held_for >= HOLD_MONTHS) or (cash != (regime == "crash"))

        ret_m = 0.0
        if not cash and cur_picks:
            mr_pos = monthly_returns.index.searchsorted(m)
            if mr_pos + 1 < len(monthly_returns.index):
                next_d = monthly_returns.index[mr_pos + 1]
                pick_rets = []
                for tk in cur_picks:
                    if tk in monthly_returns.columns and next_d in monthly_returns.index:
                        r = monthly_returns.at[next_d, tk]
                        # NaN here = the ticker stopped trading (acquisition or
                        # delisting). The honest interpretation is 0% return
                        # (cash payout from acquisition, no further P&L). Using
                        # -100% here would imply forced bankruptcy losses on
                        # every acquisition, which is incorrect for the majority
                        # of our backfilled names (AGN, ANTM, ABMD, etc. were
                        # acquired AT or ABOVE their last trade price).
                        pick_rets.append(0.0 if pd.isna(r) else float(r))
                    else:
                        pick_rets.append(0.0)
                ret_m = float((np.array(pick_rets) * cur_weights).sum())
                equity *= (1 + ret_m)

        if do_reb:
            # Apply turnover cost on rebalance
            equity *= (1 - cf)
            if regime == "crash":
                cur_picks = []
                cur_weights = np.array([])
                cash = True
            else:
                sub_panel = panel_by_asof.get(m)
                sub_ml = ml_by_asof.get(m)
                sub_chr = chr_by_asof.get(m)
                if sub_panel is None or sub_ml is None:
                    cur_picks = []
                    cur_weights = np.array([])
                else:
                    sp_set = members_g.get(m, set())
                    sub = sub_panel[sub_panel["ticker"].isin(sp_set)]
                    sub = sub[~sub["ticker"].isin(EXCLUDE)]
                    sub = sub.merge(sub_ml[["ticker", "ml_score"]], on="ticker", how="left")
                    sub = sub.dropna(subset=["ml_score"])
                    if sub_chr is not None and not sub_chr.empty:
                        sub = sub.merge(sub_chr[["ticker", "chronos_p70_3m"]],
                                        on="ticker", how="left")
                        sub = sub.dropna(subset=["chronos_p70_3m"])
                        sub["chr_p70_rk"] = sub["chronos_p70_3m"].rank(pct=True)
                        sub = sub[sub["chr_p70_rk"] >= CHRONOS_FILTER_Q]
                    sub = sub.sort_values("ml_score", ascending=False)
                    top = sub.head(K_PICKS)
                    if len(top) < K_PICKS:
                        cur_picks = []
                        cur_weights = np.array([])
                    else:
                        cur_picks = top["ticker"].tolist()
                        cur_weights = calc_invvol_weights(cur_picks, monthly_returns, m,
                                                         cap=CAP_PER_PICK)
                cash = False
            held_for = 0
        else:
            held_for += 1

        rows.append({
            "date": m, "regime": regime, "equity": equity, "ret_m": ret_m,
            "cash": cash, "n_picks": len(cur_picks),
            "picks": ",".join(cur_picks),
        })

    eq = pd.DataFrame(rows)

    # Headline
    n_months = len(eq)
    cagr_full = (eq["equity"].iloc[-1]) ** (12 / n_months) - 1 if n_months else 0.0
    ret_series = eq["ret_m"].astype(float)
    sharpe = (ret_series.mean() / max(ret_series.std(), 1e-9)) * np.sqrt(12)
    peak = eq["equity"].cummax()
    dd = (eq["equity"] - peak) / peak
    max_dd = float(dd.min())

    # SPY benchmark CAGR over same months
    spy_ret = monthly_returns["SPY"]
    next_months = pd.DatetimeIndex(eq["date"]) + pd.offsets.MonthEnd(1)
    spy_aligned = []
    for nxt in next_months:
        if nxt in spy_ret.index:
            spy_aligned.append(float(spy_ret.loc[nxt]))
        else:
            spy_aligned.append(0.0)
    spy_eq = (1 + pd.Series(spy_aligned)).cumprod()
    spy_cagr = (spy_eq.iloc[-1]) ** (12 / n_months) - 1 if n_months else 0.0

    print(f"\n[headline]")
    print(f"  n_months: {n_months}")
    print(f"  final_equity: ${eq['equity'].iloc[-1]:.2f}")
    print(f"  cagr_full: {cagr_full:.4f}")
    print(f"  spy_cagr_full: {spy_cagr:.4f}")
    print(f"  edge_pp: {(cagr_full - spy_cagr)*100:.2f}")
    print(f"  sharpe: {sharpe:.4f}")
    print(f"  max_dd: {max_dd:.4f}")
    print(f"  n_cash_months: {int(eq['cash'].sum())}")

    # Walk-forward splits
    spy_df = pd.DataFrame({"date": eq["date"], "spy_ret_m": spy_aligned})
    wf_rows = []
    for split, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)].copy()
        if len(e) == 0:
            continue
        r = e["ret_m"].astype(float)
        ec = (1 + r).cumprod()
        cv = (ec.iloc[-1]) ** (12.0 / len(ec)) - 1
        sh = (r.mean() / max(r.std(), 1e-9)) * np.sqrt(12)
        peak2 = ec.cummax()
        mdd = float(((ec - peak2) / peak2).min())
        s = spy_df[(spy_df["date"] >= lo) & (spy_df["date"] <= hi)]
        sr = s["spy_ret_m"].astype(float)
        sc = (1 + sr).cumprod()
        scgr = (sc.iloc[-1]) ** (12.0 / len(sc)) - 1
        wf_rows.append({
            "split": split, "from": str(lo.date()), "to": str(hi.date()),
            "n_m": len(e),
            "cagr": cv, "spy_cagr": scgr, "edge_pp": (cv - scgr) * 100,
            "sharpe": sh, "max_dd": mdd,
            "n_cash": int((e["regime"] == "crash").sum()),
        })
    wf = pd.DataFrame(wf_rows)

    print("\n[per-split walk-forward]")
    print(wf.round(3).to_string(index=False))

    eq.to_csv(AUG / "v5_winner_equity.csv", index=False)
    wf.to_csv(AUG / "v5_winner_walkforward.csv", index=False)

    summary = {
        "variant_name": "v5_chr_p70_q0.45_k3_invvol_cap0.4_h6_tight",
        "n_months": int(n_months),
        "final_equity": float(eq["equity"].iloc[-1]),
        "cagr_full": float(cagr_full),
        "spy_cagr_full": float(spy_cagr),
        "edge_full_pp": float((cagr_full - spy_cagr) * 100),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "n_cash_months": int(eq["cash"].sum()),
        "wf_mean_cagr": float(wf["cagr"].mean()),
        "wf_median_cagr": float(wf["cagr"].median()),
        "wf_min_cagr": float(wf["cagr"].min()),
        "wf_max_cagr": float(wf["cagr"].max()),
        "wf_mean_edge_pp": float(wf["edge_pp"].mean()),
        "wf_n_positive": int((wf["cagr"] > 0).sum()),
        "wf_n_beats_spy": int((wf["cagr"] > wf["spy_cagr"]).sum()),
        "wf_n_splits": int(len(wf)),
    }
    (AUG / "v5_winner_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] {AUG / 'v5_winner_summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
