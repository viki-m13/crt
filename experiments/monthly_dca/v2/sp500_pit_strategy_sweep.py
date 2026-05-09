"""Strategy sweep for the PIT S&P 500 universe.

Builds candidate scorers (single factor, multi-factor blends, ML models) and
sweeps K, weighting scheme, holding period, and regime gate.  Picks the
walk-forward winner.

For each candidate variant, computes:
  - Full-window CAGR, Sharpe, MaxDD
  - Walk-forward mean / median / min CAGR, edge over SPY across the 10 splits
  - Beats SPY count

Outputs:
  cache/v2/sp500_pit/sp500_pit_sweep_results.csv
  cache/v2/sp500_pit/sp500_pit_sweep_winner.json
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"
FEATURES_DIR = CACHE / "features"

EXCLUDE = {"SPY", "QQQ", "IWM", "VTI", "RSP", "DIA", "BTC-USD", "ETH-USD",
           "TQQQ", "SQQQ", "UPRO", "SPXL", "SPXS", "TZA", "TNA", "SOXL", "SOXS",
           "FAS", "FAZ", "TMF", "TMV", "UGL", "GLL", "BOIL", "KOLD"}

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


# ---------------------------------------------------------------------------
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


def classify_regime_loose(s: dict) -> str:
    """Looser crash gate — fires more often."""
    r21 = s.get("spy_ret_21d", 0.0)
    r6m = s.get("spy_mom_6_1", 0.0)
    dsma = s.get("spy_dsma200", 0.0)
    mom12 = s.get("spy_mom_12_1", 0.0)
    rsi = s.get("spy_rsi14", 50.0)
    streak = s.get("spy_below_200_streak", 0.0)
    if r21 <= -0.06 or r6m <= -0.07 or (dsma < -0.05 and rsi < 40):
        return "crash"
    if streak >= 40 and dsma > 0 and r21 > 0:
        return "recovery"
    if mom12 >= 0.10 and dsma > 0:
        return "bull"
    return "normal"


def classify_regime_none(s: dict) -> str:
    return "normal"


REGIME_GATES = {
    "tight": classify_regime_tight,
    "loose": classify_regime_loose,
    "none": classify_regime_none,
}


# ---------------------------------------------------------------------------
def load_spy_features() -> pd.DataFrame:
    rows = []
    for f in sorted(FEATURES_DIR.glob("*.parquet")):
        d = pd.Timestamp(f.stem)
        df = pd.read_parquet(f)
        if "SPY" not in df.index:
            continue
        spy = df.loc["SPY"]
        rows.append({
            "asof": d,
            "spy_dsma200": float(spy.get("d_sma200", 0.0)),
            "spy_rsi14": float(spy.get("rsi_14", 50.0)),
            "spy_mom_12_1": float(spy.get("mom_12_1", 0.0)),
            "spy_mom_6_1": float(spy.get("mom_6_1", 0.0)),
            "spy_ret_21d": float(spy.get("ret_21d", 0.0)),
            "spy_below_200_streak": float(spy.get("max_below_200_streak", 0.0)),
        })
    return pd.DataFrame(rows).set_index("asof")


# ---------------------------------------------------------------------------
def attach_score(panel: pd.DataFrame, scorer: str) -> pd.DataFrame:
    """Attach a 'score' column based on scorer name. All inputs must be _xs ranked."""
    p = panel.copy()
    if scorer == "mom_12_1":
        p["score"] = p["mom_12_1_xs"]
    elif scorer == "idio_mom":
        p["score"] = p["idio_mom_12_1_xs"]
    elif scorer == "mom_per_vol":
        p["score"] = p["mom_per_unit_vol_12_xs"]
    elif scorer == "quality_5y":
        p["score"] = (p["sharpe_5y_xs"] + p["trend_health_5y_xs"] + p["quality_score_5y_xs"]) / 3
    elif scorer == "low_vol":
        p["score"] = -p["vol_1y_xs"]
    elif scorer == "pullback_in_winner":
        # Long-term winner with recent pullback
        p["score"] = (
            0.30 * p["mom_12_1_xs"]
            + 0.20 * p["trend_health_5y_xs"]
            + 0.20 * p["sharpe_5y_xs"]
            - 0.20 * p["ret_21d_xs"]    # short-term reversal
            - 0.10 * p["rsi_14_xs"]     # oversold
        )
    elif scorer == "qmm":
        # Quality + Momentum + Mean-reversion
        p["score"] = (
            0.30 * p["mom_12_1_xs"]
            + 0.20 * p["idio_mom_12_1_xs"]
            + 0.15 * p["sharpe_5y_xs"]
            + 0.10 * p["trend_health_5y_xs"]
            + 0.10 * p["quality_score_5y_xs"]
            - 0.10 * p["ret_5d_xs"]
            - 0.05 * p["rsi_14_xs"]
        )
    elif scorer == "qmm_v2":
        # Add multibagger + earnings drift, drop weakest
        p["score"] = (
            0.25 * p["mom_12_1_xs"]
            + 0.15 * p["idio_mom_12_1_xs"]
            + 0.15 * p["mom_per_unit_vol_12_xs"]
            + 0.10 * p["sharpe_5y_xs"]
            + 0.10 * p["trend_health_5y_xs"]
            + 0.05 * p["multibagger_ratio_24m_xs"]
            + 0.05 * p["earnings_drift_proxy_xs"]
            - 0.10 * p["ret_5d_xs"]
            - 0.05 * p["rsi_14_xs"]
        )
    elif scorer == "rev_oversold":
        # Pure short-term mean-reversion
        p["score"] = -0.5 * p["ret_5d_xs"] - 0.3 * p["ret_21d_xs"] - 0.2 * p["rsi_14_xs"]
    elif scorer == "winner_oversold":
        # Combine reversal with long-term winner
        p["score"] = 0.5 * p["mom_12_1_xs"] - 0.3 * p["ret_5d_xs"] - 0.2 * p["rsi_14_xs"]
    elif scorer == "ml_filter":
        # Use existing v2 ML predictions
        ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        p = p.merge(ml.rename(columns={"pred": "score"}), on=["asof", "ticker"], how="left")
    elif scorer == "ml_retrain":
        ml = pd.read_parquet(PIT / "sp500_pit_retrain_preds.parquet")[["asof", "ticker", "pred"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        p = p.merge(ml.rename(columns={"pred": "score"}), on=["asof", "ticker"], how="left")
    elif scorer == "ml_filter_plus_qmm":
        # Blend ML score with QMM
        ml = pd.read_parquet(V2 / "ml_preds_v2.parquet")[["asof", "ticker", "pred"]]
        ml["asof"] = pd.to_datetime(ml["asof"])
        p = p.merge(ml, on=["asof", "ticker"], how="left")
        # Compute QMM
        qmm = (
            0.30 * p["mom_12_1_xs"]
            + 0.20 * p["idio_mom_12_1_xs"]
            + 0.15 * p["sharpe_5y_xs"]
            + 0.10 * p["trend_health_5y_xs"]
            + 0.10 * p["quality_score_5y_xs"]
            - 0.10 * p["ret_5d_xs"]
            - 0.05 * p["rsi_14_xs"]
        )
        # Cross-sectionally rank both, then blend
        p["ml_xs"] = p.groupby("asof")["pred"].rank(pct=True) - 0.5
        p["qmm_xs"] = qmm.groupby(p["asof"]).rank(pct=True) - 0.5
        p["score"] = 0.5 * p["ml_xs"] + 0.5 * p["qmm_xs"]
    else:
        raise ValueError(f"unknown scorer: {scorer}")
    return p


# ---------------------------------------------------------------------------
@dataclass
class Variant:
    name: str
    scorer: str
    k_normal: int
    k_recovery: int
    k_bull: int
    weighting: str          # 'ew', 'conv', 'invvol'
    regime_gate: str        # 'tight', 'loose', 'none'
    hold_months: int        # 1, 3, 6


def simulate_variant(
    panel: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    spy_features: pd.DataFrame,
    variant: Variant,
    cost_bps: float = 10.0,
    starting_cash: float = 1.0,
) -> pd.DataFrame:
    """Simulate a single variant on the full PIT panel."""
    p = attach_score(panel, variant.scorer)
    p = p.dropna(subset=["score"])
    p = p[~p["ticker"].isin(EXCLUDE)]

    months = sorted(p["asof"].unique())
    classifier = REGIME_GATES[variant.regime_gate]
    h = variant.hold_months

    monthly_returns = monthly_returns.copy()
    mr_idx = monthly_returns.index

    # Pre-compute panel by asof
    by_asof = {pd.Timestamp(d): g for d, g in p.groupby("asof")}

    equity = starting_cash
    cost_factor = cost_bps / 10000.0
    rows = []
    cur_picks: list[str] = []
    cur_weights: np.ndarray = np.array([])
    held_for = 0  # months held since last rebalance
    cash_position = False

    for i, m in enumerate(months):
        m = pd.Timestamp(m)
        # Decide rebalance: every h months OR cash transition
        do_rebalance = (i == 0) or (held_for >= h) or cash_position

        if do_rebalance:
            spy_now = spy_features.loc[m].to_dict() if m in spy_features.index else {}
            regime = classifier(spy_now)
            if regime == "crash":
                cur_picks, cur_weights = [], np.array([])
                cash_position = True
                held_for = 0
            else:
                k = {"recovery": variant.k_recovery, "bull": variant.k_bull, "normal": variant.k_normal}[regime]
                sub = by_asof.get(m, pd.DataFrame())
                if len(sub) < k:
                    cur_picks, cur_weights = [], np.array([])
                    cash_position = True
                    held_for = 0
                else:
                    top = sub.sort_values("score", ascending=False).head(k)
                    cur_picks = top["ticker"].tolist()
                    if variant.weighting == "ew":
                        cur_weights = np.ones(k) / k
                    elif variant.weighting == "conv":
                        s = top["score"].values
                        shifted = s - s.min() + 1e-6
                        cur_weights = shifted / shifted.sum()
                    elif variant.weighting == "invvol":
                        v = top["vol_1y"].values
                        v = np.where(np.isnan(v) | (v <= 0), 0.4, v)
                        invv = 1.0 / v
                        cur_weights = invv / invv.sum()
                    else:
                        raise ValueError(f"unknown weighting: {variant.weighting}")
                    cash_position = False
                    held_for = 0

        # Apply this month's return to current basket
        if cash_position or len(cur_picks) == 0:
            ret_m = 0.0
        else:
            pos1 = mr_idx.searchsorted(m)
            cands = []
            for j in (pos1 - 1, pos1):
                if 0 <= j < len(mr_idx):
                    cands.append((j, abs((mr_idx[j] - m).days)))
            cands.sort(key=lambda x: x[1])
            if not cands or cands[0][1] > 7 or cands[0][0] + 1 >= len(mr_idx):
                ret_m = 0.0
            else:
                next_d = mr_idx[cands[0][0] + 1]
                pick_rets = []
                for tk in cur_picks:
                    if tk in monthly_returns.columns:
                        r = monthly_returns.at[next_d, tk]
                        pick_rets.append(-1.0 if pd.isna(r) else float(r))
                    else:
                        pick_rets.append(-1.0)
                pick_rets = np.array(pick_rets)
                ret_m = float((pick_rets * cur_weights).sum())

        # Apply equity update; charge cost only on rebalance month
        if not cash_position and len(cur_picks) > 0:
            if do_rebalance:
                equity *= (1 + ret_m) * (1 - cost_factor)
            else:
                equity *= (1 + ret_m)
        held_for += 1

        rows.append({"date": m, "equity": equity, "ret_m": ret_m,
                     "regime": "cash" if cash_position else "active",
                     "n_picks": len(cur_picks)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def cagr_from(eq: pd.Series, start_cash: float = 1.0) -> float:
    if len(eq) == 0:
        return 0.0
    n = len(eq)
    years = max(n / 12.0, 1 / 12.0)
    return (eq.iloc[-1] / start_cash) ** (1.0 / years) - 1.0


def sharpe_monthly(ret: pd.Series) -> float:
    r = ret.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return (r.mean() / r.std()) * np.sqrt(12)


def max_drawdown_from_returns(ret: pd.Series) -> float:
    eq = (1 + ret).cumprod()
    if len(eq) == 0:
        return 0.0
    peak = eq.cummax()
    return float(((eq - peak) / peak).min())


def evaluate_variant(eq: pd.DataFrame, spy_aligned: pd.DataFrame, name: str) -> dict:
    cgr = cagr_from(eq["equity"])
    sh = sharpe_monthly(eq["ret_m"])
    mdd = max_drawdown_from_returns(eq["ret_m"])
    n_cash = int((eq["regime"] == "cash").sum())

    # Walk-forward over splits
    wf_rows = []
    for split, lo, hi in WF_SPLITS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        e = eq[(eq["date"] >= lo) & (eq["date"] <= hi)].copy()
        if len(e) == 0:
            continue
        ret = e["ret_m"].astype(float)
        eqc = (1 + ret).cumprod()
        cagr_v = (eqc.iloc[-1]) ** (12.0 / len(eqc)) - 1
        spy = spy_aligned[(spy_aligned["date"] >= lo) & (spy_aligned["date"] <= hi)]
        spy_ret = spy["spy_ret_m"].astype(float)
        spy_eq = (1 + spy_ret).cumprod()
        spy_cgr = (spy_eq.iloc[-1]) ** (12.0 / len(spy_eq)) - 1 if len(spy_eq) else 0.0
        wf_rows.append({"split": split, "cagr": cagr_v, "spy_cagr": spy_cgr,
                        "edge_pp": (cagr_v - spy_cgr) * 100})
    wf = pd.DataFrame(wf_rows)

    # Full-window SPY
    spy_ret = spy_aligned["spy_ret_m"].astype(float)
    spy_eq = (1 + spy_ret).cumprod()
    spy_cagr_full = spy_eq.iloc[-1] ** (12.0 / len(spy_eq)) - 1

    return {
        "name": name,
        "cagr_full": cgr,
        "spy_cagr_full": spy_cagr_full,
        "edge_full_pp": (cgr - spy_cagr_full) * 100,
        "sharpe": sh,
        "max_dd": mdd,
        "n_cash_months": n_cash,
        "wf_mean_cagr": float(wf["cagr"].mean()) if len(wf) else 0.0,
        "wf_median_cagr": float(wf["cagr"].median()) if len(wf) else 0.0,
        "wf_min_cagr": float(wf["cagr"].min()) if len(wf) else 0.0,
        "wf_max_cagr": float(wf["cagr"].max()) if len(wf) else 0.0,
        "wf_mean_edge_pp": float(wf["edge_pp"].mean()) if len(wf) else 0.0,
        "wf_n_positive": int((wf["cagr"] > 0).sum()) if len(wf) else 0,
        "wf_n_beats_spy": int((wf["cagr"] > wf["spy_cagr"]).sum()) if len(wf) else 0,
        "wf_n_splits": int(len(wf)),
    }


# ---------------------------------------------------------------------------
def main():
    print("=== Loading inputs ===")
    panel = pd.read_parquet(PIT / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    print(f"  panel: {panel.shape}")

    monthly_returns = pd.read_parquet(V2 / "monthly_returns_clean.parquet")
    spy_features = load_spy_features()
    print(f"  monthly_returns: {monthly_returns.shape}, spy_features: {spy_features.shape}")

    # Build SPY-aligned monthly benchmark for the full panel window
    spy_ret_series = monthly_returns["SPY"]
    full_dates = pd.DatetimeIndex(sorted(panel["asof"].unique()))
    next_month = full_dates + pd.offsets.MonthEnd(1)
    spy_aligned = pd.DataFrame({
        "date": full_dates,
        "spy_ret_m": [float(spy_ret_series.loc[nxt]) if nxt in spy_ret_series.index else 0.0
                      for nxt in next_month],
    })

    # Define candidates
    scorers = ["mom_12_1", "idio_mom", "mom_per_vol", "quality_5y", "low_vol",
               "pullback_in_winner", "qmm", "qmm_v2", "rev_oversold",
               "winner_oversold", "ml_filter", "ml_retrain", "ml_filter_plus_qmm"]

    # K-tuples: (k_normal, k_recovery, k_bull)
    k_combos = [
        ("k15_7_7", 15, 7, 7),
        ("k10_5_5", 10, 5, 5),
        ("k7_5_5",   7, 5, 5),
        ("k5_3_3",   5, 3, 3),
        ("k3_3_3",   3, 3, 3),
    ]
    weightings = ["ew", "conv", "invvol"]
    gates = ["tight", "loose", "none"]
    holds = [1, 3, 6]

    print(f"\n=== Sweep: {len(scorers)} scorers × {len(k_combos)} K × {len(weightings)} W × {len(gates)} G × {len(holds)} H = {len(scorers)*len(k_combos)*len(weightings)*len(gates)*len(holds)} variants ===")

    rows = []
    t0 = time.time()
    n_done = 0
    for scorer in scorers:
        # Pre-attach score once per scorer
        for k_lab, kN, kR, kB in k_combos:
            for w in weightings:
                for g in gates:
                    for h in holds:
                        v = Variant(
                            name=f"{scorer}|{k_lab}|{w}|{g}|h{h}",
                            scorer=scorer,
                            k_normal=kN, k_recovery=kR, k_bull=kB,
                            weighting=w, regime_gate=g, hold_months=h,
                        )
                        try:
                            eq = simulate_variant(panel, monthly_returns, spy_features, v)
                        except Exception as e:
                            print(f"  ! {v.name}: {e}")
                            continue
                        m = evaluate_variant(eq, spy_aligned, v.name)
                        m["scorer"] = scorer
                        m["k_combo"] = k_lab
                        m["weighting"] = w
                        m["gate"] = g
                        m["hold"] = h
                        rows.append(m)
                        n_done += 1
                        if n_done % 50 == 0:
                            print(f"    {n_done}/{len(scorers)*len(k_combos)*len(weightings)*len(gates)*len(holds)} "
                                  f"({time.time()-t0:.0f}s elapsed)")
    print(f"  Total variants: {len(rows)}, time {time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(PIT / "sp500_pit_sweep_results.csv", index=False)

    # Pick the winner
    print("\n=== Top 20 by full-window CAGR (filtering reasonable picks) ===")
    cand = df[(df["wf_n_splits"] >= 8) & (df["wf_min_cagr"] > -0.30) & (df["max_dd"] > -0.65)].copy()
    print(cand.sort_values("cagr_full", ascending=False).head(20)[
        ["name", "cagr_full", "edge_full_pp", "sharpe", "max_dd",
         "wf_mean_cagr", "wf_mean_edge_pp", "wf_n_beats_spy"]
    ].round(3).to_string(index=False))

    print("\n=== Top 10 by walk-forward mean CAGR ===")
    print(cand.sort_values("wf_mean_cagr", ascending=False).head(10)[
        ["name", "cagr_full", "edge_full_pp", "wf_mean_cagr", "wf_min_cagr",
         "wf_mean_edge_pp", "wf_n_beats_spy", "max_dd"]
    ].round(3).to_string(index=False))

    print("\n=== Top 10 by edge over SPY (walk-forward) ===")
    print(cand.sort_values("wf_mean_edge_pp", ascending=False).head(10)[
        ["name", "cagr_full", "edge_full_pp", "wf_mean_cagr", "wf_mean_edge_pp",
         "wf_n_beats_spy", "max_dd"]
    ].round(3).to_string(index=False))

    # Composite winner: prefer wf_mean_cagr + edge + WF-stability
    cand["composite"] = (
        cand["wf_mean_cagr"] * 100
        + cand["wf_mean_edge_pp"]
        + (cand["wf_n_beats_spy"] >= 7) * 5  # bonus for beating SPY
        + (cand["wf_min_cagr"] > 0) * 5     # bonus for never losing
    )
    print("\n=== Top 10 by composite (WF mean CAGR + WF edge + stability bonus) ===")
    print(cand.sort_values("composite", ascending=False).head(10)[
        ["name", "cagr_full", "edge_full_pp", "wf_mean_cagr", "wf_mean_edge_pp",
         "wf_n_beats_spy", "wf_n_positive", "wf_min_cagr", "max_dd", "composite"]
    ].round(3).to_string(index=False))

    winner_row = cand.sort_values("composite", ascending=False).iloc[0]
    print(f"\n=== WINNER (composite) ===")
    print(json.dumps({k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                      for k, v in winner_row.to_dict().items()}, indent=2, default=str))
    (PIT / "sp500_pit_sweep_winner.json").write_text(
        json.dumps({k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                    for k, v in winner_row.to_dict().items()}, indent=2)
    )


if __name__ == "__main__":
    main()
