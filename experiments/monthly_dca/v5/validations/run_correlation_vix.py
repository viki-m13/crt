"""Novel overlays exploring correlations and VIX-proxy signals.

After multi-asset trend rotation succeeded (§12), test five more
proprietary ideas combining risk indicators:

J. VIX-proxy DRAWDOWN: scale sleeve weight by SPY's drawdown depth from
   52-week high. Deeper drawdown → more sleeve (defensive).

K. Realized-vol scaling: scale sleeve weight by SPY trailing 60-day
   annualised vol. Higher vol → more sleeve.

L. v5-SPY correlation-conditional sizing: compute rolling 6-month
   correlation of v5 strategy returns vs SPY. High corr → v5 is
   essentially long-beta → SHIFT to sleeve. Low corr → genuine alpha → v5.

M. Picker dispersion gate: when GBM cross-sectional score dispersion is
   high (clear winners), keep low sleeve. When dispersion is low
   (uncertain), raise sleeve.

N. Stacked risk-off score: combine drawdown + vol + breadth into a single
   continuous risk indicator. Sleeve weight = f(risk_score).
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

from experiments.monthly_dca.v5.validations.harness import (
    HarnessData, load_all, pick_v5_baseline, classify_regime_tight,
    COST_BPS, HOLD_MONTHS, K_PICKS,
)
from experiments.monthly_dca.v5.validations.run_advanced_overlays import (
    sector_top_n_sleeve, summary,
)
from experiments.monthly_dca.v2.ml_strategy import EXCLUDE

RES = Path(__file__).resolve().parent / "results"

SLEEVE_ASSETS = ["XLE","XLF","XLK","XLU","XLV","XLP","XLY","XLI","XLB",
                  "TLT","EFA","EEM"]


def precompute_risk_signals(daily_spy: pd.Series, asofs: list) -> pd.DataFrame:
    """Precompute, at each asof month-end:
    - drawdown from trailing 52-week high (positive number, e.g. 0.10 = 10% off high)
    - 60-day annualised realised vol
    - 21-day annualised realised vol
    """
    rolling_max = daily_spy.rolling(252, min_periods=20).max()
    drawdown = 1.0 - daily_spy / rolling_max
    ret = daily_spy.pct_change()
    rvol60 = ret.rolling(60, min_periods=30).std() * np.sqrt(252)
    rvol21 = ret.rolling(21, min_periods=15).std() * np.sqrt(252)
    rows = []
    for a in asofs:
        # Use prior-day values to avoid look-ahead with respect to month m's
        # return. We use a (month-end) — these features at month-end are known
        # at the start of next month, but we apply them to NEXT month's
        # decision (which is what we want).
        rows.append({
            "asof": a,
            "dd_52w": float(drawdown.loc[:a].iloc[-1]) if len(drawdown.loc[:a]) else 0.0,
            "rvol60": float(rvol60.loc[:a].iloc[-1]) if len(rvol60.loc[:a]) and pd.notna(rvol60.loc[:a].iloc[-1]) else 0.15,
            "rvol21": float(rvol21.loc[:a].iloc[-1]) if len(rvol21.loc[:a]) and pd.notna(rvol21.loc[:a].iloc[-1]) else 0.15,
        })
    return pd.DataFrame(rows).set_index("asof")


def run_dynamic(data: HarnessData, sleeve_weight_fn, name: str,
                  daily_spy: pd.Series, risk_df: pd.DataFrame,
                  start: pd.Timestamp, end: pd.Timestamp,
                  hold_months: int = HOLD_MONTHS,
                  cost_bps: float = COST_BPS) -> dict:
    """Run v5 strategy with a dynamic sleeve weight provided by
    sleeve_weight_fn(prior_month_state) -> weight in [0,1].
    """
    cf = cost_bps / 1e4
    asofs = [m for m in data.asofs
             if start <= m <= end
             and m in data.spy_features.index
             and m in data.mret.index
             and m in data.members_g]
    asofs = sorted(asofs)
    daily = pd.read_parquet(Path("experiments/monthly_dca/cache/prices_extended.parquet"))

    cur_picks: list[str] = []
    cur_weights = np.array([])
    cash = False; held = 0; equity = 1.0
    strat_ret_history = []
    spy_ret_history = []
    log = []

    for i, m in enumerate(asofs):
        spy_now = data.spy_features.loc[m].to_dict() if m in data.spy_features.index else {}
        regime = classify_regime_tight(spy_now)
        do_reb = (i == 0) or (held >= hold_months) or cash

        # Strategy return
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

        # Sleeve return (multi-asset top-2 by 12m mom)
        tops = sector_top_n_sleeve(daily, m, SLEEVE_ASSETS, n=2, lookback_d=252)
        sleeve_ret = 0.0
        if tops:
            rs = [float(data.mret.at[m, s])
                   for s in tops
                   if s in data.mret.columns and m in data.mret.index
                   and pd.notna(data.mret.at[m, s])]
            if rs: sleeve_ret = float(np.mean(rs))

        # Compute sleeve weight using PRIOR month's signals (no look-ahead)
        prev_m = asofs[i - 1] if i > 0 else m
        prev_state = {
            "regime": regime,
            "dd_52w": risk_df.loc[prev_m, "dd_52w"] if prev_m in risk_df.index else 0.0,
            "rvol60": risk_df.loc[prev_m, "rvol60"] if prev_m in risk_df.index else 0.15,
            "rvol21": risk_df.loc[prev_m, "rvol21"] if prev_m in risk_df.index else 0.15,
            "strat_ret_history": strat_ret_history,
            "spy_ret_history": spy_ret_history,
        }
        sw = sleeve_weight_fn(prev_state)
        sw = max(0.0, min(1.0, sw))

        ret_m = (1 - sw) * strat_ret + sw * sleeve_ret

        # Rebalance logic
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
                    if log: ret_m -= cf

        held += 1
        equity *= (1 + ret_m)
        # Track history for correlation-conditional rules
        strat_ret_history.append(strat_ret)
        spy_ret_m = float(data.mret.at[m, "SPY"]) if (m in data.mret.index and "SPY" in data.mret.columns and pd.notna(data.mret.at[m, "SPY"])) else 0.0
        spy_ret_history.append(spy_ret_m)
        log.append({"date": str(m.date()), "regime": regime,
                     "sleeve_w": sw,
                     "dd_52w": prev_state["dd_52w"],
                     "rvol60": prev_state["rvol60"],
                     "ret_m": ret_m, "equity": equity})

    return {"log": log}


def main():
    RES.mkdir(parents=True, exist_ok=True)
    data = load_all()
    spy = data.mret["SPY"].copy(); spy.index = pd.to_datetime(spy.index)
    daily = pd.read_parquet(Path("experiments/monthly_dca/cache/prices_extended.parquet"))
    daily_spy = daily["SPY"].dropna()
    asofs = sorted([m for m in data.asofs
                     if m in data.spy_features.index
                     and m in data.mret.index
                     and m in data.members_g])
    risk_df = precompute_risk_signals(daily_spy, asofs)
    start = asofs[0]; end = asofs[-1]

    # === Sleeve-weight schemes ===
    schemes = {
        # J: VIX-proxy via 52w drawdown
        "J_dd_linear":   lambda s: min(s["dd_52w"] * 3.0, 0.75),
        "J_dd_step":     lambda s: 0.20 if s["dd_52w"] < 0.05 else
                                     0.50 if s["dd_52w"] < 0.15 else
                                     0.75,
        "J_dd_combined": lambda s: min(0.20 + s["dd_52w"] * 2.5, 0.80),
        # K: realized vol scaling
        "K_rvol_step":   lambda s: 0.20 if s["rvol60"] < 0.15 else
                                     0.40 if s["rvol60"] < 0.25 else
                                     0.70,
        "K_rvol_linear": lambda s: min(s["rvol60"] * 2.0, 0.75),
        # L: v5-SPY correlation conditional
        "L_corr_simple": lambda s: 0.50 if len(s["strat_ret_history"]) < 6
                                    else (
                                        0.30 if np.corrcoef(s["strat_ret_history"][-12:], s["spy_ret_history"][-12:])[0,1] < 0.5
                                        else 0.60
                                    ) if len(s["strat_ret_history"]) >= 12 else 0.50,
        # N: stacked risk-off score
        "N_stacked":     lambda s: min(
                            0.30 + s["dd_52w"] * 1.5 + (s["rvol60"] - 0.15) * 1.0,
                            0.80) if s["dd_52w"] > 0 else 0.30,
        # Reference: fixed 50/50
        "REF_fixed_50":  lambda s: 0.50,
    }

    rows = []
    print(f"{'Scheme':<18} {'CAGR':>7} {'Sharpe':>7} {'MDD':>7} {'YrStd':>7} {'WrstYr':>7} {'2024':>7} {'AvgSw':>7}")
    for name, fn in schemes.items():
        try:
            sim = run_dynamic(data, fn, name, daily_spy, risk_df, start, end)
        except Exception as e:
            print(f"{name}: ERROR {e}")
            continue
        r = summary(sim["log"], spy)
        df = pd.DataFrame(sim["log"])
        avg_sw = float(df["sleeve_w"].mean())
        print(f"{name:<18} {r['cagr']:>6.2f}% {r['sharpe']:>7.2f} {r['mdd']:>6.1f}% {r['edge_std']:>5.1f}pp {r['edge_min']:>+5.1f}pp {r['e2024']:>+5.1f}pp {avg_sw:>6.2f}")
        rows.append({"scheme": name, **r, "avg_sleeve_w": avg_sw})
        df.to_csv(RES / f"{name}_equity.csv", index=False)

    pd.DataFrame(rows).to_csv(RES / "correlation_vix_summary.csv", index=False)


if __name__ == "__main__":
    main()
