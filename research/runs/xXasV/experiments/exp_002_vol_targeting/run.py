"""
Exp 002: Volatility Targeting.

Take the v3 baseline (K=3, h=6m, tight) and add volatility targeting:
At each rebalance, compute realized portfolio vol from recent monthly returns.
Scale full equity exposure = vol_target / realized_vol (capped at 1.0).
Remaining capital earns T-bill rate.

Test vol_target ∈ {0.10, 0.12, 0.15, 0.18, 0.20, 0.25}
and lookback ∈ {3, 6, 12} months.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backtest"))
from engine import (
    BacktestConfig, get_monthly_returns, get_spy_features,
    load_pit_scores_panel, get_regime_at,
    compute_turnover_cost, summary_str
)

REPO = Path("/home/user/crt")
QR = REPO / "quant_research"
EXP = QR / "experiments" / "exp_002_vol_targeting"
JOURNAL = QR / "state" / "journal.jsonl"
HYP_LOG = QR / "state" / "hypotheses_tested.jsonl"


def score_ml_3plus6(grp: pd.DataFrame) -> pd.Series:
    df = grp.set_index("ticker")
    return (df["pred_3m"] + df["pred_6m"]) / 2.0


def run_vol_targeted_backtest(
    panel: pd.DataFrame,
    K: int = 3,
    hold_months: int = 6,
    vol_target: float = 0.15,
    lookback_months: int = 6,
    cost_bps: float = 5.0,
    cash_yield: float = 0.04,
    start: str = "2003-09-30",
    end: str = "2024-04-30",
) -> dict:
    """
    Walk-forward backtest with volatility targeting overlay.
    Position sizing: exposure = min(1.0, vol_target / realized_vol).
    Remainder in cash at T-bill rate.
    """
    rets = get_monthly_returns()
    spy_feats = get_spy_features()

    asofs = sorted(panel["asof"].unique())
    asofs = [a for a in asofs if pd.Timestamp(start) <= a <= pd.Timestamp(end)]

    monthly_rets = []
    dates = []
    portfolio_weights: dict = {}
    held_for = 0
    cur_w: dict = {}
    in_cash = False
    realized_returns = []   # running list of gross portfolio monthly returns

    for i, asof in enumerate(asofs):
        regime = get_regime_at(asof, spy_feats, "tight")
        is_crash = (regime == "crash")
        do_rebalance = (i == 0) or (held_for >= hold_months) or in_cash

        if do_rebalance:
            if is_crash:
                new_w = {}
                in_cash = True
            else:
                grp = panel[panel["asof"] == asof].copy()
                scores = score_ml_3plus6(grp).dropna()
                if len(scores) == 0:
                    new_w = {}
                    in_cash = True
                else:
                    top = scores.nlargest(K)
                    tickers = top.index.tolist()
                    new_w = {t: 1.0 / len(tickers) for t in tickers}
                    in_cash = False

            cost = compute_turnover_cost(portfolio_weights, new_w, cost_bps)
            cur_w = new_w
            portfolio_weights = new_w
            held_for = 1
        else:
            cost = 0.0
            held_for += 1

        # Compute realized vol from recent portfolio returns
        if len(realized_returns) >= lookback_months:
            recent = np.array(realized_returns[-lookback_months:])
            realized_vol_monthly = float(np.std(recent, ddof=1))
            realized_vol_annual = realized_vol_monthly * np.sqrt(12)
        else:
            realized_vol_annual = 0.20  # assume 20% for first few months

        # Compute exposure scaling
        if realized_vol_annual <= 0:
            exposure = 1.0
        else:
            exposure = min(1.0, vol_target / realized_vol_annual)

        # Compute gross portfolio return for NEXT month
        pos = rets.index.searchsorted(asof)
        snap_pos = None
        for cp in [pos - 1, pos]:
            if 0 <= cp < len(rets.index) and abs((rets.index[cp] - asof).days) <= 7:
                snap_pos = cp
                break

        if snap_pos is None or snap_pos + 1 >= len(rets.index):
            port_gross = 0.0
        elif in_cash and len(cur_w) == 0:
            port_gross = cash_yield / 12.0
        else:
            nm = rets.index[snap_pos + 1]
            equity_ret = sum(
                w * float(rets.loc[nm, t])
                for t, w in cur_w.items()
                if t in rets.columns and not pd.isna(rets.loc[nm, t])
            )
            cash_portion = 1.0 - exposure
            port_gross = exposure * equity_ret + cash_portion * (cash_yield / 12.0)

        port_net = port_gross - cost
        monthly_rets.append(port_net)
        dates.append(asof)
        realized_returns.append(port_gross)  # track gross for vol estimation

    ret_series = pd.Series(monthly_rets, index=pd.DatetimeIndex(dates),
                           name=f"vt{vol_target:.2f}_lb{lookback_months}")
    equity = (1 + ret_series).cumprod()
    n = len(ret_series)

    if n < 12:
        return {"cagr": 0.0, "sharpe": 0.0, "maxdd": 0.0, "n": n}

    cagr = float(equity.iloc[-1] ** (12.0 / n) - 1)
    excess = ret_series - (cash_yield / 12.0)
    sharpe = float(excess.mean() / excess.std() * np.sqrt(12)) if excess.std() > 0 else 0.0
    roll_max = equity.cummax()
    maxdd = float((equity / roll_max - 1).min())

    chunk = n // 3
    sub_sharpes = []
    for k in range(3):
        sl = ret_series.iloc[k * chunk:(k + 1) * chunk]
        ex = sl - (cash_yield / 12.0)
        ss = float(ex.mean() / ex.std() * np.sqrt(12)) if ex.std() > 0 else 0.0
        sub_sharpes.append(round(ss, 3))

    return {
        "vol_target": vol_target,
        "lookback": lookback_months,
        "cagr": cagr, "sharpe": sharpe, "maxdd": maxdd,
        "n": n, "sub_sharpes": sub_sharpes,
        "returns": ret_series, "equity": equity,
    }


def run_all():
    print("Loading PIT scores panel...")
    panel = load_pit_scores_panel()

    vol_targets = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    lookbacks = [3, 6, 12]
    results = []

    print(f"Running {len(vol_targets) * len(lookbacks)} vol-targeting configs...")
    for vt in vol_targets:
        for lb in lookbacks:
            r = run_vol_targeted_backtest(panel, vol_target=vt, lookback_months=lb)
            results.append(r)
            print(f"  vt={vt:.2f} lb={lb}: CAGR={r['cagr']:.1%} Sharpe={r['sharpe']:.3f} "
                  f"MaxDD={r['maxdd']:.1%} Sub={r['sub_sharpes']}")

    # Save results
    summary = [
        {k: v for k, v in r.items() if k not in ("returns", "equity")}
        for r in results
    ]
    with open(EXP / "results.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # Log
    n_hparams = len(vol_targets) * len(lookbacks)
    with open(HYP_LOG, "a") as fh:
        fh.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "n_hparams": n_hparams,
            "description": "exp_002 vol targeting: 6 vol_targets × 3 lookbacks",
        }) + "\n")
    with open(JOURNAL, "a") as fh:
        fh.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "exp_id": "exp_002_vol_targeting",
            "hypothesis": "Volatility targeting: scale equity exposure to achieve target annual vol",
            "what_i_did": f"Tested {n_hparams} combos on v3 baseline (K=3, h=6m, tight)",
            "result": summary,
            "hparams_tried": n_hparams,
            "next_action": "Pick best Sharpe result; if Sharpe > 1.5 then test K/h sensitivity",
        }) + "\n")

    # Find best by Sharpe
    best = max(results, key=lambda r: r["sharpe"])
    best_cagr = max(results, key=lambda r: r["cagr"])
    print(f"\nBest by Sharpe: vt={best['vol_target']:.2f} lb={best['lookback']} "
          f"CAGR={best['cagr']:.1%} Sharpe={best['sharpe']:.3f} MaxDD={best['maxdd']:.1%}")
    print(f"Best by CAGR: vt={best_cagr['vol_target']:.2f} lb={best_cagr['lookback']} "
          f"CAGR={best_cagr['cagr']:.1%} Sharpe={best_cagr['sharpe']:.3f}")

    return results


if __name__ == "__main__":
    run_all()
