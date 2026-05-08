"""No-pyramiding backtest: at each month-end, exclude any ticker that we
already hold (because we bought it in a prior month). This forces
diversification across DIFFERENT names rather than re-buying high-conviction
winners month after month.

Compares:
  - WITH pyramiding (default behavior, can re-buy same name): top-K picks
    from raw scores
  - WITHOUT pyramiding: at each month-end, drop tickers in our basket; pick
    top-K from remaining

Holding rule: 3-year fixed exit. Once a position is older than 3 years, it's
not "held" anymore so it CAN be re-bought.
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
from experiments.monthly_dca.strategies_fast import blended_pullback_momentum

CACHE = Path(__file__).resolve().parent / "cache"


def run_strategy(
    score_fn,
    top_k: int,
    no_pyramid: bool = False,
    hold_years: float = 3.0,
    start: str = "2002-01-31",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """Generate picks. If no_pyramid=True, exclude tickers in our active basket."""
    feats = load_features_long()
    panel = load_panel()
    eval_at = panel.index.max()
    asofs = sorted(feats.index.get_level_values("asof").unique())
    asofs = [a for a in asofs if pd.Timestamp(start) <= a <= pd.Timestamp(end)]

    held: list[tuple[pd.Timestamp, str]] = []  # (asof, ticker) of currently-held positions
    pick_rows = []
    hold_td = pd.Timedelta(days=int(hold_years * 365.25))

    for asof in asofs:
        # Drop expired holdings
        held = [(a, t) for (a, t) in held if (asof - a) < hold_td]
        held_tickers = {t for _, t in held}

        sub = feats.loc[asof].copy()
        if hasattr(sub.index, "get_level_values"):
            sub.index = sub.index.get_level_values("ticker")
        scores = score_fn(sub).dropna()
        scores = scores[~scores.index.isin(BENCH_EXCLUDED)]
        if no_pyramid:
            scores = scores[~scores.index.isin(held_tickers)]
        if scores.empty:
            continue
        top = scores.sort_values(ascending=False).head(top_k)
        for tkr, sc in top.items():
            entry_px = float(sub.loc[tkr, "price"]) if tkr in sub.index and "price" in sub.columns else float("nan")
            pick_rows.append({"asof": asof, "ticker": tkr, "score": float(sc), "entry_px": entry_px})
            held.append((asof, tkr))
    return pd.DataFrame(pick_rows)


def evaluate(picks: pd.DataFrame, hold_years: float = 3.0) -> dict:
    """Compute summary stats for a pick set."""
    panel = load_panel()
    eval_at = panel.index.max()
    spy = panel["SPY"]

    rets = []
    spy_rets = []
    asofs = []
    for _, r in picks.iterrows():
        asof_t = pd.Timestamp(r["asof"])
        tkr = r["ticker"]
        entry = float(r["entry_px"])
        scheduled_exit = asof_t + pd.Timedelta(days=int(hold_years * 365.25))
        eval_date = min(scheduled_exit, eval_at)
        if tkr not in panel.columns or not np.isfinite(entry) or entry == 0:
            continue
        s = panel[tkr].loc[panel.index <= eval_date].dropna()
        if s.empty:
            continue
        out_px = float(s.iloc[-1])
        ret = out_px / entry - 1.0
        rets.append(ret)
        asofs.append(asof_t)
        # SPY same window
        pos = panel.index.searchsorted(asof_t)
        spy_e = float(spy.iloc[pos])
        spy_eval_pos = panel.index.searchsorted(eval_date, side="right") - 1
        spy_eval = float(spy.iloc[spy_eval_pos])
        spy_rets.append(spy_eval / spy_e - 1.0)

    if not rets:
        return {}
    rets = np.array(rets)
    spy_rets = np.array(spy_rets)
    cf = [(t, -1.0) for t in asofs] + [(eval_at, float(np.sum(1 + rets)))]
    cf_spy = [(t, -1.0) for t in asofs] + [(eval_at, float(np.sum(1 + spy_rets)))]
    cagr = xirr(cf)
    cagr_spy = xirr(cf_spy)
    return {
        "n_picks": len(rets),
        "win_rate": float((rets > 0).mean()),
        "beat_spy_rate": float((rets > spy_rets).mean()),
        "median_ret": float(np.median(rets)),
        "cagr_dca_portfolio": float(cagr),
        "cagr_spy_dca": float(cagr_spy),
        "edge_vs_spy_dca": float(cagr - cagr_spy),
    }


def main() -> None:
    print("=== Comparing pyramiding vs no-pyramiding ===\n")

    results = {}

    for top_k in (1, 5, 10):
        for no_pyramid in (False, True):
            label = f"k={top_k}_{'no_pyramid' if no_pyramid else 'pyramid_ok'}"
            print(f"Running {label}...")
            picks = run_strategy(blended_pullback_momentum, top_k=top_k,
                                  no_pyramid=no_pyramid, hold_years=3.0)
            stats = evaluate(picks, hold_years=3.0)
            stats["label"] = label
            stats["top_k"] = top_k
            stats["no_pyramid"] = no_pyramid
            stats["unique_tickers"] = int(picks["ticker"].nunique())
            print(f"  n_picks={stats.get('n_picks')} unique_tickers={stats['unique_tickers']} "
                  f"CAGR={stats.get('cagr_dca_portfolio', 0):.3f} edge={stats.get('edge_vs_spy_dca', 0):+.3f} "
                  f"win={stats.get('win_rate', 0):.3f}")
            results[label] = stats
            picks.to_csv(CACHE / f"picks_no_pyramid_{label}.csv", index=False)

    with open(CACHE / "no_pyramid_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {CACHE/'no_pyramid_summary.json'}")


if __name__ == "__main__":
    main()
