"""Self-similarity / pattern matching score.

For each (asof, ticker), compute a 252-day normalized log-return path.
Compare against template paths from known multibaggers at their start-of-vertical
moments.  Score = max similarity to any template.

Templates:
  NVDA 2014-12-31 (start of 2015-2017 run)
  NVDA 2022-09-30 (start of AI boom)
  AAPL 2003-04-30 (start of iPod/iPhone era)
  AMZN 1998-01-31 (start of dot-com-era run)
  TSLA 2019-09-30 (start of multi-bagger run)
  NFLX 2009-04-30 (start of streaming dominance)
  AAPL 2009-01-30 (post-GFC iPhone era)
  AMD 2016-01-29 (start of Lisa Su turnaround)

Score = exp(-mean_squared_distance).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "experiments" / "monthly_dca" / "cache"
V2 = CACHE / "v2"
PIT = V2 / "sp500_pit"

WINDOW_DAYS = 252  # 1 year of trading days

TEMPLATES = [
    ("NVDA", "2014-12-31"),
    ("NVDA", "2022-09-30"),
    ("AAPL", "2003-04-30"),
    ("AMZN", "1999-01-29"),  # earliest available
    ("TSLA", "2019-09-30"),
    ("NFLX", "2009-04-30"),
    ("AAPL", "2009-01-30"),
    ("AMD",  "2016-01-29"),
]


def load_daily():
    return pd.read_parquet(CACHE / "prices_extended.parquet")


def normalized_log_path(prices: pd.Series, end_date: pd.Timestamp, window=WINDOW_DAYS):
    """Return a (window,) array of normalized log returns ending at end_date."""
    pos = prices.index.searchsorted(end_date, side="right") - 1
    if pos < 0 or pos < window:
        return None
    sub = prices.iloc[pos - window + 1: pos + 1]
    if sub.isna().any() or len(sub) < window:
        return None
    log_ret = np.log(sub.values / sub.values[0])
    return log_ret.astype(np.float32)


def build_templates(daily):
    paths = []
    for tk, d in TEMPLATES:
        if tk not in daily.columns:
            continue
        path = normalized_log_path(daily[tk].dropna(), pd.Timestamp(d))
        if path is not None:
            paths.append((tk, d, path))
            print(f"  template {tk} {d}: end log-ret = {path[-1]:.3f} (= {np.exp(path[-1])-1:+.0%})")
    return paths


def score_panel(asof: pd.Timestamp, tickers: list, daily, templates):
    """For each ticker at asof, compute max similarity to any template."""
    scores = {}
    for tk in tickers:
        if tk not in daily.columns:
            scores[tk] = np.nan; continue
        path = normalized_log_path(daily[tk].dropna(), asof)
        if path is None:
            scores[tk] = np.nan; continue
        # Compare to each template
        sims = []
        for _, _, tpath in templates:
            mse = np.mean((path - tpath) ** 2)
            # Use shape-similarity (not absolute level): also align both to end at 0 then re-comp
            sims.append(np.exp(-mse))
        scores[tk] = float(max(sims))
    return scores


def main():
    print("=== Pattern similarity score ===", flush=True)
    daily = load_daily()
    templates = build_templates(daily)
    if len(templates) == 0:
        print("  No templates loaded — abort"); return
    print(f"  {len(templates)} templates loaded", flush=True)

    # PIT panel asofs
    panel = pd.read_parquet(PIT / "sp500_pit_panel.parquet")
    panel["asof"] = pd.to_datetime(panel["asof"])
    asofs = sorted(panel["asof"].unique())

    rows = []
    for i, asof in enumerate(asofs):
        sub = panel[panel["asof"] == asof]
        scores = score_panel(asof, sub["ticker"].tolist(), daily, templates)
        for tk, sc in scores.items():
            rows.append({"asof": asof, "ticker": tk, "pattern_sim": sc})
        if i % 30 == 0:
            print(f"  asof {asof.date()} done; n_scored={sum(1 for s in scores.values() if not pd.isna(s))}",
                  flush=True)

    out = pd.DataFrame(rows)
    out.to_parquet(PIT / "ml_preds_pattern_sim.parquet", index=False)
    print(f"  saved {out.shape} to {PIT}/ml_preds_pattern_sim.parquet")


if __name__ == "__main__":
    main()
