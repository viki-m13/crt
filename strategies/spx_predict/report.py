"""Emit results/spx_forecast.json — today's forecast + the validated
track record — the single artifact a webapp/cron would consume."""
from __future__ import annotations
import json, os
from predict import load_panel, current_forecast, RESULTS
import backtest as bt


def main():
    p = load_panel()
    fc = current_forecast(p)
    out = {
        "as_of": fc["as_of"],
        "spot": fc["spot"],
        "iv_proxy": fc["iv_proxy"],
        "forecast": fc["calls"],
        "track_record": {
            "calibration": bt.calibration(p),
            "nobreach": bt.nobreach_track(p),
            "direction": bt.direction_track(p),
            "misses_by_year": bt.misses_by_year(p),
        },
        "span": [str(p.dates[0]), str(p.dates[-1])],
        "note": ("Physical = point-in-time regime-conditioned historical "
                 "probability; market = BS N(d2) with IV=realized*1.12. "
                 "See VALIDATION.md."),
    }
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "spx_forecast.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {path}")
    print(f"as_of={out['as_of']} spot={out['spot']:.2f} "
          f"forecast_calls={len(out['forecast'])}")


if __name__ == "__main__":
    main()
