# PIT (point-in-time) data archive

Consolidated copy of the point-in-time universe/membership data found across the
account's repositories. **Everything here is a COPY — no source file was moved or
removed.** Original locations are listed so the provenance stays traceable.

Why this exists: PIT membership is what makes a cross-sectional backtest
survivorship-free. These files include delisted tickers (e.g. AAMRQ, ENRNQ), so
a universe reconstructed from them contains companies that later failed — which
is the whole point. Scattering them across repos invites accidentally using a
survivorship-biased universe.

## sp500/
| file | source | notes |
|---|---|---|
| `sp500_pit_members__vol_research-data.csv` | `vol` @ `origin/claude/research-data:data/sp500_constituents/sp500_pit_members.csv` | daily rows `date,tickers` (comma-joined). Spans 1996-01-02 onward. ~5.5 MB |
| `sp500_pit_membership__bonds.csv` | `bonds/data/pit/sp500_pit_membership.csv` | independent build, same purpose. ~5.5 MB |

Two independent S&P 500 PIT builds are kept deliberately — cross-checking them
against each other is a cheap guard against a bad membership file.

## ndx/
| file | source |
|---|---|
| `ndx_pit_members__vol_research-data.csv` | `vol` @ `origin/claude/research-data:data/sp500_constituents/ndx_pit_members.csv` |
| `n100_pit_membership__bonds.csv` | `bonds/data/pit/n100_pit_membership.csv` |
| `n100_panel_member__bonds.parquet` | `bonds/data/pit/n100_panel_member.parquet` (membership matrix) |
| `ndx_pit_membership_monthly__crt.parquet` | `crt/experiments/monthly_dca/v5/qqq_pit/` |
| `ndx_pit_membership_monthly_full__crt.parquet` | `crt/experiments/monthly_dca/v5/qqq_pit/` |
| `n100_raw/changes-*.yaml` (12 files) | `vol` @ `.../sp500_constituents/n100_raw/` — raw index add/delete change logs |
| `n100_panels__bonds/n100_panel_{close,high,low,open,volume}.parquet` | `bonds/data/pit/` — PIT-aligned OHLCV panels |

## universe/
| file | source |
|---|---|
| `tiingo_universe_pit__bonds.parquet` | `bonds/dca/research/data/tiingo/tiingo_universe_pit.parquet` |
| `coverage__bonds.json`, `broad_coverage__bonds.json`, `sectors__bonds.json` | `bonds/data/pit/` |

## Deliberately NOT copied (large, and not membership data)
Left in place, noted here so they can be found:
- `bonds/data/pit/summit_panel.parquet` (48 MB) — price panel, not PIT membership
- `bonds/data/pit/prices_n100/` (5.4 MB, 39 files) — per-ticker prices

## Related PIT tooling already in this repo (not moved)
- `crt/research/validation/sp500_pit/run_pit_sp500_validation.py`
- `crt/research/validation/ndx_pit/run_pit_ndx_validation.py`
- `crt/experiments/monthly_dca/v5/spx_pit/build_sp500_pit_prices.py`
- `crt/tests/YLOka/test_pit_membership.py`
