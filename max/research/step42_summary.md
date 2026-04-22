# Step 42 — Volatility-adaptive TP grid

Universe 128 tickers (SPY excluded), 2006-04-25 → 2026-04-21. Ranker CAP5+SMA12M, monthly top-1, enter next close. Exit: first High ≥ TP else last-bar close. $1000/mo DCA, **cash idle between trades** (per spec).

## Reference

| strat | CAGR | MDD | WR | N |
|---|---|---|---|---|
| SPY DCA | **6.63%** | 35.9% | — | 241 |
| Fixed 5% × 60d | 0.01% | — | 79.4% | 228 |
| Prod CAP5+SMA12M hold-forever | +18.31% | — | — | — |

Every "pick/TP/idle" variant trails SPY DCA by design: ~$1k works for a few months per cycle, cash accumulates.

## Top-5 per formula by CAGR

**ATR** (TP = entry · (1 + k·ATR14/entry))

| combo | CAGR | WR | MDD | Calmar |
|---|---|---|---|---|
| k=5.0 ts=252 | 0.32% | 70.2% | 14.0% | 0.023 |
| k=4.0 ts=252 | 0.30% | 77.6% | 11.0% | 0.027 |
| k=3.0 ts=252 | 0.27% | 85.1% |  7.9% | 0.034 |
| k=2.5 ts=252 | 0.24% | 88.6% |  4.8% | 0.050 |
| k=4.0 ts=126 | 0.22% | 68.9% |  5.5% | 0.040 |

**Sigma** (TP = entry · (1 + k·σ60·√ts))

| combo | CAGR | WR | MDD |
|---|---|---|---|
| k=2.5 ts=252 | 1.02% | 10.5% | 24.1% |
| k=2.0 ts=252 | 0.92% | 13.2% | 24.1% |
| k=1.5 ts=252 | 0.81% | 24.1% | 24.1% |
| k=1.0 ts=252 | 0.61% | 39.0% | 24.1% |
| k=2.5 ts=126 | 0.43% |  5.7% | 11.0% |

Top-sigma rows are degenerate — TP almost never fires, so they collapse to a 1-yr hold.

**Quantile** (TP = entry · (1 + Q%-quantile of 3Y historical ts-bar fwd rets))

| combo | CAGR | WR | MDD | Calmar |
|---|---|---|---|---|
| q=80 ts=252 | 0.77% | 75.5% | 4.0% | **0.193** |
| q=70 ts=252 | 0.50% | 84.0% | 2.6% | 0.189 |
| q=80 ts=126 | 0.36% | 67.5% | 2.7% | 0.131 |
| q=60 ts=252 | 0.25% | 89.7% | 1.4% | 0.181 |
| q=70 ts=126 | 0.23% | 79.4% | 2.2% | 0.106 |

## Best by win-rate
- ATR k=1.0 ts=252 → WR **96.5%**, CAGR 0.10%, σ_tkWR 0.059 (most equalized)
- Sigma k=0.5 ts=252 → WR 68.0%, CAGR 0.41%
- Quantile q=50 ts=21 → WR **95.7%**, CAGR 0.01%, only 69 trades

## Head-to-head

| | CAGR | WR | Calmar |
|---|---|---|---|
| Fixed 5% × 60d (agent-2 ref) | 0.01% | 79.4% | — |
| ATR best | 0.32% | 70.2% | 0.023 |
| Sigma best | 1.02%\* | 10.5% | 0.043 |
| **Quantile best** | **0.77%** | 75.5% | **0.193** |

Vol-adaptive edges fixed-5%; none touch SPY DCA.

## Recommendation

**Quantile q=80 × ts=252.** Best Calmar (~8× ATR winner), sensible 75.5% WR, 4% MDD. Self-calibrating per ticker — no k to tune — and avoids the "TP never fires" degeneracy of sigma.

## Sanity

**Does TP scaling equalize WR across low-vol (KO, PG) vs high-vol (TSLA, SMCI, MARA)?** **Mostly moot** — CAP5+SMA12M top-1 lands on a ~27-ticker subset (AMD 23×, NEM 18×, NFLX 16×, GE 14×, FCX 13×…). **KO, PG, TSLA, MARA are never picked** across 228 months; only SMCI appears (5×, 80% WR under ATR k=2.5 ts=252). The ranker rejects both vol extremes. Inside the realized set, per-ticker WR at ATR k=2.5 ts=252 spans 66.7% (STZ) → 100% (CDNS, NFLX, MU, GILD, SPG, WFC, VRTX…). σ_tkWR minimum overall is **ATR k=1 ts=252 = 0.059** (near-uniform).

**Discipline gates pass:**
- k=1 ATR ts=21 WR 72.4% > k=3 ts=21 WR 39.5%. Monotone.
- ATR14/entry ≈ 2-3% → k=2 × 63d ≈ 5% TP, matches fixed 5% × 63d.
- σ60·√63 at 20% ann vol ≈ 10% → k=2 × 63d TP ≈ 20%, consistent with 8.8% WR.

## Observation

The ranker concentrates in mid-vol cyclicals + large-cap tech. Vol-adaptive TP is therefore solving the wrong problem — the ranker already pre-selected a narrow vol band, so differentiation by ATR/σ barely matters. If future work expands the pickable universe (or uses a dispersion-aware ranker), vol-adaptive TP's equalization property becomes more relevant.

## Artifacts
- `max/research/step42_atr_tp.py`
- `max/research/step42_results.json`
