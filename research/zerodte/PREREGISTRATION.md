# 0DTE forward test — pre-registration

Registered 2026-08-11, BEFORE any collected data existed. Rules below are
frozen; changing them after seeing results voids the test (a change requires
a new registration section with a new start date, and the old rules must
still be reported).

## Why forward, not backward

No free historical intraday 0DTE data exists (checked 2026-08-11: free EOD
archives end at 2013 — daily SPX/SPY expirations began in late 2022; intraday
history is paid-only). Rather than backtest a model of 0DTE quotes — which
this project's standards reject — we record real delayed quotes going forward
(collect.py, 5 snapshots/day) and score frozen rules against them.

## Instruments

SPX 0DTE options (European, cash-settled at the 4:00pm ET close — no
assignment risk). SPY recorded too, for the toll comparison only.

## Strategy S1 — "morning richness condor" (short premium, defined risk)

Thesis from the published record: the same-day variance premium is
concentrated early in the session (Almeida–Freire–Hizmeri; Dim–Eraker–Vilkov
document morning sellers earning the bulk of 0DTE premium).

- Entry: first snapshot at/after 13:30Z (~9:30-9:45am ET quote given delay).
- Structure: iron condor, today's expiration. Short call and short put at the
  strikes nearest |delta| = 0.15 on each side; long wings 0.30% of spot
  further out (nearest strike).
- Fills: worst side — shorts at bid, longs at ask, from the snapshot.
- Exit: none. Cash settlement vs the official SPX close (^GSPC).
- Skip day if: any leg has bid=0 or ask missing, or net credit <= 0, or the
  four strikes are not strictly ordered.

## Strategy S2 — "last-hour momentum ride" (long premium, defined risk)

Thesis: market intraday momentum — the first part of the day predicts the
last half hour (Gao, Han, Li, Zhou 2018; documented in SPY at the half-hour
grid).

- Signal: r = spot(19:00Z snapshot) / spot(13:30Z snapshot) − 1.
- Entry: at the 19:00Z snapshot (~2:45-3pm ET quotes), only if |r| >= 0.30%.
- Structure: 0DTE vertical debit spread in the direction of r. Long the
  strike nearest spot (ATM), short the strike ~0.30% of spot further in the
  direction of r. Calls if r > 0, puts if r < 0.
- Fills: worst side — long at ask, short at bid.
- Exit: none. Cash settlement vs the official SPX close.
- Skip day if quotes missing/crossed or debit <= 0 or debit >= strike gap.

## Sizing basis

Both scored per 1 spread, P&L as a fraction of max loss (S1: width − credit;
S2: debit). No compounding during the test window — the question is per-trade
edge, not path.

## Pre-stated evaluation

- Minimum sample: 60 trading days with valid entries before ANY conclusion.
- Success bar (per strategy): mean P&L/maxloss > 0 with block-bootstrap 90%
  CI excluding 0, AND worst day > −1.0 (no undefined risk taken).
- Kill criteria: after 60 valid days, CI includes 0 → strategy rejected,
  logged, not resubmitted with tweaked parameters.
- The delayed-quote caveat is symmetric (entries AND marks are delayed) and
  is reported alongside results, not adjusted away.

## Trial accounting

This registration adds exactly 2 trials to the project count. Parameter
values (0.15 delta, 0.30% wing/gap, 0.30% signal threshold) were fixed from
the cited literature before data collection; no sweeps were or will be run
on the forward record.
