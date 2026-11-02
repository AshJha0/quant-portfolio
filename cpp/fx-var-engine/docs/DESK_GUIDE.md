# Desk Guide

How a global FX risk function would run this engine: regional sweeps,
per-currency limits, the peg watchlist, backtests, and the regression gate
against the Python engine. Written for the risk manager who reads the
numbers. The engine handles cash, spot, forwards and European options
(Garman-Kohlhagen, repriced in every scenario; vol is held at today's
level unless the returns or covariance carry `VOL:CCY` columns and you pass
`VolScenarios`, in which case each option's vol moves with its own ATM vol
history under sticky strike, sticky delta or historical ATM); P&L explain stays in the Python engine.

## 1. Regional sweeps

FX has no single close. A typical schedule:

| Window (local close) | Run | Readers |
| --- | --- | --- |
| Tokyo, about 15:00 JST | APAC books: HS (plain and FHS), parametric, MC normal 100k, stress library | APAC risk officer |
| London, about 17:00 GMT | Global sweep; peg watchlist refresh; 10-day figures | EMEA risk, end-of-day limits |
| New York, about 17:00 EST | Official end of day: VaR/ES for the backtest series, Basel update | Head of market risk, regulatory reporting |

With the same seed, book and binary on the same platform, the three regions
get the same MC numbers. On a different libm the last bits can differ, so
pin the platform for the official run. The README lists measured
per-call latencies on the synthetic 250-position benchmark book.

**Latency-sensitive callers** (pre-trade limit checks, intraday what-if):
compile the book once (`CompiledBook`), allocate the scenario block, the
P&L vector and one selection buffer at start-up, then call
`try_pnl_batch` and `try_empirical_var_es` per request. Neither allocates
nor throws; a bad input comes back as an `fxvar::Error` whose `message()`
is the text the throwing API would have used.

**Intraday**: rerun `historical_var` and the stress library on each
material fill. Rerun the MC when exposures move past a tolerance.

**Missing fixings**: the engine refuses NaN and infinite values. The data
layer must forward-fill or drop dates explicitly, so the choice is
auditable.

## 2. Daily numbers

- **VaR 99 % 1-day (HS-FHS)**, alpha = 0.01: the headline and limit
  measure.
- **ES 97.5 % 1-day**, alpha = 0.025: watched for divergence from VaR,
  typical on books with EM shorts or peg exposure.
- **Exposure vector** (`ParametricResult::exposures`): P&L per unit factor
  move, the attribution traders act on.
- **Method disagreement**: HS vs parametric vs MC on the same book. A
  widening gap means fat tails, non-linearity or a broken history.
- **Stress ladder** (`run_stress`, worst first): canned replays, broad USD
  moves and peg-break add-ons.
- **MC standard error**: `se_var` (bootstrap) travels with every MC VaR.

## 3. Per-currency limits

The exposure vector prices a 1.00 log-return move per factor. Divide by 100
for the 1 % exposure most limit frameworks use.

- **Per-currency delta limits**, tighter for flagged pegs and EM.
- **Book VaR limit** with a soft-warning band, on the official method.
- **Stress limits**: worst canned-scenario loss. This limit binds on peg
  books.
- **IR leg limits** on forward books: exposure(IR:CCY) / 1e4 is the DV01.

## 4. Peg watchlist

The engine flags every `FX:*` factor with daily vol below 5e-4 in
`flagged_peg_factors` and `warnings`.

1. **Flag appears**: the currency joins the watchlist. Its HS and
   parametric contributions are treated as unreliable, not low.
2. **Stress add-on**: attach `peg_break_scenario(ccy, jump, contagion)`
   with direction and size from the peg's economics and any NDF market.
3. **MC overlay**: run `McDist::kJump` with `prob` as the per-day jump
   probability. The engine turns it into a Poisson count over the horizon,
   so a 10-day run sees about 10 times the jump risk of a 1-day run. On a
   pegged book the jump MC reports more than 20x the HS VaR.
4. **Exit** only by governance decision.

## 5. Backtesting

- **Daily**: `evaluate_var_backtest(pnl, var, alpha)` gives the exception
  series, Kupiec, Christoffersen, conditional coverage and the Basel light.
  alpha is the tail probability (0.01).
- **Reading the tests**: Kupiec failing high means too many exceptions.
  Christoffersen failing with Kupiec passing means clustered exceptions;
  move to FHS.
- **Traffic light**: at 250 days and 1 %, yellow is 5-9 exceptions and red
  10+. Other windows (a shortened history after a migration) zone by the
  binomial CDF and take the plus factor from the 250-day table at the same
  CDF.
- **ES**: `acerbi_szekely_z2(pnl, var, es, alpha)` below 0 means ES is
  understated on exception days, a peg-blindness signature even when VaR is
  green.

## 6. Reverse stress

`reverse_stress_for_loss(exposures, cov, loss_target)` returns the most
plausible move (fixed Mahalanobis distance) that costs the target. It is
the linear worst case on the estimated covariance. For peg books use the
peg-break scenario, since the covariance cannot represent the break.

## 7. Gate against the Python engine

- **Golden gate (CI)**: `tests/test_golden_python.cpp` pins historical,
  parametric, backtest and shared-convention values to the
  Python-generated header. A methodology change regenerates the goldens in
  the same change.
- **Method parity**: triangulation, CIP legs, ES >= VaR and Basel
  boundaries are tested in both suites.
- **Scope**: route only linear books here. Unknown factors in a book throw;
  scenario shocks for factors a book lacks are ignored by convention.
- **Audit replay**: seed, code version, book snapshot and platform
  determine every MC number.
