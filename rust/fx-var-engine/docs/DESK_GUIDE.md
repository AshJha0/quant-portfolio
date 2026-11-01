# Desk Guide

How a global FX risk function would run this engine day to day: regional
sweeps, per-currency limits, the peg watchlist, backtest governance, and the
regression gates against the Python package and the C++ engine. Written for
the risk manager reading the numbers.

All calls take `alpha` as the tail probability (`0.01` = 99 % VaR). Code or
configuration written for releases before 0.2.0 passed `0.99`; that now
returns an error, so stale configs fail loudly instead of reporting a 1 %
VaR.

## 1. Regional sweeps

| Window (local close) | Run | Consumers |
| --- | --- | --- |
| Tokyo ~15:00 JST | APAC books: HS (plain and FHS), parametric, MC normal 100k, stress library | APAC risk; handover to London |
| London ~17:00 GMT | Global sweep; peg watchlist refresh; 10-day figures for capital | EMEA risk; limit reports |
| New York ~17:00 EST | Official global EOD: VaR/ES for the backtest series, Basel update | Head of market risk; regulatory reporting |

Each engine is deterministic for a fixed seed on the same build, so a rerun
of the same book gives the same number. HS, parametric and stress cost a few
milliseconds per book on the 250-position bench book; one MC run with its
bootstrap SE costs about 0.35 s to 0.45 s at 100k scenarios on
a 12-thread laptop. The result is the same for any thread count. Data
readiness, not compute, sets the schedule.

**Intraday**: rerun `historical_var` and the stress library on material
fills; rerun MC when exposures move beyond a tolerance.

**Missing fixings**: the engine refuses NaN rather than imputing. The data
layer must forward-fill or drop the date explicitly.

## 2. The daily numbers

- **VaR 99 % 1d (FHS)**: the headline and limit measure.
- **ES 97.5 % 1d** (`alpha = 0.025`): watched for divergence from VaR.
- **MC standard error**: `MonteCarloResult::se_var` is the bootstrap SE;
  `se_var_kde` is a diagnostic. Publish the SE with the MC number.
- **Exposure vector** (`ParametricResult::exposures`): the per-factor P&L
  per unit log move, the attribution the trader acts on.
- **Method disagreement**: HS vs parametric vs MC on the same book. A
  widening gap means fat tails, non-linearity or a broken history.
- **Stress ladder** (`run_stress`, worst first): Brexit GBP, CHF depeg, JPY
  1998, broad USD moves, peg-break add-ons.

## 3. Per-currency limits

- Per-currency delta limits on `|exposure(FX:CCY)| / 100` (1 % move), tighter
  for flagged pegs and EM.
- Book VaR limit with a warning band, checked at every sweep.
- Stress limits: the one that binds on peg books.
- IR legs: `exposure(IR:CCY) / 1e4` is the DV01 of the forward book.

## 4. Peg watchlist

The engine flags every `FX:*` factor with realised daily vol below 5e-4 in
`flagged_peg_factors` and `warnings` (`returns::PEG_VOL_THRESHOLD`).

1. A flag puts the currency on the watchlist. Its HS and parametric VaR
   contribution is unreliable-low, not low.
2. Attach `stress::peg_break_scenario(ccy, jump, &contagion)` to the book's
   stress set.
3. Run the jump MC (`McDist::Jump`). `jumps.prob` is a **per-day**
   probability; the engine turns it into a Poisson count over the horizon,
   so a 10-day run carries about ten times the 1-day jump frequency. The
   overlay is uncompensated: it moves the mean as well as the tail.
   On a pegged test book the jump MC reports more than 20x the HS VaR
   (`edge_cases::peg_break_jump_produces_loss_historical_simulation_misses`).
4. A currency leaves the watchlist by governance decision only.

## 5. Backtesting

- **Daily**: `evaluate_var_backtest(&pnl, &var, alpha)` gives the exception
  series, Kupiec, Christoffersen, conditional coverage and the Basel light.
- **Traffic light**: the zone comes from `F(x)` under `Binomial(n, alpha)`
  for any window length, so a shortened history after a book migration gets
  the right zone. The yellow plus factor is the 1996 table entry at the
  equivalent 250-day count (`TrafficLight::plus_factor`).
- **Kupiec failing high**: too many exceptions. **Christoffersen failing
  with Kupiec passing**: clustering; move to FHS rather than a bigger
  multiplier.
- **ES**: `acerbi_szekely_z2` clearly negative means losses beyond VaR
  exceed forecast ES; on peg books this shows even when the VaR backtest is
  green.

## 6. Reverse stress

`stress::reverse_stress_for_loss(&exposures, &cov, loss_target)` gives the
most plausible move (fixed Mahalanobis radius) that costs the target. It is
the linear worst case on the estimated covariance; for peg books use the
peg-break scenario, which the covariance cannot represent.

## 7. Gates against the Python package and the C++ engine

- **Goldens**: `tools/gen_golden_rs.py --check`, `test_conv_golden.rs`
  (shared `conv_*` keys) and `test_golden_python.rs` (cases A, B, C). An
  unported methodology change breaks the build.
- **Fuzz**: `tests/fuzz_public_api.rs` with its registry of public functions.
- **Scope**: options are Python-only; books routed here must be linear. The
  scenario-shock map ignores unknown factor keys by convention and rejects
  the `"FX:USD"` pivot.
- **Rust vs C++ on MC**: different RNG cores (xoshiro256++ vs
  `mt19937_64`), so MC draws differ. Agreement within a few combined
  bootstrap SEs is expected; a larger gap on the same book and covariance
  points to a bug.
- **Audit**: seed, code version and book snapshot determine every MC number
  on a given build.
