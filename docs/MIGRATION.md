# Migration notes

What changed between releases and what, if anything, you need to do. Each
release has one section, newest first. Changes that alter a number say what
the old and new values are and which independent reference decided it.
The policy behind these notes is in `docs/VERSIONING.md`.

If you skip releases, read every section between your version and the one
you are moving to; the notes are cumulative, so you can migrate once.

## Unreleased (toward v3.0.0)

The next release is v3.0.0, after Round 18; there are no v2.x tags (owner
decision, 2026-11-03). `main` carries `3.0.0.dev0` until then. Work lands
in milestone pull requests; each milestone adds its notes below under its
own heading, and the v3.0.0 notes are the sum of them.

### Milestone v2.1 (Rounds 6-7)

#### Versions

- Every package now carries the repository version from `VERSION`. Before
  this, the 23 Python projects and `qp-core` said `1.0.0` (`fx-var` said
  `1.1.0`) and the two binding packages said `0.1.0`, although the
  repository was at v2.0.0.
- **What to do:** if you pinned a package by its old number
  (`qp-core==1.0.0`), pin the release instead (`qp-core==3.0.0` once
  released). Requirements written as `qp-core>=1.0` keep working.

#### Errors: `qp_core.errors`

- New module `qp_core.errors` with `QPError` and the subclasses
  `InputError` (also a `ValueError`), `ConvergenceError` (also a
  `RuntimeError` and a `ValueError`) and `DataError` (also a
  `RuntimeError`).
- **Bindings (`qp_cpp`, `qp_rust`).** Bad arguments now raise `InputError`;
  a numerical failure on well-formed input raises `ConvergenceError`. The
  cases for the second are: implied vol that does not converge; a
  covariance matrix that stays indefinite after the bounded Cholesky jitter
  (`eq_monte_carlo_var`); other engine failures the C++ engines report with
  `std::runtime_error` or the Rust engines report as `NoConvergence` /
  `Numerical`. Both classes are `ValueError`s, which is what the bindings
  raised before, so existing `except ValueError` code is unaffected.
  **What to do:** nothing. New code can catch `qp_core.errors.QPError`
  (every library error) or `ConvergenceError` to tell a failed solver from
  bad input.
- If `qp_core` is not installed next to a binding, the binding raises plain
  `ValueError` for every failure, as before.

#### Logging: `qp_core.logconfig`

- New module `qp_core.logconfig`. `configure_logging()` installs a JSON
  handler with a run context (run id, git commit, version, as-of date);
  `run_boundary()` logs one record when an entry point fails. Only
  application and study entry points call it.
- `import qp_core` now attaches a `logging.NullHandler` to the `qp_core`
  logger. No library code configures handlers, so nothing new is printed.
  **What to do:** nothing.

#### Benchmarks: tail columns relabelled

- The C++ latency harness (`cpp/core/include/qp/bench/latency.hpp`) now
  marks its p99 and max columns `p99*` and `max*` on stdout, and both the
  C++ and the Rust harness (`rust/core/src/bench.rs`, function
  `tail_basis`) add a JSON field `tail_basis`: `"per-call"` when each
  sample is one call, otherwise `"p99 of B-call batch means"` with B
  written out. The numbers are unchanged; they were always tails of B-call
  means and are not per-call tail latency.
  **What to do:** if you parse the bench JSON strictly, allow the new
  field. Do not quote these tails as per-call latency.

#### VaR engines (Round 6, lane L3)

- **Equity Monte Carlo VaR, linear book.** The C++ engine
  (`try_monte_carlo_var_factored`, which `monte_carlo_var` calls) and the
  Rust engine (`monte_carlo_pnl`, `monte_carlo_run`, `McWorkspace`) now
  compute each path's P&L in O(n) as `m (L'w)'z`, with `v = L'w` formed once
  per run, instead of forming the correlated shock vector first. The draws
  are the same; the summation order is not, so results move in the last
  bits and are no longer bit-identical to 2.0.0 or to the two-step path
  (`simulate_factor_returns` then `portfolio_pnl`). The tests hold VaR and
  ES to 1e-12 relative and each path to 1e-12 x sum |w_i r_i|. FX Monte
  Carlo VaR is non-linear and is unchanged.
  **What to do:** if you compare against stored 2.0.0 outputs with exact
  equality, compare with a relative tolerance of 1e-12 instead.
- **New non-throwing samplers** in the C++ engines:
  `RandomStream::gamma_nothrow` and `RandomStream::chi_squared_nothrow`
  (equity), `Rng::gamma_nothrow` and `Rng::chisq_nothrow` (FX). On a bad
  shape (zero, negative, NaN, infinite) they return NaN and leave the
  stream untouched; the noexcept kernels use them and report the NaN as
  the existing `McReturnsNotFinite` error. The throwing samplers are
  unchanged.
- The VaR packages' `dev` extra now includes `mpmath>=1.3`, used by the new
  independent VaR oracle tests.

#### Algorithmic execution (Round 6, lane L2)

- **Equity execution-cost numbers rise by roughly 3 to 4 times.** The
  equity simulator (`eq_algo.intraday`) divided participation by the
  number of buckets a second time, so finer slicing looked cheaper. A
  100k-share TWAP on a flat day cost 5.26 / 3.72 / 2.63 / 1.86 bp on
  13 / 26 / 52 / 104 buckets; it now costs 18.97 bp on all four
  (0.3 x 2% x sqrt 0.1, the continuous-time integral). The seed-42 race
  moves from 13.7-27.5 bp to 49.6-124.7 bp; the CHANGELOG has every row.
- **FX execution costs no longer depend on the grid.** The FX simulator
  (`fx_algo`) scaled impact by the per-bucket volatility. A 200mm two-hour
  TWAP's temporary impact ran from 0.119 pips (1-minute grid) to 0.651
  (30-minute); it is now 0.266 on every grid, and permanent impact 0.141
  (was 0.314 to 0.057). The 5-minute grid is unchanged.
  **What to do:** rerun any stored cost estimates; do not mix 2.0.0 and
  3.0.0 cost figures in one comparison.
- **Deprecated in `eq_algo`, removal no earlier than 4.0.0** (two minor
  releases at least, `docs/VERSIONING.md`):
  - the `market_volumes=` argument of `pov_schedule`: use
    `forecast_volumes=` (a forecast curve, not the realised tape). The old
    keyword still works and emits `FutureWarning`, the class the policy
    uses for behaviour aimed at notebook users.
  - the `expected_is_bps` column of `evaluate_schedules`: read
    `model_implied_is_bps`, which holds the same values. The old column is
    still filled; a DataFrame column cannot warn, so nothing is emitted.
- New in `eq_algo`: `power_law_trajectory` and `power_law_objective` (the
  optimal schedule under the simulator's own power-law impact),
  `compare_impact_laws`, and `realised_participation`.

#### Dates and calendars (Round 6, lane L1)

- **One rule-based calendar.** `qp_core.holidays` (`HolidayCalendar`,
  `CALENDARS`, `get_calendar`, `joint_calendar`) now serves schedules,
  rolls and FX dates; `get_calendar` also accepts `EUR+USD`, `EURUSD`,
  `EUR/USD` and `TARGET`. The range is 2000-2099 (was 2010-2099).
  USD, GBP, TARGET, JPY, CHF, CAD and AUD match QuantLib for every year
  2000-2030 (USD against `UnitedStates(FederalReserve)`).
- **Dates that change.** `qp_core.daycount`, `fi_rates` and `fx_rates`
  used the old partial calendars, which had no Good Friday, Thanksgiving,
  UK bank holidays or weekend substitutes. Examples: GBP modified following
  on Good Friday 18-Apr-2025 now rolls to 2025-04-22 (was 2025-04-18);
  USD Thanksgiving 27-Nov-2025 to 2025-11-28; EUR Good Friday 3-Apr-2026 to
  2026-04-07; USD Sat 31-Aug-2024 following to 2024-09-03 (was Labor Day,
  2024-09-02); 19-Jun-2020 is a USD business day (Juneteenth starts in
  2022) and 5-Jul-2021 a USD holiday. GBP 2002, 2011 and 2012 proclamation
  holidays are now closed.
  **What to do:** regenerate any stored schedules; dates that fell on these
  holidays move.
- **Deprecated: `qp_core.calendars`.** It is now a shim over
  `qp_core.holidays`; importing it emits `DeprecationWarning`. It ships in
  v3.0.0 and is removed in v4.0.0. `BusinessCalendar` there is
  `HolidayCalendar`. **What to do:** import from `qp_core.holidays`.
- **Removed: `BusinessCalendar.fixed_holidays`.** The rule-based calendar
  has no fixed-date table. **What to do:** call `.holidays(year)`.
- **FX spot dates.** `fx_rates.spot_date` now uses the pair's market lag
  when `spot_lag_days` is not given: T+1 for USDCAD (and the other T+1
  pairs in `qp_core.holidays.spot_lag`), T+2 otherwise, with the USD rule
  for crosses. USDCAD traded Fri 6-Jun-2025 now settles 2025-06-09 (was
  2025-06-10); EURJPY traded Wed 2-Jul-2025 settles 2025-07-07 (was
  2025-07-04, a USD holiday). New: `qp_core.holidays.spot_lag` and
  `fx_spot_date`.
- **FX forwards run from the spot date.** `fx_rates.MarketState` takes
  `valuation_date=` or `spot_time=`, and forwards are
  `S DF_f(t_s, T) / DF_d(t_s, T)` in outright, FX swap and cross-currency
  values. A 1y EURUSD forward (S = 1.0850, r_d = 4.5%, r_f = 2.5%, valued
  Thu 5-Jun-2025) is 1.106676, down 2.43 pips from 1.106918.
  `fx_options.forward_from_dates` gives 1.106918 for 5-Jun-2025 to
  9-Jun-2026 (365 days of carry), where reading the dates from t = 0 gave
  1.107161.
- **Deprecated: spot as a t = 0 price.** A `MarketState` with neither
  `valuation_date` nor `spot_time`, or with `spot_at_t0=True`, keeps the
  old behaviour and emits `DeprecationWarning`; removal in v4.0.0.
  **What to do:** pass `valuation_date` (or `spot_time`).
- **New:** `fx_rates.ndf` (`NDF`, base- or quote-settled, and
  `ndf_fixing_date`); `forward_for_date`, `forward_points_for_date` and
  `curve_from_forward_points` for broken dates; a `t_spot` keyword on
  `fx_options` forwards. The same calendars, spot dates, forwards and NDF
  value are ported to Rust (`rust/core/src/holidays.rs`) and C++
  (`cpp/core/include/qp/holidays.hpp`).

#### Data: `qp_data` replaces the `live.py` fetchers (Round 7)

- The `data/live.py` fetchers in `fx_options`, `fx_vol`, `fx_var`,
  `fx_rates`, `fx_pairs`, `fx_port` and `fi_rates` now delegate to
  `qp_data` and emit `DeprecationWarning`; removal in v4.0.0. Outputs are
  unchanged. **What to do:** call
  `qp_data.fetch("<key>", start, end, allow_network=True)`, then
  `qp_data.get("<key>", as_of=...)`. Frankfurter keys are
  `frankfurter:<BASE>:<SYM,SYM>`; for CCYUSD legs, invert a USD-based
  frame.
- FRED is read keyless. `api_key` is ignored (`fx_rates` warns about it;
  in `fi_rates` passing one still counts as opting in to the network). `fi_rates.data.live.load_fred_cmt`
  now needs `allow_network=True` or `QP_DATA_ALLOW_NETWORK=1`. **What to
  do:** remove `FRED_API_KEY`; pass `allow_network=True`.
- `qp_fetch.FRED_RESTRICTED` is gone; use `qp_fetch.fred_permitted(series)`.
  The two excluded-source functions in `data/fetch/qp_fetch.py` are removed
  and have no replacement.

#### Backtests: financing and excess returns (Round 7)

- `run_backtest` in `eq_port`, `eq_algo` and `fx_pairs` takes
  `financing=` and warns (`FinancingWarning`) when it is not given.
  **What to do:** to keep the old zero-rate cash silently, pass
  `financing=Financing(rate=0.0)` (`qp_core.stats.Financing`). For
  realistic results pass the daily rate series, for example
  `Financing(rate=sofr_series, borrow_spread_bp=50)`.
- Compute Sharpe ratios on `result.excess_returns` (`eq_port`, `eq_algo`)
  or `result.excess_pnl` (`fx_pairs`). `eq_algo.capacity_backtest` now
  does this for `net_sharpe` and takes `financing=`; with
  `financing=None` the number is unchanged.
- `eq_algo` ledgers have three new columns, `financing`, `risk_free` and
  `excess_ret`; `net_ret` now includes `financing`.
- `holdout.deflated_sharpe_of(r)` (research) without `n_trials` reads the
  programme tally in `research/results/trials.jsonl`. Its dict has more
  keys (`se_method`, `in_market_fraction`, `trial_sr_variance_source`).

#### Pricing engines and results (Round 7)

- `qp_cpp.eq_bs_price_batch` and `eq_bs_greeks_batch` default to the fast
  vectorised kernels (within 1e-13 / 1e-12 scaled of the exact path, not
  bit-identical). **What to do:** pass `exact=True` where you need the old
  bit-exact results, for example to compare with `eq_bs_price`.
- `eq_var.monte_carlo_risk` returns a frozen `MonteCarloRisk`;
  `res["var"]`, `keys()` and `dict(res)` still work, but
  `isinstance(res, dict)` is now False and the result cannot be mutated.
  `eq_var.BacktestResult` is frozen. **What to do:** build a new object
  with `dataclasses.replace`.

#### FX dates and swaps (Round 7)

- `fx_options.cuts.spot_date` and `expiry_and_delivery` default to
  `holidays="auto"` (was `()`, weekends only), which uses
  `qp_core.holidays.fx_spot_date`. EURUSD traded Wed 25-Nov-2026 settles
  Fri 27-Nov (was Mon 30-Nov). `cuts.spot_lag("USDPHP")` is 1. **What to
  do:** pass `holidays=()` for the old weekends-only dates.
- `CrossCurrencySwap` and `BasisSwap` with `include_initial_exchange=True`
  value the exchange on the spot date (`exchange_time`, default
  `market.spot_time`). EURUSD, 150k quote notional off market, valued
  5-Jun-2025: 150,000.00 to 149,926.05. **What to do:** nothing, unless you
  relied on the undiscounted value. For `CrossCurrencySwap`,
  `exchange_time=0.0` values the exchange today as before.
- New `FixedRateBond.calendar` (default `USD-GOVT`) and `payment_dates()`;
  prices and accrual still use the unadjusted schedule.

#### Testing and CI (contributors only)

- Hypothesis settings profiles in the root `conftest.py`: `ci`
  (derandomised: two runs on one commit draw the same examples), `nightly`
  (random, ten times the examples) and `dev` (defaults). Select with
  `QP_HYPOTHESIS_PROFILE` or `--hypothesis-profile`.
- The QuantLib cross-checks (Hull-White, ISDA CDS, Heston, the holiday
  calendars including USD-GOVT, and the FX spot dates) run in a required `quantlib` CI job against QuantLib 1.44 installed from
  hash-locked wheels, and fail rather than skip.
- New CI jobs: `py data` (python/data on Python 3.11 to 3.13) and
  `learning` (notebook and career-prep tests, nbmake on the solution
  notebooks). The root workspace adds `python/data` and a `notebooks`
  dependency group (nbmake, ipykernel).

## 2.0.0 (2026-11-03)

The first release with this file. See `CHANGELOG.md` for what changed in
2.0.0 and earlier.
