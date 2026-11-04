# Changelog

All notable changes to this repository are recorded here. The format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) and versions
follow [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

A release is cut by `.github/workflows/release.yml`, which tags a commit only
when the `ci-ok` check passed on that exact commit. The release notes are the
section below whose heading matches the version.

## [Unreleased]

## [3.0.0] - 2026-11-04

This release collects Rounds 6 to 18 and the work first planned for v3.1. Numbers come from seeded
synthetic data unless an entry names a reference. What callers need to do
is in `docs/MIGRATION.md`.

### Fixed

- **Data before 2000.** `qp_data` can store rows dated before 2000, so
  `cboe:VIX` and `french:*` load. Before 2000, release dates count every
  weekday as a publishing day and follow today's rules, so they are notional
  and can be early. `qp_core.holidays` still raises outside 2000-2099. A live
  `cboe:VIX` fetch may meet a redirect; that is untested.
- **`qp_data.fetch` timeout.** `fetch` takes `timeout=` (default 30 seconds).
- **Rust feed handler.** `SyncEngine::on_snapshot` reloaded a REST snapshot
  no newer than the live book, which dropped levels beyond the snapshot
  depth while the book still reported fresh: 1,083 of 1,130 fresh states
  differed from the reference on one live recording. It now ignores such a
  snapshot, as C++ does.
- **AAD cost ratio.** `rust/aad/examples/cost_ratio.rs` let the compiler
  fold most of the plain Black-Scholes price, so Round 17's 13-34x was
  really about 10-12x.
- **Shared CRR tree speed (C++ and Rust).** Back to the speed of the old FX
  trees with the same results: the C++ American tree is about 5x faster,
  the Rust one about 2x.
- **ISDA CDS accrued-on-default (`eq_credit.isda_cds`).** The premium leg
  had no ISDA half-day bias, so wide-spread names were mispriced: at a flat
  17% hazard, 5y, the clean upfront was 379 per 10mm away from QuantLib's
  `IsdaCdsEngine` with `HalfDayBias`; with the bias it is 0.23 (at a 2%
  hazard: 12.6 before, 0.012 after). `IsdaCds(half_day_bias=True)` is the
  default; `False` gives the unbiased integral (QuantLib `NoBias`). The
  QuantLib cross-check now runs both settings.
- **Post-trade settlement (`qp_posttrade.settlement`).** It carried 2026-only
  holiday tables, so any trade from 31 Dec 2026 raised and USDCAD settled
  T+2. It now uses `qp_core.holidays` (Fed, TARGET2, UK, ... and
  `fx_spot_date`, so USDCAD is T+1) and an NYSE Rule 7.2 generator with the
  one-off closures, 2000-2099. The 40 hand-worked 2026 cases are unchanged.
- **Cross-platform parity fixtures.** `rates_vol_parity` (Rust) and the
  Python freshness checks of `exotics_golden.txt` and
  `xva_capital_parity.txt` compared last bits exactly, which failed on the
  Linux runner (libm `ln`/`erf` differ by a few ulp from the machine that
  wrote them). They now hold every number to 1e-12 relative, the bound the
  C++ and Rust parity tests already use; labels and counts stay exact.
- **Calendar property test (`qp_core`).** `test_adjusted_is_business_day`
  asserted that no adjustment moves a date more than 7 days. Tokyo's 2019
  accession Golden Week closed banks for 10 days (27 Apr to 6 May), so the
  bound was wrong, not the calendar; the JPY bound is now 10 and the 2019
  closure is pinned by its own test.

- **Holiday calendars (`qp_core.daycount`, `fi_rates`, `fx_rates`).** These
  used partial fixed-date tables with no Good Friday, Thanksgiving, UK bank
  holidays or weekend substitutes, so rolls could land on closed days. Old
  to new, each pinned by a test written first: GBP Good Friday 18-Apr-2025,
  modified following, 2025-04-18 to 2025-04-22; EUR Good Friday 3-Apr-2026,
  2026-04-03 to 2026-04-07; USD Thanksgiving 27-Nov-2025, 2025-11-27 to
  2025-11-28; USD Sat 31-Aug-2024, following, 2024-09-02 (Labor Day) to
  2024-09-03; USD 19-Jun-2020 holiday to business day (Juneteenth from
  2022); USD Mon 5-Jul-2021 business day to holiday; GBP Christmas 2027
  roll 2027-12-27 to 2027-12-29. In `qp_core.holidays`, the GBP
  proclamation holidays of 2002, 2011 and 2012 are now closed (and
  2002-05-27 and 2012-05-28 open), and the years 2000-2009 are supported
  instead of raising. References: gov.uk bank holidays, ECB TARGET2
  closing days, the Federal Reserve holiday list; QuantLib 1.44 agrees for
  USD (FederalReserve), GBP, TARGET, JPY, CHF, CAD and AUD on every year
  2000-2030.
- **FX spot dates (`fx_rates.spot_date`).** The default lag is now the
  pair's market lag. USDCAD traded Fri 6-Jun-2025 settles 2025-06-09
  (T+1), not 2025-06-10; EURJPY traded Wed 2-Jul-2025 settles 2025-07-07,
  not 2025-07-04, a USD holiday (Clark 2011, ch. 1; Wystup, ch. 1). 31 of
  the 32 spot-table cases are cross-checked against the same rule on
  QuantLib calendars (USDTRY has none).
- **FX forwards carry from the spot date (`fx_rates.MarketState`).** A 1y
  EURUSD forward (S = 1.0850, r_d = 4.5%, r_f = 2.5%, valued Thu
  5-Jun-2025) is 1.106676, 2.43 pips below the old 1.106918, which carried
  from today instead of from spot Mon 9-Jun. `fx_options.forward_from_dates`
  for 5-Jun-2025 to 9-Jun-2026 gives 1.106918 (365 days of carry), where a
  t = 0 reading gave 1.107161 (369 days). Closed form
  S exp((r_d - r_f)(T - t_s)).
- **Equity execution simulator (`eq_algo.intraday`).** Participation was
  divided by the number of buckets a second time, so finer slicing looked
  cheaper. A 100k-share TWAP on a flat day cost 5.26 / 3.72 / 2.63 /
  1.86 bp on 13 / 26 / 52 / 104 buckets and now costs 18.97 bp on all four
  (0.3 x 2% x sqrt 0.1); the bucket cost converges to the continuous
  integral (error times n below 0.5). Seed-42 race, model-implied IS: VWAP
  13.7 to 49.6 bp, TWAP 14.3 to 53.0, Almgren-Chriss 15.1 to 57.1, POV
  17.2 to 68.0, Aggressive 27.5 to 124.7; the order is unchanged. TCA
  aggregate trading cost 18.6 to 57.8 bp, total 14.3 to 53.5 bp. Costs
  rise by roughly 3 to 4 times. The Almgren-Chriss frontier figures do not
  use the simulator and are unchanged.
- **FX execution simulator (`fx_algo`).** Impact was scaled by the
  per-bucket volatility, so it depended on the grid. A 200mm two-hour
  overlap TWAP: temporary impact 0.119 (1-minute grid) to 0.651 pips
  (30-minute) is now 0.266 on every grid; permanent 0.314 to 0.057 is now
  0.141. The 5-minute grid is unchanged. Fix study (1-minute grid):
  fix-targeting tracking error +0.498 to +1.015 pips, self-impact 0.078 to
  0.035, standard deviation unchanged.
- **C++ VaR engines, noexcept paths.** `RandomStream::gamma` and
  `Rng::chisq` throw and were reachable from noexcept kernels, where a
  throw ends in `std::terminate`. New `gamma_nothrow` /
  `chi_squared_nothrow` (equity) and `gamma_nothrow` / `chisq_nothrow`
  (FX) return NaN on a bad shape without consuming the stream, and the
  kernels report it as `McReturnsNotFinite`. `qp::gamma_marsaglia_tsang`
  loops forever on a NaN shape and recurses without end on a shape <= 0;
  it is now never called unguarded.
- **Deflated Sharpe ratio (research, `qp_core.stats`).** The DSR used a
  per-study trial count (3, hard-coded, for FX pairs) and V = 1/(T-1). It
  now uses the programme-wide count from `research/results/trials.jsonl`
  (13) and the variance of the 30 logged trial Sharpes (1.15e-4, was
  2.06e-4), with the Lo (2002) standard error when the series is in the
  market on fewer than half the days. FX pairs v1 training DSR: 0.616 to
  0.374. There is one implementation, `deflated_sharpe_detail`.
- **Backtest cash at a zero rate (`eq_port`, `eq_algo`, `fx_pairs`).** The
  levered `eq_port` book borrowed for free and the `eq_algo` and
  `fx_pairs` capital earned nothing. `run_backtest(financing=Financing(...))`
  now finances the cash residual or the posted capital at a daily rate plus
  stated spreads, calendar-day accrual, and the results expose excess
  returns. Hand fixture: 150% invested at 4% + 50 bp over a weekend,
  ACT/360, financing -1.8753e-4 (was 0). `eq_algo.capacity_backtest`
  reports `net_sharpe` on excess returns (equal to net when
  `financing=None`).
- **Look-ahead in research rates.** Monthly rates were used inside their
  own month. `research/lib/excess_returns.as_known` applies m+2 to monthly
  series and d+1 to daily fixings; `qp_data.release.check_release` raises
  when a series is used before its release (June 2024 FEDFUNDS on 28 June;
  DGS10 for 1 July on 1 July).
- **Cross-currency initial exchange (`fx_rates.xccy`).** With
  `include_initial_exchange=True` the exchange is valued on the spot date,
  not today. EURUSD, quote notional 150k off market, valued Thu
  5-Jun-2025, r_d 4.5%: 150,000.00 to 149,926.05 (150k x P_d(t_s)). An
  at-market `BasisSwap`: -2,377.82 USD to 0.00 (Clark 2011, ch. 1).
- **FX option spot dates (`fx_options.cuts`).** `spot_date` and
  `expiry_and_delivery` ignored holidays by default. EURUSD traded Wed
  25-Nov-2026: Mon 30-Nov to Fri 27-Nov; EURJPY 2-Jul-2025: Fri 4-Jul (a
  USD holiday) to Mon 7-Jul; GBPUSD 16-Apr-2025: Fri 18-Apr (Good Friday)
  to Tue 22-Apr. `spot_lag("USDPHP")` is T+1 (was T+2).
- **FRED loaders needed a key (`fi_rates`, `fx_rates` `data.live`).** They
  now read FRED keyless, and FRED's empty-cell missing marker is parsed
  as well as `.`.

### Changed

- **FX Monte Carlo VaR (C++ and Rust).** Each scenario draws from its own
  Philox4x32-10 stream with AS 241 normals, as equity does. The two engines
  give the same bits on any thread count, and the C++ simulation is about
  4x faster. Numbers move within Monte Carlo error; see `docs/MIGRATION.md`.
- **Bootstrap standard errors (Rust).** Faster with the same results: about
  20% on the equity 250-day case and about 25% on the FX 100k case (local
  A/B).
- **Heston.** The characteristic function lives once, in
  `qp_core.vol.heston.heston_cf`; the `eq_surface` and `fx_surface` versions
  are shims. Prices are bit-identical, checked against both old copies, the
  kernel corpus and QuantLib `AnalyticHestonEngine`.
- **AAD.** Black-Scholes reuses a per-thread tape and records fused nodes.
  The cost ratio falls to 4.85-4.9x in Rust (from about 10-12x) and
  12.5-13x in Python (from 20-21x), measured on a loaded laptop. The Rust
  margin under 5x is thin.
- **Feed handlers (C++ and Rust).** Books sit on the flat L3 book with a
  per-product tick window, order-id interning and an exact side store. A
  non-allocating Binance decoder gives zero heap allocations on that path
  (tested). Coinbase `l2update` messages have non-allocating decoders too
  (Rust `fast::coinbase_level2_into`, C++ `decode_level2`), checked against
  the generic parser on generated, mutated and recorded input. The C++
  handler's duplicate filter and map book can still allocate, the Rust decoder
  has no production caller yet, and `full` uses the generic parser.
- **Workspace.** `python/backtest` (qp-bt), `python/oms` (qp-oms),
  `python/execution` (qp-exec) and `python/posttrade` (qp-posttrade) are uv
  workspace members, type-checked by mypy and tested in CI as projects of
  their own; a project's workspace dependencies (e.g. qp-bt for the nine
  migrated backtesters) are installed by `discover_projects.py`.
- **Rust workspace.** `rust/ull/feed`, `rust/ull/gateway` and `rust/aad`
  joined the workspace (one `Cargo.lock`); new CI jobs `rust-extra`
  (tests, clippy, loom on the SPSC ring), `cpp-extra` (exec-core,
  rates-curves, aad, ULL feed/gateway/bench, plus the Python book against
  the exec-core replay) and `java` (Temurin 21, `./mvnw -B verify`). The
  research job runs every `research/` test folder; the learning job adds
  `docs/learn/tools` and the research solution notebooks; the lint job
  checks the model inventory.
- **`fi_rates` exports** `make_schedule`, `FixingStore`, `SOFR`, `ESTR`,
  `SONIA`, `EURIBOR_3M`, `EURIBOR_6M`, `TERM_SOFR_3M`, `FixedLeg`,
  `IborLeg`, `OvernightLeg`, `Swap` and `futures_convexity_adjustment`.
- **`qp_data`** registers the `french:<dataset>` source, with a
  conservative 45-business-day release bound.
- **Model tags.** 31 more model entrypoints carry `@model`; the two
  `eq_surface` entries (SSVI, Dupire) are `decorated = false` until the
  surface rework lands.

- **Equity Monte Carlo VaR, linear book (C++ and Rust).** Each path costs
  one O(n) dot product with `v = L'w`, formed once per run, instead of an
  O(n^2) triangular matvec. Same draws, different summation order: results
  are no longer bit-identical to 2.0.0 or to the two-step path; VaR and ES
  agree to 1e-12 relative and each path to 1e-12 x sum |w_i r_i|. Rust, 10k
  paths x 50 factors, local release build: 50.9 ms to 20.1 ms. FX Monte
  Carlo VaR is non-linear and unchanged.
- **One rule-based calendar.** `qp_core.holidays` (`HolidayCalendar`,
  `CALENDARS`, `get_calendar`, `joint_calendar`) serves every package;
  range 2000-2099. Ported to Rust (`rust/core/src/holidays.rs`) and C++
  (`cpp/core/include/qp/holidays.hpp`) and tested in both against one
  QuantLib extract for 2000-2030.
- **Versions.** Every package carries the repository version,
  `3.0.0.dev0`, from `VERSION` (the projects said `1.0.0`, `fx-var`
  `1.1.0`, the bindings `0.1.0`); `tests/test_repo_policy.py` enforces it.
  Every Python package, `qp-core` and both bindings included, ships
  `py.typed`.
- **Errors.** New `qp_core.errors`: `QPError`, `InputError` (a
  `ValueError`), `ConvergenceError` (a `RuntimeError` and a `ValueError`),
  `DataError` (a `RuntimeError`). The bindings raise `InputError` for bad
  arguments and `ConvergenceError` for numerical failures; both remain
  `ValueError`s.
- **Benchmarks.** The C++ and Rust latency harnesses label their tails
  "p99 of B-call batch means" (JSON field `tail_basis`; `per-call` when
  B = 1). The README, ARCHITECTURE and BENCHMARKS tables say which column
  is which. No number changes.
- **Release workflow.** `release.yml` is manual dispatch only, with a dry
  run, a `VERSION` check, cibuildwheel wheels for Linux, Windows and macOS
  arm64, sdists, a CycloneDX 1.5 SBOM, `SHA256SUMS` and build provenance
  attestation. GitHub Release only.
- **Docs.** `docs/LEARN.md`: all 532 answers fold under `<details>` with a
  tags line; the range-estimator efficiencies are sourced (Parkinson 4.91
  derived, Garman-Klass 7.4, Rogers-Satchell about 6 by simulation); the
  pairs-funnel figures are re-run at seed 7. `docs/index.html` gains a
  dark theme. The equity C++ VaR docs say "to 1e-12" where they said "bit
  for bit".
- **C++ binding batch pricing.** `qp_cpp.eq_bs_price_batch` and
  `eq_bs_greeks_batch` default to the vectorised kernels: 3.9 times faster
  for prices (75.0 to 19.4 ns per option) and 2.5 times for Greeks (97.6
  to 38.9 ns), median of 15 runs of a 100,000-option call, WSL, GCC 15.2.
  Prices stay within 1e-13 max(1, S, K) of the exact path and each Greek
  within 1e-12 max(1, S, K, |exact|). `exact=True` gives the old results;
  `eq_options.engine.bs_price_batch` gains `exact=`. MSVC builds still run
  the exact path. The Rust batch is unchanged.
- **Array-tolerant scalar pricers.** `eq_options.bs_price` and
  `fx_options.gk_price` accept arrays and return arrays, within 1e-15
  max(1, S, K) of the scalar calls.
- **Frozen results.** `eq_var.monte_carlo_risk` returns a frozen
  `MonteCarloRisk` (dict-style access kept); `eq_var.BacktestResult` is
  frozen.
- **FX option spot-date default.** `fx_options.cuts.spot_date` and
  `expiry_and_delivery` default to `holidays="auto"`, which delegates to
  `qp_core.holidays.fx_spot_date` (was `()`, weekends only).
- **`data/fetch/qp_fetch.py`** refuses any FRED series not on a
  public-domain allow-list (`fred_permitted`); it used to refuse five named
  series.
- **Docs.** The RLS hedge tracker is no longer called "Kalman-lite";
  `docs/CAREER_PREP.md` is new; README, ARCHITECTURE, DIAGRAMS and
  CONTRIBUTING describe `python/data` and the two new CI jobs.

### Added

- FRTB SBM delta and vega for CSR non-securitisation, equity and commodity
  (`qp_core.capital.frtb_classes`, MAR21.51-95), curvature for GIRR, CSR
  non-securitisation, equity, commodity and FX (`frtb_curvature`, MAR21.5
  and 21.96-101), and checksum-tested parameter tables (`frtb_tables`).
  Commodity vega is per commodity and option maturity, and curvature per
  commodity: delivery location is not a factor (MAR21.13(2)-(3)).
- Rust `qp_core::frtb`, `qp_core::collateral` and C++ `qp/frtb.hpp`,
  `qp/collateral.hpp`: scenario aggregation, curvature charge, `net_cvr`, margin
  balance, lag and exposure, matching Python to 1e-12 relative on a shared
  fixture. The per-class wrappers stay in Python.
- `tests/independent/frtb`: a second implementation of GIRR and FX delta
  and vega from the MAR21 text, agreeing with `frtb_sbm` to 1e-10 relative
  on 1,750 seeded cases, and a second oracle for CSR, equity and commodity
  delta and vega and curvature for all five classes on 4,000 more. It found
  the commodity location error above before release.
- `qp_core.xva.collateral`: exposure with variation margin (margin period
  of risk, thresholds, MTA) and initial margin, checked against Brownian
  closed forms and limiting cases.
- `qp_exec.mm`: toxicity-aware spreads (markout EWMA with decay, VPIN) and
  a fill-rate and markout report by counterparty tier, on synthetic tiered
  flow. In this simulator markout widening also fires on benign flow,
  because a maker's own fill moves the displayed mid.
- `tools/recorder`: a free, no-account recorder for Binance diff-depth and
  Coinbase `level2_batch`, three short live recordings with a SHA-256
  manifest, and a CI job replaying them through the C++ and Rust handlers
  against an independent Python reference book. Full-day recordings have
  not been made.
- Rust `qp_core::risk` and C++ `qp/risk.hpp`: quote-space risk through the
  bootstrap Jacobian, par DV01, quote gamma with the bootstrap-curvature
  term, vol grids with bucket vegas, and P&L explain, matching Python to
  1e-12 on a shared fixture. Also `Smile`, `smile_delta`, `smile_gamma`,
  `premium_adjusted_delta`, `rank_smile_dynamics`, `tenor_weights` and
  `reporting_conversion`; book revaluation stays in Python. `smile_gamma` is
  held to the tolerance of the two deltas it differences.
- `smile_dynamics=` on the flat-vol option risk calls of `eq_options`,
  `fx_options` and `fx_var.book.Book`. All three values give the same
  numbers there, because those calls use one flat vol.
- Simulated power files for the combination panel, UST and TSMOM drafts.
  The combination design reaches 80% power only with an IC of about 0.05 and
  30 currencies; the free data has a median of 21.
- Model inventory grows from 33 to 74 entries, covering the shared stacks,
  each tagged in code with a section in `VALIDATION.md`; a port that reproduces
  a Python fixture is not counted as independent. Three COOKBOOK recipes (curvature of a short
  call, exposure under a margin period of risk, toxicity-aware quotes).
- `qp_data`: OECD MEI interbank rates and CPI (CC BY 4.0) through keyless
  FRED, released at month m+2.
- `research/exploratory/r18`: the 2000-2015 FX carry, value and momentum
  panel and a SOFR 2024 compounding check against the NY Fed SOFR Index
  (within 0.0009 bp). No preregistration, so neither supports a claim.
- `qp_core.holidays.spot_lag` and `fx_spot_date`; `MarketState`
  `valuation_date=` / `spot_time=` and `spot_today`; `forward_for_date`,
  `forward_points_for_date`, `curve_from_forward_points` (reprices an input
  points strip to 1e-10); `fx_rates.ndf` (`NDF`, `ndf_fixing_date`); a
  `t_spot` keyword and `forward_from_dates` in `fx_options.forwards`.
- `eq_algo.power_law_trajectory` and `power_law_objective`: the optimal
  schedule under the simulator's power-law impact (Newton on the discrete
  Euler-Lagrange equations, one tridiagonal sweep per step; equals the
  Almgren-Chriss closed form to 1e-8 of the order at delta = 1). Seed-42
  race: 55.5 bp, standard deviation 71.6 (Almgren-Chriss 57.1 and 74.5).
  Also `compare_impact_laws` and `realised_participation`.
- An independent VaR oracle (`python/equity/03-var-es-engine/tests/golden/
  var_oracle.py`, mpmath at 50 digits, no engine imports): 196 values per
  project, emitted as C++ and Rust constants, checked in all three
  languages. It reproduces all 11 cells of the BCBS (1996) traffic-light
  table and 7 of 9 cells of the Kupiec (1995) table; the other two are
  explained by the LR test and pinned. Every engine agreed with it.
- `qp_core.logconfig` (JSON logging with a run context, `run_boundary`).
- Hypothesis profiles `ci`, `nightly` and `dev` in the root `conftest.py`;
  a nightly CI run; a required `quantlib` CI job (QuantLib 1.44 from
  hash-locked wheels, no skips) and an `api diff` job for pull requests.
- `docs/VERSIONING.md`, `docs/MIGRATION.md` and `docs/CONTRIBUTING.md`.
- `qp_data` (`python/data`): one keyless data layer. Sources: FRED CSV,
  ECB reference rates and the euro short-term rate, Frankfurter, NY Fed
  SOFR, BoE SONIA, US Treasury par curve, CBOE VIX, Coinbase and Binance
  public daily candles. Retries with backoff; a SHA-256 content-addressed
  store with a manifest; publication-lag rules from issuer release
  calendars and `check_release`; bitemporal `get(series, as_of=...)`.
  Tests run on recorded responses; live tests are opt-in.
- `research/lib/`: CIP excess returns with publication lags, a
  point-in-time 38-currency universe (`research/universe/fx.csv`), tiered
  costs at 0x/1x/2x with the breakeven multiple, concentration diagnostics,
  the one-day fill lag, per-leg pairs P&L, LRV and Fama-MacBeth panel
  tools, and a power gate (a single series is monitoring only; a
  confirmatory test needs a panel at 80% power).
- `qp_core.stats`: `Financing`, `FinancingWarning`, `warn_no_financing`,
  `deflated_sharpe_detail`, `deflated_sharpe_from_moments`,
  `in_market_fraction`.
- `eq_pairs.kalman.kalman_hedge_ratio` (`KalmanHedgeResult`): a Kalman
  hedge ratio with observation and state noise estimated by maximum
  likelihood, checked against the exact Gaussian likelihood (1e-10
  relative) and the ridge-OLS posterior when Q = 0 (1e-8).
- `fx_pairs.BacktestResult`: `leg1_pnl`, `leg2_pnl`, `financing_pnl`,
  `risk_free_pnl`, `excess_pnl`; `run_backtest(capital=)`.
- `USD-GOVT` (alias `SIFMA`), the SIFMA US bond-market calendar, in
  `qp_core.holidays` with Rust and C++ ports, matching QuantLib 1.44
  `UnitedStates(GovernmentBond)` 2000-2030 apart from three listed SIFMA
  closures QuantLib omits; `qp_core.holidays.CALENDAR_CODES`;
  `fi_rates.calendars.TREASURY_CALENDAR` and `SWAP_CALENDAR`;
  `FixedRateBond.calendar` and `payment_dates()` (prices and accrual are
  unchanged).
- `fx_rates.swaps`: `DatedFXSwap`, `price_fx_swap`, `fair_swap_points`,
  `tom_next_swap`, `roll_position`; Rust and C++ `fx_leg_value`,
  `fx_swap_far_rate` and `xccy_initial_exchange`. `xccy` swaps gain
  `exchange_time`; `CrossCurrencySwap.cashflows` takes an optional
  `market`. `fx_options.forward_from_dates` is exported at package level.
- `eq_var.engine.monte_carlo_var(..., engine=)` for linear books.
- Four learning notebooks (Black-Scholes and implied vol, VaR and ES,
  curve bootstrap, Almgren-Chriss), each as exercise and solution, checked
  against independent computations; `career-prep/` with a 60-problem
  probability pack (answers kept separate, each recomputed exactly and by
  seeded simulation), a mental-maths drill and a market-making game.
- CI jobs `py data` and `learning` (nbmake on the solution notebooks); the
  `quantlib` job also runs the USD-GOVT parity test.

### Deprecated

Each still works in v3.0.0 and is removed no earlier than v4.0.0
(`docs/VERSIONING.md`).

- `qp_core.calendars`: a shim over `qp_core.holidays`; importing it emits
  `DeprecationWarning`.
- `fx_rates.MarketState` with neither `valuation_date` nor `spot_time`, or
  with `spot_at_t0=True`: spot is treated as a t = 0 price;
  `DeprecationWarning`.
- `eq_algo.pov_schedule(market_volumes=)`: use `forecast_volumes=`;
  `FutureWarning`.
- The `expected_is_bps` column of `eq_algo.evaluate_schedules`: read
  `model_implied_is_bps` (same values; a column cannot warn).
- The `data/live.py` fetchers in `fx_options`, `fx_vol`, `fx_var`,
  `fx_rates`, `fx_pairs`, `fx_port` and `fi_rates`: use `qp_data`;
  `DeprecationWarning`, outputs unchanged.

### Removed

- `BusinessCalendar.fixed_holidays`: the rule-based calendar has no
  fixed-date table; use `.holidays(year)`. This is a public API break,
  allowed at a major version and recorded in `docs/MIGRATION.md`.
- The two excluded-source functions in `data/fetch/qp_fetch.py` and
  `qp_fetch.FRED_RESTRICTED` (use `fred_permitted`).
- The unsigned order-book execution draft registration (withdrawn
  2026-11-03, never signed, no data scored; `research/withdrawals/`).

## [2.0.0] - 2026-11-03

One release, v2.0.0, is planned. Until then changes collect here. Results
quoted below come from seeded synthetic data unless an entry names a
dataset (the research entries use ECB reference rates).

### Changed

- All Rust crates (the `rust/` workspace and the PyO3 binding) move to
  edition 2024 with `rust-version = "1.99"`, matching the pinned 1.99.1
  toolchain, and the workspace resolver to 3. Code changes are what the
  edition brings: rustfmt's 2024 style, and four nested `if let` blocks in
  `fx-var-engine` collapsed into let chains. No numeric output changes; all
  605 workspace tests and the 85 binding tests pass.
- The Black-Scholes maths the equity and FX options code carried twice now
  lives once per language, in cost-of-carry form (b = r - q = r_d - r_f):
  `qp_core.black_scholes` and `qp_core.iv_core` (Python), `qp/black_scholes.hpp`
  and `qp/black_scholes_vm.hpp` (C++), `qp_core::black_scholes` (Rust). The
  equity and FX packages keep their names, signatures and validation and call
  it. Every in-scope output is unchanged bit for bit on a 6,000-quote grid
  (Python, C++ on GCC and Clang, Rust), signed zeros included; no golden
  moves. `tests/test_carry_twin.py` pins equity-with-yield against
  FX-with-foreign-rate.
- FX `analytic_greeks` (Python) now takes `N(phi d1)`, `N(phi d2)` and
  `n(d1)` from `qp_core.normal` (`0.5 erfc(-x / sqrt 2)`), the normal the
  shared kernel uses, instead of `scipy.stats.norm`. On 5,944 quotes price,
  deltas, theta and both rhos move by at most 2.13e-14 relative to
  `max(1, |value|)` (price 2.13e-14, deltas 2.2e-16, theta 2.66e-15, rhos
  2.5e-15); gamma, vega, vanna and volga do not move; no FX options golden moves. The
  fields are now Python floats rather than `numpy.float64`. Table in
  `python/fx/01-options-pricing/docs/VALIDATION.md` section 7.
- Deprecated P&L-level `filtered_historical_var` (Python, C++, Rust equity
  VaR engines) now reads its tail with the order statistic
  L(ceil(n alpha)), like every other VaR path, instead of numpy type-7
  interpolation. Golden `a_fhs_var_5pct_lam094` moves from
  109.3777910164513 to 112.05212695060598; no other golden moves. Table and
  references in docs/MARKET_RISK.md.

### Added

- Signed the forward-only registrations `research/prereg/pairs-fx-g10-v2.md`
  (sha256 78da5f57...c6042b) and `research/prereg/momentum-fx-g10.md` (sha256
  623f8034...dab31a) on 2026-11-02, both through the power gate in `prereg.py sign`;
  decision rules unchanged. Only fixes from 2026-11-03 are scored; evaluation on
  or after 2028-11-13. Their formation/lookback history is one ECB EXR file,
  `research/forward/history_1999-01-04_2026-11-02.csv` (7,127 fixes, sha256
  d2e62948442b8c42c3739f95f6db04ca4894bdb20cce11b50ea322c9a943b6ea), fetched at signing.
- Optional QuantLib cross-checks, skipped unless QuantLib is installed:
  Hull-White zero-bond options and Jamshidian swaptions against QuantLib's
  `HullWhite` model (1e-15 abs and 1e-7 rel), and Heston COS and
  Gil-Pelaez prices against `AnalyticHestonEngine` (1e-9 and 1e-12 rel).
- Documentation: nine new COOKBOOK recipes (NumPy array pricing and Greeks,
  the shared carry-form kernel for equity, FX and futures, consuming the
  C++ engines with `find_package(QuantPortfolio)`, installing CI wheels,
  the hash-checked install, the bench gate, the QuantLib checks, the trials
  log, interim forward figures); LEARN Set 19 (21 questions on the carry
  form, VaR quantile conventions, optional stopping, power, false discovery
  rate, hashed installs and latency measurement); DIAGRAMS gains the CI
  pipeline as it now runs, the shared-kernel layout and the research
  lifecycle. README, START_HERE, ARCHITECTURE and the site updated to match.

#### Packaging

- `cpp/CMakeLists.txt` installs the four C++ engines and `qp::core` as one
  CMake package, `QuantPortfolio` 1.0.0 (targets `qp::core`, `qp::eqopt`,
  `qp::eqvar`, `qp::fxopt`, `qp::fxvar`). The engines' include paths are now
  wrapped in `BUILD_INTERFACE`/`INSTALL_INTERFACE`; standalone engine builds
  are unchanged. A new required CI job, `cmake install`, builds
  `cpp/tests/find_package_consumer` against the installed prefix with GCC 15
  and Clang 21.
- CI job `wheels`: cibuildwheel wheels for `qp-cpp` and `qp-rust`, CPython
  3.11 to 3.13, manylinux_2_34 x86_64 and Windows amd64, each smoke-tested
  by `bindings/wheel_smoke.py`. Artefacts only, never published; gated by
  `ci-ok` after its first green run.
- `eqopt::bs_price_batch_fast` and `fxopt::gk_price_batch_fast`: batch
  pricers that vectorise (branch-free exp, log and erfc from `qp/vmath.hpp`,
  plus an AVX2 clone on x86-64 Linux). Not bit-exact; tested within
  `1e-13 * max(1, S, K)`. Measured about 4x over `bs_price_batch` with
  GCC 15 and 1.2-2.4x with Clang 21 (noisy laptop, WSL2); see the engine
  METHODOLOGY. `bs_greeks_batch_fast` / `gk_greeks_batch_fast` added on the
  same kernels (about 3-4x with GCC, tested within `1e-12 * max(1, S, K,
  |exact|)`); `fxopt_batch_bench` added. On MSVC the `_fast` functions use
  the exact path, because there they measured 3-4x slower. `qp::vm::log_batch` added; `qp::vm::erfc_batch` now actually
  vectorises (its sign select had compiled to a branch).
- `eq_options.bs_price_array`, `eq_options.bs_greeks_array`,
  `fx_options.gk_price_array` and `fx_options.gk_greeks_array`: NumPy entry points with keyword-only
  arguments after `K`. About 19x faster than a scalar loop for 10 000
  options. Scalar functions unchanged.

#### Research results (ECB reference rates)

- Every training, stress and calibration run now appends a line to
  `research/results/trials.jsonl` (git commit, registration hash, params,
  metric), so trial counts come from the log rather than memory.
- Pair-selection calibration (`research/studies/robustness.py`): 53 pairs
  selected in 2003-2021 against 22 expected under a bootstrap null
  (p = 0.010), but only 4 of 19 years beat the null at 5 % and
  Benjamini-Hochberg keeps 14 pairs. Cost (1/3/5 bp) and event-day stresses
  for pairs v2 and momentum: both stay negative at every cost in training.
- `pairs-fx-g10` withdrawn on 2026-11-02, before any hold-out data was read
  (DR-R4-5 in `research/README.md`): on simulated pairs that are
  cointegrated by construction its Kalman one-step z almost never passed 2,
  so a hold-out result would have been uninformative. The harness and
  `prereg.py check` refuse a pairs hold-out run or result.
- `regime-fx-basket` hold-out run once (2022-01-03 to 2025-12-31, commit
  68ae51c): 161 of 1,019 days high-vol, annualised next-day basket return
  +12.5% high against -4.0% low, HAC t = +0.94 (4 lags). The rule needs
  t < -1.645, so H0 is not rejected; the effect has the opposite sign to
  the hypothesis. Result: `research/results/regime_fx_basket_holdout.json`.
  The hold-out is spent.
- Simulated power check for the momentum draft
  (`research/studies/momentum_power.py`, `research/power/momentum-fx-g10.json`):
  about 13% rejection at a true Sharpe of about 0.4, 50% near 1.3. A fail
  on the forward data is the expected outcome. `prereg.py sign` refuses a
  registration that names no power file.

#### Hash-checked CI installs

- Every Python install in CI (`py core`, the 23 project jobs on 3.11 to 3.13,
  bindings, engine differential, research, forward log) now comes from
  `uv.lock` with sha256 hashes checked. `.github/scripts/hashed_install.sh`
  exports the locked closure of the named projects with `uv export`, installs
  it with `pip --require-hashes --only-binary :all: --no-deps`, then installs
  the local packages with `--no-deps --no-build-isolation` and runs
  `pip check`. A wheel whose hash is not in the lock fails the job.
- The root `pyproject.toml` gains four dependency groups locked in `uv.lock`:
  `test` (pytest, hypothesis), `lint` (ruff, mypy, stubs), `backend`
  (setuptools) and `build` (nanobind, scikit-build-core, maturin). The lint
  job installs them with `uv sync --group test --group lint`.
- The bindings are built without build isolation, from the hashed backends,
  so `PIP_CONSTRAINT` is gone from CI.
- `requirements/ci-constraints.txt` stays for local `pip install -c`; the lint
  job fails if it pins a different version from `uv.lock`
  (`.github/scripts/check_constraints_lock.py`).

#### CDS cross-check against QuantLib

- `python/equity/06-credit-risk/tests/test_isda_cds_quantlib.py`: three
  flat-curve cases against QuantLib `IsdaCdsEngine` (NoBias, Taylor).
  Protection legs agree to 1e-12 relative, risky annuity and par spread to
  1e-7 relative, trade-date value to 1e-8. Skipped unless QuantLib is
  installed; QuantLib is not a CI dependency.

#### Shared normal CDF

- `qp_core.normal` (`norm_cdf`, `norm_pdf`): one definition for the option
  packages, which carried their own copies. Same `erfc` expression, so
  results are unchanged. The foundations replication project keeps its
  from-scratch copy on purpose. The new FX model modules use
  `scipy.special.ndtr` and define no copy of their own.

#### CDS business-day calendars

- `qp_core.holidays`: the rules-based settlement calendars (USD, EUR, GBP,
  JPY, CHF, CAD, AUD, 2010 to 2099) moved here from `fx_options.holidays`,
  which is now a thin re-export, so `eq_credit` can use them without an FX
  dependency. New `is_business_day`, `adjust_following` and
  `add_business_days` take a centre or joint code (`"USD"`, `"EURGBP"`).
  One-off closures: `add_closure` / `clear_closures` at runtime and the
  `extra_closures` context manager as a test hook; the docstring gives the
  procedure for making a closure permanent.
- `eq_credit.isda_cds`: `IsdaCds`, `bootstrap_isda` and
  `upfront_from_quoted_spread` take `calendar=`; coupon payment dates, the
  accrual start and `T + 3` cash settlement roll over that centre's
  holidays. Default `None` keeps the earlier weekends-only behaviour, so
  existing results are unchanged. `eq_credit` now depends on `qp-core`.
  fx_credit has no CDS instrument, so nothing changed there.
#### Forward log

- `research/forward/forward.py` and `.github/workflows/forward-log.yml`:
  a weekday job (16:30 UTC) appends one JSON line per ECB business day to
  `forward/log.jsonl` on the unprotected branch `forward-log`, with the
  rates, the two registered strategies' state as registered, the commit and
  the SHA-256 of the ECB response. It refuses any date before 2026-11-03,
  so the 2022-2025 hold-out can never enter it. Idempotent per date; first
  value wins. Least privilege (`contents: write`), pinned actions,
  actionlint clean. Off until the repository variable
  `FORWARD_LOG_ENABLED` is set to `true`. 16 tests on synthetic data, including
  exact parity with `fx_studies.pair_returns` / `pairs_study`. Docs:
  `research/forward/README.md`.
#### FX model twins

- `fx_surface.dupire`: Dupire local vol for FX (total-variance route and
  GK-price route with `q = r_f`), local vol at a quoted delta under all
  four delta conventions, and local-vol Monte Carlo under the domestic
  measure.
- `fx_surface.ssvi`: SSVI/eSSVI with `fit_ssvi_from_pillars`, which reads
  `theta` at the forward rather than at the DNS ATM strike, and
  `ssvi_strike_at_delta`. Finding: SSVI misses five-point USDJPY smiles by
  up to about one vol point (fx_surface VALIDATION F10).
- `fx_surface.bates`: Bates (1996) for FX with the Lewis integral, a
  Merton-series oracle using GK, and an exact `xi -> 0` limit.
- `fx_vol.har_rv`: HAR-RV with realised variance on 17:00 New York FX
  days (DST-aware, weekends dropped) and 260-day annualisation.
- `fx_var.fx_attribution`: currency mapping for Euler VaR/ES attribution,
  option exposure with the premium-adjusted delta for base-currency
  reporting, and additive grouping.
- Each module is checked against its equity twin (agreement to 1e-12 or
  better where the maths coincides). 44 new tests.
#### Forward-hold-out pre-registrations

- Two new draft registrations, unsigned, whose hold-out is forward ECB data
  from 2026-11-03 (evaluated from the forward log after 24 months):
  `research/prereg/pairs-fx-g10-v2.md` (signed selection rules with a
  level-based z instead of the Kalman one-step z, trial count 7, simulated
  power check in `research/studies/pairs_power.py` including JPY-like price
  levels) and `research/prereg/momentum-fx-g10.md` (monthly cross-sectional
  spot momentum, 63-day lookback, long 3 / short 3, trial count 1). Both
  take formation history from one ECB file to 2026-11-02 fetched at
  signing, and score 2026-11-03 to 2028-11-02.
- `research/studies/forward_studies.py` runs both on training data only;
  results in `research/results/pairs_fx_g10_v2_train.json` and
  `momentum_fx_g10_train.json` (training net Sharpe: pairs v2 -0.11,
  momentum -0.20).
- `research/forward/SPEC.md`: the contract between the registrations and
  the forward log's schema 1, the history-file rule and the
  recompute-from-rates evaluation.
- `research/forward/evaluate.py` scores both forward registrations from the
  log and the history file; before 2028-11-13, with a gap in the log or
  while unsigned it prints interim numbers only, never a verdict.
- `research/tools/prereg.py sign` fills the signature block and refuses to
  sign a registration that names no recorded simulated power result
  (`research/power/*.json` with per-cell `reps` and `rejection_rate`).

#### Option VaR and American pricing

- Option VaR vol scenario modes. `eq_var.revaluation` and
  `fx_var.revaluation` `full_revaluation_pnl(..., vol_mode=...)` take
  `"sticky_strike"` (default, unchanged results), `"sticky_delta"` (the
  smile moves with spot through a per-leg `skew`, new `OptionLeg.skew`
  field) and `"historical_atm"` (vol shocks are historical log changes of
  ATM vol, applied proportionally to every strike); new `shocked_vol`.
  The same three modes are in all four compiled engines (C++
  `full_revaluation_pnl(..., VolMode, skews)`, Rust
  `full_revaluation_pnl_vol_mode`), pinned to the Python reference at
  1e-12 relative.
- Option VaR drivers take vol scenarios in all four compiled engines. The
  vol shocks are extra factors on the spot axis: equity uses a combined
  panel or covariance of 2J columns (spot block, then one vol column per
  factor; C++ `historical_option_var_es_vol` / `monte_carlo_option_var_vol`,
  Rust `historical_option_var_es_vol` / `monte_carlo_option_run_vol`); FX
  names them `VOL:CCY` beside `FX:CCY` (C++ `historical_var` /
  `monte_carlo_var` overloads taking `VolScenarios`, Rust
  `historical_option_var_vol` / `monte_carlo_option_var_vol`). Historical
  VaR/ES pinned to Python at 1e-11 for all three modes; Monte Carlo checked
  against a closed-form vol-only quantile. Sticky strike with a zero vol
  block reproduces the old drivers bit for bit.
- `eq_options.alo`: the `q < r < 0` double boundary is now solved by
  Chebyshev collocation (Andersen and Lake 2021 approach; closing time as
  an extra unknown fixed by smooth fit), about 15 ms a price and within
  1e-5 of the extrapolated tree. The time march stays as
  `double_boundary_method="march"`, the reference and the fallback.
  `alo_double_boundary` takes the same `method` switch.
- `fi_rates.DiscountCurve(..., "monotone_convex", positive_forwards=True)`:
  the Hagan-West (2006) Section 5 positivity collar on the node forwards.
  Off by default.
- Fixed: `fi_rates` monotone convex: two equal adjacent discrete forwards (equal
  up to round-off) made the forward jump at the pillar, because every
  Hagan-West shape squeezes its transition to a point there. That boundary
  now uses the plain quadratic: continuous, pillars still reprice.
- Fixed: `eq_options.alo` double-boundary march: with fine grids (`n_march` 1600
  and up) the first nodes sit at tau ~ 1e-7 and `K - x - put` cancelled to
  noise, so the march stopped after one node and returned a wrong price. The
  residual now goes through put-call parity with `expm1`, and the Newton
  tolerance scales with tau.
- Fixed: mypy errors in `eq_var.stress_testing` and `fx_var.monte_carlo_var`
  (numpy scalar types; no behaviour change).

#### Fixes: options

- `eq_options.alo` now prices negative rates. Single boundary for `r > 0`
  (any `q`) and `r = 0 > q`; European where the carry `r K - q S` never
  pays (`r <= 0`, `q >= r`); and the double boundary of the `q < r < 0`
  put (`r < q < 0` call), where the exercise band is found from the two
  value-matching integral equations (Andersen and Lake 2021) by marching
  in time. New `alo_double_boundary` returns the band. Eight new tree
  comparisons, all within 2e-5.
- `eq_options.sabr_pde`: arbitrage-free SABR (Hagan et al. 2014) by a
  conservative finite-volume scheme on a grid uniform in Hagan's `z`, with
  TR-BDF2 or implicit Euler steps (Le Floc'h and Kennedy 2014). Mass and
  forward conserved to 1e-12, non-negative density and convex call prices
  in the case where the expansion's density is negative.
- `sharpe_tests` (equity and FX): `sharpe_diff_hac(kernel="qs_pw")`, the
  Andrews-Monahan prewhitened quadratic-spectral HAC with Andrews'
  automatic bandwidth that Ledoit and Wolf (2008) use, plus
  `hac_cov_qs_prewhitened`, `qs_kernel` and `qs_bandwidth`.
  `sharpe_diff_bootstrap` is now the Ledoit-Wolf studentised bootstrap by
  default; `studentize=False` gives the previous centred bootstrap.
- Research hold-out entry point: `fx_studies.py --mode holdout` scores 2022-2025 once, behind signed-hash, clean-tree and write-once guards, with provenance in every result; `--dry-run` runs the same path on synthetic fake-dated data; the pairs study now reports the deflated Sharpe (trial count 3). `prereg.py check` validates hold-out provenance.

#### Engine switch

- `qp_core.engines` (`check_engine`, `load_engine`, `engine_available`) and
  an `engine` module in `eq_options`, `fx_options`, `eq_var` and `fx_var`
  whose functions take `engine="python" | "cpp" | "rust"`. `"python"` is the
  default and calls the reference unchanged. An unknown name raises
  `ValueError`; a missing extension raises `ImportError` with the install
  command.
- `tests/test_engine_differential.py`: each compiled engine against the
  Python reference on 10,000 random inputs per group, at the binding-test
  tolerances. CI job `engine differential` builds both bindings and fails
  if any compiled case is skipped.
- The four packages now declare `qp-core` as a dependency, because their
  `engine` modules import it.

#### Fixes: FX smile and calendars

- `fx_options.smile.smile_strangle_from_broker` and
  `broker_strangle_from_smile`: broker (market) 25-delta strangle to smile
  strangle conversion and back (Reiswich and Wystup 2010, section 3.2;
  Clark 2011), with `broker_strangle_premium`.
- `fx_options.holidays`: rules-based settlement calendars for USD, EUR,
  GBP, JPY, CHF, CAD and AUD, 2010 to 2099, deterministic and offline;
  `holidays="auto"` in `spot_date` and `expiry_and_delivery` uses the
  pair's joint calendar.

#### Research platform

- `data/`: free-data fetchers (`data/fetch/qp_fetch.py`) for ECB reference
  rates and FRED series, each download stored under the git-ignored
  `data/raw/` with a `.sha256` file. FRED third-party series are refused. A
  parser for LOBSTER files placed by hand.
- `research/`: pre-registrations `pairs-fx-g10`, `regime-fx-basket` and
  `execution-lobster`; `research/tools/prereg.py` (`hash`, `check`); the
  Johansen trace test and a Kalman hedge ratio in `research/studies/models.py`
  (oracles: statsmodels `coint_johansen`; recursive least squares); the
  `training_only` guard; training-period pairs and regime results on ECB
  data, 1999 to 2021. CI job `research (prereg check, tests)`.
- `pairs-fx-g10` and `regime-fx-basket` signed by the maintainer on
  2026-11-02. `execution-lobster` stays unsigned.
- Training-period results are not out-of-sample evidence. Pairs: net Sharpe
  0.24, block-bootstrap 95% interval [-0.17, 0.46], a position on 79 of
  4,867 days, trial count 3. Regime: the hypothesis (negative mean return
  in the high-volatility state) is not supported in training, HAC t = +0.27.

#### Options

- `eq_options`: `lr_tree` (Leisen-Reimer), `pde` (Crank-Nicolson, American),
  `alo` (Andersen-Lake-Offengeim), `dividends` (escrowed spot, RGW, tree),
  `term_structure`, `bachelier`, `sabr`, `greeks_higher` (charm, speed,
  colour, zomma, veta), `risk_grids`.
- `fx_options`: `smile` (three-point smile from ATM, RR and BF), `premium`
  (premium styles), `cuts` (spot lag, expiry and delivery from tenor, cut
  times, ACT/365F time to the cut).
- C++ `eqopt` and Rust `eq-options-engine` ports of Leisen-Reimer, Bachelier
  and the higher-order Greeks, matched to the Python values to 1e-12
  relative (1e-13 for Bachelier).
- Each model has an oracle test; the list is in the README and in the
  package `docs/VALIDATION.md`.

#### Volatility and rates

- Dupire local vol (`eq_surface.dupire`), SSVI and eSSVI (`eq_surface.ssvi`),
  Bates (`eq_surface.bates`), HAR-RV (`eq_vol.har_rv`), OIS multi-curve
  bootstrap (`fi_rates.multicurve`), Black and Bachelier caps and swaptions
  (`fi_rates.rate_options`), Hull-White one-factor with Jamshidian swaptions
  (`fi_rates.hull_white`), float-float cross-currency swaps with constant
  and MtM notional (`fx_rates.xccy_float`).

#### Credit and portfolio

- `eq_credit`: `lgd_ead`, `calibration` (central tendency, margin of
  conservatism, CRR 3 bp floor), `migration` (cohort matrix, lifetime PD,
  generator), `cds` (hazard bootstrap with accrual on default).
- `fx_credit`: `xva` (bilateral CVA/DVA, CSA threshold, MTA and MPoR,
  wrong-way risk), `saccr` (SA-CCR for FX, CRE52). `fx_credit.cva` stays
  the deprecated alias.
- `eq_port` and `fx_port` (same code in both): `hrp`, `robust_mvo`
  (resampled and robust), `cost_qp` (L1 and quadratic costs, turnover cap),
  `factor_model`, `tracking_error`, `sharpe_tests` (HAC delta method,
  circular block bootstrap).

#### VaR and ES

- `eq_var` and `fx_var` (same file in both): `risk_attribution` (Euler
  marginal, component and incremental VaR and ES), `frtb_ima`
  (liquidity-horizon cascade, stressed ES, IMCC, capital, desk backtest,
  PLA test), `stressed_var` (stress-window search, Basel 2.5 capital),
  `evt` (POT GPD fit, VaR and ES), `copula_mc` (t-copula MC), `revaluation`
  (option full revaluation, Taylor P&L, P&L explain), `limits`.
- `option_reval` in all four compiled VaR engines (C++ and Rust): BSM price,
  Greeks, full-revaluation and delta-gamma P&L, matched to Python goldens to
  1e-11.
- Option VaR in all four compiled VaR engines : historical and
  Monte Carlo VaR/ES reprice every option leg in full in every scenario
  (BSM for equity, Garman-Kohlhagen for FX), with the linear path unchanged
  for books without options. Equity: `historical_option_var_es`,
  `monte_carlo_option_var` (C++) / `monte_carlo_option_run` (Rust),
  `option_book_pnl`. FX: option overloads of `historical_var` and
  `monte_carlo_var` (C++), `historical_option_var` and
  `monte_carlo_option_var` (Rust). Historical VaR, ES and scenario P&L match
  the Python revaluation at 1e-11 relative (goldens from
  `cpp/equity-var-engine/tools/gen_option_var_golden.py`); Monte Carlo VaR
  lands within 4 bootstrap SEs of a closed-form quantile. Vol is held at
  today's level per scenario; FX option books must report in USD and
  historical option VaR takes `horizon_days = 1` only.
- The BCBS MAR paragraph numbers quoted in `frtb_ima` have not been
  re-checked against the bis.org text.

#### Fixes: rates and credit

- `fi_rates.curve`: `monotone_convex` interpolation (Hagan and West 2006,
  Section 5) and `DiscountCurve.instantaneous_forward`. Pillars reprice
  exactly; the forward is continuous, local and keeps the monotonicity of
  the inputs.
- `fi_rates.hull_white`: two-curve swaptions (OIS discounting, separate
  forward curve, Henrard 2010 deterministic spread) and
  `calibrate_hull_white` to a swaption vol grid (normal or Black), with
  `(a, sigma)` or `sigma` alone; checked by Monte Carlo and round trip.
- `fx_rates.xccy_float`: MtM convexity adjustment (`MtmConvexity`:
  lognormal spot, Hull-White quote rate, correlation; checked by Monte
  Carlo) and seasoned valuation between resets (`elapsed` with the current
  fixings).
- `eq_credit.isda_cds`: CDS on ISDA standard-model conventions (dated IMM
  schedule, step-in, ACT/360, exact accrual on default, a real discount
  curve), dated par-spread bootstrap and quoted-spread to upfront
  conversion. Not checked against ISDA's licensed test grids.

#### Performance kernels

- `qp::core` and `qp-core`: Philox4x32-10 (Random123 known-answer vectors
  pass in both languages), O(n) quantile selection, cache-blocked GEMM and
  `D^T D` (bit-identical to the naive loops), branch-free batch `exp`,
  `erfc` and normal CDF (documented tolerances, opt-in), a persistent Rust
  thread pool, and the C++ `bench_core` latency bench. CI job
  `cpp core (<preset>)` builds and tests `cpp/core` and runs `bench_core`
  report-only.
- Measured on one laptop (SSE2 target): selection 7.3x faster than a full
  sort and blocked GEMM 3.2x faster than naive at p50, but the batch `erfc`
  and `exp` are slower than libm (0.66x and 0.38x). They are not used by
  any engine.

#### Shared code, bindings and build (earlier work)

- `qp-core` 1.0.0 at `python/core` (import `qp_core`): statistics,
  execution fill rule, calendars, day counts, validation and metrics shared
  by 12 Python projects, which declare `qp-core>=1.0`. Their old import paths
  re-export it. The other 11 projects have no shared copies and are
  unchanged.
- Root `pyproject.toml` with a uv workspace (`python/core` and the 23
  projects) and `uv.lock`. The mypy configuration is in the same file.
- `qp::core` at `cpp/core`: a header-only CMake target with the code the four
  C++ engines used to copy (normal CDF, implied-vol core, RNG, special
  functions, matrix and linear algebra, Brent, thread pool, `expected`,
  latency bench helper). `CMakePresets.json` in each engine and in
  `cpp/core` with the presets `gcc`, `clang`, `gcc16-cxx26`, `asan`, `tsan`
  and `msvc`. `QP_CXX26=ON` compiles with C++26 (contracts, `std::simd`
  where available). Root `.clang-format` and clang-tidy configuration.
- Rust: `rust/` is one Cargo workspace with a fifth crate, `qp-core` 0.1.0
  (standard library only, `forbid(unsafe_code)`): RNGs, special functions,
  implied-vol core, `brentq`, Cholesky, thread partitioning. One
  `Cargo.lock`; toolchain pinned in `rust/rust-toolchain.toml` (1.99.1). The
  bench binaries take `--out <path>`.
- `qp-cpp` 0.1.0 (`bindings/cpp-nanobind`, nanobind) and `qp-rust` 0.1.0
  (`bindings/rust-pyo3`, PyO3, abi3 for Python 3.10+): Python bindings for
  the four C++ engines and the four Rust crates. Each is tested against the
  Python reference packages.
- CI jobs: `py core`, `lint` (`uv lock --check`, ruff, mypy), C++ presets on
  GCC and Clang, a GCC 16 container in C++26 mode, MSVC on Windows,
  `clang-format` and `clang-tidy`, a Rust job per workspace crate including
  `qp-core`, `rust fmt`, `bind <package>` and `bench` (fails when a kernel
  whose committed p50 is above 1 us is more than 2x slower).
- CI pins for ruff, mypy, nanobind, scikit-build-core and maturin.
  `pandas-stubs` and `scipy-stubs` are pinned in
  `requirements/ci-constraints.txt` for Python 3.12 and later only; the
  `lint` job installs them for mypy.

#### Infrastructure

- CI runs every Python project in its own environment on Python 3.10, 3.11,
  3.12 and 3.13, the four C++ engines with GCC 15 and Clang 21 plus an
  AddressSanitizer and UBSan build, and the four Rust crates on a pinned
  toolchain (1.99.1). A final `ci-ok` job is the single required check.
- `goldens --check` job: regenerated C++ headers and Rust modules must match
  the committed golden JSON.
- `counts.json` CI artefact with the executed pass counts per project.
- `release.yml`: tag-on-green releases with notes taken from this file.
- `requirements/ci-constraints.txt`: one resolved set of dependency versions
  for CI, valid on Python 3.10 to 3.13. `mpmath` added for the options oracle.
- Dependabot for pip, cargo and GitHub Actions. CHANGELOG, SECURITY,
  CONTRIBUTING, CODEOWNERS, `.editorconfig`, `ruff.toml` (report-only) and a
  pre-commit configuration.
- `hypothesis` and `mpmath` (Python) and `proptest` (Rust) as test-only
  dependencies.

#### Options (equity and FX; Python, C++, Rust)

- Golden vectors from an mpmath oracle: 456 equity and 434 FX records, each
  output with its own tolerance (records of `[value, tol]`). The Python
  consumer is `tests/test_golden.py`.
- Differential tests from Python against the compiled engines through JSONL
  shims: Rust `eq_jsonl`/`fx_jsonl`, C++ `eqopt_jsonl`/`fxopt_jsonl`, located
  by `QP_RS_EQ_JSONL`, `QP_RS_FX_JSONL`, `QP_CPP_EQ_JSONL`, `QP_CPP_FX_JSONL`.
  They skip when the binaries are absent.
- C++ FX: `Seed` and `McConfig` types for Monte Carlo.
- CMake options `EQOPT_WERROR`, `FXOPT_WERROR`, `EQVAR_WERROR` and
  `FXVAR_WERROR`. CI sets them ON.

#### VaR and ES (equity and FX; Python, C++, Rust; foundations risk metrics)

- `cornish_fisher_valid(skew, ex_kurt, alpha)`: closed-form validity check on
  [-z_a, z_a].
- `cholesky(cov, repair=None|"clip"|"higham")` with Higham (2002)
  alternating projections. Jitter is capped at 1e-6 times the mean diagonal.
- `acerbi_szekely_z2` with the paper's sign and a simulated p-value
  (Python); z2 only in Rust and C++.
- `monte_carlo_risk`: one simulation returns P&L, VaR, ES and a bootstrap SE.
  `se_var_kde` stays as a diagnostic.
- Per-factor filtered historical simulation
  (`factor_filtered_historical_var` in equity).
- `KupiecInput` struct in Rust and C++. The Basel result carries
  `plus_factor`.
- 49 `conv_*` golden keys shared by both families, plus a `versions` block.
- Input fuzz tests over a registry of every public entry point.

#### Volatility, surfaces, rates, replication

- `robust_std_errors`, `default_hac_lags`, `egarch_unconditional_variance`,
  `yang_zhang_var`.
- `smile_quotes_from_broker`, `broker_strangle_from_smile`,
  `butterfly_report`, `delta_calendar_report`.
- FX rates: `calendars` module, `BasisSwap`, `basis_discount_curve`,
  `tenor_dates`, `money_market_year_fraction`, `carry_rolldown(repo_rate=)`,
  `frn_price_from_curve(current_fixing=)`.
- The Black-Scholes replication pipeline takes `--out` and `--plot`.

#### Alpha research (pairs, execution, regime, foundations backtest)

- `ExecutionConfig(mode="next"|"close_auction", fill_fraction=0.5,
  auction_cost_bp=...)` in all seven backtests.
- `research_stats` module in all seven: Lo Newey-West Sharpe SE, deflated
  Sharpe ratio, PBO by CSCV, block bootstrap.
- eq_pairs: `CostModel` impact, financing and short-rebate fields;
  `COST_COLUMNS`.
- fx_pairs: `cointegration_funnel`, `hedged_vol_scale`.
- eq_algo: `capacity_backtest`, `DailyPanel.trailing_adv_dollars`, ledger
  `borrow` column, `BacktestConfig.borrow_bps_annual`, schedule columns
  `expected_is_bps` and `se_mean_is_bps`.
- fx_algo: `BARS_PER_YEAR`, `fix_print`, `fix_self_impact_pips`, ledger
  `fill_price`, summary key `sharpe_se`.
- eq_regime: `backtest_from_detection`, `random_timing_weights`,
  `vol_target_weights`, `causal_min_duration`, `regimes_known_at_decision`,
  `equal_drift_params`.
- fx_regime: `QUOTES`, `PIP_SIZE`, `half_spreads`, `quote_levels`,
  `generate_roro_panel(equal_drift=)`, `random_timing_regimes`,
  `DetectionConfig.cold_restart_every`.
- Example pipelines write to a temp directory or `--out`, never the tree.

#### Credit and portfolio

- eq_port: `max_sharpe_qp`, `MaxSharpeResult`, `sharpe_difference_test`.
- fx_port: `sleeve_turnover`, `hedge_ratio_inference`, `out_of_sample_hedge`.
- eq_credit: `pd_backtest_by_grade`, `finite_portfolio_quantile`,
  `granularity_adjustment`, `mc_quantile_ci`.
- fx_credit: `pd_backtest_by_grade`, `bcbs_window_hours`,
  `expected_sign_check`, `ConvergenceWarning`, `expected_exposure_loss`,
  `expected_exposure_loss_for_forward`.

#### Docs

- README, CONVENTIONS and the site describe the three-stage plan (study
  portfolio, research platform, desk library), point at CI for status and
  carry an engine latency table for the latency pass to fill.
- ARCHITECTURE and DIAGRAMS describe the differential-test pipeline and the
  CI jobs.

### Changed

- One latency method for the C++ and Rust benches (`qp/bench/latency.hpp`,
  `qp_core::bench`): batches of back-to-back calls sized to at least 2 us,
  median with p10/p90, nearest-rank percentiles. The CI bench job now builds
  the base and head commits on one runner, runs them in three alternating
  rounds and fails only when the head median is more than 25% slower with
  head p10 above base p90 (`.github/scripts/bench_ab.py`, replacing
  `bench_compare.py` and its 2x rule against laptop numbers). README latency
  tables are labelled as the older laptop run. See `docs/BENCHMARKS.md`.

- C++ equity and FX `try_mc_price` (and `mc_price`) run their chunks inline
  below 16,384 independent draws (`kMcParallelMinDraws`) instead of on the
  thread pool. At 10,000 paths, 4 threads measured 531 us against 424 us on
  one thread; results are bit-identical. Crossover table in each engine's
  `docs/METHODOLOGY.md`.

- Python 3.10 dropped (end of life 2026-10-01). Every project and both
  bindings now declare `requires-python = ">=3.11"`; the CI matrix runs 3.11
  to 3.13; ruff targets `py311`. `uv.lock` and
  `requirements/ci-constraints.txt` were regenerated for 3.11 and later,
  which only removes the 3.10-only pins (no version changed for 3.11+).
  Python 3.15 is not in the matrix yet.
- Toolchain upkeep (2026-11-02): rechecked current stable releases (GCC 16.2,
  Clang 23.1.2, MSVC 14.51 in Visual Studio 2026 18.10, Rust 1.99.1, Python
  3.15.2). `uv.lock`, `requirements/ci-constraints.txt` and both
  `Cargo.lock` files were already at the latest compatible versions, so no
  pin moved. Contracts and `std::simd` are still GCC-only on cppreference
  and stay behind their feature-test macros; `cpp/core/README.md` records
  the dated status.

#### Later work

- Equity VaR sample covariance (C++ and Rust) runs on the blocked `D^T D`.
  Results are bit-identical; no golden changed.
- `sharpe_diff_bootstrap` (equity and FX `sharpe_tests`) now studentises by
  default, following Ledoit and Wolf (2008), so its p-value for the same
  inputs differs from earlier versions. Pass `studentize=False` to get the old
  centred-bootstrap result.
- No other existing function, golden or pinned number changed in the later work.

#### Earlier work

- Equity EWMA covariance seeds from the zero-mean second moment mean(r r')
  of the first min(30, T) rows (convention S8) in Python, C++ and Rust, so
  the forecast at a date uses no later data.
- Rust eq-options `fast::try_mc_price` rejects `threads == 0` with
  `ErrorCode::BadThreads`, as the C++ engine does. The other entry points
  keep their behaviour; CONVENTIONS.md lists which reject 0 and which treat
  it as 1.
- Rust crate versions: eq-options 0.2.0, fx-options 1.1.0, eq-var 0.3.0,
  fx-var 0.3.0. The engines take their shared numerics from `qp-core`. No
  number moves: a bit-identity harness and the JSONL shims give
  byte-identical output against the previous main.
- The C++ engines take their shared code from `qp::core` with byte-identical
  results. CMake 3.25 is now the minimum. All C++ is formatted with the root
  `.clang-format` and builds clean under MSVC `/W4 /WX`.
- `HestonParams` uses one field order, `(v0, kappa, theta, xi, rho)`, in
  `eq_surface` and `fx_surface`. Pass the fields by keyword.
- Python: ruff and mypy report no findings over `python/`. No result
  changed.
- `bindings/rust-pyo3/Cargo.lock` includes the `qp-core` crate.

#### VaR and ES (behaviour and golden changes)

- `alpha` is the tail probability in (0, 0.5) everywhere. FX defaults move
  from 0.99 to 0.01 and from 0.975 to 0.025. A positional 0.99 now raises.
  Rust and C++ FX functions take the tail probability and return `Err` or
  throw on 0.99.
- Quantile: type-7 interpolation is replaced by the order statistic
  L(ceil(n alpha)). ES is the exact Acerbi-Tasche estimator with the
  fractional atom. Foundations ES moves from the naive tail mean to
  Acerbi-Tasche. Golden: `a_hist_var_1pct` 122.4129 to 141.7568,
  `a_hist_var_5pct` 104.5523 to 108.2349.
- Kupiec: keyword-only `kupiec_pof(*, n_obs, n_exceptions, alpha)`.
  `n_exceptions > n_obs` raises. Kupiec and Christoffersen goldens move in
  the last digits (log1p).
- Basel traffic light: one rule for any window and alpha (binomial CDF
  cut-offs 0.95 and 0.9999, plus factor from the 250-day equivalent).
  `basel_zone_250d` becomes `basel_zone`.
- Acerbi-Szekely Z2 uses the paper's sign (negative means ES understated) and
  returns a p-value. Golden `conv_as_z2` is -3.18545 (rounded).
- EWMA variance is seeded with the mean squared return of the first
  min(30, n) observations (`seed_window=`).
- Monte Carlo VaR: Poisson jump counts over the horizon, log-spot shocks,
  floored vol, bootstrap SE as the primary `se_var`.
- An empty book raises in every engine.
- Rust FX functions return `Result`.
- Equity full revaluation is vectorised (59 s to 0.03 s on the synthetic demo
  book).
- Reverse stress: the delta case is closed form. The delta-gamma solver works
  in whitened coordinates and raises when it finds no feasible point.
- `statsmodels` is no longer a dependency of eq_var. fx_var is version 1.1.0;
  the Rust VaR crates are 0.2.0.

#### Options

- All six engines share one out-of-the-money Halley implied-vol core with no
  bisection stage (mean 2.6 price evaluations, max 10).
- FX implied vol: a Brent failure throws. The premium-adjusted bracket is
  derived from the Mills ratio.
- Antithetic plus control-variate Monte Carlo estimates beta on pair
  averages (SE 0.0247 to 0.0087 on the test case).
- CRR trees precompute node spots.
- Monte Carlo with too few samples raises (3 with a control variate).
- An S = 0 American put with r < 0 returns K e^{-rT}. Overflow raises. C++ FX
  sigma = 25 throws.

#### Volatility, surfaces, rates, replication

- fx_vol fitters raise `ConvergenceError` on failure.
- Diebold-Mariano HAC lag default is max(h-1, floor(4 (n/100)^(2/9))) with
  the HLN correction. Variance targeting uses the mean squared return.
- EGARCH initialisation and unconditional-variance semantics corrected. FX
  GJR allows a negative gamma subject to alpha + gamma >= 0.
- `FXVolSurface` and `build_surface` default to `delta_convention="native"`.
- FX `implied_vol` no longer returns a bracket edge. `price_cos` uses L = 24.
  `calibrate_heston` uses N = 512 and `n_starts=3`.
- Equity `fit_svi` is constrained by default (`arbitrage_free=False` gives
  the old fit). The wing bound is b(1 + |rho|) <= 2 on total variance (Lee).
- Equity Heston QE uses `martingale_correction=True`.
- Fixed income: `generate_schedule` and `FixedRateBond` default to
  `eom=True`; government bonds use ACT/ACT-ICMA.
- FX rates: the spot date counts business days. Basis quotes are basis-swap
  spreads.
- Black-Scholes replication: `implied_volatility` takes `option_type` and
  warns on NaN. The normal CDF uses `math.erfc`.

#### Alpha research

- Default fill timing in all seven backtests moves from the decision close to
  a t+1 modelled fill p_t + f (p_{t+1} - p_t), f = 0.5. The old timing is
  `ExecutionConfig("close_auction", auction_cost_bp=0)`.
- eq_algo: impact coefficient default 0.1 to 0.7 (`BacktestConfig` and
  `capacity_curve`). Intraday temporary impact uses bucket volume.
- eq_regime: `run_ledger` charges 50 bp a year on leverage by default.
- fx_regime: `SPREAD_PIPS` values are in market pips. Spot P&L is on simple
  returns, carry on ACT/365. `detect_columns` drops `usd_str`.
- fx_pairs: `fit_ou_ols` and `fit_ou_mle` raise for phi <= 0.
  `sharpe_se_lo` is the Lo Newey-West value with lag floor(4 (T/100)^(2/9)).
- `fit_hmm` appends the final likelihood when it stops at `max_iter`.
- foundations-02: scipy is a runtime dependency.

#### Credit and portfolio

- `max_sharpe_constrained` falls back to minimum variance when max mu <= rf.
  It used to maximise volatility.
- `sharpe_lo` uses Newey-West with a T^(1/3) Bartlett lag.
- FX solvers raise on failure. FX closed forms reject matrices that are not
  positive definite.
- `basel_k`: 5 bp PD floor, M clipped to [1, 5], 1.25 multiplier for
  financial institutions.
- FX WoE sign is good over bad. Equity categorical WoE raises on unseen
  categories. WoE smoothing is renormalised.
- `stepwise_select` raises on non-convergence. `carry_sizing` returns 0 for a
  non-positive mean.

#### Docs and tests

- `tests/test_docs_counts.py` no longer compares hand-typed test totals. Test
  counts come from `counts.json` and the docs do not repeat them. The test
  checks every quoted LEARN question count and COOKBOOK recipe count against
  the files, and fails on a hand-typed test total.
- LEARN.md answers and COOKBOOK.md recipes follow the current APIs. The C++
  and Rust VaR engines are described as revaluing linear positions and
  repricing European option legs in full.

### Deprecated

Each stays for one release with a `DeprecationWarning` (Python),
`#[deprecated]` (Rust) or `[[deprecated]]` (C++).

- Positional `HestonParams(...)` in `eq_surface` (old order
  `(v0, kappa, theta, rho, xi)`) and `fx_surface`.
- Options: Python `implied_vol` arguments `tol`, `max_iter`, `sigma_lo`,
  `sigma_hi` (ignored); `IV_PRICE_TOL`; the old C++ `implied_vol` overloads.
- VaR: positional `kupiec_pof`; the FX `confidence=` keyword; FX
  `es_backtest_acerbi_szekely`; the two-argument `cornish_fisher_valid`;
  `safe_cholesky`, `robust_cholesky` and Rust `cholesky_jitter`; P&L-level
  FHS; positional equity `basel_traffic_light`; the type-7 quantile helper.
- Volatility and rates: `calendar_arbitrage_report(deltas=)` (use
  `delta_calendar_report`); FX rates calendar-day spot; `tenor_to_years` for
  ON/TN/SN; `basis_model="zero_spread"`.
- Alpha: eq_algo `DailyPanel.adv_dollars`; eq_regime
  `detection.min_duration_filter` (not causal); fx_regime `PIP_FRACTION`;
  `carry_accrual` without `year_fraction`.
- Credit: `hosmer_lemeshow` without `out_of_sample`; FX `cva` and
  `cva_for_forward`. Use `expected_exposure_loss` and
  `expected_exposure_loss_for_forward`: an expected loss on exposure under a
  real-world PD, not a CVA.

### Removed

- C++ FX `mc_price` overload taking a raw `uint64` seed (breaking). Use
  `Seed{}` or `McConfig`.
- The equity Acerbi-Szekely `reject` key and the fixed -0.70 flag.
- From the docs: hand-typed test totals, "flagship", "twins" for the VaR
  engines, review-history narration (now only in this file) and the banned
  hiring-process word in CONVENTIONS.md.

### Fixed

- CI previously stopped at the first failing project and installed no test
  extras, so 22 of 23 Python projects had never run in CI.
- Odd `n_paths` with antithetics in FX Heston Monte Carlo now has a test
  (`test_odd_n_paths_with_antithetic_raises`).
- The Cornish-Fisher example (skew 0.122, excess kurtosis -0.427) is valid at
  alpha = 0.01 under the closed-form rule on [-z_a, z_a]. Its minimum
  derivative is at the endpoint z = -4.
- Docs contradictions C1 to C19 from the 2026-11-01 review: golden tolerance
  statements, "independent" codebases, bootstrap SE coverage, stale
  Cornish-Fisher grid text, dangling references, the "Real Data" title of a
  synthetic-data project, and the descriptions of CI and the count mechanism.

### Known issues

These are gaps in this release that a user can hit. Each is stated so the
number it affects can be read with the right caveat.

- The ISDA CDS standard model (`eq_credit.isda_cds`) has not been compared
  with the ISDA published test grid, because downloading the grid needs a
  licence click-through. Its in-repo checks are internal (identities and
  hand examples) plus an optional QuantLib 1.44 `IsdaCdsEngine` test
  (skipped unless QuantLib is installed; not run in CI).
- The ISDA CDS module defaults to a weekends-only calendar; pass
  `calendar=` to roll over a centre's holidays.
- One-off closures (a state funeral, say) must be added with
  `qp_core.holidays.add_closure` until a dated rule is written.
- `fx_surface.ssvi` misses five-point FX smiles by up to about one vol point
  on USDJPY (about 0.5 on EURUSD); eSSVI does not help (fx/09-vol-surface
  VALIDATION F10).
- The double-boundary Andersen-Lake-Offengelden price (`q < r < 0` put)
  takes about 1 s per price, because it marches the two boundaries in time.
- Option VaR holds implied vol at today's level in every scenario; there is
  no vol factor. FX option books must report in USD, and historical option
  VaR needs a one-day horizon (pass h-day returns for longer horizons).
- The positivity collar of the monotone-convex curve (Hagan and West) is not
  implemented, so forwards can go negative on steep inputs.
- Dupire local vol, SSVI, Bates, HAR-RV and Euler attribution exist for
  equity only; the FX packages do not have them.
- The `pairs-fx-g10` pre-registered strategy is nearly dormant under its
  registered variance rule: in training it trades in 3 of 19 test years (79
  active days). That is the registered design, not a code fault; changing the
  rule would be a new registration. The reasoning is in `research/README.md`.

### Known limitations

- The `engine=` switch does not cover Monte Carlo pricing, MC VaR, the
  backtests or the newer compiled ports (Leisen-Reimer, Bachelier,
  higher-order Greeks).
- Not done because the data needs an account: equity pairs on a
  point-in-time universe (Nasdaq Data Link) and the LOBSTER execution study,
  whose pre-registration is an unsigned draft.
- One out-of-sample result exists (the regime hold-out, H0 not rejected).
  The pairs and momentum drafts are scored on forward data only and cannot
  be evaluated before 2028-11-13.
- Engines do not use Philox or the batch kernels. Switching would change
  every Monte Carlo golden, and the batch kernels move prices by up to
  about 1e-15 relative.
- FRTB NMRF/SES (an input), the default risk charge and risk-factor
  eligibility are not implemented.
- The newer Python modules (for example `eq_options.sabr_pde`,
  `fx_options.holidays`, `eq_credit.isda_cds`) are not re-exported from the
  package `__init__`; import them by module path.
- The bindings do not cover the FX VaR Monte Carlo, book-level VaR, deltas
  and strikes from delta, Black-76, Christoffersen or Acerbi-Szekely.
- The `gcc16-cxx26` preset has not been run yet. GCC 16 is not installed
  locally; CI runs the preset in a `gcc:16` container.
- The C++ `uniform_open53` draw can return exactly 1.0 with probability
  2^-53. Fixing it changes the Monte Carlo goldens, so it waits for a release
  that regenerates them.

## [1.1.0] - 2026-09-06

The tag `v1.1.0` was created before CI had run on its commit, and that run
failed. No GitHub Release was published for it.

### Fixed

Code review between 2026-08-19 and 2026-09-06 found four numerical bugs. None
of the fixes changed a golden-vector value.

| Bug | Affected engines | Symptom before the fix | Fix | Regression test |
|---|---|---|---|---|
| Implied-vol solver stopped when Newton progress stalled | Options, equity and FX, Python/C++/Rust (6) | At S=K=100, T=25y, sigma=300% the vol error was 1.8e-2, not the reported 2.4e-5. The FX solver could return a vol that did not reprice the input. | The solver always finishes with Brent or bisection on its bracket. | `test_round_trip_long_dated_high_vol_flat_vega`, `test_long_dated_deep_itm_high_vol_flat_plateau_raises` |
| Cornish-Fisher monotonicity checked on a grid | VaR, equity and FX, Python/C++/Rust (6) | An 801-point grid accepted skew 0.122, excess kurtosis -0.427. That expansion is not monotone near z = 3.1. | The check uses the closed-form minimum of the derivative, a quadratic in z. | `test_domain_check_is_exact_not_grid_resolution_dependent` |
| Monte Carlo VaR standard error biased low | Monte Carlo VaR, all 6 | The density-based SE was 9-17% below the SE measured across replications when alpha >= 0.995 or the scenario count was small. | Added a bootstrap SE (`var_standard_error_bootstrap`). The density-based SE is unchanged. | `test_bootstrap_se_matches_true_sampling_se_at_default_scenario_count` |
| Odd `n_paths` broke antithetic pairing | FX Heston Monte Carlo | The reported SE was too small. | `antithetic=True` with odd `n_paths` raises `ValueError`. | none in 1.1.0 |

### Known limitations

- The Kupiec test uses the chi-squared(1) approximation. At 250 days and 99%
  its actual size is 9.5%, not 5%. Treat p-values near 0.05 as inconclusive.

[Unreleased]: https://github.com/AshJha0/quant-portfolio/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/AshJha0/quant-portfolio/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/AshJha0/quant-portfolio/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/AshJha0/quant-portfolio/releases/tag/v1.1.0
