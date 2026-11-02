# Changelog

All notable changes to this repository are recorded here. The format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) and versions
follow [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

A release is cut by `.github/workflows/release.yml`, which tags a commit only
when the `ci-ok` check passed on that exact commit. The release notes are the
section below whose heading matches the version.

## [Unreleased]

One release, v2.0.0, is planned at the end of all phases unless the owner
decides otherwise. Until then changes collect here. All results quoted below
come from seeded synthetic data.

### Added

#### Round 4: CDS business-day calendars

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
#### Round 4: forward log

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
#### Round 4: FX model twins (fxmodels lane)

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
#### Round 4: forward-hold-out pre-registrations (prereg2 lane)

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

#### Round 3 fixes: options lane

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

#### Round 3: engine switch (lane A)

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

#### Round 3 fixes: FX smile and calendars (fx lane)

- `fx_options.smile.smile_strangle_from_broker` and
  `broker_strangle_from_smile`: broker (market) 25-delta strangle to smile
  strangle conversion and back (Reiswich and Wystup 2010, section 3.2;
  Clark 2011), with `broker_strangle_premium`.
- `fx_options.holidays`: rules-based settlement calendars for USD, EUR,
  GBP, JPY, CHF, CAD and AUD, 2010 to 2099, deterministic and offline;
  `holidays="auto"` in `spot_date` and `expiry_and_delivery` uses the
  pair's joint calendar.

#### Round 3: research platform, Phase 4 (lane C)

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
  2026-11-02 under owner decision 9. `execution-lobster` stays unsigned.
- Training-period results are not out-of-sample evidence. Pairs: net Sharpe
  0.24, block-bootstrap 95% interval [-0.17, 0.46], a position on 79 of
  4,867 days, trial count 3. Regime: the hypothesis (negative mean return
  in the high-volatility state) is not supported in training, HAC t = +0.27.

#### Round 3: options, Phase 5 (lane D)

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

#### Round 3: volatility and rates (lane F)

- Dupire local vol (`eq_surface.dupire`), SSVI and eSSVI (`eq_surface.ssvi`),
  Bates (`eq_surface.bates`), HAR-RV (`eq_vol.har_rv`), OIS multi-curve
  bootstrap (`fi_rates.multicurve`), Black and Bachelier caps and swaptions
  (`fi_rates.rate_options`), Hull-White one-factor with Jamshidian swaptions
  (`fi_rates.hull_white`), float-float cross-currency swaps with constant
  and MtM notional (`fx_rates.xccy_float`).

#### Round 3: credit and portfolio (lane G)

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

#### Round 3: VaR and ES, Phase 5 (lane E)

- `eq_var` and `fx_var` (same file in both): `risk_attribution` (Euler
  marginal, component and incremental VaR and ES), `frtb_ima`
  (liquidity-horizon cascade, stressed ES, IMCC, capital, desk backtest,
  PLA test), `stressed_var` (stress-window search, Basel 2.5 capital),
  `evt` (POT GPD fit, VaR and ES), `copula_mc` (t-copula MC), `revaluation`
  (option full revaluation, Taylor P&L, P&L explain), `limits`.
- `option_reval` in all four compiled VaR engines (C++ and Rust): BSM price,
  Greeks, full-revaluation and delta-gamma P&L, matched to Python goldens to
  1e-11.
- Option VaR in all four compiled VaR engines (fix round): historical and
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

#### Round 3 fixes: rates and credit

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

#### Round 3: performance kernels, Phase 6 (lane B)

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

#### Shared code, bindings and build (Round 2)

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

- Toolchain upkeep (2026-11-02): rechecked current stable releases (GCC 16.2,
  Clang 23.1.2, MSVC 14.51 in Visual Studio 2026 18.10, Rust 1.99.1, Python
  3.15.2). `uv.lock`, `requirements/ci-constraints.txt` and both
  `Cargo.lock` files were already at the latest compatible versions, so no
  pin moved. Contracts and `std::simd` are still GCC-only on cppreference
  and stay behind their feature-test macros; `cpp/core/README.md` records
  the dated status.

#### Round 3

- Equity VaR sample covariance (C++ and Rust) runs on the blocked `D^T D`.
  Results are bit-identical; no golden changed.
- `sharpe_diff_bootstrap` (equity and FX `sharpe_tests`) now studentises by
  default, following Ledoit and Wolf (2008), so its p-value for the same
  inputs differs from Round 2. Pass `studentize=False` to get the old
  centred-bootstrap result.
- No other existing function, golden or pinned number changed in Round 3.

#### Round 2

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

These are gaps in Round 3 work that a user can hit. Each is stated so the
number it affects can be read with the right caveat.

- The ISDA CDS standard model (`eq_credit.isda_cds`) has not been compared
  with the ISDA published test grid, because downloading the grid needs a
  licence click-through. Its checks are internal (identities and hand
  examples).
- The ISDA CDS module uses a weekends-only calendar, so a date that falls on
  a public holiday is not rolled.
- The rules-based holiday calendars omit one-off closures announced after the
  code was written (a state funeral or a royal wedding, say).
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
  backtests or the Round 3 compiled ports (Leisen-Reimer, Bachelier,
  higher-order Greeks).
- Not done because the data needs an account: equity pairs on a
  point-in-time universe (Nasdaq Data Link) and the LOBSTER execution study,
  whose pre-registration is an unsigned draft.
- No hold-out run yet. The research results are training-period only. The
  hold-out entry point exists and has only been run with `--dry-run`.
- Engines do not use Philox or the batch kernels. Switching would change
  every Monte Carlo golden, and the batch kernels move prices by up to
  about 1e-15 relative.
- FRTB NMRF/SES (an input), the default risk charge and risk-factor
  eligibility are not implemented.
- The Round 3 Python modules (for example `eq_options.sabr_pde`,
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

[Unreleased]: https://github.com/AshJha0/quant-portfolio/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/AshJha0/quant-portfolio/releases/tag/v1.1.0
