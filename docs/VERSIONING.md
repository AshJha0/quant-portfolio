# Versioning and API policy

This page says what a version number promises, what counts as the public
API, how long a deprecated name survives, and what the project is built and
tested on. `docs/MIGRATION.md` lists what changed in each release and what
to do about it.

## One version for the whole repository

The top-level `VERSION` file holds the version. Every Python package
(`python/core`, `python/data` and the 23 projects), both binding packages (`qp-cpp`,
`qp-rust`), the Rust binding crate and every `__version__` attribute carry
the same number. `tests/test_repo_policy.py` fails on any mismatch.

Between releases the version is a development release of the next
release, `3.0.0.dev0` today (PEP 440; Cargo spells it `3.0.0-dev.0`). The
release commit changes it to the plain version (`3.0.0`), and `release.yml`
refuses to tag unless `VERSION` equals the version being released.

There is one release ahead: **v3.0.0, after Round 18** (owner decision,
2026-11-03). There are no v2.x tags. On the way, work lands in milestone
pull requests: milestone v2.1 is Rounds 6 and 7, then v2.2, v2.3 and v2.4.
A milestone is a merge to `main` when CI is green, not a release: nothing
is tagged or published, and `release.yml` runs only when started by hand
for v3.0.0 (or as a dry run). `docs/MIGRATION.md` collects the notes under
"Unreleased (toward v3.0.0)", one sub-heading per milestone.

## What a version number promises

The project follows Semantic Versioning 2.0.0 (semver.org):

- **Patch** (3.0.0 to 3.0.1): bug fixes that do not change a documented
  result, plus documentation.
- **Minor** (3.0 to 3.1): new features; deprecations; and fixes that change
  numbers because the old numbers were wrong. A wrong answer is a bug, so
  correcting it does not wait for a major release. Each such change is
  listed in `docs/MIGRATION.md` with the old and new value and the
  independent reference that decided which is right. Where a caller could
  depend on the old behaviour, it stays reachable behind a deprecated flag
  for the deprecation period below.
- **Major** (3.x to 4.0): removal of deprecated names and any other change
  that breaks code written against the public API.

## The public API

A name is public if, and only if, it is listed in the `__all__` of a
package or of a public module (a module whose name does not start with an
underscore). Everything else, including modules starting with `_`,
attributes of public objects that are not documented, and the layout of
`tests/`, `examples/` and `research/`, can change in any release.

For the bindings, the public API is the set of functions exported by
`qp_cpp` and `qp_rust` (their `__init__.pyi` stubs list them), their
argument names and defaults, and the error classes they raise.

The C++ and Rust engines are libraries inside this repository. Their
headers and crates are not published packages and carry no compatibility
promise beyond what the bindings expose.

### How the API is checked

The `api-diff` CI job runs griffe (the `api` dependency group, hash-locked
in `uv.lock`) on every pull request. It compares each package's public API
at the pull request's base commit with the head and fails on a breaking
change: a removed or renamed `__all__` name, a removed parameter, a
parameter that lost its default, or a changed parameter kind. Adding names
or optional parameters passes.

A deliberate break (a removal scheduled in the deprecation table, a major
release) is allowed by putting the label `api-break` on the pull request.
The pull request must then add the matching `docs/MIGRATION.md` entry; the
reviewer checks it.

## Deprecation

Following PEP 387:

1. A deprecated name keeps working for at least **two minor releases**
   after the release that first deprecates it, and is removed no earlier
   than the next major release after that.
2. Using it emits `DeprecationWarning` (`FutureWarning` if the behaviour is
   aimed at end users of a notebook rather than library callers), naming
   the replacement and the release that removes it.
3. The deprecation is recorded in `docs/MIGRATION.md` under the release
   that introduces it, and in the table below.

| Name | Deprecated in | Earliest removal | Replacement |
|---|---|---|---|
| `market_volumes=` (`eq_algo.pov_schedule`; `FutureWarning`) | 3.0.0 (milestone v2.1) | 4.0.0 | `forecast_volumes=` |
| `expected_is_bps` column (`eq_algo.evaluate_schedules`; no warning, a column) | 3.0.0 (milestone v2.1) | 4.0.0 | `model_implied_is_bps` |
| `qp_core.calendars` (module) | 3.0.0 (milestone v2.1) | 4.0.0 | `qp_core.holidays` |
| The `data/live.py` fetchers in `fx_options`, `fx_vol`, `fx_var`, `fx_rates`, `fx_pairs`, `fx_port` and `fi_rates` (seven modules; `DeprecationWarning`) | 3.0.0 (milestone v2.1) | 4.0.0 | `qp_data.fetch` and `qp_data.get` |
| `fx_rates.MarketState` with neither `valuation_date` nor `spot_time`, or `spot_at_t0=True` | 3.0.0 (milestone v2.1) | 4.0.0 | `valuation_date=` or `spot_time=` |

A name deprecated on the way to v3.0.0 has never warned in a released
version, so it ships in v3.0.0 still working, with its warning, and the
two-minor-release clock starts there. Names removed at v3.0.0 are the ones
deprecated in v2.0.0 or earlier.

One of those is kept on purpose:

| Name | Deprecated in | Earliest removal | Replacement |
|---|---|---|---|
| P&L-level `filtered_historical_var` (`eq_var`, C++ `eqvar::`, Rust `eq_var_engine::historical::`) | 2.0.0 or earlier | 4.0.0 | `factor_filtered_historical_var` (Python), `filtered_historical_var_es` (C++), `filtered_historical_var_factors` (Rust) |

It stays because the three languages share one golden value
(`a_fhs_var_5pct_lam094` in `tests/golden/golden_vectors.json`) and the
model inventory lists it under `EQ-VAR-HIST`; removing it means retiring
that key and the inventory entry in the same change.

### Removed in 3.0.0

Each removal below warned in v2.0.0. A test asserts that the old form now
fails: `tests/test_removed_in_3_0.py` in `eq_var` and `fx_var`, the
`*_removed_in_3_0` tests in the other packages, and concept-based
`static_assert`s in the C++ test files.

| Package | Removed | Use instead |
|---|---|---|
| `eq_options`, `fx_options` | `implied_vol(tol=, max_iter=, sigma_lo=, sigma_hi=)` (ignored keywords) | `implied_vol(...)` without them |
| `eq_var` | positional `kupiec_pof`, `basel_traffic_light`, positional `alpha` in `acerbi_szekely_z2` | keyword forms |
| `eq_var` | `ewma_volatility(init=)`, `simulate_garch_returns(omega_frac=)` (model parameters now keyword-only) | the default seed window; keywords |
| `eq_var` | `safe_cholesky`, `cornish_fisher_domain_ok` | `cholesky(cov, repair=)`, `cornish_fisher_valid(skew, excess_kurt, alpha)` |
| `fx_var` | `confidence=` keyword on every VaR/ES function, positional `kupiec_pof`/`basel_traffic_light`, `es_backtest_acerbi_szekely`, `robust_cholesky(max_tries=)`, `cornish_fisher_domain_ok` | `alpha=` (tail probability), keyword forms, `acerbi_szekely_z2`, `robust_cholesky(cov, repair=)`, `cornish_fisher_valid` |
| `eq_credit`, `fx_credit` | `hosmer_lemeshow` without `out_of_sample` | pass `out_of_sample=True` or `False` (keyword-only, required) |
| `fx_credit` | `cva`, `cva_for_forward` aliases | `expected_exposure_loss`, `expected_exposure_loss_for_forward` |
| `eq_algo` | `DailyPanel.adv_dollars` (full-sample mean, look-ahead) | `trailing_adv_dollars(window)` |
| `eq_surface`, `fx_surface` | positional `HestonParams(...)` | keywords |
| `fx_surface` | `calendar_arbitrage_report(deltas=)` | `delta_calendar_report(deltas=)` |
| `eq_regime` | `detection.min_duration_filter` (not causal) | `strategy.causal_min_duration` |
| `fx_regime` | `carry_accrual` without `year_fraction` (ACT/252 fallback) | pass `year_fraction` |
| `fx_rates` | `spot_date(calendar_days=True)`, `tenor_to_years("ON"/"TN"/"SN")` (now raises), `MarketState(basis_model="zero_spread")` | business-day spot, `tenor_dates`, `"float_float"` |
| Rust `eq-var-engine` | `kupiec_pof_positional`, `basel_traffic_light_250`, `historical::linear_quantile`, `Matrix::cholesky_jitter`, `monte_carlo::var_order_statistic_se`, `parametric::cornish_fisher_domain_ok` | `kupiec_pof(&KupiecInput)`, `basel_traffic_light(&BaselInput)`, `tail::empirical_var`, `cholesky_repair`, `var_bootstrap_se` / `var_kde_se`, `cornish_fisher_valid` |
| Rust `fx-var-engine` | `kupiec_pof_confidence`, `basel_traffic_light_confidence`, `cornish_fisher_domain_ok` | struct forms, `cornish_fisher_valid` |
| Rust `eq-options-engine` | `IV_PRICE_TOL` | none (the solver has no price tolerance) |
| C++ `eqvar`, `fxvar` | positional `kupiec_pof`/`basel_traffic_light` overloads, `eqvar::quantile_linear`, `eqvar::cholesky(cov, jitter, max_tries)`, `fxvar::robust_cholesky`, `cornish_fisher_domain_ok` | `KupiecInput`/`BaselInput` forms, `var_es_order_statistic`, `cholesky(cov, Repair)`, `cornish_fisher_valid` |
| C++ `eqopt`, `fxopt` | `implied_vol` overloads with tuning parameters | the 7-argument `implied_vol` |

## Errors

Every error a package raises on purpose is a `qp_core.errors.QPError`:

| Class | Meaning | Also a |
|---|---|---|
| `InputError` | an argument is outside the model's domain | `ValueError` |
| `ConvergenceError` | a numerical method failed on well-formed input | `RuntimeError` and `ValueError` |
| `DataError` | market or reference data is missing, stale or malformed | `RuntimeError` |

The built-in base is the one the Python packages already used for that kind
of failure, so existing `except ValueError` and `except RuntimeError`
clauses keep catching. Packages move their raise sites to these classes one
at a time; each move is listed in `docs/MIGRATION.md`.

The bindings raise the same classes. The C++ binding maps the engines'
`std::runtime_error` (solver or factorisation failure) to
`ConvergenceError` and every other engine failure to `InputError`; the Rust
binding maps the engines' `NoConvergence` and `Numerical` variants to
`ConvergenceError` and the rest to `InputError`. If `qp_core` is not
installed next to a binding, both fall back to `ValueError`.
`bindings/*/tests/test_errors.py` checks each mapping.

Library code raises; it does not log. Entry points log once at the boundary
with `qp_core.logconfig` (one JSON record carrying run id, git commit,
version and as-of date).

## Typing

Every package ships a `py.typed` marker (PEP 561), so type checkers use the
inline annotations. `python/core` is checked with mypy in strict mode;
strict mode is extended to the other packages one at a time, and the
`[tool.mypy]` overrides in the root `pyproject.toml` show which are strict.

## Supported platforms

What CI builds and tests on every pull request (GitHub-hosted runners
only):

| | Supported |
|---|---|
| Python | 3.11, 3.12, 3.13 (CPython) |
| Operating systems | Linux x86_64 (Ubuntu 24.04) for everything; Windows x86_64 (Windows Server 2025) for the C++ engines and the binding wheels; macOS arm64 for the release wheels |
| C++ | C++23; GCC 15, Clang, MSVC (Visual Studio 2022 toolset); GCC 16 with `-std=c++26` as an opt-in job |
| Rust | the channel pinned in `rust/rust-toolchain.toml` |
| Binding wheels | CPython 3.11-3.13 (`qp-rust` is abi3 from 3.10), manylinux x86_64, Windows amd64, macOS arm64 |

A platform outside the table may work but is not tested, and a fix for it
is not a reason for a release.

## Where releases are published

Releases are GitHub Releases on this repository. Each carries the source
distribution and wheel of every Python package, the binding wheels for the
platforms above, a CycloneDX SBOM, and build-provenance attestations
(SLSA v1.0, verifiable with `gh attestation verify`). Nothing is published
to PyPI or TestPyPI (owner decision D7, 2026-11-03). Install a release by
downloading its wheels and running `pip install <wheel>`.

## References

- Semantic Versioning 2.0.0, semver.org.
- PEP 387, Backwards Compatibility Policy.
- PEP 440, Version Identification and Dependency Specification.
- PEP 561, Distributing and Packaging Type Information.
- SLSA v1.0 build provenance; CycloneDX 1.6 specification.
