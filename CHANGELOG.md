# Changelog

All notable changes to this repository are recorded here. The format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) and versions
follow [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

A release is cut by `.github/workflows/release.yml`, which tags a commit only
when the `ci-ok` check passed on that exact commit. The release notes are the
section below whose heading matches the version.

## [Unreleased]

### Added

- CI runs every Python project in its own environment on Python 3.10, 3.11,
  3.12 and 3.13, the four C++ engines with GCC 15 and Clang 21 plus an
  AddressSanitizer and UBSan build, and the four Rust crates on a pinned
  toolchain. A final `ci-ok` job is the single required check.
- `counts.json` CI artefact with the executed pass counts per project.
- `release.yml`: tag-on-green releases with notes taken from this file.
- `requirements/ci-constraints.txt`: one resolved set of dependency versions
  for CI, valid on Python 3.10 to 3.13.
- Dependabot for pip, cargo and GitHub Actions. CHANGELOG, SECURITY,
  CONTRIBUTING, CODEOWNERS, `.editorconfig`, `ruff.toml` and a pre-commit
  configuration.

### Changed

- `tests/test_docs_counts.py` compares the docs with the executed counts in
  `counts.json`. Without that file it falls back to collected counts and
  warns that they are approximate.

### Fixed

- CI previously stopped at the first failing project and installed no test
  extras, so 22 of 23 Python projects had never run in CI.

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
