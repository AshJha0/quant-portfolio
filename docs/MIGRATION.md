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
  marks its p99 and max columns `p99*` and `max*` and adds a JSON field
  `tail_basis`: `"per-call"` when each sample is one call, otherwise
  `"p99 of B-call batch means"` with B written out. The numbers are
  unchanged; they were always tails of B-call means and are not per-call
  tail latency.
  **What to do:** if you parse the bench JSON strictly, allow the new
  field. Do not quote these tails as per-call latency.

#### VaR engines (Round 6, lane L3)

- **Equity Monte Carlo VaR, linear book.** The C++ engine now computes each
  path's P&L in O(n) as `v = L'w` (the exposure vector mapped once through
  the Cholesky factor) instead of forming the correlated shock vector
  first. The algebra is identical; floating-point summation order is not,
  so results move in the last bits (about 1e-12 relative) and are no
  longer bit-identical to 2.0.0.
  **What to do:** if you compare against stored 2.0.0 outputs with exact
  equality, compare with a relative tolerance of 1e-12 instead.
- **New non-throwing samplers** in the C++ engines: `gamma_nothrow` and
  `chi_squared_nothrow` (equity), `gamma_nothrow` and `chisq_nothrow` (FX).
  They return an error value instead of throwing, for the noexcept kernel
  path. The throwing samplers are unchanged.
- The VaR packages' `dev` extra now includes `mpmath>=1.3`, used by the new
  independent VaR oracle tests.

#### Algorithmic execution (Round 6, lane L2)

- **Execution-cost numbers rise by roughly 3 to 4 times** after the
  bucket-width fix in `python/{equity,fx}/08-algo-execution`. The old
  numbers were too low; the CHANGELOG entry for the fix gives the old and
  new values and the reference that verified them.
  **What to do:** rerun any stored cost estimates; do not mix 2.0.0 and
  3.0.0 cost figures in one comparison.
- **Deprecated, removal no earlier than 4.0.0** (two minor releases at
  least, `docs/VERSIONING.md`):
  - the `market_volumes=` argument of the POV schedule: use
    `forecast_volumes=`;
  - `expected_is_bps`: use `model_implied_is_bps`.
  Both still work in v3.0.0 and emit `DeprecationWarning`.

#### Testing and CI (contributors only)

- Hypothesis settings profiles in the root `conftest.py`: `ci`
  (derandomised: two runs on one commit draw the same examples), `nightly`
  (random, ten times the examples) and `dev` (defaults). Select with
  `QP_HYPOTHESIS_PROFILE` or `--hypothesis-profile`.
- The QuantLib cross-checks (Hull-White, ISDA CDS, Heston) run in a
  required `quantlib` CI job against QuantLib 1.44 installed from
  hash-locked wheels, and fail rather than skip.

## 2.0.0 (2026-11-03)

The first release with this file. See `CHANGELOG.md` for what changed in
2.0.0 and earlier.
