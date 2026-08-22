# Validation

How the engine was validated, and where it fails. Everything below is
enforced by the test suite (`RUSTFLAGS="-D warnings" cargo test --release`:
9 integration-test files, 88 tests, plus 10 rustdoc examples — 98 in total,
all offline and deterministic) — validation claims that are not unit-tested do not appear
here.

## 1. Cross-language golden tests (the headline)

`tests/test_cross_language.rs` pins the engine to the Python research
reference (`eq_var`, numpy / scipy). Three deterministic cases whose inputs
are **closed-form (sin/cos) series every language regenerates
independently** — no data files, no serialisation ambiguity, and the same
inputs/constants as `cpp/equity-var-engine/tests/test_cross_language.cpp`:

- **Case A** — `pnl[t] = 100*sin(3t+1) + 0.5*t*cos(t)`, t = 0..99:
  historical VaR at 1 % / 5 % (122.4129222375264 / 104.5522835374927),
  empirical ES at 1 % / 5 % (141.7568107549531 / 122.6373207703405),
  BRW VaR (108.2348601407293), FHS VaR (109.3777910164513), and the 5-element
  BRW weight vector.
- **Case B** — fixed 3-asset book (w = [1e6, -5e5, 2e5], vols
  [1 %, 1.5 %, 2 %], full correlation matrix): portfolio sigma
  (9962.429422585637), parametric VaR normal/t(6)/10-day
  (23176.07650751402 / 25563.37478743866 / 73289.18899006478), closed-form
  normal and t ES at 2.5 %, Cornish-Fisher VaR (28230.62626871169).
- **Case C** — deterministic 60x3 returns panel: six sample- and
  EWMA-covariance entries, parametric VaR off the estimated matrices, Kupiec
  LR/p at (250, 7) and (250, 0), Christoffersen independence and conditional
  coverage on a planted 9-exception pattern.

Tolerance: **1e-9 relative** (+1e-12 absolute floor). The observed residual
is libm-vs-numpy ulp noise in sin/cos — far below any risk tolerance. The
constants were regenerated from the live Python stack on 2026-08-18
(`PYTHONPATH=src python3 -c "..."` against `eq_var.historical_var`,
`eq_var.expected_shortfall`, `eq_var.parametric_var`, `eq_var.backtesting`)
and matched the values already committed in the C++ engine's golden test
digit-for-digit — i.e. Python, C++ and Rust independently agree.

## 2. Analytic identities and hand-computed values

| check | tolerance | test |
|---|---|---|
| `Phi^{-1}(0.975) = 1.959963984540054` | 1e-9 | `test_stats::normal_ppf_known_quantiles` |
| `Phi(Phi^{-1}(p)) = p`, symmetry across p in [1e-8, 0.9999] | 1e-9 | `test_stats::normal_ppf_symmetry_and_round_trip` |
| normal ES identity `ES = sigma*phi(z)/alpha` vs Simpson quadrature | 1e-10 rel | `test_expected_shortfall::normal_es_identity_vs_numerical_quadrature` |
| parametric closed forms (normal, variance-matched t) | 1e-12 rel | `test_parametric::parametric_var_*` |
| Cholesky `A = LL'` reconstruction on SPD input | 1e-12 | `test_matrix::cholesky_reconstructs_spd` |
| empirical quantile / ES exact on hand-solved tiny arrays | exact | `test_historical::quantile_linear_hand_exact_tiny_arrays`, `test_expected_shortfall::empirical_es_hand_exact_tiny_arrays` |
| BRW weights: sum = 1, ratio = 1/lambda, monotone in recency | 1e-12 | `test_historical::brw_weights_*` |
| EWMA vol: seed = ddof-0 variance, recursion uses x[t-1] only (no look-ahead) | 1e-14 | `test_historical::ewma_volatility_no_lookahead_and_riskmetrics_recursion` |
| Kupiec LR from first-principles log-likelihoods (T=250, x in {0,5}) | 1e-12 | `test_backtest::kupiec_*` |
| `chi^2(1) sf(3.841458...) = 0.05` | 1e-11 | `test_stats::incomplete_gamma_chi2_pvalues` |
| Student-t quantiles vs `scipy.stats.t.ppf` | 1e-8 | `test_stats::student_t_quantiles_vs_scipy` |
| binomial CDF vs scipy at the Basel boundaries | 1e-11 | `test_stats::incomplete_beta_binomial_cdf` |
| Christoffersen transition counts on a hand-traced pattern | exact | `test_backtest::christoffersen_transition_counts_on_tiny_pattern` |
| Basel zones: all boundaries 0/4/5/9/10 and the full add-on ladder | exact | `test_backtest::basel_exact_zone_boundaries_at_250_obs` |

## 3. Monte Carlo convergence and determinism

- **MC → parametric**: 200k-path normal MC VaR within **3 order-statistic
  SE** of the closed form (SE itself asserted < 2 % of VaR); MC ES within
  4 SE of the normal-ES identity. The bench run prints the live numbers:
  68 584 vs 69 002 at SE 422 (< 1 SE apart).
- **Student-t fatter at 99 %**: t(5) MC VaR and ES strictly above normal MC
  on the same seed, and within 3 SE of the variance-matched t closed form.
- **Moment matching**: the simulated 200k-path covariance matches the target
  Sigma entry-wise (5 % rel on variances, 10 % of sigma_i*sigma_j on
  covariances) for both normal and t — i.e. the `(nu-2)/nu` scale correction
  is verified, not assumed.
- **Bitwise determinism**: same seed → exact equality (`assert_eq!`, not an
  epsilon comparison) on VaR and ES; different seed → different result.
  This is checked *within* the Rust engine only — see the cross-engine
  caveat below.
- **RNG health**: uniforms strictly inside (0,1); Box–Muller normal sample
  moments at ~4 SE bands; chi^2 mean/variance vs (df, 2df).
- **Bootstrap SE cross-check**: `var_bootstrap_se` (distribution-free,
  resample-with-replacement) agrees with `var_order_statistic_se` within a
  3x band on a 100k-path sample (`bootstrap_se_within_3x_of_order_statistic_se_on_large_sample`);
  same-seed calls are bitwise identical, and a constant P&L slice returns an
  exact `0.0`, never `NaN` (see item 10 below).

All in `tests/test_monte_carlo.rs`.

## 4. Statistical power checks (planted-pathology tests)

- **Christoffersen detects planted clustering**: a run of 10 consecutive
  exceptions in 250 days rejects independence at p < 0.001 while the same 10
  exceptions spread every 25 days give p > 0.10.
- **Kupiec monotonicity**: LR increases with excess exceptions; 10/250 at
  1 % rejects at 5 %; exact-coverage samples give LR = 0, p = 1; the
  degenerate x = 0 case matches `-2*250*ln(0.99)` analytically.
  Note (Python reference, `eq_var` `docs/VALIDATION.md` item 9): the
  chi2(1) asymptotic this LR is compared against is itself oversized at
  exactly this window — the exact rejection probability of a nominal-5%
  test at n=250, alpha=0.01 is ~9.5%, not 5%, because the expected
  exception count (2.5) is too small for the asymptotic to be accurate.
  The LR formula is unchanged (it is what the golden tests pin); this is a
  read-the-p-value caveat, not a bug.
- **BRW regime response**: a crash on the most recent day moves VaR to the
  full crash size, the same crash 100 days ago does not.
- **FHS regime response**: damping the last 40 % of the sample cuts FHS VaR
  below plain historical VaR (and FHS is exactly positive-homogeneous).

All in `tests/test_backtest.rs` and `tests/test_historical.rs`.

## 5. Edge cases (documented **and** tested — `tests/test_edge_cases.rs`)

| edge case | behaviour |
|---|---|
| empty inputs (pnl, exposures, panel) | `EqVarError::InvalidInput` with an informative message, everywhere |
| alpha not in (0, 0.5), incl. 0, 0.5, 1 | errors in every VaR/ES entry point |
| single-asset portfolio | collapses exactly to scalar formulas; MC within 3 SE; long/short symmetric |
| zero-variance asset in Sigma (singular PSD) | Cholesky jitter path engages (plain Cholesky fails, jitter fallback succeeds), MC still close to parametric |
| all-zero covariance | sigma = 0, parametric VaR = 0, ES = 0 exactly |
| constant P&L series | VaR = -constant (a gain floor), moments 0 not NaN, FHS floored (no 0/0) |
| NaN / Inf in P&L | errors — never silently dropped |
| NaN / Inf **anywhere in the covariance, exposures, return panel, vols or correlations** | errors at every entry point: `portfolio_sigma`, `parametric_var(_full)`, `parametric_es(_full)`, `monte_carlo_{var,es,pnl}`, `simulate_factor_returns`, `Matrix::cholesky(_jitter)`, `sample_covariance`, `ewma_covariance`, `covariance_from_vols`, `normal_es`, `student_t_es`, `cornish_fisher_var` (with the domain check on **and** off), `TailModel::StudentT { df }` |
| NaN / Inf in a **backtest** P&L or VaR series | `exceptions_from_pnl` errors instead of scoring the day as "no exception" |
| rank-deficient covariance (a factor that is an exact linear combination of two others) | plain Cholesky fails, the jitter fallback repairs it at the first rung, and the resulting MC VaR sits within 10 % of the parametric number |
| materially indefinite covariance (a genuinely negative eigenvalue) | `EqVarError::Numerical` naming the materiality cap — never silently repaired |
| one- and two-element samples | `linear_quantile` is exact on 1 and 2 points; every higher-level estimator refuses samples below its documented minimum; `sample_covariance` refuses a 1-row panel and is exact on a 2-row one |
| fewer than `MIN_OBS` (50) obs for historical methods | errors, boundary exact |
| alpha << 1/n | estimate pinned to the worst observed loss (type-7 interpolation inside the first gap), never extrapolated |
| Cornish-Fisher outside monotonicity domain (e.g. S = -0.5, K = 0) | errors with an explanation; `check_domain=false` override exists for diagnostics only |
| badly indefinite "covariance" | `EqVarError::Numerical` after the jitter ladder — corrupt input is fatal by design |
| non-finite or non-positive `jitter` argument | `EqVarError::InvalidInput` before any factorisation is attempted |
| portfolio/panel shape mismatch | errors with the exact dimensions in the message |

## 6. Known failure modes and limits

1. **Tail resolution**: 250 observations put ~2.5 obs in a 1 % tail; the
   99 % historical VaR SE is large (the MC SE machinery quantifies the same
   effect). Below `MIN_OBS = 50` the engine refuses. For alpha < 1/n the
   estimate degrades to the sample minimum — tested, documented, and the
   reason parametric/MC exist.
2. **sqrt-time scaling understates stressed multi-day risk** (vol
   clustering): known bias, stated where used; do not feed the 10-day number
   to anything that assumes it is conservative.
3. **Cornish-Fisher validity region is small**: |S| >~ 0.3 with K = 0 already
   fails on |z| <= 3.5. The engine errors rather than degrades — this is a
   feature; the Python reference behaves identically. The domain check
   itself is exact, not grid-sampled: `dz_cf/dz` is a quadratic in `z`, so
   its minimum on `[-z_range, z_range]` is found in closed form (the vertex
   if it falls in range, else an endpoint) rather than by sampling a fixed
   grid, which could otherwise miss a thin non-monotone dip between two
   grid nodes (`cornish_fisher_domain_check_is_exact_not_grid_resolution_dependent`).
4. **RNG streams differ across engines**: cross-language / cross-engine MC
   agreement is statistical (within SE bars), never bitwise — the golden
   tests therefore pin the deterministic estimators and closed forms, and MC
   is validated against closed forms *within* each engine. A given seed's
   VaR is only reproducible against a previous run of *this* Rust build.
5. **EWMA lambda is a hyperparameter**, not estimated: lambda = 0.94/0.98 are
   RiskMetrics/BRW conventions. A mis-tuned lambda shows up as Christoffersen
   clustering failures — that is the monitoring loop, by construction.
6. **Linear P&L only**: no gamma/vega. Options books need the Python twin's
   full revaluation; this engine's scope stops at the delta map.
7. **NaN is the failure mode this engine guards hardest against — and the
   one a naive guard misses.** A validation written as
   `if x < 0.0 { return Err(...) }` silently **accepts** NaN, because every
   IEEE-754 comparison against NaN is false. Rust does not help here:
   `f64::NAN < 0.0` is `false` exactly as in C and C++. Worse, `f64::max`
   propagates the *other* operand, so `NaN.max(0.0)` is `0.0`. Three
   concrete holes were closed in the hardening pass, each of which turned a
   corrupt input into a *plausible-looking* risk number rather than an
   error:

   * `portfolio_sigma` computed `w' Sigma w`, tested it with
     `var < -tol` (false for NaN) and then returned
     `var.max(0.0).sqrt()` — so a single NaN covariance entry reported a
     portfolio sigma, a parametric VaR and a parametric ES of exactly
     **zero**. A broken feed looked like a riskless book.
   * `Matrix::cholesky` tested its pivot with `sum <= 0.0` (false for NaN),
     so a NaN covariance factorised into an all-NaN `L`, and every Monte
     Carlo scenario, P&L and VaR downstream was NaN with no diagnostic.
     `is_symmetric` had the same shape of hole (`|NaN - NaN| > tol` is
     false, i.e. a NaN matrix was "symmetric").
   * `exceptions_from_pnl` tested `pnl_t < -var_t`. With a NaN on either
     side that comparison is false, so the day counted as "no exception" —
     a model whose VaR feed had broken to NaN recorded **zero breaches**
     and passed Kupiec, Christoffersen and the Basel traffic light in the
     green zone. The backtest certified the broken model.

   All validation is now written with `is_finite()` rather than an ordering
   comparison, and the contract is pinned by
   `test_edge_cases::non_finite_covariance_and_exposures_are_rejected_not_silently_zeroed`
   and
   `test_edge_cases::backtests_reject_non_finite_series_instead_of_scoring_them_green`.
8. **The Cholesky jitter ladder repairs rounding noise, not indefiniteness.**
   `cholesky_jitter(jitter, max_tries)` escalates the diagonal jitter by
   10x per attempt. With the module defaults (`1e-10`, 12 tries) the old
   ladder topped out at **ten times** the mean variance — enough to
   "repair" a materially indefinite matrix and return a factor whose
   simulated risk bears no relation to the caller's covariance, with no
   diagnostic at all. The escalation now stops at
   `matrix::MAX_RELATIVE_JITTER` (1e-6) times the mean diagonal and returns
   `EqVarError::Numerical` naming that cap. PSD-but-singular covariances (a
   riskless leg, perfectly correlated factors, an exactly rank-deficient
   factor block) are repaired at the *first* rung and are unaffected — the
   cap only bites on matrices that were never a covariance. A matrix that
   needs more than a rounding-noise repair is a data problem (stale
   correlation block, mis-signed loading, skipped shrinkage) and must be
   surfaced.
9. **Student-t quantile via bisection** costs ~200 CDF evaluations (~µs).
   Irrelevant at desk call rates; would matter only inside a per-path loop,
   where it is never called (the MC uses Box–Muller normal + chi^2 mixing,
   not `student_t_ppf`).
10. **`var_order_statistic_se` undersmooths in the deep tail / at modest
    scenario counts.** Its density estimate is a local finite difference
    with bandwidth `ceil(sqrt(alpha * n))` around the quantile rank — a
    window sized for the bulk of the distribution. Deep in the tail (`alpha`
    at or below 1 %) or with a modest path count, that window is narrow
    enough that the local finite difference under-resolves the true local
    density, and the reported SE can understate the true sampling error of
    the quantile by roughly 10-15 %. It is not wrong in the sense of a bug —
    the asymptotic formula and the finite-difference density estimate are
    both textbook — it is a bandwidth trade-off that biases low precisely
    where callers care most (the risk tail).
    [`var_bootstrap_se`](../src/monte_carlo.rs) is the fix: it resamples the
    scenario P&L with replacement and reports the empirical standard
    deviation of the re-estimated quantile directly, carrying no bandwidth
    assumption at all (distribution-free), at the cost of `n_boot` extra
    quantile evaluations. It is the desk-standard way to attach an error bar
    to an MC or historical VaR precisely because it does not inherit this
    bias. Running both and comparing (`bootstrap_se_within_3x_of_order_statistic_se_on_large_sample`
    in `tests/test_monte_carlo.rs`) is the recommended cross-check; a caller
    who needs a defensible SE deep in the tail or off a small scenario count
    should prefer the bootstrap number over the order-statistic one. Mirrors
    the Python reference's `var_standard_error_bootstrap`
    (`eq_var.monte_carlo_var`), which is the desk-standard estimator there
    for the same reason.

## 7. Toolchain hygiene

Built and tested with `RUSTFLAGS="-D warnings" cargo test --release`
(rustc warnings-as-errors) on the 2021 edition; `#![deny(missing_docs)]` at
the crate root means every public item has rustdoc, and every rustdoc code
example is itself an executed test (10 of them, `cargo test --doc`).
`#![warn(clippy::all)]` is set at the crate root as a standing lint; the
crate compiles clean under plain `cargo build`/`cargo test`.
