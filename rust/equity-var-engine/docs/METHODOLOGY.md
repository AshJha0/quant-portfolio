# Methodology

The engine mirrors the Python reference (`eq_var`) function-for-function so
the two stacks are twins: Python for research iteration, Rust (like the
sibling C++ engine) for the latency/determinism-critical production path.
This file states the maths, **why each model was chosen against
alternatives**, and the assumptions register. Conventions throughout: `alpha`
is the tail probability (`alpha = 0.01` → 99 % VaR), P&L is in currency units
with losses negative, VaR/ES are reported **positive for a loss**, daily
horizon unless scaled.

## 1. Historical simulation family (`src/historical.rs`)

**Plain historical VaR** is minus the empirical `alpha`-quantile of the P&L
sample using NumPy's default *linear* (Hyndman–Fan **type-7**) interpolation:

    h = (n-1) alpha,   Q = x_(floor h) + (h - floor h) (x_(floor h + 1) - x_(floor h))

*Why type-7?* It is what `np.quantile` does by default, hence what every
research notebook and the Python twin produce — cross-language agreement to
1e-9 requires bit-matching the interpolation rule, not just "a quantile".
Alternatives considered: *lower/higher order statistic* (types 1/3) differ by
O(1/n) and make VaR a step function of alpha (bad for limit monitoring);
*Harrell–Davis* weights all order statistics (smoother, but no longer matches
any regulator-recognisable convention and is ~50x more expensive). The
interpolation choice moves 99 % VaR on 250 days by well under 1 σ of the
estimator's own sampling noise; it is a convention, and we pick the dominant
one.

**BRW age-weighted VaR** (Boudoukh–Richardson–Whitelaw): observation `i`
(0 = oldest) gets weight `(1-λ) λ^{n-1-i} / (1-λ^n)`, and VaR is the
step-function inversion of the *weighted* empirical CDF (smallest P&L whose
cumulative weight reaches alpha, stable-sorted). With λ = 0.98 and n = 250,
the most recent day carries ~2 % weight — a crash yesterday enters VaR
immediately instead of at weight 1/250.

**Filtered historical simulation (FHS)**: devolatilise P&L by one-step-ahead
EWMA (RiskMetrics) volatility forecasts

    sigma^2_t = lambda sigma^2_{t-1} + (1-lambda) x^2_{t-1}     (seeded with the ddof-0 sample variance)

then rescale the standardised innovations to tomorrow's forecast
`sigma^2_{T+1} = lambda sigma^2_T + (1-lambda) x^2_T` and take the empirical
type-7 quantile. FHS keeps the empirical (fat, skewed) tail *shape* but
re-levels it to the current volatility regime — the best-performing member of
the family in the Python project's 500-day backtest.

*Why this family at all (vs parametric only)?* Historical simulation makes no
distributional assumption in the tail, prices in observed cross-asset
dependence for free, and is the desk lingua franca. Its weaknesses (ghost
effects, unresponsiveness) are exactly what BRW and FHS address, so all three
are shipped and compared rather than picking one.

**Square-root-of-time**: `VaR_h = VaR_1 * sqrt(h)` — valid only under i.i.d.
zero-drift returns; documented as an *approximation with known bias* under
volatility clustering (understates multi-day risk in stressed regimes; see
VALIDATION.md). The alternative — overlapping h-day P&L
([`historical::overlapping_horizon_pnl`](../src/historical.rs)) — needs h×
the history and autocorrelates the sample; for the 10-day regulatory number,
sqrt-time on 1-day VaR is the standard desk compromise and is what we scale
by default.

## 2. Covariance estimation (`src/matrix.rs`)

- **Sample covariance** (ddof = 1) of the (T x n) returns panel — the
  unbiased baseline, equally weighting the window.
- **EWMA (RiskMetrics) covariance**: `Sigma <- lambda Sigma + (1-lambda) r_t r_t'`
  iterated over all rows, seeded with the sample covariance, λ = 0.94 daily.
  Zero-mean returns assumed (standard at daily horizon — the daily mean is
  ~20x smaller than the daily vol and estimating it adds noise).

*Why not shrinkage (Ledoit–Wolf) or a factor model?* For n = 100, T = 250
the sample matrix is invertible but noisy; however the VaR use-case only
needs `w'Sigma w` (a quadratic form, noise-averaging) and a Cholesky factor —
neither inverts Sigma, so shrinkage's main benefit (conditioning of
Sigma^-1) is moot. The Python research stack is the place to iterate on
estimators; the engine mirrors the two the reference implements.
Near-singularity is handled at the Cholesky, not the estimator.

## 3. Linear algebra: dense `Matrix` + Cholesky with jitter

A deliberately minimal row-major dense [`Matrix`](../src/matrix.rs) over one
contiguous `Vec<f64>` — cache-friendly, allocation-light hot loops, no
expression-template machinery to audit. *Why not `nalgebra` / `ndarray`?* The
project rule is zero external dependencies; the operations needed (matvec,
quadratic form, Cholesky, rank-1 updates) are ~50 lines each and trivially
verified, and the MC hot loop is memory-bound on the triangular product
either way — a general-purpose linear-algebra crate buys nothing here and
adds a dependency the risk numbers of record would then depend on.

**Cholesky with diagonal-jitter fallback** (mirrors
`eq_var.monte_carlo_var.safe_cholesky` / the C++ engine's `cholesky_jitter`):
attempt plain `A = LL'`; on a non-positive pivot (exactly singular PSD
inputs — perfectly correlated factors, zero-variance assets — or slight
asymmetry-rounding indefiniteness) add `jitter * mean(diag)` to the diagonal
and retry, escalating x10 up to 12 times. Starting at 1e-10 of the average
variance, the perturbation is orders of magnitude below Monte Carlo noise.
*Alternative*: eigenvalue clipping is more surgical but needs an
eigendecomposition (O(n^3) with a big constant, plus code we'd have to write
dependency-free); jitter achieves the same effect for PSD-but-singular
matrices at zero extra cost, and *badly indefinite* input still returns
`EqVarError::Numerical` — it should, because that is corrupt data, not
numerics.

## 4. Parametric (variance–covariance) VaR (`src/parametric.rs`)

    sigma_p = sqrt(w' Sigma w),   VaR = -(mu*h + z_alpha * sigma_p * sqrt(h))

with three tail models ([`TailModel`](../src/lib.rs)):

- **Normal**: `z = Phi^{-1}(alpha)`.
- **Variance-matched Student-t**: `z = t_nu^{-1}(alpha) * sqrt((nu-2)/nu)`,
  df > 2. The rescaling matches the *variance* so that switching the tail
  model changes only tail shape, never the sigma — otherwise t-VaR and
  normal-VaR differences would conflate two effects. df = 6 is a typical
  equity daily-return fit (caller-supplied, not hard-coded as a default in
  the API).
- **Cornish–Fisher**: `z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36`
  with sample skewness S and excess kurtosis K, **guarded by an explicit
  monotonicity-domain check**
  ([`cornish_fisher_domain_ok`](../src/parametric.rs)): the quartic expansion
  is only a quantile function where `dz_cf/dz > 0` on `|z| <= 3.5`; outside
  that region (e.g. S = -0.5, K = 0 already fails) the "quantile" is not a
  quantile and the engine returns `EqVarError::InvalidInput` rather than a
  number. This check is inherited verbatim from the Python reference and is
  unit-tested on both sides.

*Why keep parametric at all?* It is the only closed-form method — the
analytic anchor every simulation method is validated against, the fastest
(sub-µs marginal cost per exposure update → intraday incremental VaR), and
the basis of Euler risk decomposition (`dVaR/dw = z Sigma w / sigma_p`).

## 5. Monte Carlo VaR (`src/monte_carlo.rs`)

Simulate factor returns `r = L z` (L the Cholesky factor), P&L `= w . r`,
read VaR/ES off the scenario distribution.

- **Multivariate normal**: `z ~ N(0, I)` via [`rng::Rng::standard_normal`]
  (Box–Muller transform, pair-cached).
- **Multivariate Student-t**: common-mixing construction
  `r = L z / sqrt(W/nu)`, `W ~ chi^2_nu`, with scale matrix `Sigma*(nu-2)/nu`
  so the simulated **covariance equals Sigma exactly** while tails fatten —
  the same variance-matching discipline as the parametric family. A *single*
  mixing variable per path (not per factor) is what makes the joint tails
  fat — independent per-factor t draws would kill tail dependence, which is
  the point of using t.
- **Determinism**: one seeded [`rng::Rng`] (xoshiro256++, Blackman & Vigna
  2019, seeded via SplitMix64) drives every draw; gaussians via Box–Muller
  (not an inverse-CDF transform, unlike the C++ engine — Box–Muller is exact
  and marginally cheaper, and since the Rust engine owns its whole RNG stack
  end to end there is no cross-platform `std::normal_distribution` hazard to
  avoid), chi² via Marsaglia–Tsang gamma built on the same normal primitive.
  Given a seed, VaR is **bitwise** reproducible on a given build
  (unit-tested: `tests/test_monte_carlo.rs::bitwise_seed_determinism`).
  **Rust's RNG stream is its own** — it agrees with neither NumPy's PCG64
  nor the C++ engine's `mt19937_64`, so cross-engine MC agreement is
  statistical (within order-statistic SE bars), never bitwise; only the
  deterministic estimators (historical, parametric, ES closed forms) are
  golden-tested bit-for-bit across languages.
- **Error bars**: order-statistic asymptotic SE
  `sqrt(alpha(1-alpha)/n) / f_hat(q)`
  ([`var_order_statistic_se`](../src/monte_carlo.rs)), with the density
  estimated by a symmetric order-statistic finite difference of bandwidth
  `ceil(sqrt(alpha n))`. Every MC number ships with an SE the caller can
  compute alongside it; convergence tests assert within 3 SE.

*Why MC when the portfolio is linear?* For a linear book, MC with a normal
factor model must reproduce parametric VaR — that identity is a validation
asset, not redundancy. MC is the only method that extends to fat-tailed
joint factors (t), and the architecture (simulate panel → revalue → tail
metrics) is the one that generalises to non-linear revaluation.

## 6. Expected Shortfall (`src/expected_shortfall.rs`)

**Empirical ES** is the *exact* integral of the empirical quantile function
over `(0, alpha]`: with `k = floor(alpha n)`,

    ES = -(1/(alpha n)) [ sum_{i<=k} x_(i) + (alpha n - k) * x_(k+1) ]

i.e. fractional weight on the boundary order statistic — not "mean of
observations beyond VaR", which is biased for non-integer `alpha n`.
Closed forms: **normal** `ES = sigma*phi(z_alpha)/alpha - mean` (the identity
validated against quadrature to 1e-10) and **variance-matched Student-t**
`ES = sigma * f_nu(q)(nu+q^2)/((nu-1)alpha) * sqrt((nu-2)/nu) - mean`.
ES >= VaR always (tested); ES is subadditive where VaR need not be, and FRTB
made 97.5 % ES the capital measure — both closed forms take `alpha = 0.025`
in the parametric convenience wrapper.

## 7. Backtesting (`src/backtest.rs`)

An exception is a day with `pnl < -VaR`. Three regulator-standard tests:

- **Kupiec POF (1995)**: `LR_uc = -2 ln[ L(alpha) / L(x/T) ]` on the binomial
  likelihood, chi^2(1) under H0; `0*ln 0 = 0` convention for degenerate
  counts.
- **Christoffersen (1998) independence**: LR of an i.i.d. exception process
  against a first-order Markov alternative on the transition counts
  `n00, n01, n10, n11`; clustering inflates `n11` and rejects. **Conditional
  coverage**: `LR_cc = LR_uc + LR_ind ~ chi^2(2)` — right rate *and*
  independence jointly.
- **p-values** from the chi^2 survival function via the **regularized upper
  incomplete gamma** (series + continued fraction, `src/stats.rs`) — no
  statistics library needed; validated at the classical critical value
  `chi^2(1) = 3.8415 -> p = 0.05` to 1e-11.
- **Basel traffic light** on the 250-day window with the *exact* regulatory
  count boundaries: 0-4 green (k = 3.0), 5-9 yellow (add-ons
  0.40/0.50/0.65/0.75/0.85), 10+ red (k = 4.0). The exact binomial CDF (via
  the regularized incomplete beta) reports where the observed count sits
  under a correct model — e.g. 4 exceptions is the 89.2nd percentile.

*Why LR tests and not just counting?* The traffic light *is* counting — but
it has known low power; Kupiec adds a calibrated significance level, and
Christoffersen catches the failure counting cannot see: 10 exceptions in one
volatility cluster versus 10 spread over the year (unit-tested with planted
patterns of both kinds).

## 8. Special functions (`src/stats.rs`)

- `Phi^{-1}`: Acklam's rational approximation (|rel err| < 1.15e-9) + one
  Halley refinement step against the erfc-based CDF -> near machine precision
  observed in the tested range.
- Regularized incomplete beta (Lentz continued fraction) → Student-t CDF and
  exact binomial CDF; incomplete gamma (series + CF) → chi^2 survival and
  `erfc` (`erfc(x) = Q(1/2, x^2)`).
- Student-t quantile by bisection on the CDF to a machine-precision bracket
  (matches `scipy.stats.t.ppf` to < 1e-9; a 200-step bisection at ~µs cost is
  irrelevant next to any consumer, and has no tuning parameters to defend).
- Moments: `stdev` ddof = 1; skew/kurtosis as biased moment ratios
  (`scipy bias=True`) because that is what the Python reference feeds
  Cornish-Fisher.

## 9. Why Rust for this engine, specifically

The C++ engine already exists as the latency-critical production twin; this
Rust engine is not a second production path but a **second, independently
implemented cross-check** on the same methodology, written against the same
Python golden vectors from scratch (own RNG, own special functions, own
dense linear algebra) rather than transliterated line-by-line from the C++
source. Three implementations (Python semantics, C++ hand-rolled maths, Rust
hand-rolled maths) agreeing to 1e-9 on the same closed-form inputs is
materially stronger evidence than any one of them being "obviously correct"
— transcription bugs and off-by-one interpolation errors tend not to survive
three independent re-derivations. Where Rust adds engineering value beyond
that: `Result`-typed error handling makes "did this call actually validate
its inputs" a compile-time-checked question (an unhandled `Result` is a
compiler warning, not a silently-swallowed exception), and the ownership
model rules out the aliasing bugs that dense in-place linear algebra in C or
C++ is classically prone to.

## Assumptions register

Each assumption states *what breaks if violated*.

1. **Linear (delta) P&L mapping**: `pnl = w' r`. Violated by optionality/gamma
   → all methods mis-state risk on gap moves; the Python twin's full
   revaluation shows delta-gamma missing materially on a large gap move.
   This engine is for the linear (cash equity / futures) book or
   delta-mapped views.
2. **Daily returns are (locally) stationary over the window**. Violated in
   regime breaks → plain historical VaR lags ~250 days; use FHS/BRW (built
   for exactly this) and monitor with Christoffersen.
3. **Zero-mean daily returns** (EWMA covariance, default `mean = 0`).
   Violated for strongly trending factors → second-order at daily horizon
   (mean ~ vol^2 * years); the `mean` parameter exists where it matters
   (`parametric_var_full`, `parametric_es_full`).
4. **i.i.d. for sqrt-time scaling**. Violated under vol clustering → 10-day
   VaR understated in stressed regimes; documented bias, use overlapping
   P&L (`overlapping_horizon_pnl`) in the research stack when it matters.
5. **Student-t df > 2** (finite variance) for variance matching. df <= 2 has
   no variance to match → `EqVarError::InvalidInput` by design.
6. **Cornish-Fisher inside its monotonicity domain**. Outside → not a
   quantile function; the engine refuses (domain check) instead of returning
   a number.
7. **Covariance input is symmetric PSD up to rounding**. Small negative
   eigenvalues → jitter absorbs them; a badly indefinite matrix (data error)
   → `EqVarError::Numerical`, deliberately fatal.
8. **250-day, 99 % convention for Basel zones**; the count boundaries are
   specific to that window — other windows change the binomial, and the
   engine reports the exact binomial CDF for whatever `n_obs` is passed.
9. **RNG stream is Rust-specific**. A given seed reproduces bitwise *within*
   this engine on a given build, but not against the Python or C++ engines'
   RNG streams — cross-engine MC agreement is checked statistically (within
   SE), and any code change to `rng.rs` that alters the stream (even a
   refactor that changes call order) invalidates previously logged
   audit-trail seeds; `rng.rs` is treated as a frozen interface.
