# Methodology

The engine mirrors the Python reference (`eq_var`) function-for-function so
the two stacks are twins: Python for research iteration, C++ for the
latency/determinism-critical production path. This file states the maths,
**why each model was chosen against alternatives**, and the assumptions
register. Conventions throughout: `alpha` is the tail probability
(`alpha = 0.01` → 99 % VaR), P&L is in currency units with losses negative,
VaR/ES are reported **positive for a loss**, daily horizon unless scaled.

## 1. Historical simulation family (`eqvar/historical.hpp`)

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

    sigma²_t = λ sigma²_{t-1} + (1-λ) x²_{t-1}     (seeded with the ddof-0 sample variance)

then rescale the standardised innovations to tomorrow's forecast
`sigma²_{T+1} = λ sigma²_T + (1-λ) x²_T` and take the empirical type-7
quantile. FHS keeps the empirical (fat, skewed) tail *shape* but re-levels it
to the current volatility regime — the best-performing member of the family
in the Python project's 500-day backtest.

*Why this family at all (vs parametric only)?* Historical simulation makes no
distributional assumption in the tail, prices in observed cross-asset
dependence for free, and is the desk lingua franca. Its weaknesses (ghost
effects, unresponsiveness) are exactly what BRW and FHS address, so all three
are shipped and compared rather than picking one.

**Square-root-of-time**: `VaR_h = VaR_1 · sqrt(h)` — valid only under i.i.d.
zero-drift returns; documented as an *approximation with known bias* under
volatility clustering (understates multi-day risk in stressed regimes; see
VALIDATION.md). The alternative — overlapping h-day P&L — needs h× the
history and autocorrelates the sample; for the 10-day regulatory number,
sqrt-time on 1-day VaR is the standard desk compromise and is what we scale.

## 2. Covariance estimation (`eqvar/matrix.hpp`)

- **Sample covariance** (ddof = 1) of the (T x n) returns panel — the
  unbiased baseline, equally weighting the window.
- **EWMA (RiskMetrics) covariance**: `Σ ← λ Σ + (1-λ) r_t r_tᵀ` iterated over
  all rows, seeded with the sample covariance, λ = 0.94 daily. Zero-mean
  returns assumed (standard at daily horizon — the daily mean is ~20x smaller
  than the daily vol and estimating it adds noise).

*Why not shrinkage (Ledoit–Wolf) or a factor model?* For n = 100, T = 250
the sample matrix is invertible but noisy; however the VaR use-case only
needs `wᵀΣw` (a quadratic form, noise-averaging) and a Cholesky factor —
neither inverts Σ, so shrinkage's main benefit (conditioning of Σ⁻¹) is moot.
The Python research stack is the place to iterate on estimators; the engine
mirrors the two the reference implements. Near-singularity is handled at the
Cholesky, not the estimator.

## 3. Linear algebra: dense Matrix + Cholesky with jitter

A deliberately minimal row-major dense `Matrix` over one contiguous
`std::vector<double>` — cache-friendly, allocation-free hot loops, no
expression-template machinery to audit. *Why not Eigen?* The project rule is
zero external dependencies; the operations needed (matvec, quadratic form,
Cholesky, rank-1 updates) are 50 lines each and trivially verified, and the
MC hot loop is memory-bound on the triangular product either way.

**Cholesky with diagonal-jitter fallback** (mirrors
`eq_var.monte_carlo_var.safe_cholesky`): attempt plain `A = LLᵀ`; on a
non-positive pivot (exactly singular PSD inputs — perfectly correlated
factors, zero-variance assets — or slight asymmetry-rounding indefiniteness)
add `jitter · mean(diag)` to the diagonal and retry, escalating ×10 up to 12
times, reporting the jitter actually used. Starting at 1e-10 of the average
variance, the perturbation is orders of magnitude below Monte Carlo noise.
*Alternative*: eigenvalue clipping is more surgical but needs an
eigendecomposition (O(n³) with a big constant, plus code we'd have to write
dependency-free); jitter achieves the same effect for PSD-but-singular
matrices at zero extra cost, and *badly indefinite* input still throws — it
should, because that is corrupt data, not numerics.

## 4. Parametric (variance–covariance) VaR (`eqvar/parametric.hpp`)

    sigma_p = sqrt(wᵀ Σ w),   VaR = -(mu·h + z_alpha · sigma_p · sqrt(h))

with three tail models:

- **Normal**: `z = Phi^{-1}(alpha)`.
- **Variance-matched Student-t**: `z = t_nu^{-1}(alpha) · sqrt((nu-2)/nu)`,
  df > 2. The rescaling matches the *variance* so that switching the tail
  model changes only tail shape, never the sigma — otherwise t-VaR and
  normal-VaR differences would conflate two effects. df = 6 default (typical
  equity daily-return fit).
- **Cornish–Fisher**: `z_cf = z + (z²-1)S/6 + (z³-3z)K/24 - (2z³-5z)S²/36`
  with sample skewness S and excess kurtosis K, **guarded by an explicit
  monotonicity-domain check**: the quartic expansion is only a quantile
  function where `dz_cf/dz > 0` on `|z| ≤ 3.5`; outside that region (e.g.
  S = -0.5, K = 0 already fails) the "quantile" is not a quantile and the
  engine throws rather than returning nonsense. This check is inherited
  verbatim from the Python reference and is unit-tested on both sides.

*Why keep parametric at all?* It is the only closed-form method — the
analytic anchor every simulation method is validated against, the fastest
(sub-µs marginal cost per exposure update → intraday incremental VaR), and
the basis of Euler risk decomposition (`∂VaR/∂w = z Σw / sigma_p`).

## 5. Monte Carlo VaR (`eqvar/monte_carlo.hpp`)

Simulate factor returns `r = L z` (L the Cholesky factor), P&L `= wᵀr`, read
VaR/ES off the scenario distribution.

- **Multivariate normal**: `z ~ N(0, I)` via inverse-CDF on 53-bit uniforms.
- **Multivariate Student-t**: common-mixing construction `r = L z / sqrt(W/nu)`,
  `W ~ chi²_nu`, with scale matrix `Σ·(nu-2)/nu` so the simulated
  **covariance equals Σ exactly** while tails fatten — the same
  variance-matching discipline as the parametric family. A *single* mixing
  variable per path (not per factor) is what makes the joint tails fat —
  independent per-factor t draws would kill tail dependence, which is the
  point of using t.
- **Determinism**: one seeded `std::mt19937_64`; gaussians via our own
  `Phi^{-1}(u)` (Acklam + one Halley step), chi² via Marsaglia–Tsang gamma
  built on the same primitives. `std::normal_distribution` and
  `std::gamma_distribution` are deliberately avoided: their algorithms are
  implementation-defined, so results would differ across standard libraries.
  Given a seed, VaR is **bitwise** reproducible (unit-tested).
- **Error bars**: order-statistic asymptotic SE
  `sqrt(alpha(1-alpha)/n) / f_hat(q)`, with the density estimated by a
  symmetric order-statistic finite difference of bandwidth `ceil(sqrt(alpha n))`.
  Every MC number ships with its SE; convergence tests assert within 3 SE.

*Why MC when the portfolio is linear?* For a linear book, MC with a normal
factor model must reproduce parametric VaR — that identity is a validation
asset, not redundancy. MC is the only method that extends to fat-tailed
joint factors (t), and the architecture (simulate panel → revalue → tail
metrics) is the one that generalises to non-linear revaluation.

## 6. Expected Shortfall (`eqvar/expected_shortfall.hpp`)

**Empirical ES** is the *exact* integral of the empirical quantile function
over `(0, alpha]`: with `k = floor(alpha n)`,

    ES = -(1/(alpha n)) [ Σ_{i≤k} x_(i) + (alpha n - k) · x_(k+1) ]

i.e. fractional weight on the boundary order statistic — not "mean of
observations beyond VaR", which is biased for non-integer `alpha n`.
Closed forms: **normal** `ES = sigma·phi(z_alpha)/alpha - mean` (the identity
validated against quadrature to 1e-10) and **variance-matched Student-t**
`ES = sigma · f_nu(q)(nu+q²)/((nu-1)alpha) · sqrt((nu-2)/nu) - mean`.
ES ≥ VaR always (tested); ES is subadditive where VaR need not be, and FRTB
made 97.5 % ES the capital measure — both closed forms take `alpha = 0.025`.

## 7. Backtesting (`eqvar/backtest.hpp`)

An exception is a day with `pnl < -VaR`. Three regulator-standard tests:

- **Kupiec POF (1995)**: `LR_uc = -2 ln[ L(alpha) / L(x/T) ]` on the binomial
  likelihood, chi²(1) under H0; `0·ln 0 = 0` convention for degenerate counts.
- **Christoffersen (1998) independence**: LR of an i.i.d. exception process
  against a first-order Markov alternative on the transition counts
  `n00, n01, n10, n11`; clustering inflates `n11` and rejects. **Conditional
  coverage**: `LR_cc = LR_uc + LR_ind ~ chi²(2)` — right rate *and*
  independence jointly.
- **p-values** from the chi² survival function via the **regularized upper
  incomplete gamma** (series + continued fraction, `src/stats.cpp`) — no
  statistics library needed; validated at the classical critical value
  `chi²(1) = 3.8415 → p = 0.05` to 1e-12.
- **Basel traffic light** on the 250-day window with the *exact* regulatory
  count boundaries: 0–4 green (k = 3.0), 5–9 yellow (add-ons
  0.40/0.50/0.65/0.75/0.85), 10+ red (k = 4.0). The exact binomial CDF (via
  the regularized incomplete beta) reports where the observed count sits
  under a correct model — e.g. 4 exceptions is the 89.2nd percentile.

*Why LR tests and not just counting?* The traffic light *is* counting — but
it has known low power; Kupiec adds a calibrated significance level, and
Christoffersen catches the failure counting cannot see: 10 exceptions in one
volatility cluster versus 10 spread over the year (unit-tested with planted
patterns of both kinds).

## 8. Special functions (`eqvar/stats.hpp`)

- `Phi^{-1}`: Acklam's rational approximation (|rel err| < 1.15e-9) + one
  Halley refinement → ~1e-13 observed; upper half routed through the lower
  tail (`p > 0.5 → -Phi^{-1}(1-p)`, exact for doubles by Sterbenz), which
  makes symmetry bitwise and avoids erfc cancellation near 1.
- Regularized incomplete beta (Lentz continued fraction) → Student-t CDF and
  exact binomial CDF; incomplete gamma (series + CF) → chi² survival.
- Student-t quantile by bisection on the CDF to a machine-precision bracket
  (matches `scipy.stats.t.ppf` to < 1e-9; a 200-step bisection at ~µs cost is
  irrelevant next to any consumer, and has no tuning parameters to defend).
- Moments: `stdev` ddof = 1; skew/kurtosis as biased moment ratios
  (`scipy bias=True`) because that is what the Python reference feeds
  Cornish-Fisher.

## Assumptions register

Each assumption states *what breaks if violated*.

1. **Linear (delta) P&L mapping**: `pnl = wᵀr`. Violated by optionality/gamma
   → all methods mis-state risk on gap moves; the Python twin's full
   revaluation shows delta-gamma missing 37 % on a 2020 replay. This engine
   is for the linear (cash equity / futures) book or delta-mapped views.
2. **Daily returns are (locally) stationary over the window**. Violated in
   regime breaks → plain historical VaR lags ~250 days; use FHS/BRW (built
   for exactly this) and monitor with Christoffersen.
3. **Zero-mean daily returns** (EWMA covariance, default `mean = 0`).
   Violated for strongly trending factors → second-order at daily horizon
   (mean ≈ vol²·years); the `mean` parameter exists where it matters.
4. **i.i.d. for sqrt-time scaling**. Violated under vol clustering → 10-day
   VaR understated in stressed regimes; documented bias, use overlapping
   P&L in the research stack when it matters.
5. **Student-t df > 2** (finite variance) for variance matching. df ≤ 2 has
   no variance to match → throws by design.
6. **Cornish-Fisher inside its monotonicity domain**. Outside → not a
   quantile function; the engine refuses (domain check) instead of returning
   a number.
7. **Covariance input is symmetric PSD up to rounding**. Small negative
   eigenvalues → jitter absorbs them; a badly indefinite matrix (data error)
   → `std::runtime_error`, deliberately fatal.
8. **250-day, 99 % convention for Basel zones**; the count boundaries are
   specific to that window — other windows change the binomial, and the
   engine reports the exact binomial CDF for whatever `n_obs` is passed.
