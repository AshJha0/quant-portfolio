# Methodology

This engine is the C++ port of the Python engine `eq_var` (linear
exposures plus European options repriced in full per scenario, see the
README's Option VaR section). The
Python engine revalues each position in full. This engine takes a vector of
linear factor exposures `w` and uses `P&L = w . r`. This file states the
conventions, the maths, the model choices and the assumptions.

## 0. Conventions (shared by the Python, Rust and C++ VaR engines)

| item | convention |
|---|---|
| alpha | Tail probability: 0.01 = 99 % VaR. Every public function rejects alpha outside (0, 0.5) with `std::invalid_argument("alpha is the tail probability (e.g. 0.01 for 99 %) ...")`. |
| sign | P&L in currency units, loss negative. VaR and ES are positive for a loss. |
| empirical VaR | Losses L = -pnl sorted descending. k = n alpha, m = ceil(k - 1e-9) clamped to [1, n]. VaR = L(m). No interpolation. |
| empirical ES | ES = [ sum_{i<m} L(i) + (k - (m-1)) L(m) ] / k (Acerbi-Tasche, exact fractional atom). |
| weighted VaR / ES | Weights normalised to 1, stable-sorted with the losses. W_j cumulative. m = first j with W_j >= alpha - 1e-12. VaR = L(m), ES = [ sum_{i<m} w_i L(i) + (alpha - W_{m-1}) L(m) ] / alpha. |
| Kupiec | `kupiec_pof(KupiecInput{.n_obs, .n_exceptions, .alpha})`. Throws for alpha outside (0, 0.5), n_obs < 1, n_exceptions < 0 or > n_obs. |
| Basel | F(x) = P(X <= x), X ~ Binomial(n_obs, alpha) by pmf recursion. Green if F < 0.95, yellow if F < 0.9999, red otherwise. Yellow plus factor from the 250-day table at the smallest x' in 5..9 with F250(x') >= F(x) - 1e-12. |
| Acerbi-Szekely | Z2 = sum_t pnl_t I_t / (T alpha ES_t) + 1, I_t = 1{pnl_t < -VaR_t}. Negative means ES is understated. |
| Cornish-Fisher | Valid iff g'(z) > 0 on [-z_a, z_a], z_a = Phi^{-1}(1 - alpha). Non-finite moments throw. |
| Cholesky | Symmetry within 1e-10 max abs entry. Jitter from 1e-12 to 1e-6 x mean(diag), x10 per step. Past the cap: throw, or repair by eigenvalue clip or Higham. |
| EWMA | s2_0 = mean of r^2 over the first min(30, n) observations, s2_t = lam s2_{t-1} + (1 - lam) r_{t-1}^2. |
| Monte Carlo SE | Bootstrap of the order-statistic VaR (primary, n_boot >= 2). Silverman KDE at the reported quantile (diagnostic). |
| empty book | An empty or all-zero exposure vector throws "book has no positions". |
| inputs | NaN, +inf, -inf, empty, too-short and negative-where-invalid inputs throw `std::invalid_argument`. |

## 1. Historical simulation (`eqvar/historical.hpp`)

**Plain historical VaR** is the order statistic L(m) above. It is computed
with `std::nth_element` (O(n)) and a partial sum of the m - 1 larger losses,
with no full sort. The order statistic is a value from the sample, so it is
the same in every language that rounds the inputs the same way. Type-7
interpolation (`quantile_linear`) is deprecated and no VaR path uses it.

**BRW age-weighted VaR and ES** (Boudoukh, Richardson, Whitelaw):
observation i (0 = oldest) gets weight (1-lam) lam^{n-1-i} / (1-lam^n), and
the weighted rule above applies. With lam = 0.98 and n = 250 the most recent
day carries about 2 % of the weight.

**Filtered historical simulation, per factor**
(`filtered_historical_var_es`). For each factor j, sigma_{t,j} is the EWMA
volatility of the factor returns (section 2), z_{t,j} = r_{t,j} / sigma_{t,j},
and the rescaled scenario is r*_{t,j} = z_{t,j} sigma_{T+1,j}, with
sigma_{T+1,j} the one-step-ahead forecast. Scenario P&L = w . r*. VaR and ES
then follow the order-statistic rule. The P&L-level `filtered_historical_var`
devolatilises the book P&L instead of the factors. It is deprecated and kept
for one release with its full-sample seed. Its tail now uses the same order
statistic as every other path; it used type-7 interpolation before, which
moved golden `a_fhs_var_5pct_lam094` from 109.3777910164513 to
112.05212695060598 (docs/MARKET_RISK.md has the before/after table).

**Square-root-of-time**: VaR_h = VaR_1 sqrt(h). It holds only for i.i.d.
zero-drift returns and understates multi-day risk under volatility
clustering.

## 2. EWMA volatility

RiskMetrics zero-mean recursion:

    s2_0 = mean(r_0^2 .. r_{w-1}^2),  w = min(30, n)
    s2_t = lam s2_{t-1} + (1 - lam) r_{t-1}^2,   sigma_t = sqrt(s2_t)

sigma_t is the forecast for day t from data up to t - 1. The seed reads the
first w observations, so sigma_t has look-ahead only inside the seed window.
`ewma_volatility_next` returns sqrt(lam s2_{n-1} + (1 - lam) r_{n-1}^2).
Variances are floored at 1e-32 to keep r / sigma finite on all-zero data.

## 3. Covariance and linear algebra (`eqvar/matrix.hpp`)

- **Sample covariance**, ddof = 1. The returns are centred once and the
  cross-products come from `qp::syrk_upper_blocked`, a cache-blocked
  `D^T D` that adds the rows in their original order. The result is the
  same double as the row-streaming loop it replaced. The Rust engine uses
  the same kernel (`qp_core::gemm`).
- **EWMA covariance**: Sigma <- lam Sigma + (1 - lam) r r' over all rows,
  lam = 0.94. The seed is the zero-mean second moment mean(r r') of the
  first min(seed_window, T) rows, seed_window = 30 (convention S8), so the
  seed uses no later data. It matches the Python `ewma_covariance` and the
  `c_ewma_cov_*` golden values.

**Cholesky.** Plain factorisation first. On a non-positive pivot the engine
adds eps I with eps = 1e-12 mean(diag), then x10 per try up to
1e-6 mean(diag). Any jitter is reported in `CholeskyResult::jitter_added`
and `warning`. A matrix that needs more than the cap has a real negative
eigenvalue, and factoring it would simulate a different covariance from the
one supplied. With `Repair::None` (default) the engine throws
`std::runtime_error` and names the most negative eigenvalue.

**Repair.** `Repair::Clip` eigen-decomposes, clips eigenvalues to
max(lam, 1e-10 mean(diag)), rebuilds, and rescales by D^{1/2} so the
original variances are kept. `Repair::Higham` runs Higham's (2002)
alternating projections with Dykstra correction on the correlation matrix
(unit-diagonal projection, then PSD projection), stops when successive
iterates differ by less than 1e-12 in relative Frobenius norm, fails after
1000 iterations, and rescales by the original vols. Both use a cyclic Jacobi
eigen solver (`symmetric_eigen`), which is accurate and simple at the
factor counts used here.

## 4. Parametric VaR (`eqvar/parametric.hpp`)

    sigma_p = sqrt(w' Sigma w),   VaR = -(mu h + z sigma_p sqrt(h))

- **Normal**: z = Phi^{-1}(alpha).
- **Variance-matched Student-t**: z = t_nu^{-1}(alpha) sqrt((nu-2)/nu),
  nu > 2. Switching the tail model then changes tail shape only, not sigma.
- **Cornish-Fisher**: z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36
  with skewness S and excess kurtosis K. The map is a quantile function only
  where it is monotone. `cornish_fisher_valid(S, K, alpha)` checks g' > 0 on
  the closed interval [-z_a, z_a], z_a = Phi^{-1}(1 - alpha), in closed form:
  g' is quadratic in z, so its minimum is at the vertex (if inside and the
  parabola opens upward) or at an endpoint. Example: (S, K) = (0.122, -0.427)
  has a concave g' with its minimum at the left endpoint. On the old fixed
  interval [-4, 4] it fails (g'(-4) = -9e-4). At alpha = 0.01 (z_a = 2.326)
  and alpha = 0.001 (z_a = 3.090) it is valid. `cornish_fisher_var` throws
  when the check fails, and always throws on non-finite moments.

## 5. Monte Carlo VaR (`eqvar/monte_carlo.hpp`)

Simulate r = L z, P&L = w . r, then VaR and ES by the order statistic.

- **Multivariate normal**: z ~ N(0, I) by inverse CDF on 53-bit uniforms.
- **Multivariate Student-t**: r = L z / sqrt(W / nu), W ~ chi2_nu, scale
  matrix Sigma (nu-2)/nu so the simulated covariance equals Sigma. One
  mixing variable per path keeps the joint tails dependent.
- **Repeatability**: one seeded `std::mt19937_64`. Gaussians come from the
  engine's own Phi^{-1} (Acklam plus one Halley step), chi2 from a
  Marsaglia-Tsang gamma sampler. A seed repeats exactly on one platform.
  Results can differ in the last bits across platforms because `std::log`,
  `std::exp` and `std::erfc` are not correctly rounded.
- **Simulate once**: `monte_carlo_var` simulates one scenario set and
  derives VaR, ES and both standard errors from it.
- **Standard error**: the bootstrap resamples the P&L with replacement
  n_boot times (default 200, at least 2), recomputes L(m) with
  `std::nth_element` and reports the ddof = 1 standard deviation. It reuses
  one buffer, so memory is O(n). The KDE value
  sqrt(alpha(1-alpha)/n) / f_hat(q), with a Silverman bandwidth
  0.9 min(sd, IQR/1.34) n^{-1/5} at the reported quantile q, is a
  diagnostic. It is biased low in deep tails.

For a linear book with a normal factor model, MC must reproduce parametric
VaR. The tests use that identity.

## 6. Expected Shortfall (`eqvar/expected_shortfall.hpp`)

Empirical ES follows the convention table. Closed forms: normal
ES = sigma phi(z_alpha) / alpha - mean, variance-matched Student-t
ES = sigma f_nu(q) (nu + q^2) / ((nu - 1) alpha) sqrt((nu-2)/nu) - mean.
ES >= VaR for every sample (tested).

## 7. Backtesting (`eqvar/backtest.hpp`)

An exception is a day with pnl < -VaR.

- **Kupiec POF**: LR = -2 ln[(1-a)^{n-x} a^x] + 2 ln[(1-p)^{n-x} p^x],
  p = x/n, 0 ln 0 = 0, chi2(1) p-value.
- **Christoffersen independence** on the transition counts n00, n01, n10,
  n11, and **conditional coverage** LR_cc = LR_uc + LR_ind, chi2(2).
- **Basel traffic light** for any window and alpha (convention table). At
  n = 250 and alpha = 0.01 the binomial zones give green 0-4, yellow 5-9 and
  red 10+, which is the 1996 table. On other windows the zone comes from
  F(x), and the yellow plus factor comes from the 250-day count with the
  same F. Example: n = 1000, x = 15 gives F = 0.9521, yellow, plus 0.40.
- **Acerbi-Szekely Z2** (convention table). The C++ engine returns Z2 only.
  The Python engine also simulates a p-value.

## 8. Special functions (`eqvar/stats.hpp`)

`eqvar/stats.hpp` is a thin wrapper: the functions below live in
`cpp/core/include/qp/var_stats.hpp`, shared with the FX engine, and the
Rust twins in `qp_core::stats`.

- Phi^{-1}: Acklam's rational approximation plus one Halley step. The upper
  half is routed through the lower tail (1 - p is exact for p in [0.5, 1)).
- `log_gamma`: Lanczos (g = 7, 9 terms), about 1e-15 relative, with
  reflection below 0.5. The library never calls `std::lgamma`, which writes
  glibc's global `signgam` and is not thread-safe.
- Regularised incomplete beta (Lentz continued fraction) for the Student-t
  CDF; incomplete gamma (series and continued fraction) for chi2.
- Binomial CDF by the pmf recursion pmf(0) = exp(n log1p(-p)),
  pmf(j+1) = pmf(j) (n-j)/(j+1) p/(1-p). It falls back to the incomplete
  beta when pmf(0) underflows.
- Student-t quantile by bisection on the CDF.
- Moments: stdev with ddof = 1; skewness and kurtosis as biased moment
  ratios, as the Python engine feeds Cornish-Fisher.

## 9. Hot-path implementation (`eqvar/kernels.hpp`)

The noexcept kernels compute the same numbers as the throwing API, which
now wraps them. The rules that keep them the same:

- **Selection, not sorting.** The order-statistic VaR / ES partitions the
  loss vector with `std::nth_element` around rank m: O(n) on average. The
  tail sum then adds the m - 1 largest losses in the order the partition
  leaves them, as before. The deprecated `quantile_linear` used a full sort;
  it now uses `nth_element` for x(lo) and the minimum of the upper part for
  x(lo + 1), the same two order statistics. `var_es_weighted` (BRW) still
  sorts, because it needs the cumulative weight in loss order.
- **Summation order is never changed.** Every dot product and reduction
  adds terms left to right as before. `try_matvec`, which is also the
  scenario P&L kernel (pnl = panel x exposures), works on four rows at a
  time with four accumulators. That adds instruction-level parallelism
  without reordering any row's sum. Blocking the Cholesky or reordering the
  covariance loops would change the last bits, so they keep their loop
  order; all matrices are row-major and the inner loops walk contiguous
  rows.
- **Simulate once, store less.** `try_monte_carlo_var_factored` takes the
  Cholesky factor L and forms v = L'w once per run. For a linear book a
  path's P&L is w'(m L z) = m (L'w)'z, so each path costs one O(n) dot
  product instead of an O(n^2) triangular matvec, and the n_paths x n
  scenario panel is never stored. The draws come in the same order as
  `simulate_factor_returns` (n normals, then the chi-square mixing draw),
  but the sums run in a different order, so the P&L vector equals the
  two-step path (`simulate_factor_returns` then `portfolio_pnl`) to
  1e-12 x sum_i |w_i r_i| per path, not bit for bit, and VaR and ES agree
  to 1e-12 relative. `monte_carlo_var` calls the same kernel, so the two
  public entry points still agree bit for bit. VaR, ES, the bootstrap SE
  and the KDE SE all read that one P&L vector.
- **No allocation.** Selection buffers, the Cholesky factor and its
  jittered copy, and the Monte Carlo vectors are caller storage. The
  bootstrap reuses one resample buffer across replicates.
- **One thread.** The Monte Carlo is sequential (one `mt19937_64` stream),
  so there is no thread pool and no generator change in this engine.

## Assumptions register

Each line states what breaks if the assumption fails.

1. **Linear P&L, pnl = w . r.** Options and other non-linear positions are
   mis-stated on large moves. Use the Python engine for those books.
2. **Locally stationary daily returns.** Plain historical VaR lags regime
   breaks. BRW and FHS respond faster. Christoffersen detects the lag.
3. **Zero-mean daily returns** in EWMA. Strongly trending factors add a
   second-order error at the daily horizon.
4. **i.i.d. returns for sqrt-time.** Ten-day VaR is understated under
   volatility clustering.
5. **Student-t df > 2.** No finite variance to match below that; throws.
6. **Cornish-Fisher inside its validity region.** Outside it the map is not
   a quantile function; the engine throws.
7. **Covariance symmetric PSD up to rounding.** Small negative eigenvalues
   are absorbed by jitter below 1e-6 mean(diag). Larger ones throw unless a
   repair is requested.
8. **Basel plus factors** come from the 1996 250-day table. For other
   windows the mapping through F(x) is this engine's convention, not a
   regulatory rule.
