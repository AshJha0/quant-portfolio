# Methodology

C++23 FX VaR/ES engine. It is the linear-book port of the Python engine
`python/fx/03-var-es-engine` (`fx_var`): cash, spot and forwards with full
revaluation, no options. This file states the conventions, the reasons for
a compiled engine, the model choices, and the assumptions. The Python
`docs/METHODOLOGY.md` carries the longer derivations.

## 0. Conventions (shared by the Python, Rust and C++ VaR engines)

| item | convention |
|---|---|
| alpha | Tail probability: 0.01 = 99 % VaR. Values outside (0, 0.5) throw `std::invalid_argument("alpha is the tail probability (e.g. 0.01 for 99 %) ...")`. Until this release the FX engine took a confidence level (0.99). The same parameter now means the tail, and 0.99 throws. Defaults are 0.01. |
| sign | P&L in base currency, loss negative. VaR and ES are positive losses. |
| empirical VaR | Losses L = -pnl sorted descending. k = n alpha, m = ceil(k - 1e-9) clamped to [1, n]. VaR = L(m). Uniform weights use `std::nth_element`, no full sort. |
| empirical ES | ES = [ sum_{i<m} L(i) + (k - (m-1)) L(m) ] / k. |
| weighted VaR / ES | Weights normalised, stable-sorted with the losses. m = first j with W_j >= alpha - 1e-12. ES = [ sum_{i<m} w_i L(i) + (alpha - W_{m-1}) L(m) ] / alpha. |
| closed forms | Normal VaR = -mu + sigma z, ES = -mu + sigma phi(z) / alpha, z = Phi^{-1}(1 - alpha). Student-t standardised to unit variance. |
| Kupiec | `kupiec_pof(KupiecInput{.n_obs, .n_exceptions, .alpha})`. The old `(n_exceptions, n_obs, alpha)` overload is deprecated. |
| Basel | F(x) by Binomial(n_obs, alpha) pmf recursion. Green F < 0.95, yellow F < 0.9999, red otherwise. Yellow plus factor from the 250-day table at the smallest x' in 5..9 with F250(x') >= F(x) - 1e-12. |
| Acerbi-Szekely | Z2 = sum_t pnl_t I_t / (T alpha ES_t) + 1. Negative means ES is understated. |
| Cornish-Fisher | Valid iff g'(z) > 0 on [-z_a, z_a], z_a = Phi^{-1}(1 - alpha). Non-finite moments throw. |
| Cholesky | Symmetry within 1e-10 max abs entry. Jitter 1e-12 to 1e-6 x mean(diag), x10 per step. Past the cap: throw, or repair (`Repair::kClip`, `Repair::kHigham`). |
| EWMA | s2_0 = mean of r^2 over the first min(30, n) rows; s2_t = lam s2_{t-1} + (1 - lam) r_{t-1}^2. |
| MC jumps | Per-day jump probability p becomes the intensity lam = -ln(1 - p). Over horizon h the count is N ~ Poisson(lam h). Jump return = sum of N iid N(mu_J, sigma_J). Not compensated: adds drift lam h mu_J. |
| MC SE | Simulate once. Bootstrap SE (primary, n_boot >= 2), Silverman KDE SE at the reported quantile (diagnostic). |
| empty book | Throws "book has no positions". A book of base-currency cash only has zero risk and returns 0. |
| inputs | NaN, +inf, -inf, empty and invalid negative inputs throw `std::invalid_argument`. |

## 1. Why a compiled engine for FX risk

- **Intraday books.** The compiled book (struct-of-arrays legs, one `exp()`
  per leg per scenario) revalues a synthetic 250-position / 50-factor book
  scenario by scenario; the README carries measured per-call latencies.
- **Batch windows.** One binary, no interpreter start-up, flat memory.
- **Executable contract with the Python engine.** Golden values generated
  by the Python engine are compiled into the tests; any drift fails the
  build.

## 2. Factor model and book

Same as `fx_var.common` and `fx_var.book`:

- Pairs are quoted BASE/QUOTE (EURUSD = USD per 1 EUR).
- Each currency has one USD factor: `FX:CCY` is the daily log return of
  CCYUSD. USD has none. A EURJPY position becomes a long-EUR and a short-JPY
  leg, so cross risk follows from the USD-leg covariance (tested to 1e-12).
- `IR:CCY` is an absolute shock to the continuously compounded ACT/365 zero
  rate, flat per currency.
- **Forwards** are spot plus two deposit legs:
  V_usd = N e^{-r_f T} S_f - N K e^{-r_d T} S_d.
- **P&L**: PnL = V1_usd / S1_base - V0_usd / S0_base, with the base
  currency's own USD price shocked consistently.
- **Options are out of scope** and stay in the Python engine.

`CompiledBook` flattens positions into USD legs once: discounted value
v0 = amount S0 e^{-r0 T}, an FX index, an IR index and -T. A scenario
revalues a leg as v0 exp(dfx - T dr): spot S exp(r) on log returns.

## 3. VaR methods

- **Historical simulation**: plain, BRW age-weighted, and per-factor
  filtered (each factor devolatilised by its EWMA sigma and rescaled to the
  one-step-ahead forecast). Full revaluation under each scenario.
- **Parametric**: finite-difference exposures, then normal or standardised
  Student-t on the same sigma, plus Cornish-Fisher with the alpha-aware
  validity check. Example: (S, K) = (0.122, -0.427) fails on the old fixed
  interval [-4, 4] (g'(-4) = -9e-4, at the endpoint) and is valid at
  alpha = 0.01 and 0.001.
- **Monte Carlo**: normal, covariance-matched t, and the jump mixture
  (Poisson count over the horizon). The jump overlay adds variance that the
  covariance cannot contain: the peg-break case.
- **ES** with every method.

## 4. Assumptions register

| # | Assumption | What breaks if violated |
| --- | --- | --- |
| A1 | Cross risk is captured by the two USD legs. | Crosses away from triangulated parity (CNH vs CNY) are misstated; add a separate factor. |
| A2 | `FX:*` are daily log returns; scenarios exponentiate. | Simple returns would double-count convexity on large shocks. |
| A3 | Flat zero curve per currency, ACT/365. | Curve-shape risk is invisible; long-dated forwards misprice. |
| A4 | CIP holds in revaluation. | In stressed funding the cross-currency basis widens; forwards need a basis factor. |
| A5 | Square-root-of-time to multi-day horizons. | 10-day VaR is understated for books with autocorrelated unwinds. |
| A6 | Covariance is stationary within the window. | Regime breaks; EWMA recovers with a lag; the peg screen and stress add-on cover the gap. |
| A7 | i.i.d. scenarios in plain HS. | Christoffersen failures under clustering; use FHS. |
| A8 | Pegged currencies: history has no break. | HS and parametric VaR near 0 on the peg. The engine flags it and prices it through jumps and peg-break scenarios. |
| A9 | Repeatability uses library-owned transforms over `mt19937_64`. | `std::normal_distribution` would change streams between standard libraries. Last-bit differences across libms remain. |
| A10 | Linear instruments only. | Option books need the Python engine. |
| A11 | Jumps are uncompensated. | With mu_J != 0 the jump overlay shifts the mean by lam h mu_J. |

## 5. Design choices

- **Warnings as data.** Peg flags, Cholesky jitter and repair notes are
  returned in result structs, not written to stderr.
- **Refuse, never impute.** NaN or infinite inputs, invalid alpha, empty
  books, invalid Cornish-Fisher moments, indefinite covariances beyond the
  jitter cap and impossible stress moves all throw.
- **Capped jitter.** Singular covariances are routine in FX (two currencies
  pegged to one anchor), so a jitter up to 1e-6 x mean(diag) is applied and
  reported. Beyond that the matrix has a real negative eigenvalue and the
  engine throws unless a repair is requested. The deprecated
  `robust_cholesky` used to escalate to 1e-5 x mean(diag) without this
  check; it now forwards to `cholesky`.
- **Scale-relative guards.** The PSD test in `portfolio_sigma` and the
  Cholesky symmetry test are relative to the magnitudes involved.
- **Two API layers.** The throwing API validates and throws
  `std::invalid_argument`. Under it, `fxvar/kernels.hpp` exposes `noexcept`
  kernels that return `std::expected<T, fxvar::Error>` and work in
  caller-provided buffers, so a caller on a latency budget sizes its
  buffers once and never allocates or throws per call. The arithmetic is
  shared, so both layers return the same bits.
- **Selection, not sorting.** Uniform-weight VaR/ES and the KDE standard
  error's interquartile range use `std::nth_element` (O(n)). Only the
  weighted VaR/ES sorts, because the cumulative weight must be walked in
  loss order.
- **Row-major linear algebra.** Matrices are dense row-major. Mat-vec,
  quadratic form, Cholesky and the scenario product L z read rows
  contiguously. Summation order is never changed for speed, since that
  would change results in the last bits.
- **Special functions in the library.** Acklam plus Halley inverse normal,
  Lanczos `log_gamma` (thread-safe; `std::lgamma` writes glibc's global
  `signgam`), incomplete gamma and beta, binomial CDF by pmf recursion.
