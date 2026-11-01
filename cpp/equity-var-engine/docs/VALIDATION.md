# Validation

How the engine is validated, and where it fails. Every claim below is
enforced by the test suite: `ctest` runs 115 tests in 11 executables,
offline, in about 7 s serial (GCC 15.2, Release). The same suite passes
under Clang 21.1 and under GCC with `-fsanitize=address,undefined`.

## 1. Cross-language golden tests

`tests/test_cross_language.cpp` pins the engine to the Python engine
(`eq_var`, Python 3.12.10, numpy 2.5.4, scipy 1.18.1). All inputs are
closed-form series that both languages rebuild, so there are no data files.

- **Case A**: pnl[t] = 100 sin(3t + 1) + 0.5 t cos(t), t = 0..99.
  Historical VaR at 1 % / 5 % (141.75681075495305 / 108.2348601407293),
  ES at 1 % / 5 % (141.75681075495305 / 122.6373207703405), BRW VaR
  (108.2348601407293), the deprecated P&L-level FHS VaR (109.3777910164513)
  and the 5-element BRW weight vector. With n = 100 and alpha = 0.01, k = 1,
  so VaR and ES at 1 % are the same worst loss.
- **Case B**: fixed 3-asset book. Portfolio sigma (9962.429422585637),
  parametric VaR normal / t(6) / 10-day, closed-form ES at 2.5 %,
  Cornish-Fisher VaR (28230.62626871169).
- **Case C**: 60 x 3 returns panel. Sample and EWMA covariance entries,
  parametric VaR on them, Kupiec at (250, 7) and (250, 0), Christoffersen
  independence and conditional coverage on a planted pattern.
- **Shared-convention keys (`conv_*`)**, identical in the equity and FX
  JSONs: VaR and ES at alpha 0.010 / 0.025 / 0.050 on the 250-point sample
  P; Basel zone, multiplier and CDF on nine (n, x, alpha) windows; Kupiec LR
  on three windows; Acerbi-Szekely Z2; five Cornish-Fisher validity flags;
  Higham and clip repairs of [[1, .9, .9], [.9, 1, -.9], [.9, -.9, 1]]; the
  EWMA one-step vol.

Tolerances: cases A to C use 1e-9 relative plus 1e-12 absolute (libm vs
numpy differences in sin/cos). The `conv_*` tolerances are stated per key
in the test: 1e-12 relative for order statistics, partial sums and EWMA,
1e-12 absolute for the Basel CDF, 1e-10 for Kupiec LR (the two Python
engines differ by 3e-15 there), 1e-8 for Higham (the spec value). The header
is generated: `tools/gen_golden_header.py --check` fails when it does not
match the JSON.

## 2. Analytic identities and hand-computed values

| check | tolerance | test |
|---|---|---|
| Phi^{-1}(0.975) = 1.959963984540054 | 1e-12 | `NormalPpf.KnownQuantiles` |
| normal ES identity vs Simpson quadrature | 1e-10 rel | `NormalEs.IdentityVsNumericalQuadratureTo1e10` |
| order statistic and fractional ES on n = 10 by hand (VaR 8, ES 9.2) | exact / 1e-15 | `OrderStatistic.FractionalAtomByHand` |
| weighted rule on uniform and concentrated weights | 1e-14 | `OrderStatistic.WeightedMatchesUniformAndHandCase` |
| historical VaR equals L(m) of the sorted losses | exact | `HistoricalVar.IsOrderStatisticAndIsMonotoneInAlpha` |
| EWMA seed = mean r^2 over the seed window; recursion uses r[t-1] | 1e-14 | `EwmaVolatility.NoLookAheadAndRiskMetricsRecursion` |
| Kupiec LR from the log-likelihoods | 1e-12 | `Kupiec.*` |
| Basel: n = 500, x = 8 green; n = 1000, x = 15 yellow +0.40; 1996 table at n = 250 | exact | `Basel.ZonesByBinomialCdfOnAnyWindow` |
| Acerbi-Szekely Z2 on a 4-day hand case (-1.25) | 1e-15 | `AcerbiSzekely.Z2ByHand` |
| Cornish-Fisher validity at alpha 0.01 and 0.001 | exact | `CornishFisher.AlphaAwareValidity` |
| Jacobi eigenvalues of [[2, 1], [1, 2]] | 1e-15 | `Cholesky.JacobiEigenByHand` |
| Higham and clip repairs vs numpy | 1e-8 / 1e-12 | `Cholesky.IndefiniteThrowsUnlessRepaired` |
| log_gamma vs closed forms | 1e-13 | `LogGamma.LanczosMatchesClosedForms` |
| per-factor FHS: homogeneity and a one-factor hand build | 1e-9 / 1e-12 | `Fhs.*` |
| chi2(1) sf(3.8415) = 0.05 | 1e-12 | `IncompleteGamma.Chi2PValues` |
| Student-t quantiles vs scipy | 1e-9 | `StudentT.QuantilesVsScipy` |

## 3. Monte Carlo convergence

- **MC vs parametric**: 200k-path normal MC VaR within 3 SE of the closed
  form; MC ES within 4 SE of the normal ES identity.
- **Student-t**: t(5) MC VaR and ES above normal MC at 99 % and within 3 SE
  of the variance-matched closed form.
- **Moment matching**: the simulated covariance matches the target for
  normal and t, which checks the (nu-2)/nu scale.
- **Standard error**: on N(0, 1) with n = 100k and alpha = 0.01 the
  bootstrap SE is within 20 % of the asymptotic value
  sqrt(a(1-a)/n) / phi(z_a) and the KDE value within 10 %
  (`MonteCarloSe.BootstrapNearAsymptoticAndNeedsTwoReplicates`). The 20 %
  covers the 3.5 % sampling error of a 400-replicate SE and the small-sample
  bias of the quantile bootstrap.
- **Repeatability**: same seed gives `EXPECT_EQ` on VaR, ES and SE; a
  different seed gives a different result. This holds on one platform. Other
  libms can change the last bits.

## 4. Planted-pathology checks

- Christoffersen rejects 10 consecutive exceptions in 250 days at p < 0.001
  and accepts the same 10 spread every 25 days (p > 0.10).
- Kupiec LR grows with excess exceptions; x = 0 matches -2 * 250 ln(0.99).
  At n = 250 and alpha = 0.01 the chi2(1) approximation is oversized: the
  exact rejection rate of a nominal 5 % test is about 9.5 %.
- BRW: a crash on the last day moves VaR to the crash size; the same crash
  100 days earlier does not.

## 5. Edge cases (`tests/test_edge_cases.cpp`, `tests/test_fuzz.cpp`)

| edge case | behaviour |
|---|---|
| NaN, +inf or -inf in any input of any public function | `std::invalid_argument` (fuzz test walks every function) |
| empty inputs, too-short samples, negative where invalid | `std::invalid_argument` |
| alpha outside (0, 0.5), including 0.99 | throws everywhere |
| empty or all-zero exposure vector | throws "book has no positions" |
| `mc_bootstrap_se` with n_boot < 2 | throws |
| zero-variance asset (singular PSD) | jitter below 1e-6 mean(diag), MC within 3 SE |
| all-zero covariance | sigma = 0, Cholesky returns L = 0 |
| constant P&L | VaR = -constant, moments 0, KDE SE 0 |
| fewer than 50 observations for historical methods | throws |
| alpha below 1/n | VaR and ES equal the worst observed loss |
| Cornish-Fisher outside its validity region | throws; non-finite moments always throw |
| covariance needing more than 1e-6 mean(diag) jitter | `std::runtime_error` naming the most negative eigenvalue, unless `Repair::Clip` or `Repair::Higham` |
| indefinite matrix reaching `portfolio_sigma` | w' Sigma w < 0 throws |
| NaN VaR in the backtest | throws; a NaN compares false and would count as no exception |

## 6. Known failure modes and limits

1. **Tail resolution.** 250 observations put about 2.5 points in a 1 %
   tail. The engine refuses fewer than 50.
2. **KDE SE is biased low in deep tails.** It is reported as a diagnostic
   next to the bootstrap SE.
3. **Square-root-of-time understates stressed multi-day risk.**
4. **Jitter is a rounding repair.** Above 1e-6 mean(diag) the engine throws
   or applies the requested PSD repair. A repaired matrix is a different
   covariance; `CholeskyResult::repaired` and `warning` say so.
5. **RNG streams differ from NumPy.** Cross-language MC agreement is
   statistical. Golden tests pin deterministic estimators only.
6. **EWMA lambda is a fixed parameter** (0.94 / 0.98), not estimated.
7. **Linear P&L only.** No gamma or vega.
8. **Bootstrap cost.** 200 resamples of 100k paths add about 0.4 s to a
   100k-path run.

## 7. Toolchain

Built and tested with GCC 15.2 and Clang 21.1, C++20, CMake 4.2,
GoogleTest 1.15.2, with `-DEQVAR_WERROR=ON` (`-Wall -Wextra -Wpedantic
-Werror`), plus one GCC build with `-fsanitize=address,undefined`. Test
binaries add `-Wno-unused-result` (`EXPECT_THROW` discards `[[nodiscard]]`
values) and `-Wno-deprecated-declarations` (tests call the deprecated
entry points on purpose).
