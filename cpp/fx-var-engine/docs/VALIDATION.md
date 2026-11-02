# Validation

How the C++ engine is validated: analytic identities, golden values from
the Python engine, ordering tests, an input fuzz walk, and known failure
modes. Every claim below is enforced by `ctest --test-dir build`: 133 tests
in 15 executables, about 3 s serial, passing under GCC 15.2 and Clang 21.1
with `-DFXVAR_WERROR=ON` and under GCC 15.2 and Clang 21.1 with
`-fsanitize=address,undefined` (131 tests there: the allocation-counting
binary replaces `operator new`, which the sanitizer runtimes also define,
so sanitizer builds skip it).

## 1. Analytic identities

| Identity | Tolerance | Test |
| --- | --- | --- |
| Triangulation: long N EURJPY equals long N EURUSD plus the USDJPY leg, per scenario | 1e-12 rel | `TriangulationIdentityEURJPY` |
| Forward at the CIP strike has zero value | 1e-10 | `ForwardZeroValueAtInceptionAndCip` |
| Forward equals two discounted deposit legs under FX shocks | 1e-10 | `ForwardMatchesTwoLegDepositDecomposition` |
| Base-currency cash has zero P&L | 1e-9 x notional | `BaseCurrencyPositionHasZeroRisk` |
| Empirical VaR/ES on 10-point samples, fractional atom, weighted case | exact | `test_expected_shortfall.cpp`, `OrderStatistic.FractionalAtomByHand` |
| Normal ES = sigma phi(z) / alpha, z = Phi^{-1}(1 - alpha); 2.3263 / 2.6652 at alpha 0.01 | 1e-10 / 1e-9 | `NormalVarEsIdentity` |
| Inverse normal CDF vs scipy | 1e-9 | `InverseCdfBelow1eMinus9` |
| log_gamma vs closed forms | 1e-13 | `LogGamma.LanczosMatchesClosedForms` |
| Kupiec and Christoffersen LR from the log-likelihoods | 1e-12 | `test_backtest.cpp`, `Kupiec.StructInputAndTailAlpha` |
| Basel: 1996 table at n = 250; n = 1000, x = 15 yellow +0.40; n = 500, x = 9 +0.50 | exact | `Basel.PlusFactorMappedThrough250DayTable` |
| Acerbi-Szekely Z2 on a 4-day hand case (-1.25) | 1e-15 | `AcerbiSzekely.Z2ByHand` |
| Jacobi eigenvalues of the 3 x 3 indefinite reference matrix (-0.8, 1.9, 1.9) | 1e-14 | `Cholesky.HighamReferenceCase` |
| EWMA seed = mean r^2 over the first 30 rows | 1e-15 | `Ewma.SeedIsMeanSquareOverFirst30` |
| Reverse stress closed form; loss-target inversion | 1e-10 | `test_stress.cpp` |

## 2. Cross-model and ordering tests

- **MC vs parametric**: 100k-scenario normal MC VaR within 3 SE of the
  closed form on the same book.
- **Tail ordering at 99 %**: Student-t(4) and jump-mixture VaR and ES
  exceed normal MC at the same covariance and seed.
- **Jump horizon**: with a per-day jump probability of 0.05 over 10 days,
  the share of scenarios with at least one jump is 1 - 0.95^10 = 0.401
  within 5 binomial SE, and the mean jump return is -0.1 lam h within 5 SE
  (`Jumps.PoissonCountScalesWithHorizon`). The old Bernoulli overlay gave
  0.049.
- **Standard error**: on N(0, 1) with 100k draws at alpha = 0.01 the
  bootstrap SE is within 20 % of sqrt(a(1-a)/n) / phi(z) and the KDE SE
  within 10 % (`MonteCarloSe.BootstrapPrimaryKdeDiagnostic`).
- **Repeatability**: same seed gives `EXPECT_EQ` on VaR, ES and P&L entries
  on one platform. Other libms can change the last bits because
  `std::log`, `std::exp` and `std::erfc` are not correctly rounded.
- **FHS regime response**: on a history whose second half is 3x more
  volatile, filtered HS VaR exceeds plain HS VaR.
- **Reverse stress**: an independent projected-gradient search matches the
  closed form to 1e-6 in loss.

## 3. Cross-language golden values (Python `fx_var`)

`tests/test_golden_python.cpp` asserts the generated
`tests/golden_vectors.hpp` (75 values):

- **Case A**: 3-position book on a 300-day sinusoidal history. One scenario
  P&L, plain HS VaR/ES at alpha 0.01 and 0.025, BRW VaR/ES at decay 0.995.
  1e-6 absolute on values near 6e4.
- **Case B**: `var_covar` normal and t(5) at alpha 0.01, 10-day normal.
  1e-8 relative (Acklam plus Halley vs Cephes inverse CDF).
- **Case C**: Kupiec (250, 8), Christoffersen on a fixed pattern, Basel CDF
  at x = 4, 5, 10. 1e-10 to 1e-12.
- **Shared `conv_*` keys** (identical in the equity JSON): VaR/ES at alpha
  0.010 / 0.025 / 0.050 on the 250-point sample P; Basel zone, multiplier
  and CDF on nine windows; Kupiec LR; Z2; five Cornish-Fisher flags; Higham
  and clip repairs; the EWMA one-step vol. Per-key tolerances are stated in
  the test (1e-12 relative for order statistics and EWMA, 1e-12 absolute
  for the Basel CDF, 1e-10 for Kupiec, 1e-8 for Higham).

Regenerate:

```bash
cd python/fx/03-var-es-engine
PYTHONPATH=src python3 tests/golden/generate_golden.py   # -> tests/golden/golden_vectors.json
cd ../../../cpp/fx-var-engine
python3 tools/gen_golden_header.py --check              # exit 0 = header matches the JSON
python3 tools/gen_golden_header.py                      # rewrite tests/golden_vectors.hpp
```

## 4. Failure modes

- **F1 Peg blindness.** A pegged currency has near-zero historical moves, so
  HS and var-covar report near-zero VaR. The peg screen flags `FX:*`
  factors with daily vol below 5e-4; the jump-mixture MC on the same book
  reports more than 20x the HS VaR
  (`MonteCarlo.PegBreakJumpProducesLossHistoricalSimulationMisses`).
  The risk is detected and priced; no historical method can remove it.
- **F2 Singular covariances.** Pegged blocks make Sigma rank-deficient. The
  jitter ladder (at most 1e-6 x mean(diag)) factorises and reports it. A
  larger negative eigenvalue throws unless `Repair::kClip` or
  `Repair::kHigham` is requested (`Cholesky.JitterIsCappedAt1e6MeanDiag`).
- **F3 Cornish-Fisher outside its region.** `cornish_fisher_valid(S, K,
  alpha)` checks monotonicity on [-z_a, z_a] in closed form.
  (0.122, -0.427) fails on the old fixed [-4, 4] interval, where g'(-4) =
  -9.15e-4 at the endpoint, and is valid at alpha = 0.01 and 0.001
  (`CornishFisher.DomainCheckIsExactNotGridResolutionDependent`).
- **F4 Square-root-of-time on carry books.** Serially correlated unwinds
  break i.i.d. aggregation; 10-day figures are understated.
- **F5 KDE standard error.** The Silverman bandwidth is set by the bulk of
  the distribution, so at deep tails or below about 20 000 scenarios the
  KDE SE is biased low. The bootstrap SE is the primary figure in
  `MonteCarloResult::se_var`; the KDE value is in `se_var_kde`.
- **F6 Scale-relative guards.** The PSD test in `portfolio_sigma` and the
  Cholesky symmetry test are relative to the magnitudes involved
  (`PortfolioSigma.HedgedBillionDollarBookAgainstRankDeficientCovariance`,
  `Cholesky.SymmetryToleranceIsRelativeToScale`).
- **F7 Non-finite inputs.** Every public function throws
  `std::invalid_argument` on NaN, +inf or -inf (`test_fuzz.cpp`).
- **F8 Kupiec size at 250 days.** At n = 250 and alpha = 0.01 the expected
  count is 2.5, and the exact rejection rate of a nominal 5 % chi2(1) test
  is about 9.5 %. It falls to about 5.5 % at n = 1000.

## 5. Edge cases

| edge case | behaviour |
| --- | --- |
| empty book | throws "book has no positions" |
| base-currency cash only | zero exposure and zero P&L |
| 1- and 2-scenario samples | VaR and ES collapse onto the observed loss; SE functions refuse fewer than 10 |
| alpha 0.99 (old confidence convention) | throws |
| alpha at 1e-6 and 0.5 - 1e-12 | accepted |
| horizon 0, negative, NaN or inf | throws |
| jump probability 1 or more | throws (intensity -ln(1 - p) is infinite) |
| bootstrap with n_boot < 2 | throws |
| indefinite Sigma beyond the jitter cap | `std::runtime_error` naming the most negative eigenvalue |
| NaN or inf in Sigma, exposures, returns, shocks, positions or rates | `std::invalid_argument` |

## 6. Low-latency API and benchmark

| Claim | Check |
| --- | --- |
| Every `try_*` kernel is `noexcept` | `static_assert`s in `tests/test_latency_api.cpp` |
| `try_*` results equal the throwing API bit for bit | `LatencyApi.*` (VaR/ES and SE on 5 sample sizes x 5 alphas, closed forms, mat-vec, quadratic form, sigma, var-covar, Cholesky, normal and t simulation) |
| Batch book revaluation equals `CompiledBook::pnl(ReturnsMatrix)` bit for bit | `BatchApi.BookRevaluationEqualsLegacyBitForBit` (5,000 Student-t scenarios) |
| Errors carry the throwing API's message text | `LatencyApi.ErrorsCarryTheThrowingMessage` |
| No heap allocation on the kernel success path | `test_no_alloc` (counting global `operator new`) |
| No numerical change from the refactor | 87,761 hex-float results and exception texts of the legacy API are identical against the old and new library, GCC 15.2 and Clang 21.1 (a scratch program; not part of ctest) |

`var_standard_error` used to sort a copy of the sample to read two
interquartile order statistics. It now reads them with `std::nth_element`
plus a minimum over the upper part, which returns the same two numbers in
O(n). The bootstrap SE reuses one selection buffer across replicates
instead of allocating a loss vector per replicate.

The README describes the benchmark (synthetic book, per-call latency
percentiles) and carries the measured table.

## 7. Reproducing

```bash
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
./build/fxvar_bench --out /tmp/fxvar_latency.json
```

No network beyond the optional GoogleTest fetch and no data files: every
test input is generated in the source.
