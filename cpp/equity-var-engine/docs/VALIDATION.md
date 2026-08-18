# Validation

How the engine was validated, and where it fails. Everything below is
enforced by the test suite (`ctest`: 9 suites, 77 tests, ~320 assertions,
< 1 s, offline, deterministic) — validation claims that are not unit-tested
do not appear here.

## 1. Cross-language golden tests (the headline)

`tests/test_cross_language.cpp` pins the engine to the Python research
reference (`eq_var`, numpy 2.4.4 / scipy 1.17.1). Three deterministic cases
whose inputs are **closed-form (sin/cos) series both languages regenerate
independently** — no data files, no serialisation ambiguity:

- **Case A** — `pnl[t] = 100·sin(3t+1) + 0.5·t·cos(t)`, t = 0..99:
  historical VaR at 1 % / 5 % (122.4129222375264 / 104.5522835374927),
  empirical ES at 1 % / 5 % (141.7568107549531 / 122.6373207703405),
  BRW VaR (108.2348601407293), FHS VaR (109.3777910164513), and the 5-element
  BRW weight vector.
- **Case B** — fixed 3-asset book (w = [1e6, -5e5, 2e5], vols
  [1 %, 1.5 %, 2 %], full correlation matrix): portfolio sigma
  (9962.429422585637), parametric VaR normal/t(6)/10-day
  (23176.07650751402 / 25563.37478743866 / 73289.18899006478), closed-form
  normal and t ES at 2.5 %, Cornish-Fisher VaR (28230.62626871169).
- **Case C** — deterministic 60×3 returns panel: six sample- and
  EWMA-covariance entries, parametric VaR off the estimated matrices, Kupiec
  LR/p at (250, 7) and (250, 0), Christoffersen independence and conditional
  coverage on a planted 9-exception pattern.

Tolerance: **1e-9 relative** (+1e-12 absolute floor). The observed residual
is libm-vs-numpy ulp noise in sin/cos — far below any risk tolerance. The
constants were regenerated from the live Python stack on 2026-08-18 and
matched the committed values digit-for-digit; the provenance command sits in
the test header so model validation can re-run it at will.

## 2. Analytic identities and hand-computed values

| check | tolerance | test |
|---|---|---|
| `Phi^{-1}(0.975) = 1.959963984540054` | 1e-12 (spec 1e-8) | `NormalPpf.KnownQuantiles` |
| `Phi(Phi^{-1}(p)) = p`, symmetry across p ∈ [1e-8, 0.9999] | 1e-12 / 1e-9 | `NormalPpf.SymmetryAndRoundTrip` |
| normal ES identity `ES = sigma·phi(z)/alpha` vs Simpson quadrature | 1e-10 rel | `NormalEs.IdentityVsNumericalQuadratureTo1e10` |
| parametric closed forms (normal, variance-matched t) | 1e-12 rel | `ParametricVar.*` |
| Cholesky `A = LLᵀ` reconstruction on SPD input | 1e-12 | `Cholesky.ReconstructsSpdTo1e12` |
| empirical quantile / ES exact on hand-solved tiny arrays | exact (`EXPECT_DOUBLE_EQ`) | `QuantileLinear.HandExactTinyArrays`, `EmpiricalEs.HandExactTinyArrays` |
| BRW weights: sum = 1, ratio = 1/λ, monotone in recency | 1e-12 | `BrwWeights.*` |
| EWMA vol: seed = ddof-0 variance, recursion uses x[t-1] only (no look-ahead) | 1e-14 | `EwmaVolatility.NoLookAhead...` |
| Kupiec LR from first-principles log-likelihoods (T=250, x∈{0,5}) | 1e-12 | `Kupiec.*` |
| `chi²(1) sf(3.841458...) = 0.05` | 1e-12 (spec 1e-4) | `IncompleteGamma.Chi2PValues` |
| Student-t quantiles vs `scipy.stats.t.ppf` | 1e-9 | `StudentT.QuantilesVsScipy` |
| binomial CDF vs scipy at the Basel boundaries | 1e-12 | `IncompleteBeta.BinomialCdf` |
| Christoffersen transition counts on a hand-traced pattern | exact | `Christoffersen.TransitionCountsOnTinyPattern` |
| Basel zones: all boundaries 0/4/5/9/10 and the full add-on ladder | exact | `Basel.ExactZoneBoundariesAt250Obs` |

## 3. Monte Carlo convergence and determinism

- **MC → parametric**: 200k-path normal MC VaR within **3 order-statistic
  SE** of the closed form (SE itself asserted < 2 % of VaR); MC ES within
  4 SE of the normal-ES identity. The bench run prints the live numbers:
  68 681 vs 69 002 at SE 401 (0.8 SE).
- **Student-t fatter at 99 %**: t(5) MC VaR and ES strictly above normal MC
  on the same seed, and within 3 SE of the variance-matched t closed form.
- **Moment matching**: the simulated 200k-path covariance matches the target
  Σ entry-wise (5 % rel on variances, 10 % of sigma_i·sigma_j on
  covariances) for both normal and t — i.e. the `(nu-2)/nu` scale correction
  is verified, not assumed.
- **Bitwise determinism**: same seed → `EXPECT_EQ` (not NEAR) on VaR, ES and
  SE; different seed → different result. This holds across standard
  libraries because the engine uses only `mt19937_64` (bit-specified by the
  standard) and its own inverse-CDF transforms.
- **RNG health**: uniforms strictly inside (0,1); gaussian sample moments at
  ~4 SE bands; chi² mean/variance vs (df, 2df).

## 4. Statistical power checks (planted-pathology tests)

- **Christoffersen detects planted clustering**: a run of 10 consecutive
  exceptions in 250 days rejects independence at p < 0.001 while the same 10
  exceptions spread every 25 days give p > 0.10.
- **Kupiec monotonicity**: LR increases with excess exceptions; 10/250 at
  1 % rejects at 5 %; exact-coverage samples give LR = 0, p = 1; the
  degenerate x = 0 case matches `-2·250·ln(0.99)` analytically.
- **BRW regime response**: a crash on the most recent day moves VaR to the
  full crash size, the same crash 100 days ago does not.
- **FHS regime response**: damping the last 40 % of the sample cuts FHS VaR
  below plain historical VaR (and FHS is exactly positive-homogeneous).

## 5. Edge cases (documented **and** tested — `tests/test_edge_cases.cpp`)

| edge case | behaviour |
|---|---|
| empty inputs (pnl, exposures, prices, panel) | `std::invalid_argument` with informative message, everywhere |
| alpha ∉ (0, 0.5), incl. 0, 0.5, 1 | throws in every VaR/ES entry point |
| single-asset portfolio | collapses exactly to scalar formulas; MC within 3 SE; long/short symmetric |
| zero-variance asset in Σ (singular PSD) | Cholesky jitter path engages (jitter > 0, ≤ 1e-6 of variances), MC still within 3 SE of parametric |
| all-zero covariance | sigma = 0, parametric VaR = 0, ES = 0 exactly |
| constant P&L series | VaR = −constant (a gain floor), moments 0 not NaN, FHS floored (no 0/0) |
| NaN / Inf in P&L | throws — never silently dropped |
| fewer than 50 obs for historical methods | throws (`kMinHistObs` guard, boundary exact) |
| alpha ≪ 1/n | estimate pinned to the worst observed loss (type-7 interpolation inside the first gap), never extrapolated |
| Cornish-Fisher outside monotonicity domain (e.g. S = −0.5, K = 0) | throws with an explanation; `check_domain=false` override exists for diagnostics only |
| badly indefinite "covariance" | `std::runtime_error` after the jitter ladder — corrupt input is fatal by design |
| non-positive prices in return calc | throws |

## 6. Known failure modes and limits

1. **Tail resolution**: 250 observations put ~2.5 obs in a 1 % tail; the
   99 % historical VaR SE is large (the MC SE machinery quantifies the same
   effect). Below `kMinHistObs = 50` the engine refuses. For alpha < 1/n the
   estimate degrades to the sample minimum — tested, documented, and the
   reason parametric/MC exist.
2. **sqrt-time scaling understates stressed multi-day risk** (vol
   clustering): known bias, stated where used; do not feed the 10-day number
   to anything that assumes it is conservative.
3. **Cornish-Fisher validity region is small**: |S| ≳ 0.3 with K = 0 already
   fails on |z| ≤ 3.5. The engine throws rather than degrades — this is a
   feature; the Python reference behaves identically.
4. **RNG streams differ from NumPy's**: cross-language MC agreement is
   statistical (within SE bars), never bitwise — the golden tests therefore
   pin the deterministic estimators and closed forms, and MC is validated
   against closed forms *within* each language.
5. **EWMA λ is a hyperparameter**, not estimated: λ = 0.94/0.98 are
   RiskMetrics/BRW conventions. A mis-tuned λ shows up as Christoffersen
   clustering failures — that is the monitoring loop, by construction.
6. **Linear P&L only**: no gamma/vega. Options books need the Python twin's
   full revaluation; this engine's scope stops at the delta map.
7. **Student-t quantile via bisection** costs ~200 CDF evaluations (~µs).
   Irrelevant at desk call rates; would matter only inside a per-path loop,
   where it is never called (the MC uses inverse-normal + chi² mixing).

## 7. Compiler / toolchain hygiene

Built and tested with g++ 13.3, `-Wall -Wextra -Werror -O2`, C++20, cmake
3.28, GoogleTest 1.14 (system). Test binaries additionally pass
`-Wno-unused-result` only because `EXPECT_THROW(f(x), …)` legitimately
discards `[[nodiscard]]` values when asserting that `f` throws; the library
itself compiles warning-free with the full flag set.
