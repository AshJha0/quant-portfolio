# Validation

How the C++ engine was validated: analytic identities, cross-language
golden constants against the Python reference, statistical/ordering
tests, benchmarks, and known failure modes. Every claim below is enforced
by the test suite (`ctest --test-dir build`): **83 tests / ~340 assertions,
all passing under `-Wall -Wextra -Werror`**.

---

## 1. Analytic identities

| Identity | Tolerance | Test |
| --- | --- | --- |
| Triangulation: long N EURJPY ≡ long N EURUSD + long N·X(EURUSD) USDJPY, per scenario | 1e-12 (rel) | `test_book.cpp` / `TriangulationIdentityEURJPY` |
| Forward at CIP strike has zero inception value | 1e-10 | `ForwardZeroValueAtInceptionAndCip` |
| Forward ≡ two discounted deposit legs (+N e^{−r_f T} BASE, −N K e^{−r_d T} QUOTE) under FX shocks | 1e-10 | `ForwardMatchesTwoLegDepositDecomposition` |
| Forward rate-leg deltas = ∓T·N·e^{∓rT}·S closed form | 1e-4 rel (FD) | `ForwardRateLegSensitivities` |
| Base-ccy cash has zero P&L under any FX shock | 1e-9·notional | `BaseCurrencyPositionHasZeroRisk` |
| Empirical VaR/ES hand-exact on 10-point samples (incl. fractional Acerbi–Tasche atom, weighted BRW case) | exact | `test_expected_shortfall.cpp` |
| Normal ES ≡ σ·φ(z_α)/(1−α); classic values (2.3263, 2.6652 at 99%) | 1e-10 / 1e-9 | `NormalVarEsIdentity` |
| Standardised t: fatter than normal at equal σ; df→∞ recovers normal | ordering / 1e-5 | `StudentTFatterThanNormalAtEqualSigma` |
| Stratified normal grid: empirical estimator → closed form | 2e-4 / 2e-3 | `EmpiricalConvergesToNormalIdentity` |
| Inverse normal CDF vs AS241/scipy quantiles (0.95/0.975/0.99/0.999/1e-6…) | 1e-9 (< 1e-13 measured) | `InverseCdfBelow1eMinus9` |
| Student-t quantile vs scipy (`t.ppf`), Cauchy q(0.75)=1, CDF round-trips | 1e-10 / 1e-11 | `CdfQuantileRoundTripAndKnownValues` |
| χ² SF vs scipy at 1/2 df; P(0.5,x)=erf(√x); χ²(2) sf = e^{−x/2} | 1e-10–1e-12 | `RegularisedIncompleteAndChi2`, `SurvivalFunctionSpotChecks` |
| Cholesky reconstruction L·Lᵀ = Σ; exact-zero-pivot rank-1 matrix engages jitter and reports it | 1e-12 | `test_matrix_stats.cpp` |
| var_covar ≡ hand-computed w'Σw quantile; √time scaling of VaR and ES (h=10) | 1e-12 / 1e-10 | `test_parametric.cpp`, `test_historical.cpp` |
| Kupiec & Christoffersen LR ≡ hand-computed log-likelihoods; LR_cc = LR_uc + LR_ind | 1e-12 | `test_backtest.cpp` |
| Basel zones exact at 250d/99%: green 0–4, yellow 5–9 (3.40/3.50/3.65/3.75/3.85), red 10+ (4.0) — from the cumulative binomial, not a table | exact | `ExactRegulatoryBoundaries` |
| Reverse stress closed form: loss = k·σ_p, shock reproduces loss via −w'dx; loss-target inversion | 1e-10–1e-12 | `test_stress.cpp` |

## 2. Cross-model and ordering tests

- **MC → parametric**: 100k-scenario normal MC VaR within **3 standard
  errors** of the var-covar closed form on the same book (the SE itself is
  KDE-based and asserted positive).
- **Tail ordering at 99%**: with identical covariance and seed,
  Student-t(4) and jump-mixture VaR *and* ES strictly exceed normal MC —
  pure tail shape, no variance leakage (the t is covariance-matched by
  construction, verified on 200k draws).
- **Bitwise determinism**: identical seed ⇒ `EXPECT_EQ` (not `NEAR`) on
  VaR, ES and sampled P&L entries; different seed ⇒ different draw. Holds
  across platforms because all transforms (inverse-CDF normal,
  Marsaglia–Tsang gamma) are library-owned over the standard-specified
  `mt19937_64` stream.
- **FHS regime response**: on a history whose second half is 3× more
  volatile, filtered HS VaR > plain HS VaR (plain dilutes the current
  regime).
- **Reverse stress**: independent projected-gradient search in whitened
  coordinates (seeded random start, finite-difference gradients, never
  assumes the analytic optimum) matches the closed form to **1e-6** in
  loss and 1e-4 in the shock vector.

## 3. Cross-language golden constants (Python `fx_var`)

Three fully deterministic cases were run in the Python reference package
and their outputs embedded as constants in `tests/test_golden_python.cpp`
(provenance header in the file; repr() 17-digit precision; 2026-08-18):

- **Case A — book + historical VaR.** 3-position book (10m EURUSD spot,
  5m USDJPY ATM CIP forward, −3m EURJPY cross) on a 300-day sinusoidal
  history reproduced bit-for-bit in both languages. Checked: single
  scenario P&L (58 177.374 898…), plain HS VaR/ES at 99% and 97.5%, BRW
  age-weighted VaR/ES at λ=0.995 — agreement ≤ 1e-6 absolute on ~6e4
  magnitudes (≈ 1e-11 relative; residual is libm ulp noise).
- **Case B — parametric closed forms.** Fixed exposure vector and 3×3
  covariance through `var_covar`: normal and t(5) 99% VaR/ES, 10-day
  normal — 1e-8 relative (differences only in inverse-CDF
  implementations: Cephes vs Acklam+Halley).
- **Case C — backtest statistics.** `kupiec_pof(8, 250, 0.99)` LR and
  χ² p-value, Christoffersen LR/p on a fixed 250-day 9-exception pattern,
  Basel cumulative binomial probabilities at x = 4, 5, 10 — 1e-10 to
  1e-12.

Regeneration: run the snippet below from
`python/fx/03-var-es-engine` with `PYTHONPATH=src`; it is the exact
generator used (also documented in the test header):

```python
import numpy as np, pandas as pd
from fx_var.book import Book, Market, Spot, Forward
from fx_var.historical_var import historical_var
from fx_var.parametric_var import var_covar
from fx_var.backtesting import kupiec_pof, christoffersen_independence, basel_traffic_light

market = Market(spot_usd={"EUR": 1.10, "JPY": 0.0090, "GBP": 1.27},
                rates={"USD": 0.050, "EUR": 0.030, "JPY": 0.001})
book = Book([Spot("EURUSD", 10e6), Forward("USDJPY", 5e6, 0.5),
             Spot("EURJPY", -3e6)], base="USD")
factors = book.factors(market)          # FX:EUR FX:JPY IR:JPY IR:USD
scales = {"FX:EUR": .006, "FX:JPY": .007, "IR:JPY": .0004, "IR:USD": .0005}
t = np.arange(300.)
rets = pd.DataFrame({f: scales[f]*(np.sin(.1*t+j)+.5*np.cos(.05*t*(j+1.)))
                     for j, f in enumerate(factors)})
print(book.pnl(market, rets.iloc[17].to_dict()))
print(historical_var(book, market, rets, alpha=.99, warn_pegs=False))
w = pd.Series({"FX:EUR": 11e6, "FX:JPY": -4.5e6, "IR:USD": -2.4e6})
cov = pd.DataFrame([[3.6e-5,1.1e-5,-2e-6],[1.1e-5,4.9e-5,-1e-6],
                    [-2e-6,-1e-6,2.5e-7]], index=w.index, columns=w.index)
print(var_covar(w, cov, .99), var_covar(w, cov, .99, dist="t", df=5.),
      var_covar(w, cov, .99, horizon_days=10.))
print(kupiec_pof(8, 250, .99), basel_traffic_light(5, 250, .99))
```

## 4. Failure modes (mirrored from the Python reference)

- **F1 — Peg blindness (the headline).** A pegged currency contributes
  ~zero historical scenarios; HS and var-covar report ≈0 VaR while the
  true risk is a rare revaluation jump (CHF 2015: no daily USDCHF move
  over 1.9% in the prior 250 days, then +15–30% in hours). *Engine
  behaviour, tested end to end
  (`MonteCarlo.PegBreakJumpProducesLossHistoricalSimulationMisses`)*:
  the peg screen flags `FX:*` factors with daily vol < 5e-4 into
  `flagged_peg_factors`/`warnings`; the jump-mixture MC on the same book
  reports > 20× the HS VaR; `peg_break_scenario` supplies the stress
  add-on. The failure mode is *detected and priced, not fixed* — no
  historical method can be.
- **F2 — Singular covariances.** Pegged blocks make Σ rank-deficient;
  plain Cholesky aborts. Escalating jitter factorises and *reports*
  (`cholesky_warning`), tested with an exactly-rank-1 matrix.
- **F3 — Cornish–Fisher out of domain.** For |S|, K outside the Maillard
  monotonicity region the expansion is not a quantile function (99% "VaR"
  can undercut 95%). The engine checks the domain and throws; the escape
  hatch is explicit (`check_domain=false`). Tested at S=2.5. The check
  itself is exact, not grid-sampled: `dz_cf/dz` is a quadratic in `z`, so
  its minimum on `[-z_range, z_range]` is found in closed form rather than
  by scanning `z_cf` values on a fixed grid, which could miss a thin
  non-monotone dip between two grid nodes — confirmed with a constructed
  counterexample, `(S, K) = (0.122, -0.427)`, that the previous 801-point
  grid reported as monotone while the true minimum derivative on
  `[-4, 4]` is ≈ -9.15e-4 (`CornishFisher.DomainCheckIsExactNotGridResolutionDependent`).
- **F4 — √time scaling on carry books.** Negatively skewed, serially
  correlated unwinds violate i.i.d. aggregation; 10-day figures inherit
  the 1-day tail shape. Documented limitation (assumption A5); the
  scaling itself is tested so at least it is the *documented* error.
- **F5 — KDE-based VaR standard error** degrades in extremely discrete
  P&L distributions (near-degenerate books): the bandwidth floor guards
  the division but the SE is then conservative. MC convergence tests use
  well-spread books; samples below 10 scenarios are refused outright.
  Separately (see the Python `fx_var` reference, `docs/VALIDATION.md` F9,
  for the quantified benchmark this engine shares): the fixed-bandwidth
  (Silverman) KDE density estimate is tuned to the bulk of the P&L
  distribution, not the tail it is evaluated at, so at deep confidence
  levels (>= 99.5%) or with fewer than ~20,000 scenarios it *systematically
  underestimates* the true sampling SE by 10-20% — directionally
  overconfident, not just noisy. This engine now carries the same
  distribution-free cross-check the Python reference does:
  `var_standard_error_bootstrap` resamples the P&L vector with replacement
  and applies the identical order-statistic VaR rule to each resample
  (no bandwidth to choose, so it does not share the KDE's tail bias —
  unbiased to ~1-2% in the same benchmark, at the cost of higher
  trial-to-trial variance unless `n_boot` is generous). Prefer it to
  cross-check `var_standard_error` whenever `alpha >= 0.995` or scenario
  counts are modest, rather than trusting the KDE figure at face value.
- **F6 — Scale sensitivity of the numerical guards (fixed).** Two guards
  used *absolute* thresholds and misfired on legitimate large-notional
  books:
  * `portfolio_sigma` rejected a hedged multi-billion book against a
    rank-deficient (single-driver) covariance, because the quadratic form
    rounds to ~1e-16 of |w|² |Σ| — larger than the old −1e-12 threshold.
    The tolerance is now relative to |w|² max|Σ| and the rounding is
    clamped to sigma = 0 (regression test
    `PortfolioSigma.HedgedBillionDollarBookAgainstRankDeficientCovariance`).
  * `robust_cholesky` rejected covariances quoted in large units as
    "not symmetric" — they are symmetric only to ~1e-16 *relative*. The
    symmetry test is now scaled by max|entry|
    (`Cholesky.SymmetryToleranceIsRelativeToScale`).
  Genuinely indefinite matrices are still rejected at any scale
  (`Cholesky.IndefiniteCovarianceStillFails`).
- **F7 — Non-finite inputs are refused, never propagated.** NaN/Inf in a
  covariance previously produced either a NaN sigma or a misleading
  "not factorisable" `runtime_error` after the full jitter ladder; both
  paths now throw `std::invalid_argument` immediately, as does a
  non-finite exposure and a `simple_to_log` move of −100% or worse (which
  would inject −inf into every position of a stress report).
- **F8 — `kupiec_pof`'s chi2(1) reference is oversized exactly at the
  regulatory window** (see the Python `fx_var` reference,
  `docs/VALIDATION.md` F10, for the exact-probability derivation this
  engine's LR formula shares): at `n_obs=250, alpha=0.99` the expected
  exception count is only 2.5, and a nominally-5%-size chi2(1) test there
  has an *exact* rejection probability of ≈9.5% under a correctly
  calibrated model — falling to ≈5.5% by `n_obs=1000`. The LR formula
  itself is unchanged (it is what the golden tests pin); read a p-value
  near 0.05 at the 250-day window with that in mind.

## 4b. Edge cases (documented **and** tested)

| edge case | behaviour |
| --- | --- |
| empty book | `std::invalid_argument` — a VaR on nothing is a configuration error, not a zero |
| single-currency book in its own base ccy | exactly zero exposure and zero P&L under any shock; the same balance in a USD-base book is fully exposed (`Book.SingleCurrencyBookInNonUsdBaseHasNoFxRisk`) |
| identity triangulation | `EURJPY = EURUSD · USDJPY`, `pair · inverse = 1`, and forward triangulation consistent with CIP through the USD pivot, all to 1e-12 (`Market.TriangulationIdentitiesAreExact`) |
| 1- and 2-scenario samples | `empirical_var`/`empirical_es` collapse onto the observed loss for every alpha with no out-of-bounds read; `var_standard_error` refuses fewer than 10 scenarios |
| empty P&L sample, mismatched or all-zero weights, NaN P&L | throw |
| boundary alpha (1e-6, 1 − 1e-12) | accepted and usable; 0, 1, negative and NaN rejected; horizon ≤ 0 and NaN rejected |
| rank-deficient / pegged Σ | jitter path engages, records `jittered`/`warning`; hedged books report sigma = 0 rather than being rejected |
| indefinite Σ | `robust_cholesky` throws `std::runtime_error`; `portfolio_sigma` throws `std::invalid_argument` |
| NaN/Inf in Σ or exposures | `std::invalid_argument` at the boundary |
| pure base-ccy cash book | factorless: VaR = ES = 0 by construction |
| degenerate Cornish-Fisher grid (`n_grid < 3`) | throws |

## 5. Benchmarks

Single thread, g++ 13 `-O2`, Intel Xeon @ 2.80 GHz; 250-position /
50-factor deterministic book, 500-day history (`./build/fxvar_bench`,
numbers also quoted in README):

```
book: 250 positions, 50 factors
historical  VaR (500 scen):  2.0 ms
parametric  VaR (normal)  :  1.2 ms
monte carlo VaR 100k normal: 776 ms   (var 102.7k vs parametric 103.0k)
monte carlo VaR 100k t(5)  : 848 ms   (var 115.8k — fatter, as required)
```

The MC-vs-parametric agreement on the bench book (~0.3%) is itself a
daily cross-model consistency check: the two numbers share only the
covariance estimator.

## 6. Reproducing

```bash
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure
./build/fxvar_bench
```

No network, no data files: every test input is generated deterministically
in-source (sinusoidal histories, fixed seeds).
