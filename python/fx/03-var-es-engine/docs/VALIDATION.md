# Validation: FX VaR and Expected Shortfall Engine

Contract items **3** (how it was validated) and **4** (where it fails), with
the numbers produced by the committed seeds on synthetic data. Reproduce
everything with `python -m pytest -q` (867 tests, about 90 s, offline) and
`python examples/run_pipeline.py` (about 30 s). Timings: Windows 11,
Python 3.12.10, numpy 2.5, scipy 1.18. All `alpha` values below are tail
probabilities (0.01 = 99 %).

---

## 1. Analytic and identity checks

| Check | Tolerance | Test |
|---|---|---|
| EURJPY P&L = EURUSD + USDJPY leg decomposition (triangulation), arbitrary joint shocks | 1e-6 USD abs (machine-level) | `test_book.py::test_triangulation_identity_eurjpy`, `::test_triangulation_cross_equals_two_usd_positions` |
| Base-ccy cash carries zero risk; base-change consistency `PnL_EUR·S¹_EUR = PnL_USD` | 1e-9 | `test_book.py` |
| Forward: engine value = hand-built deposit legs = discounted `(F−K)` under joint FX+IR shocks (CIP) | 1e-6 abs | `test_forwards.py::test_exact_revaluation_equals_deposit_legs` |
| ATM forward has zero initial value; DV01 = `N·K·T·e^{−rT}·1e-4` | 1e-6 / 1e-4 rel | `test_forwards.py` |
| GK put-call parity `c − p = S e^{−r_f T} − K e^{−r_d T}` | 1e-10 | `test_gk_options.py` |
| GK delta/gamma/vega vs central finite differences | 1e-7 / 1e-4 | `test_gk_options.py` |
| Normal ES identity `ES alpha = sigma phi(z)`, `z = Phi^{-1}(1 - alpha)` | **1e-10** | `test_expected_shortfall.py::test_normal_es_identity_1e10` |
| Normal & t ES vs numerical integration of the quantile function | 1e-8 | `test_expected_shortfall.py` |
| HS quantile = order statistic on known arrays (n=100/250/500, incl. fractional-tail Acerbi-Tasche ES) | exact | `test_expected_shortfall.py`, `test_historical_var.py` |
| Kupiec LR(n=250, x=5, p=1%) = 1.9568; Christoffersen LR hand sequence = 0.0900139; CC = UC + IND | 1e-12 vs hand formula | `test_backtesting.py` |
| Basel zones at 250 / 1 %: green 0-4, yellow 5-9, red 10+, multipliers 3.00/3.40/3.50/3.65/3.75/3.85/4.00. Off-window cases (500, 1000 days; alpha 0.025) against the 250-day equivalent count and scipy's binomial CDF | exact / 1e-12 | `test_backtesting.py`, `test_conventions.py` |
| Shared convention vectors `conv_*` (VaR/ES at k = 2.5, 6.25, 12.5; Basel; Kupiec; Z2; CF flags; Higham and clip repair; EWMA) against independent hand computations | exact to 1e-12 | `test_golden_vectors.py::TestConventionVectors` |
| Reverse stress closed form: loss = `k sqrt(w' Sigma w)`, shock on the ellipsoid boundary. SLSQP in whitened coordinates (success flag and constraint residual checked) matches direction and loss | 1e-12 / 1e-4 | `test_stress_testing.py` |
| EWMA vol/cov recursions vs hand-computed values | 1e-12 | `test_historical_var.py`, `test_parametric_var.py` |

## 2. Convergence and statistical validation

* **MC to closed form**: 200k linear-P&L scenarios, 99 % VaR within **3
  standard errors** of the exact normal quantile. Full-revaluation MC on a spot book matches the var-covar closed
  form within 3 SE + a 1% convexity allowance (`exp()` P&L is convex in the
  log shock: a real, measured 0.5-0.6% bias at 99%, not noise).
  Synthetic demo-book bootstrap SE at 100k scenarios: **3,261 USD, 0.51 % of
  the 635k USD VaR**.
* **Variance matching**: the Student-t simulator reproduces the target
  covariance within 5% at 300k draws: so t-vs-normal VaR differences are
  pure tail shape.
* **Calibration replication** (loose tolerance, seeded): simulator sample
  vols within 10% of `ANNUAL_VOLS`; G10 block correlation 0.55 ± 0.10, EM
  0.45 ± 0.10; EWMA σ̂ correlates > 0.7 with the true GARCH conditional σ;
  GARCH squared-return ACF(1) > 0.10 vs |ACF(1)| < 0.05 for the iid sim.
* **Cross-model consistency**: on Gaussian constant-vol G10 data, HS,
  parametric and MC 99% VaR agree within 15% (same quantile, three
  estimators). Pipeline numbers (99%/1d, demo book): HS $691k, age $683k,
  FHS $629k, param-N $652k, param-t5 $730k, MC-N $635k.
* **Method disagreement where it should disagree** (EM fat tails): with a
  common t(3) mixing variable at matched vols, HS 99% VaR exceeds
  parametric-normal by >10% (seeded ratio 1.245) while the 95% ordering
  *reverses*: exactly the t-tail signature. MC on the demo EM book at equal
  covariance (synthetic): normal 1.212m USD, t(5) 1.356m USD (+12 %),
  compound-Poisson jump mixture 2.888m USD (+138 %). **Normal MC
  underestimates the EM 99 % tail.**
* **Backtest engine on itself** (500-day rolling, 99%, GARCH+regime data,
  seed 29): parametric-normal 14 exceptions, Kupiec p=0.0009, independence
  p=0.0002, CC p<1e-4, **Basel red (4.00)**; plain HS 8 exceptions, yellow;
  **FHS 7 exceptions, all p>0.05, green (3.00)**. Acerbi-Szekely on the
  synthetic FHS run: Z2 = -0.400, p = 0.185 (not rejected).
* **ES coherence**: subadditivity counterexample from two independent 0.9%
  peg-jump assets: VaR₉₉(A)=VaR₉₉(B)=0 but VaR₉₉(A+B)=10 (non-subadditive);
  ES₉₉(A)=9.0 each and ES₉₉(A+B)=10.08 ≤ 18 (subadditive, exact hand
  values). ES ≥ VaR verified property-style across distributions and levels.

## 3. Failure modes (known, reproduced, documented)

### F1: Peg blindness (the CHF 2015 case study)

On 14 Jan 2015 a 250-day HS window on USDCHF contained **no daily move larger
than ~1.9%**: the SNB floor had capped realised vol near 2.5% annualised for
three years. The next day CHF gapped ~+15% close-to-close (+30% intraday) vs
EUR: **the 99% HS VaR was off by a factor of >10, and no amount of window
tuning could have fixed it**: the information was not in the data.

Engine reproduction (seeded): a long-HKD book over 500 band-noise days gets
HS 99% VaR ≈ $17.6k per $50m notional; the −30% peg-break scenario produces a
**$15.0m loss: 850× the HS figure** (`test_stress_testing.py::
test_peg_break_supplies_the_loss_hs_missed`). Mitigation is structural, not
statistical: `PegBlindnessWarning` on any FX factor with daily σ < 0.05%
(tested), mandatory `peg_break_scenario` add-on, jump-mixture MC for a
quantile with the break priced in. On the demo book the peg-break loss is
21.7× the HS 99% VaR.

### F2: Correlation breakdown in crises

Sample covariance from a calm window assumes the calm regime persists. In
stress, G10/EM correlations jump (0.55→0.75 in the simulator's calibration)
and JPY's correlation to carry currencies flips negative: a JPY-hedged AUD
book is hedged in the mean regime and doubly exposed in the crisis regime.
Reproduced: the regime-switching simulator is exactly what makes
parametric-normal go **Basel red** in §2 while FHS (which rescales but keeps
the empirical copula) stays green. Parametric users should re-run with
`default_correlation(regime="stress")` as a stressed-VaR overlay.

### F3: Delta-vega mapping on short-gamma books

Measured: delta-only mapping error grows quadratically (ratio ≈ 4.0 when the
shock doubles, tested with 35% tolerance); for a long option the mapping
*overstates* losses (conservative), for a short option it *understates* them
(dangerous). Full revaluation is the default; the mapping exists for speed
and for exposure reporting.

### F4: √time scaling with carry trades

`VaR(10d) = √10·VaR(1d)` is exact only for iid returns. Two documented
violations: (i) vol clustering: under GARCH, large days follow large days,
so √time *underestimates* multi-day risk in turbulent states (the same
mechanism behind F2's backtest failure); (ii) carry books are short a
negatively skewed risk premium: EM carry P&L "goes up by the stairs and
down by the elevator", so the 10-day loss tail is fatter than √10 of the
1-day tail. The engine reports √time numbers because desks and regulators
expect them, and flags them as scaled (the result object carries
`horizon_days`); overlapping-window or simulated multi-step MC is the
upgrade path.

### F5: EM data quality

EM fixings embed stale prints, onshore/offshore splits (CNY/CNH, official vs
parallel TRY/ARS rates at times), holiday gaps that differ from G10, and
managed-float smoothing that understates true vol. Consequences: sample σ
too low, HS windows too quiet, correlations biased toward zero. Engine
policy: NaN **and ±inf** inputs **raise** (never silently filled: tested;
the P&L gates in `expected_shortfall._tail` and
`backtesting.evaluate_var_backtest` are full `np.isfinite` checks, not
`isnan`-only, so an `inf` loss can no longer yield an `inf` ES or a `+inf`
P&L day be silently scored as "no exception": `TestInfinityIsRefusedLikeNaN`),
near-zero-vol
factors are flagged (the same machinery as pegs catches over-smoothed EM
series), and the EM crisis composite scenario + jump MC provide the fat-tail
floor that the polluted history cannot.

### F6: Cornish-Fisher out of domain

The CF map is a quantile function only where it is increasing. The engine
checks this on `[-z_a, z_a]`, `z_a = Phi^{-1}(1 - alpha)`, the interval the
requested VaR actually uses (`cornish_fisher_valid`). The derivative is a
quadratic in `z`, so the minimum is at an endpoint or the vertex and the
check is exact. Outside the domain `cornish_fisher_var` raises.
`check_domain=False` is an explicit override.

| (S, K) | alpha | Valid | Where the derivative is smallest |
|---|---|---|---|
| (0.122, -0.427) | 0.01 | yes | left endpoint z = -2.326 |
| (0.122, -0.427) | 0.001 | yes | left endpoint z = -3.090 |
| (0.122, -0.427) | fixed \|z\| <= 4 (deprecated predicate) | no | z = -4, derivative -9.15e-4 |
| (1, 1) | 0.01 | yes | |
| (1, 1) | 0.001 | no | left endpoint |
| (-3, 0) | 0.01 | no | |

The (0.122, -0.427) case is concave, so its minimum is at an endpoint, not
inside the interval. Tests: `test_parametric_var.py::
test_cf_counterexample_minimum_is_at_the_endpoint_and_alpha_aware` and
`::test_cf_valid_tightens_with_smaller_alpha`.

### F7: Non-finite market data was absorbed, not refused (fixed)

The engine's stated NaN policy is *refuse, never impute*, but it was only
enforced on the factor-**return** history, not on the `Market` snapshot
itself. Two silent paths existed and are now closed:

* A **NaN interest rate** propagated straight through to a NaN P&L and a
  NaN VaR: visibly wrong, but only if someone looked.
* A **NaN implied vol** was far worse: `gk_d1_d2` decides the degenerate
  branch with `sig_sqrt_t > 1e-12`, and `NaN > 1e-12` is `False`, so a NaN
  vol was silently treated as the **zero-vol** case. The option priced at
  forward intrinsic and the book reported a P&L of *exactly 0.0*: a
  plausible-looking number with no warning attached.

`Market.__post_init__` now rejects non-finite (and negative) rates and vols
at construction, and `fx_var.gk._validate` rejects non-finite `strike`,
`expiry` and `vol` directly. Neither change alters any value for finite
inputs: it converts two silent-wrong-answer paths into `ValueError`s
(`test_edge_cases_extra.py::TestNonFiniteMarketData`).

A third, smaller gap: `option_method` was only validated inside the
per-position branch, so a typo (`"delta"` for `"delta_vega"`) was accepted
silently on any book without options. It is now validated up front in
`Book.value_usd`.

### F8: Singular and indefinite covariance

Two currencies pegged to the same anchor give a singular covariance.
`robust_cholesky` adds diagonal jitter from 1e-12 to 1e-6 times the mean
variance and warns (`NumericalWarning`). Past the cap it raises and names the
most negative eigenvalue. `repair="clip"` (eigenvalue clip, variances kept)
and `repair="higham"` (nearest correlation matrix, Higham 2002) repair an
indefinite matrix explicitly and warn. Tests: `test_conventions.py`.

### F9: The KDE standard error is biased at deep tails and small samples

`var_standard_error` uses `SE = sqrt(alpha (1-alpha) / n) / f(q)` with a
Silverman-bandwidth Gaussian KDE at the reported quantile. A fixed bandwidth
smooths the tail. `monte_carlo_var` therefore reports the bootstrap SE
(`var_standard_error_bootstrap`) and the KDE figure is a diagnostic.

Ratio of the mean estimated SE to the true sampling SE of the VaR (standard
normal P&L, 20 trials, true SE from 2,000 independent samples, synthetic):

| alpha | Scenarios | KDE / true | Bootstrap / true |
|---|---|---|---|
| 0.01 | 2,000 | 0.89 | 1.00 |
| 0.01 | 50,000 | 0.96 | 0.96 |
| 0.001 | 50,000 | 1.04 | 1.11 |

The bootstrap SE of an order statistic is discrete and noisy at small `n`.
It is not a confidence interval. The C++ and Rust ports have both a KDE and
a bootstrap estimator. Test: `test_monte_carlo.py::TestVarStandardErrorAccuracy`.

### F10: Kupiec's chi2(1) reference is oversized exactly at the
regulatory window

`kupiec_pof`'s LR statistic is asymptotically chi2(1), but at `n_obs=250,
alpha=0.99` (the Basel backtesting window: tail probability 1%, expected
exception count 2.5) that asymptotic is a poor approximation for a rare
binomial: the *exact* rejection probability of a nominally-5%-size test,
computed by summing the exact `Binomial(250, 0.01)` mass over `{x : LR(x)
> chi2.ppf(0.95, 1)}`, is **≈9.5%**: a correctly calibrated model is
flagged "reject at 5%" roughly twice as often as the p-value implies. The
effect is driven by the expected exception count, not `n_obs` alone: it
falls to ≈5.5% at `n_obs=1000` (expected count 10), close to nominal. This
is not a bug in the LR formula (it is the textbook asymptotic reference,
unchanged, and is what the C++/Rust golden tests pin), but it is an
unstated caveat that changes how a p-value near 0.05 should be read at the
regulatory window: documented and made exact (no simulation, no RNG) in
`test_backtesting.py::
test_kupiec_asymptotic_chi2_reference_is_oversized_at_regulatory_window`,
mirroring the equivalent finding and test in the `eq_var` equity engine
(`docs/VALIDATION.md` item 9 there). Christoffersen's independence and CC
tests share the same asymptotic chi2 machinery and are subject to the
same class of caveat at low exception counts.

## 4. Edge cases (documented **and** unit-tested)

Empty book raises; a book with only base-currency cash raises (no risk
factors); single-currency book; alpha outside (0, 0.5) raises, including an
old confidence level such as 0.99; insufficient history raises; NaN inputs raise
everywhere; missing factor columns raise with the factor names; T=0 forwards
degenerate to spot; T=0 / sigma=0 options price intrinsic; a shocked vol
below zero is floored at zero with a `NumericalWarning`, so one Monte Carlo
path cannot abort a run; df <= 2 Student-t raises; jump prob outside [0, 1)
raises; zero-risk book
reverse stress raises; Basel inputs x>n raise; Kupiec x=0 handled via the
0·log0 convention; Christoffersen with no exceptions returns LR=0; the
degenerate symmetric exception sequence (π₀₁=π₁₁) returns LR=0 exactly.

Extended coverage in `tests/test_edge_cases_extra.py`:

* **Non-finite market data**: NaN/Inf rates and vols, negative vols and
  non-positive spots all raise at `Market` construction; `gk_price` rejects
  non-finite strike/expiry/vol directly (see F7).
* **Base-currency duality**: base-ccy cash is riskless in USD, EUR *and*
  JPY reporting; foreign cash P&L matches `N·S·(e^Δ−1)` to 1e-12; and
  translating a book's P&L between reporting currencies uses the
  **post-shock** rate (`PnL_EUR = PnL_USD / (S_EUR·e^{Δ_EUR})`), which is
  the classic off-by-one-rate error in multi-currency P&L.
* **Foreign-domestic inversion**: long 1m EUR vs USD and short 1.08m USD
  vs EUR give identical P&L across a ±3% shock grid.
* **Cross triangulation**: EURJPY carries no factor of its own, so an
  equal log move in both USD legs is exactly flat.
* **CIP consistency**: ATM forwards value to zero at 0.5y/1y/2y across
  EURUSD, USDJPY and the EURJPY cross; the FX delta of a 1y forward equals
  `N·e^{−r_f T}·S` to 1e-5; the two IR legs have opposite signs; book-level
  put-call parity holds at the ATM-forward strike; T=0 options price
  intrinsic.
* **Pegged pairs end to end**: `PegBlindnessWarning` fires on band noise,
  HS 99% VaR comes in under 0.05% of the USD-equivalent leg, the −30%
  peg-break scenario exceeds it by more than 1000×, an *upward* break flips
  the sign (the CHF-2015 direction), and `warn_pegs=False` suppresses the
  warning cleanly.
* **Tiny samples and extreme alpha**: empty P&L samples raise; a single
  scenario returns itself; alpha in {0, 0.5, 0.99, 1, -0.1, 1.5, 2, NaN}
  raises; horizons <= 0 raise; alpha = 0.001 on 5 scenarios collapses the
  tail onto the single worst observation.
* **ES coherence, property-style**: ES ≥ VaR over 50 random t(4) samples ×
  3 confidence levels; the two-peg-jump subadditivity counterexample shows
  VaR violating subadditivity while ES does not; constant weights reproduce
  the uniform-weight ES exactly.
* **Scaling laws**: VaR is monotone in α, scales exactly as √h in the
  horizon and exactly linearly in notional; t-VaR exceeds normal VaR at 99%
  at equal σ but agrees within 10% at 90%; full-revaluation MC on a linear
  spot book matches the closed-form variance-covariance VaR within 4 SE.

### 4.1 Non-finite scalars: the guards that silently accepted NaN
(`tests/test_nan_guards.py`, 36 test functions, 125 collected cases)

The NaN policy was enforced on *data* (market snapshots, factor-return
frames) but not on the long tail of bare scalar arguments. Those were
guarded only by inequalities: `if horizon_days <= 0`, `if sigma < 0`,
`if df <= 2`, `if strike <= 0`, `if k <= 0`: and **every comparison
against NaN is False**, so each guard passed NaN straight through and the
engine returned `nan` as the VaR. That is the worst possible failure mode
for this number: a NaN does not trip a limit, does not colour a Basel
traffic light and does not look obviously wrong on a report.

Closed, with a test per path:

| Path | Was | Now |
|---|---|---|
| `validate_horizon(nan)` | `sqrt(nan)` scaling → NaN VaR/ES | raises |
| `normal_var/normal_es/t_var/t_es(sigma=nan)` or `mean=nan` | NaN loss | raises |
| `t_var/t_es(df=nan)` | `sqrt((df−2)/df)` → NaN | raises (df validated before any t special function) |
| `portfolio_sigma` / `var_covar` with a NaN covariance entry or exposure | NaN σ → NaN VaR | raises |
| `cornish_fisher_var(sigma/skew/kurtosis/mean = nan)` | misleading "non-monotone domain" error | raises naming the argument |
| `robust_cholesky` on a non-finite covariance | NaN Cholesky factor → NaN scenarios | raises |
| `simulate_factor_returns(dist="t", df=nan)` | NaN scenario matrix | raises |
| `JumpSpec(mean=…nan, std=…nan)` | NaN jump sizes → NaN VaR | raises |
| `Cash/Spot/Forward/Option` with a NaN notional, strike, expiry or entry rate | NaN book value | raises at construction |
| A single NaN cell in a shock mapping or scenario `DataFrame` | NaN P&L vector → NaN quantile | raises naming the factor |
| `usd_broad_move(pct=nan)`, `peg_break_scenario(jump/vol_spike/contagion = nan)` | NaN scenario shocks | raises |
| `reverse_stress_linear(radius=nan / loss_target=nan)` | NaN worst-case shock | raises |
| `acerbi_szekely_z2` with non-finite P&L/VaR/ES | NaN test statistic | raises |

`tests/test_fuzz_public_api.py` extends this to every name in
`fx_var.__all__`: a registry gives each public callable one or more
injection sites, the valid call must succeed, and NaN, +inf and -inf must
raise. A new public function without a registry entry fails the test.

Companion positive tests confirm the guards did not over-reject: √h scaling,
the closed-form normal VaR constant, jump-overlay variance inflation, shock
broadcasting over a scenario `DataFrame`, and an end-to-end four-position
book (cash + spot + forward + option) returning finite `ES ≥ VaR > 0` from
both historical and parametric methods.

### 4.2 Cross-language golden vectors (`tests/test_golden_vectors.py`)

`tests/golden/generate_golden.py` writes `golden_vectors.json`, which the C++
and Rust ports assert. The JSON records the Python, numpy and
scipy versions. The Python test rebuilds the JSON and compares every value
to 16 ulps of `max(|value|, scale)`. The scale is 2e7 for case A, because
those values are P&L differences of book values near 1e7 USD and their
rounding error follows the gross value. Higham repair values are compared
at 1e-8 absolute (iterative solver, stopped at 1e-12 relative change). The
`conv_*` keys are shared with the equity engine and are checked against
hand computations (section 1).

| Case | Content |
|---|---|
| A | factor order `[FX:EUR, FX:JPY, IR:JPY, IR:USD]`, one-scenario book P&L, plain HS 99 % and 97.5 % VaR/ES, BRW (decay 0.995) VaR/ES |
| B | `var_covar` normal and t(5) at alpha 0.01, 1 and 10 days |
| C | Kupiec LR (8 of 250), Christoffersen LR, Basel CDF and zone at 4, 5, 10 exceptions |
| conv | VaR/ES at alpha 0.010/0.025/0.050, nine Basel cases, three Kupiec LRs, Z2, five CF flags, Higham and clip entries, EWMA vol |

## 5. Out of scope

Smile/skew vol risk (one ATM vol per pair: A5), tenor-bucketed curves (A3),
cross-currency basis (A4), intraday VaR, and American/exotic options. Each
is an extension point, not an accident: the assumptions register in
METHODOLOGY.md states what breaks and how to add it.

## Round 3 validation

Every new model has at least one oracle test. Files under `tests/`:

| Test file | Oracle | Evidence class |
|---|---|---|
| `test_risk_attribution.py` | hand two-asset example; Euler sum to 1e-13; marginal vs central finite difference; incremental vs marginal for a small trade; scenario ES and VaR contributions converge to the closed form on 400,000 draws | closed form, convergence |
| `test_frtb_ima.py` | hand-computed cascade; sqrt(12) telescoping; plus-factor table; capital by hand; desk thresholds at 12/13 and 30/31; Spearman and KS equal scipy to 1e-12; PLA zones | closed form, published table, library cross-check |
| `test_stressed_var.py` | planted 250-day crisis found; window VaR equals the 3rd-worst loss at alpha n = 3; capital by hand | closed form, planted signal |
| `test_evt.py` | POT VaR equals the exact GPD quantile and POT ES the integrated quantile, for xi in (-0.2, 0, 0.25, 0.6); MLE recovers xi and beta on 200,000 draws; t_4 tail gives xi near 1/4; flat mean excess for the exponential | closed form, convergence |
| `test_copula_mc.py` | Kendall tau = (2/pi) arcsin(rho); empirical tail dependence vs the closed form; Gaussian limit VaR vs closed form | closed form, convergence |
| `test_revaluation.py` | Greeks vs finite differences; put-call parity; cross-check against the package's own Garman-Kohlhagen `gk_price`/`gk_*`; linear book exact; explain residual scales by about 8 when moves halve | closed form, library cross-check, convergence |
| `test_limits.py` | statuses, ordering and aggregation by hand | exact |

Compiled `option_reval` (C++ and Rust, both VaR engines): golden values from
`fx_var.revaluation` to 1e-11, put-call parity, input rejection. Compiled
option VaR (`test_option_var` in all four engines): historical VaR, ES and
scenario P&L against `fx_var.revaluation.full_revaluation_pnl` plus the S2
order statistic at 1e-11 relative (goldens from
`cpp/equity-var-engine/tools/gen_option_var_golden.py`); Monte Carlo VaR
within 4 bootstrap SEs of the closed-form quantile of a monotone one-factor
book; a book with no options gives bit-identical results to the linear
path.
