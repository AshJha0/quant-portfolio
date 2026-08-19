# Validation — FX VaR & Expected Shortfall Engine

Contract items **3** (how it was validated) and **4** (where it fails), with
the numbers produced by the committed seeds. Reproduce everything with
`python -m pytest tests -q` (365 tests, ~7s, offline) and
`python examples/run_pipeline.py` (~4s).

---

## 1. Analytic and identity checks

| Check | Tolerance | Test |
|---|---|---|
| EURJPY P&L = EURUSD + USDJPY leg decomposition (triangulation), arbitrary joint shocks | 1e-6 USD abs (machine-level) | `test_book.py::test_triangulation_identity_eurjpy`, `::test_triangulation_cross_equals_two_usd_positions` |
| Base-ccy cash carries zero risk; base-change consistency `PnL_EUR·S¹_EUR = PnL_USD` | 1e-9 | `test_book.py` |
| Forward: engine value = hand-built deposit legs = discounted `(F−K)` under joint FX+IR shocks (CIP) | 1e-6 abs | `test_forwards.py::test_exact_revaluation_equals_deposit_legs` |
| ATM forward has zero initial value; DV01 = `N·K·T·e^{−rT}·1e-4` | 1e-6 / 1e-4 rel | `test_forwards.py` |
| GK put–call parity `c − p = S e^{−r_f T} − K e^{−r_d T}` | 1e-10 | `test_gk_options.py` |
| GK delta/gamma/vega vs central finite differences | 1e-7 / 1e-4 | `test_gk_options.py` |
| Normal ES identity `ES·(1−α) = σφ(z_α)` | **1e-10** | `test_expected_shortfall.py::test_normal_es_identity_1e10` |
| Normal & t ES vs numerical integration of the quantile function | 1e-8 | `test_expected_shortfall.py` |
| HS quantile = order statistic on known arrays (n=100/250/500, incl. fractional-tail Acerbi–Tasche ES) | exact | `test_expected_shortfall.py`, `test_historical_var.py` |
| Kupiec LR(n=250, x=5, p=1%) = 1.9568; Christoffersen LR hand sequence = 0.0900139; CC = UC + IND | 1e-12 vs hand formula | `test_backtesting.py` |
| Basel zones **exactly** green≤4 / yellow 5–9 / red≥10 with multipliers 3.00/3.40/3.50/3.65/3.75/3.85/4.00; binomial cdf(4)=0.8922, cdf(5)=0.9588 | exact / 1e-3 | `test_backtesting.py` |
| Reverse stress closed form: loss = `k√(w'Σw)`, shock on the ellipsoid boundary; SLSQP numerical optimum matches direction and loss | 1e-12 / 1e-4 | `test_stress_testing.py` |
| EWMA vol/cov recursions vs hand-computed values | 1e-12 | `test_historical_var.py`, `test_parametric_var.py` |

## 2. Convergence and statistical validation

* **MC → closed form**: 200k linear-P&L scenarios, 99% VaR within **3 standard
  errors** of the exact normal quantile (SE via `√(α(1−α)/n)/f̂(q)`, KDE
  density). Full-revaluation MC on a spot book matches the var-covar closed
  form within 3 SE + a 1% convexity allowance (`exp()` P&L is convex in the
  log shock — a real, measured 0.5–0.6% bias at 99%, not noise).
  Demo-book MC SE at 100k scenarios: **$3,045 = 0.48% of the $635k VaR**.
* **Variance matching**: the Student-t simulator reproduces the target
  covariance within 5% at 300k draws — so t-vs-normal VaR differences are
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
  *reverses* — exactly the t-tail signature. MC on the demo EM book at equal
  covariance: normal $1.212m, t(5) $1.356m (+12%), jump-mixture $2.907m
  (+140%) — **normal MC underestimates the EM 99% tail**.
* **Backtest engine on itself** (500-day rolling, 99%, GARCH+regime data,
  seed 29): parametric-normal 14 exceptions, Kupiec p=0.0009, independence
  p=0.0002, CC p<1e-4, **Basel red (4.00)**; plain HS 8 exceptions, yellow;
  **FHS 7 exceptions, all p>0.05, green (3.00)**. The Acerbi–Szekely ES
  backtest on the FHS run: Z=+0.42, p=0.17 (accepted).
* **ES coherence**: subadditivity counterexample from two independent 0.9%
  peg-jump assets — VaR₉₉(A)=VaR₉₉(B)=0 but VaR₉₉(A+B)=10 (non-subadditive);
  ES₉₉(A)=9.0 each and ES₉₉(A+B)=10.08 ≤ 18 (subadditive, exact hand
  values). ES ≥ VaR verified property-style across distributions and levels.

## 3. Failure modes (known, reproduced, documented)

### F1 — Peg blindness (the CHF 2015 case study)

On 14 Jan 2015 a 250-day HS window on USDCHF contained **no daily move larger
than ~1.9%** — the SNB floor had capped realised vol near 2.5% annualised for
three years. The next day CHF gapped ~+15% close-to-close (+30% intraday) vs
EUR: **the 99% HS VaR was off by a factor of >10, and no amount of window
tuning could have fixed it** — the information was not in the data.

Engine reproduction (seeded): a long-HKD book over 500 band-noise days gets
HS 99% VaR ≈ $17.6k per $50m notional; the −30% peg-break scenario produces a
**$15.0m loss — 850× the HS figure** (`test_stress_testing.py::
test_peg_break_supplies_the_loss_hs_missed`). Mitigation is structural, not
statistical: `PegBlindnessWarning` on any FX factor with daily σ < 0.05%
(tested), mandatory `peg_break_scenario` add-on, jump-mixture MC for a
quantile with the break priced in. On the demo book the peg-break loss is
21.7× the HS 99% VaR.

### F2 — Correlation breakdown in crises

Sample covariance from a calm window assumes the calm regime persists. In
stress, G10/EM correlations jump (0.55→0.75 in the simulator's calibration)
and JPY's correlation to carry currencies flips negative — a JPY-hedged AUD
book is hedged in the mean regime and doubly exposed in the crisis regime.
Reproduced: the regime-switching simulator is exactly what makes
parametric-normal go **Basel red** in §2 while FHS (which rescales but keeps
the empirical copula) stays green. Parametric users should re-run with
`default_correlation(regime="stress")` as a stressed-VaR overlay.

### F3 — Delta–vega mapping on short-gamma books

Measured: delta-only mapping error grows quadratically (ratio ≈ 4.0 when the
shock doubles, tested with 35% tolerance); for a long option the mapping
*overstates* losses (conservative), for a short option it *understates* them
(dangerous). Full revaluation is the default; the mapping exists for speed
and for exposure reporting.

### F4 — √time scaling with carry trades

`VaR(10d) = √10·VaR(1d)` is exact only for iid returns. Two documented
violations: (i) vol clustering — under GARCH, large days follow large days,
so √time *underestimates* multi-day risk in turbulent states (the same
mechanism behind F2's backtest failure); (ii) carry books are short a
negatively skewed risk premium — EM carry P&L "goes up by the stairs and
down by the elevator", so the 10-day loss tail is fatter than √10 of the
1-day tail. The engine reports √time numbers because desks and regulators
expect them, and flags them as scaled (the result object carries
`horizon_days`); overlapping-window or simulated multi-step MC is the
upgrade path.

### F5 — EM data quality

EM fixings embed stale prints, onshore/offshore splits (CNY/CNH, official vs
parallel TRY/ARS rates at times), holiday gaps that differ from G10, and
managed-float smoothing that understates true vol. Consequences: sample σ
too low, HS windows too quiet, correlations biased toward zero. Engine
policy: NaN inputs **raise** (never silently filled — tested), near-zero-vol
factors are flagged (the same machinery as pegs catches over-smoothed EM
series), and the EM crisis composite scenario + jump MC provide the fat-tail
floor that the polluted history cannot.

### F6 — Cornish–Fisher out of domain

For |skew| beyond the monotonicity domain (e.g. S=−3), the CF expansion's
"quantile" is non-monotone in α. The engine checks monotonicity numerically
on [−4,4] and **raises** instead of returning a number
(`test_parametric_var.py::test_cf_domain_check_rejects_extreme_moments`);
`check_domain=False` is an explicit, documented override.

### F7 — Non-finite market data was absorbed, not refused (fixed)

The engine's stated NaN policy is *refuse, never impute*, but it was only
enforced on the factor-**return** history, not on the `Market` snapshot
itself. Two silent paths existed and are now closed:

* A **NaN interest rate** propagated straight through to a NaN P&L and a
  NaN VaR — visibly wrong, but only if someone looked.
* A **NaN implied vol** was far worse: `gk_d1_d2` decides the degenerate
  branch with `sig_sqrt_t > 1e-12`, and `NaN > 1e-12` is `False`, so a NaN
  vol was silently treated as the **zero-vol** case. The option priced at
  forward intrinsic and the book reported a P&L of *exactly 0.0* — a
  plausible-looking number with no warning attached.

`Market.__post_init__` now rejects non-finite (and negative) rates and vols
at construction, and `fx_var.gk._validate` rejects non-finite `strike`,
`expiry` and `vol` directly. Neither change alters any value for finite
inputs — it converts two silent-wrong-answer paths into `ValueError`s
(`test_edge_cases_extra.py::TestNonFiniteMarketData`).

A third, smaller gap: `option_method` was only validated inside the
per-position branch, so a typo (`"delta"` for `"delta_vega"`) was accepted
silently on any book without options. It is now validated up front in
`Book.value_usd`.

### F8 — Singular covariance

Two currencies pegged to the same anchor produce a singular covariance;
plain Cholesky fails. `robust_cholesky` escalates diagonal jitter from
1e-12·mean(diag) and warns (`NumericalWarning`); the factorisation error and
the simulated lockstep correlation (>0.999) are tested.

## 4. Edge cases (documented **and** unit-tested)

Empty book (VaR=ES=0); single-currency book; base-currency cash (zero risk);
alpha ∈ {0, 1, <0, >1} raise; insufficient history raises; NaN inputs raise
everywhere; missing factor columns raise with the factor names; T=0 forwards
degenerate to spot; T=0 / σ=0 options price intrinsic; negative shocked vol
raises; df≤2 Student-t raises; jump prob outside [0,1] raises; zero-risk book
reverse stress raises; Basel inputs x>n raise; Kupiec x=0 handled via the
0·log0 convention; Christoffersen with no exceptions returns LR=0; the
degenerate symmetric exception sequence (π₀₁=π₁₁) returns LR=0 exactly.

Extended coverage in `tests/test_edge_cases_extra.py`:

* **Non-finite market data** — NaN/Inf rates and vols, negative vols and
  non-positive spots all raise at `Market` construction; `gk_price` rejects
  non-finite strike/expiry/vol directly (see F7).
* **Base-currency duality** — base-ccy cash is riskless in USD, EUR *and*
  JPY reporting; foreign cash P&L matches `N·S·(e^Δ−1)` to 1e-12; and
  translating a book's P&L between reporting currencies uses the
  **post-shock** rate (`PnL_EUR = PnL_USD / (S_EUR·e^{Δ_EUR})`), which is
  the classic off-by-one-rate error in multi-currency P&L.
* **Foreign–domestic inversion** — long 1m EUR vs USD and short 1.08m USD
  vs EUR give identical P&L across a ±3% shock grid.
* **Cross triangulation** — EURJPY carries no factor of its own, so an
  equal log move in both USD legs is exactly flat.
* **CIP consistency** — ATM forwards value to zero at 0.5y/1y/2y across
  EURUSD, USDJPY and the EURJPY cross; the FX delta of a 1y forward equals
  `N·e^{−r_f T}·S` to 1e-5; the two IR legs have opposite signs; book-level
  put–call parity holds at the ATM-forward strike; T=0 options price
  intrinsic.
* **Pegged pairs end to end** — `PegBlindnessWarning` fires on band noise,
  HS 99% VaR comes in under 0.05% of the USD-equivalent leg, the −30%
  peg-break scenario exceeds it by more than 1000×, an *upward* break flips
  the sign (the CHF-2015 direction), and `warn_pegs=False` suppresses the
  warning cleanly.
* **Tiny samples / degenerate α** — empty P&L samples raise; a single
  scenario returns itself; α ∈ {0, 1, −0.1, 1.5, 2.0} raise; horizons ≤ 0
  raise; α = 0.999 on 5 scenarios collapses the tail onto the single worst
  observation; α = 0.001 averages 99.9% of the mass and correctly reports a
  *negative* VaR (the boundary atom is the best outcome).
* **ES coherence, property-style** — ES ≥ VaR over 50 random t(4) samples ×
  3 confidence levels; the two-peg-jump subadditivity counterexample shows
  VaR violating subadditivity while ES does not; constant weights reproduce
  the uniform-weight ES exactly.
* **Scaling laws** — VaR is monotone in α, scales exactly as √h in the
  horizon and exactly linearly in notional; t-VaR exceeds normal VaR at 99%
  at equal σ but agrees within 10% at 90%; full-revaluation MC on a linear
  spot book matches the closed-form variance-covariance VaR within 4 SE.

### 4.1 Non-finite scalars: the guards that silently accepted NaN
(`tests/test_nan_guards.py`, 119 tests)

The NaN policy was enforced on *data* (market snapshots, factor-return
frames) but not on the long tail of bare scalar arguments. Those were
guarded only by inequalities — `if horizon_days <= 0`, `if sigma < 0`,
`if df <= 2`, `if strike <= 0`, `if k <= 0` — and **every comparison
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
| `es_backtest_acerbi_szekely` with non-finite P&L/VaR/ES | NaN test statistic | raises |

Companion positive tests confirm the guards did not over-reject: √h scaling,
the closed-form normal VaR constant, jump-overlay variance inflation, shock
broadcasting over a scenario `DataFrame`, and an end-to-end four-position
book (cash + spot + forward + option) returning finite `ES ≥ VaR > 0` from
both historical and parametric methods.

### 4.2 Cross-language golden vectors, now locked from Python
(`tests/test_golden_vectors.py`)

`cpp/fx-var-engine` and `rust/fx-var-engine` assert hard-coded constants
generated from this package, but nothing on the Python side pinned them —
a refactor here would have moved the reference silently, surfacing only as
a failure in a different language's test suite. The three golden cases are
now reproduced and asserted in Python at the same tolerances:

* **Case A** — factor enumeration order `[FX:EUR, FX:JPY, IR:JPY, IR:USD]`
  (the C++/Rust engines index the returns matrix positionally, so the order
  is part of the contract); single-scenario book P&L 58 177.374 898 100 74;
  plain HS 99% VaR/ES 61 919.808 905 876 24 / 62 006.120 062 248 47; 97.5%
  61 237.426 008 895 97 / 61 777.936 082 718 57; BRW age-weighted
  (λ = 0.995) 61 874.262 685 311 49 / 61 977.524 965 941 09.
* **Case B** — `var_covar` on fixed exposures/covariance: normal 99% 1-day
  153 339.504 419 629 17 / 175 675.629 720 028 5; t(5) 99%
  171 803.123 890 914 05 / 227 327.531 497 414 4; normal 99% 10-day
  484 902.089 247 483 8 / 555 535.119 299 658 3 (and exactly √10 × the
  1-day figures).
* **Case C** — Kupiec LR(8/250, 99%) = 7.733 550 724 494 520 5
  (p = 0.005 420 405 194 127 799 4); Christoffersen LR on the fixed
  9-exception pattern = 1.006 361 033 931 412 4 (p = 0.315 776 203 762 249 9);
  Basel cumulative probabilities 0.892 187 626 903 624 9 (4, green),
  0.958 816 815 930 151 4 (5, yellow, ×3.40) and 0.999 946 101 370 953
  (10, red, ×4.00).

## 5. What is deliberately out of scope

Smile/skew vol risk (one ATM vol per pair — A5), tenor-bucketed curves (A3),
cross-currency basis (A4), intraday VaR, and American/exotic options. Each
is an extension point, not an accident: the assumptions register in
METHODOLOGY.md states what breaks and how to add it.
