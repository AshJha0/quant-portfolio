# Validation — FX VaR & Expected Shortfall Engine

Contract items **3** (how it was validated) and **4** (where it fails), with
the numbers produced by the committed seeds. Reproduce everything with
`python -m pytest tests -q` (182 tests, ~7s, offline) and
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

### F7 — Singular covariance

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

## 5. What is deliberately out of scope

Smile/skew vol risk (one ATM vol per pair — A5), tenor-bucketed curves (A3),
cross-currency basis (A4), intraday VaR, and American/exotic options. Each
is an extension point, not an accident: the assumptions register in
METHODOLOGY.md states what breaks and how to add it.
