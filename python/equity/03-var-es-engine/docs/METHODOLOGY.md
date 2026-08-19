# Methodology — Equity VaR & Expected Shortfall Engine

This document answers contract items 1 and 2: **why these models** (against
alternatives, with trade-offs) and **what assumptions they rest on** (with
what breaks when each is violated).

## 1. The problem

Estimate, each day, how much a portfolio of equities, index futures and
options can lose over a 1-day (and 10-day) horizon at 95 % / 99 % confidence
— and quantify the *average* loss beyond that point (Expected Shortfall).
Conventions used throughout:

- `alpha` is the **tail probability** (`alpha = 0.01` → 99 % VaR).
- VaR and ES are reported as **positive numbers for losses**:
  `VaR_a = -Q_a(P&L)`, `ES_a = -E[P&L | P&L ≤ Q_a]`.
- Price-factor scenarios are simple returns; implied-vol scenarios are
  absolute changes in decimal vol. P&L is in currency units.
- Day count ACT/365F; option rates/dividends continuously compounded; vol
  quoted on log-returns, annualised (per CONVENTIONS.md).

## 2. Why three VaR families coexist

No single VaR estimator dominates; every real risk department runs at least
two and reconciles them. The engine implements all three so the
disagreement itself becomes a diagnostic (see the method-disagreement table
in `examples/run_pipeline.py` and VALIDATION.md §3).

| | Historical simulation | Parametric (var-covar) | Monte Carlo |
|---|---|---|---|
| Distributional assumption | none (empirical) | normal / t / Cornish-Fisher | any simulable factor model |
| Non-linear positions | exact (full reval of scenarios) | linear (delta) only; delta-gamma via moments | exact (full reval per path) |
| Fat tails | captured *if in the window* | only via t / CF corrections | captured if the factor model has them |
| Responds to current vol | plain HS: no; **FHS: yes** | yes with EWMA covariance | yes with EWMA covariance |
| Estimation noise at 99 % | high (~2-3 tail points per year of data) | low (uses whole sample for σ) | controllable (paths ↑) but model risk remains |
| Cost | trivial | trivial | expensive for exotic books |
| Main failure | window myopia ("great moderation" problem) | wrong tail shape | wrong model, sampling error |

**Chosen defaults and why:**

- **Filtered Historical Simulation (FHS)** is the flagship historical
  estimator (Barone-Adesi/Hull-White devolatilisation): standardise each
  past P&L by its one-step-ahead EWMA vol forecast, then rescale all
  innovations to *tomorrow's* forecast. It keeps the empirical tail shape
  while reacting to the volatility regime — the combination that makes it
  the industry workhorse. Alternatives considered: plain HS (rejected as
  default: exceptions cluster in vol regimes — demonstrated in
  VALIDATION.md §4, Christoffersen independence p = 0.02 on GARCH data);
  age-weighted BRW (implemented as a middle ground: reweights but does not
  rescale, so it fades old regimes without adapting the scale).
- **Parametric VaR** uses `σ_p = sqrt(wᵀΣw)` with dollar exposures `w`
  from the factor mapping and sample or EWMA (λ=0.94) covariance. Normal
  quantiles are the baseline; variance-matched Student-t and Cornish-Fisher
  provide tail corrections. CF is guarded by an explicit **monotonicity
  domain check** (§5) because outside its validity region the expansion is
  not a quantile function.
- **Monte Carlo VaR** simulates factor returns (multivariate normal or
  variance-matched multivariate t via `Z/√(W/ν)`) from the same covariance,
  with Cholesky + escalating diagonal jitter for singular matrices, then
  **fully revalues** the book per path. It is the only family that combines
  a chosen tail model with exact treatment of option convexity, and the SE
  of its quantile estimate is quantified (bootstrap + order-statistics CI).

### Quantile interpolation choice

Plain historical VaR uses NumPy's type-7 (`"linear"`) interpolation between
order statistics — the most common desk convention and the default of every
major statistical package. Age-weighted VaR necessarily uses the weighted
step-CDF inversion (interpolating weighted order statistics has no standard
definition). The two differ by at most one order statistic (unit-tested:
95.0 vs 95.05 on a 100-point grid); on 250+ day windows the difference is
immaterial relative to the sampling error of a 1 % quantile.

## 3. Portfolio representation and P&L

`Portfolio` maps positions onto named `RiskFactor`s (stock prices, index
level, implied vol). Linear positions (equity, index futures) are revalued
**exactly**: P&L = dollar-exposure × return. Options are revalued two ways:

- **Full revaluation**: Black-Scholes at shocked spot and vol (T fixed).
- **Delta-gamma-vega**: `dV ≈ Δ·dS + ½Γ·dS² + ν·dσ`.

Both are exposed so the approximation error is *measured*, not assumed
(`Portfolio.approximation_error`); the stress table in the pipeline shows it
reaching $93k (37 % of P&L) in the COVID scenario — quantifying exactly when
delta-gamma is unsafe (VALIDATION.md §6).

## 4. Expected Shortfall and the FRTB shift

ES fixes VaR's two structural defects:

1. **VaR ignores tail severity** — it is a threshold, not a loss estimate.
2. **VaR is not subadditive** — merging desks can *increase* reported VaR,
   so VaR limits can be gamed by splitting books. The classic two-bond
   counterexample is implemented and unit-tested: two independent positions
   each with VaR₉₅ = −5 (a gain) combine to VaR₉₅ = 95, while ES stays
   subadditive on the same book.

ES is coherent (Artzner et al.), which is why FRTB replaced 99 % VaR with
**97.5 % ES** for market-risk capital. The calibration was chosen for
continuity: for normal P&L, ES₉₇.₅ = σ·φ(z₀.₀₂₅)/0.025 ≈ 2.338σ vs
VaR₉₉ = 2.326σ — within 0.5 % (unit-tested). Under fat tails ES₉₇.₅ is
*larger*, which is precisely the regulatory intent: banks with fat-tailed
books hold more capital.

The trade-off (documented, not hidden): ES at a given alpha is estimated
from *averages of few tail points* and backtesting it is harder — we
implement the Acerbi-Székely Z₂ statistic (chosen over the exception
severity z-test because it uses every exception's magnitude relative to
ex-ante ES and needs no normality assumption; its 5 % critical value ≈
−0.70 is an approximation from the AS paper, since exact p-values require
simulating under the model's own distribution).

Empirical ES uses the exact tail integral of the empirical quantile
function (fractional weight on the boundary order statistic), which
guarantees `ES ≥ VaR` and is exact on known arrays (unit-tested).

## 5. Cornish-Fisher and its validity region

CF adjusts the normal quantile with sample skew `S` and excess kurtosis `K`:
`z_cf = z + (z²−1)S/6 + (z³−3z)K/24 − (2z³−5z)S²/36`. It is cheap and uses
information the plain normal ignores — but the cubic polynomial is only a
quantile function where it is **monotone**. We check the analytic derivative
`1 + zS/3 + (3z²−3)K/24 − (6z²−5)S²/36 > 0` on a dense grid over
`|z| ≤ 3.5` and **raise** by default outside the region (e.g. S=3 or K=10
flags; S=−0.3, K=2 passes — unit-tested). Outside the region the 99 %
"quantile" can cross the 95 % one; a number produced there is not a VaR.

## 6. Multi-day horizons

Two 10-day estimators, both with documented caveats:

- **√t scaling**: exact for i.i.d. zero-drift returns. Under volatility
  clustering the error has **no fixed sign** — which VaR you scale decides
  the direction, and conflating the two cases is a common desk error:
  - scaling a *conditional* VaR measured in a **calm** state **understates**
    risk, because variance mean-reverts upward over the horizon, so the
    10-day variance exceeds 10× today's (tested:
    `test_sqrt_time_understates_from_a_calm_conditional_state`);
  - scaling an *unconditional* VaR estimated on a full fat-tailed sample
    **overstates** the directly measured 10-day quantile, because temporal
    aggregation pulls the 10-day distribution toward normality. On the
    20k-day GARCH sample the excess kurtosis falls from **4.76 (1-day) to
    1.94 (10-day)**, and √t lands ~8 % above the direct estimate (tested:
    `test_temporal_aggregation_thins_the_tail_of_garch_returns`,
    `test_sqrt_time_overstates_the_unconditional_10day_var_under_garch`).

  A non-zero drift adds a further bias in either direction, since the mean
  scales linearly with `h` while sigma scales with `√h` (tested:
  `test_nonzero_mean_breaks_pure_sqrt_scaling`).
- **Overlapping 10-day windows**: model-free but serially dependent —
  effective sample ≈ n/10, so the quantile SE is far larger than the
  nominal count suggests. Pipeline shows 171.6k (√t) vs 140.3k
  (overlapping) for the same book: the gap *is* the message, and its sign
  is the unconditional-aggregation case above.

## 7. Backtesting stack

- **Kupiec POF** (LR, χ²₁): correct unconditional exception frequency.
- **Christoffersen independence** (Markov LR, χ²₁) and **conditional
  coverage** (= UC + IND, χ²₂): exceptions must not cluster.
- **Basel traffic light**: 250-day 99 % exception count → green 0-4
  (multiplier 3.0), yellow 5-9 (add-ons 0.40/0.50/0.65/0.75/0.85), red 10+
  (4.0). The boundaries are exact Binomial(250, 0.01) statements: a correct
  model is green with p ≈ 0.892 and red with p ≈ 2.5×10⁻⁴ (unit-tested).
- **ES backtest**: Acerbi-Székely Z₂ (§4).

## 8. Assumptions register

| # | Assumption | Where | What breaks if violated |
|---|---|---|---|
| A1 | Factor returns capture all P&L drivers (no residual/basis risk; futures basis carry ignored) | portfolio.py | Idiosyncratic gaps (halts, single-name news) hit P&L but not VaR; add per-name factors or specific-risk add-on |
| A2 | Option P&L computed with T fixed (no theta over the horizon) | portfolio.py | Overstates long-option risk slightly at 10d (theta bleed is deterministic P&L, not risk); material only for short-dated books |
| A3 | Implied vol is a single parallel factor per underlier | portfolio.py | Skew/term-structure twists (crash reshapes the smile) mis-measured; extend to a vol-surface factor set |
| A4 | Daily P&L mean ≈ 0 | all VaR modules | With material drift (e.g. carry books) VaR is biased by −μ·h; pass `mean` explicitly |
| A5 | i.i.d. returns within the estimation window (plain HS, sample cov) | historical/parametric | Vol clustering → clustered exceptions, Kupiec/Christoffersen failures — exactly what the GARCH backtest demonstrates; use FHS/EWMA |
| A6 | EWMA λ=0.94 (RiskMetrics daily) is the right decay | historical/parametric | Too-slow decay lags regime shifts; too-fast is noisy. λ is a parameter everywhere, sensitivity should be checked quarterly |
| A7 | √t scaling for 10-day VaR | historical/parametric | Understates multi-day risk under clustering/autocorrelation (see §6); regulators accepted it pre-FRTB, FRTB moved to liquidity-horizon scaling |
| A8 | Student-t df (default 6) matched to variance | parametric/MC | Wrong df → wrong tail: df too high reverts to normal (VaR understated), df ≤ 2 has no variance (rejected with `ValueError`) |
| A9 | Covariance matrix is estimable and roughly stationary | parametric/MC | Correlations spike toward 1 in crashes → diversification benefit evaporates exactly when needed; stress tests (not VaR) cover this |
| A10 | MC sampling error is acceptable at the chosen path count | monte_carlo_var | Quantified, not assumed: bootstrap SE and order-statistic CI reported; 100k paths → SE ≈ 0.8 % of the 99 % VaR on the demo book |
| A11 | Historical stress shocks approximate the named episodes | stress_testing | They are **approximations of published moves** (flagged in code + DESK_GUIDE); scenario P&L is indicative, not a replay of actual desk P&L |
| A12 | Basel zones assume 250 obs at 99 % | backtesting | Different window/alpha invalidates the 0-4/5-9/10+ mapping; `basel_zone_probabilities` recomputes the binomial table for any (n, alpha) |

## 9. References

- Kupiec (1995), *Techniques for Verifying the Accuracy of Risk Measurement
  Models*.
- Christoffersen (1998), *Evaluating Interval Forecasts*.
- Boudoukh, Richardson, Whitelaw (1998), *The Best of Both Worlds* (BRW).
- Barone-Adesi, Giannopoulos, Vosper (1999); Hull & White (1998) — FHS.
- Artzner, Delbaen, Eber, Heath (1999), *Coherent Measures of Risk*.
- Acerbi & Székely (2014), *Backtesting Expected Shortfall*.
- BCBS (1996), *Supervisory framework for the use of backtesting*; BCBS
  (2019), *Minimum capital requirements for market risk* (FRTB).
