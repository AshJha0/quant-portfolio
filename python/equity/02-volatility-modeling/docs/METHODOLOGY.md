# Methodology — Equity Volatility Modeling & Forecasting

This document answers contract items 1 and 2 of `CONVENTIONS.md`: **why these
models**, compared against the live alternatives, and **an explicit
assumptions register** with what breaks when each assumption fails.

Conventions used throughout: daily log-returns in decimal units, zero
conditional mean (standard for daily equity data, where `|mu| dt` is an order
of magnitude below `sigma sqrt(dt)`), volatility annualised with `sqrt(252)`.

---

## 1. The models, and the maths

### 1.1 Historical (realized) estimators — `eq_vol.historical`

Rolling close-to-close: `vol_t = sqrt(252 * mean(r^2, window))`. Range-based
one-day variance estimators, all unbiased for driftless GBM:

| Estimator | Formula (per day) | Relative efficiency vs r² | Drift-robust |
|---|---|---|---|
| Close-to-close | `r_t^2` | 1.0x | no (bias `+mu^2 dt`) |
| Parkinson (1980) | `ln(H/L)^2 / (4 ln 2)` | ~4.9x | no |
| Garman–Klass (1980) | `0.5 ln(H/L)^2 − (2 ln2 − 1) ln(C/O)^2` | ~7.4x | no |
| Rogers–Satchell (1991) | `ln(H/C)ln(H/O) + ln(L/C)ln(L/O)` | ~6x | **yes** |

Efficiency means lower sampling variance of the variance estimate — a 21-day
Parkinson estimate is roughly as precise as a ~100-day close-to-close
estimate, which matters enormously when vol moves. The price: assumptions of
continuous monitoring (discrete trading biases the observed range *down* by
`O(1/sqrt(intraday steps))`), no overnight gaps, and no microstructure noise
(bid–ask bounce biases the range *up*). Both biases are demonstrated and
tolerance-tested in `tests/test_historical.py`.

### 1.2 EWMA (RiskMetrics 1996) — `eq_vol.ewma`

`sigma2_t = lambda sigma2_{t-1} + (1−lambda) r_{t-1}^2`, `lambda = 0.94`
(half-life 11.2 days). This is IGARCH(1,1) with zero intercept: persistence is
*exactly* one, so the k-step forecast is **flat** at the 1-step forecast —
there is no long-run level to revert to. That is simultaneously its virtue
(no parameters to estimate, never badly mis-calibrated) and its defect (no
term structure, permanent response to every shock).

### 1.3 GARCH(1,1) (Bollerslev 1986) — `eq_vol.garch`

`sigma2_t = omega + alpha r_{t-1}^2 + beta sigma2_{t-1}`, with `omega > 0`,
`alpha, beta ≥ 0`, `alpha + beta < 1`. Key derived quantities:

* unconditional variance `omega / (1 − alpha − beta)`;
* persistence `alpha + beta`; half-life of a variance shock
  `ln(1/2) / ln(alpha+beta)`;
* k-step forecast: geometric decay of the 1-step forecast toward the
  unconditional level at rate `alpha + beta`.

Estimation is exact MLE (Gaussian or standardised Student-t) via L-BFGS-B on
smooth parameter transforms — `omega = exp(u0)`, persistence
`P = 0.9999·sigmoid(u1)`, `alpha = P·sigmoid(u2)`, `beta = P − alpha` — so
every optimiser iterate satisfies positivity and stationarity by
construction, with optional **variance targeting** (`omega` pinned so the
model's unconditional variance equals the sample variance). Standard errors
come from the inverse numerical Hessian of the negative log-likelihood in
natural parameter space. The variance recursion is an AR(1) with exogenous
input and is computed by `scipy.signal.lfilter` — no Python loop — which is
what makes 20,000-observation MLE run in ~0.1 s.

### 1.4 EGARCH(1,1) (Nelson 1991) — `eq_vol.egarch`

`ln sigma2_t = omega + beta ln sigma2_{t-1} + alpha(|z_{t-1}| − E|z|) +
gamma z_{t-1}`. Because the recursion lives in **log**-variance, positivity
of `sigma2` holds for *any* real parameters — the only constraint is
`|beta| < 1` for stationarity. `gamma < 0` produces the leverage effect
(negative shocks raise vol more). Multi-step forecasts have no practical
closed form (they require `E[exp(alpha|z| + gamma z)]` products; a
semi-analytic Gaussian expression exists but does not generalise across
innovation distributions), so `eq_vol.forecasting` uses seeded Monte Carlo.

### 1.5 GJR-GARCH(1,1) (Glosten–Jagannathan–Runkle 1993) — `eq_vol.gjr`

`sigma2_t = omega + (alpha + gamma·1[r_{t-1}<0]) r_{t-1}^2 + beta
sigma2_{t-1}`. `gamma > 0` is the leverage effect (note the *opposite* sign
convention from EGARCH — both are stated and unit-tested). Stationarity under
symmetric innovations: `alpha + gamma/2 + beta < 1`; unconditional variance
`omega / (1 − alpha − gamma/2 − beta)`. The reparameterisation splits
persistence into the positive-shock load `alpha ≥ 0` and the negative-shock
load `alpha + gamma ≥ 0`, so `gamma` is genuinely free to take either sign —
leverage-sign recovery in tests is meaningful, while positivity and
stationarity always hold.

---

## 2. Why the GARCH family? (contract item 1: alternatives & trade-offs)

The realistic alternatives for a daily equity vol forecast:

**Alternative A — Stochastic volatility (e.g. Heston, log-SV).**
SV models treat variance as its own latent stochastic process, which is the
theoretically "right" description and what option pricing ultimately needs.
The cost: the likelihood involves an unobserved state and requires filtering
(particle filters, MCMC, or quasi-ML via Kalman approximations) — an order of
magnitude more machinery, harder diagnostics, and materially harder model
governance. For **one-step-ahead P&L-relevant vol forecasting from daily
returns**, the empirical literature (Hansen–Lunde 2005, "Does anything beat a
GARCH(1,1)?") finds essentially no forecast gain over GARCH-family models.
GARCH gives an *exact, closed-form likelihood* — every number in this repo is
auditable to machine precision, which we exploit in the arch cross-validation.

**Alternative B — Realized-volatility / HAR models (Corsi 2009).**
If clean intraday data are available, realized variance + HAR regression
typically *beats* daily GARCH out of sample, and it would be the preferred
production choice on liquid index futures. Trade-offs: it requires a reliable
tick/5-min data pipeline (survivorship of exchange feeds, session handling,
microstructure-noise corrections), does not produce a full conditional
*distribution* without further assumptions, and degrades to nothing on assets
without intraday history. This project's scope is daily OHLC-and-close data —
the range estimators in `historical.py` are exactly the "poor man's realized
vol" appropriate to that data budget, and the harness in `forecasting.py`
would accept an RV proxy unchanged if one were available.

**Alternative C — Implied volatility.**
Where liquid options exist, implied vol is forward-looking and hard to beat
for direction. But it embeds a (time-varying) variance risk premium — it is a
biased forecast by construction, it is unavailable for most single names and
custom baskets, and it cannot be used to *mark* the very options it comes
from without circularity. Desks use implied and model vol together; this
package supplies the model leg.

**Within the family**, the escalation Historical → EWMA → GARCH → EGARCH/GJR
is a controlled increase in structure: each step adds exactly one economic
feature (exponential weighting; mean reversion; asymmetry), each is testable
against the previous by likelihood ratio / information criteria / DM tests,
and the pipeline demonstrates the value of each step on data where the
feature is truly present (`examples/run_pipeline.py`).

---

## 3. Assumptions register (contract item 2)

Each assumption states **what breaks if violated** and where it is tested or
documented.

1. **Zero conditional mean.** Returns are modelled as `r_t = sigma_t z_t`.
   *If violated*: variance estimates absorb `mu^2`; for daily equities the
   bias is second-order (`(mu dt)^2` vs `sigma^2 dt`, well under 1% of
   variance for |mu| ≤ 20%/yr). For strongly trending series use the demeaned
   estimators (`realized_vol(..., demean=True)`) or Rogers–Satchell
   (drift-robust) — both tested in `test_historical.py`.

2. **iid innovations `z_t`** (after scaling by `sigma_t`). *If violated*
   (remaining serial dependence): standard errors and forecasts are wrong;
   the Ljung-Box test on squared standardised residuals and the ARCH-LM test
   in `evaluation.py` are exactly the checks that detect this, and the fitted
   models pass them on their own simulated data (`test_evaluation.py`).

3. **Correctly specified innovation distribution.** Gaussian QMLE remains
   *consistent* for the variance parameters even under fat tails
   (Bollerslev–Wooldridge), but tail quantiles (VaR/ES) computed from a
   Gaussian assumption will be badly wrong when the truth is t-distributed.
   Mitigation: the Student-t likelihood (`dist="t"`, `nu` recovered within
   0.2 of truth on 20k simulated obs). See VALIDATION.md failure mode F3.

4. **Covariance stationarity** (`alpha + beta < 1`, `alpha + gamma/2 + beta <
   1`, `|beta| < 1`). *If violated*: no unconditional variance exists, the
   forecast term structure never flattens, and `unconditional_variance()`
   **raises** rather than returning a negative number (tested). Estimated
   persistence hitting the 0.9999 transform ceiling is a red flag surfaced by
   the parameter table, not hidden.

5. **No structural breaks** — parameters constant over the estimation window.
   *If violated*: a level shift in variance masquerades as persistence ≈ 1
   (IGARCH illusion) and forecasts mean-revert to a stale level. This is the
   single most important practical failure mode; it gets a dedicated case
   study (COVID-style regime jump) in VALIDATION.md and
   `test_edge_cases.py::TestCrisisRegimeJump`.

6. **One-period-ahead information structure.** `sigma2_t` is measurable with
   respect to information at `t−1`; the rolling harness never lets a
   forecast see its own realisation (no look-ahead — explicitly tested in
   `test_forecasting.py::test_forecast_alignment_no_lookahead`).

7. **Squared returns are a valid (unbiased, noisy) proxy for realised
   variance** in evaluation. *If violated* (e.g. proxy contaminated by
   microstructure): naive loss rankings flip. Mitigation: QLIKE and MSE are
   Patton (2011)-robust losses, which is *why* they are the only two losses
   offered (`evaluation.py` docstring).

8. **252 trading periods per year; equidistant observations.** *If violated*
   (weekly/monthly data, gaps): annualisation constants and half-lives are
   wrong, and low-frequency samples are usually too short for MLE — see
   VALIDATION.md F5 (low-frequency limits); fitters refuse < 100 obs.

9. **The pre-sample variance is a nuisance parameter whose influence dies
   out.** GARCH/GJR/EGARCH start from an arch-compatible exponentially
   weighted backcast (`init_method="backcast"`, decay 0.94 over ≤75 obs);
   EWMA defaults to the full-sample mean of squared returns. *If violated*
   (persistence near 1, or a short sample): the initial condition decays
   geometrically at rate β, so with β ≈ 0.95 it still contributes ~1% of the
   variance level after 90 days — on samples of a few hundred observations
   the choice of `init_method` visibly moves ω. Two consequences a reviewer
   should know: (i) `init_method="backcast"` vs `"sample"` agree on long
   samples but not short ones (tested in
   `test_garch.py::test_init_methods_agree_on_long_sample`); (ii) EWMA's
   default `init_var` uses the *whole* sample, so an in-sample EWMA path is
   not a strictly causal filter at its first observations. This is a level
   effect on the initial condition only, and the out-of-sample harness never
   inherits it — `rolling_one_step_forecasts` re-runs the filter on
   `returns[0..t-1]` alone at every date, which is what
   `test_forecast_alignment_no_lookahead` verifies. Pass an explicit
   `init_var` when a strictly causal in-sample path is required.

10. **Variance is scale-equivariant.** The models contain no absolute
    threshold: rescaling returns by `c` must rescale every variance by `c²`
    while leaving α, β and persistence untouched. *If violated*, some
    hard-coded scale (a tolerance, a floor, a starting value) has leaked into
    the recursion or the likelihood, and results would silently depend on
    whether the desk feeds decimal or percent returns. This is asserted to
    1e-12 across EWMA, realized vol, the GARCH/EGARCH recursions and the
    fitted parameters — see VALIDATION.md §6.1
    (`test_properties.py::TestScaleEquivariance`).

---

## 4. References

* Bollerslev (1986), *Generalized Autoregressive Conditional
  Heteroskedasticity*, J. Econometrics.
* Nelson (1991), *Conditional Heteroskedasticity in Asset Returns*, Econometrica.
* Glosten, Jagannathan, Runkle (1993), *On the Relation between the Expected
  Value and the Volatility of the Nominal Excess Return on Stocks*, J. Finance.
* Engle, Ng (1993), *Measuring and Testing the Impact of News on Volatility*, J. Finance.
* Diebold, Mariano (1995), *Comparing Predictive Accuracy*, JBES;
  Harvey, Leybourne, Newbold (1997) small-sample correction.
* J.P. Morgan/Reuters (1996), *RiskMetrics — Technical Document*, 4th ed.
* Hansen, Lunde (2005), *A Forecast Comparison of Volatility Models: Does
  Anything Beat a GARCH(1,1)?*, J. Applied Econometrics.
* Patton (2011), *Volatility Forecast Comparison Using Imperfect Volatility
  Proxies*, J. Econometrics 160(1).
* Corsi (2009), *A Simple Approximate Long-Memory Model of Realized
  Volatility* (HAR), J. Financial Econometrics.
