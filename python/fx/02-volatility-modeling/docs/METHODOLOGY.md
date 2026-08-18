# Methodology — FX Volatility Modeling & Forecasting

This document answers documentation-contract items 1 (model choice, with
alternatives and trade-offs) and 2 (assumptions register), and sets out where
FX volatility genuinely differs from the equity textbook treatment.

---

## 1. The models and the maths

All models act on daily zero-mean log returns `r_t` of a BASE/QUOTE FX pair,
`r_t = sigma_t z_t`, with `z_t` i.i.d. standard Gaussian or standardized
Student-t(nu) (unit variance, nu > 2). Units: daily decimal returns;
annualized vol = `sqrt(252 * var_daily)` (see §4 on 252 vs 260).

| Model | Variance equation | Parameters |
|---|---|---|
| Historical | rolling `std(r)`; Parkinson `(4 ln 2)^-1 mean(ln(H/L)^2)`; Garman–Klass OHLC | window |
| EWMA (RiskMetrics) | `s2_t = λ s2_{t-1} + (1-λ) r_{t-1}^2` | λ (0.94) |
| GARCH(1,1) | `s2_t = ω + α r_{t-1}^2 + β s2_{t-1}` | ω, α, β (, ν) |
| GARCH-X | `s2_t = ω + γ_x' x_t + α r_{t-1}^2 + β s2_{t-1}` | + γ_x ≥ 0 |
| GJR-GARCH | `s2_t = ω + (α + γ 1[r_{t-1}<0]) r_{t-1}^2 + β s2_{t-1}` | + γ ≥ 0 |
| EGARCH | `ln s2_t = ω + β ln s2_{t-1} + α(|z_{t-1}|-E|z|) + γ z_{t-1}` | γ sign-free |

Estimation is from-scratch maximum likelihood (`arch` is used **only** as a
cross-validation oracle in the tests):

- **Constraint transforms.** ω = exp(u); stationarity via persistence
  p = expit(u_p) ∈ (0,1) split into α = p·s, β = p·(1−s) (GARCH), or a
  softmax split across (α, γ/2, β) (GJR); EGARCH β = tanh(u); ν = 2.05 + exp(u);
  γ_x = exp(u). The optimizer (multi-start L-BFGS-B) works unconstrained, so
  no penalty hacks and no infeasible iterates.
- **Scaling.** Returns are internally rescaled to unit variance; ω, γ_x and
  σ² map back by the variance scale, the log-likelihood by the
  change-of-variables constant `−n·log(s)`. This makes decimal vs percent
  inputs equivalent (tested against `arch`'s percent convention) and lets the
  same code fit pegged pairs with basis-point vols.
- **Standard errors** from the numerical Hessian of the negative
  log-likelihood in *natural* parameter space (central differences, pinv
  fallback; NaN at boundaries rather than fake precision).
- **Variance targeting** (optional): ω fixed at `Var(r)(1−α−β)`, one fewer
  free parameter, unconditional variance matched to sample by construction.
- **Initialization**: arch-style exponentially weighted backcast (λ=0.94 over
  the first 75 observations), so cross-checks compare like for like.

Forecasting: GARCH/GJR multi-step forecasts are analytic
(`E s2_{T+h} = s2_bar + p^{h-1}(s2_{T+1} − s2_bar)`, GJR persistence
`α + γ/2 + β` under symmetric innovations); EGARCH has no closed form for
`E[s2]` (expectations of exponentials of |z|), so it is simulated with a
seeded Generator; EWMA is flat by construction.

## 2. Why this model family (contract item 1)

**Chosen: the GARCH family (GARCH-t as the G10 default, EGARCH/GJR for EM),
plus EWMA and realized estimators as benchmarks.** Compared against:

**Alternative A — EWMA/RiskMetrics only.** One parameter, no estimation risk,
industry-legible. But persistence is hard-wired to 1: no mean reversion, so
*every* multi-step forecast is flat, term vol forecasts are wrong by
construction, and after a shock the model never de-escalates on its own.
The 500-day OOS race (VALIDATION §4) shows EWMA is QLIKE-dominated on both
test pairs. Kept as the benchmark every candidate must beat.

**Alternative B — Stochastic volatility (e.g. Heston-type discrete SV, or
realized-vol HAR models).** SV separates the vol shock from the return shock,
which is more realistic and links directly to options pricing. Costs: the
likelihood involves an intractable latent-state integral (particle filtering
or MCMC — an order of magnitude more machinery to validate and to explain to
model governance), and daily-frequency forecast gains over GARCH-t are small
and unstable in the literature. HAR needs intraday realized measures we do
not assume available for all pairs (EM crosses in particular). Trade-off
taken: GARCH's one-shock structure is a known simplification; we buy exact
likelihoods, second-long fits, analytic forecasts and easy governance.

**Alternative C — Implied vol from the options market.** Forward-looking and
incorporates event calendars automatically. But it *contains a risk premium*
(persistently above subsequent realized — that is the vol-selling business,
DESK_GUIDE §3), is unavailable or stale for many EM crosses, and cannot be
used to mark the very premium we want to measure. Used as a *comparator*
(`fx_vol.vol_premium`), not as the forecaster.

Within the family, the pipeline's empirical ranking logic (run_pipeline.py,
real numbers in VALIDATION §3): Student-t innovations are first-order
(ΔAIC ≈ 100 on the G10-style pair, ≈ 900 on the EM-style pair); asymmetry is
immaterial for the G10 pair (EGARCH-t within 0.6 AIC of GARCH-t, leverage
γ = +0.007 ± 0.011) but first-order for the EM pair (EGARCH-t beats GARCH-t
by ΔAIC ≈ 19.5, γ = +0.046 ± 0.017).

## 3. Where FX differs from equity (first-class in this package)

1. **Quote direction is a modelling choice.** An equity index has one
   orientation; an FX pair has two. Inverting BASE/QUOTE negates log returns:
   volatility is invariant (`test_returns.py::TestInversionInvariance`), but
   *asymmetry flips sign*. USDJPY safe-haven behaviour (yen bid in risk-off →
   big *negative* pair returns carry the vol) becomes positive-side asymmetry
   in JPYUSD. Consequence: GJR's γ ≥ 0 sees asymmetry in only one quote
   direction — the pipeline shows GJR-t finding γ = 0 on USDMXN but
   γ = +0.123 ± 0.040 on inverted MXNUSD, while sign-free EGARCH catches it
   either way. Desk rule: fit the quote direction you trade, and check γ's
   sign against the economics.
2. **Leverage is weak for G10, strong for EM.** There is no debt-equity
   channel in FX; G10 asymmetry is a risk-appetite effect and is small
   (γ_EGARCH ≈ 0.007 on the G10-style pair vs ≈ 0.046 on EM-style). EM pairs
   quoted USD/EM have a genuine one-sided depreciation-jump channel.
3. **Scheduled events dominate the calendar.** FOMC/ECB/BoJ days are known
   years ahead and carry a variance premium. GARCH-X puts the event calendar
   directly in the variance equation with dummy x_t known at t−1; the dummy
   coefficient is recovered on simulation within 3% (γ_x = 4.89e−5 vs true
   5e−5, t = 23.9) and forecasts can price a known meeting into day h
   (run_pipeline §5: day-3 FOMC lifts the day-3 ann. vol forecast from 8.8%
   to 14.2%).
4. **Weekly seasonality and the 24h5d market.** FX trades continuously
   Monday–Friday: no overnight gaps intra-week (range estimators unusually
   effective), but a weekend gap Monday. Day-of-week vol factors (quiet
   Monday, heavy Wednesday/Friday — FOMC/NFP) are estimable
   (`day_of_week_vol_factors`, recovered within 8% on simulation) and can
   pre-whiten returns before GARCH fitting. Annualization: 252 (default,
   comparable with equity desks and most vendor quotes) vs 260 = 52×5 (FX
   convention counting every weekday); the ratio is a constant
   sqrt(260/252) ≈ 1.6% of vol — immaterial for dynamics, material when
   comparing against a counterparty's implied quote. Every function takes
   `periods_per_year` explicitly.
5. **Triangular consistency.** Cross vols are constrained by
   `σ_x² = σ_1² + σ_2² + 2 c_1 c_2 ρ σ_1 σ_2` (signs from quote directions).
   This has no equity analogue and gives a free consistency check and a way
   to mark illiquid crosses from liquid legs (`cross_volatility`, exact
   in-sample identity, tested with positive and negative correlation).
6. **Pegs and depegs.** Managed floats and hard pegs produce basis-point vols
   and boundary fits; depegs produce 50-sigma days. Both are edge-case tested
   (VALIDATION §5) — equity models rarely face a "the central bank stopped
   defending the price" regime.

## 4. Assumptions register (contract item 2)

| # | Assumption | What breaks if violated |
|---|---|---|
| A1 | Zero conditional mean of daily returns | Daily FX drift (carry) is ~1–2 bp vs vol ~60 bp, second-order for variance MLE. A large ignored drift (EM with high carry, crawling pegs) biases ω up: variance absorbs mean². Demean or model the mean explicitly if |mean| > ~10% of vol. |
| A2 | i.i.d. innovations z_t after variance filtering | Remaining serial dependence (intraday seasonality aggregated badly, weekly patterns) invalidates the likelihood factorization; detected by Ljung-Box/ARCH-LM on standardized residuals (both wired in and tested). |
| A3 | One-regime parameters over the estimation window | Structural breaks (depeg, EM crisis, intervention-regime change) contaminate estimates — the depeg tests show persistence pinning near 1. Use rolling windows (the OOS harness refits every 100–125 days) and the failure modes in VALIDATION §5. |
| A4 | Symmetric innovation distribution (Gaussian/t) | GJR multi-step persistence uses E1[z<0] = 1/2; skewed innovations (EM jump series) bias multi-step forecasts. EGARCH simulation forecasting uses the same symmetric draw — a skewed-t extension is the natural upgrade. |
| A5 | Finite variance: ν > 2 (enforced ν ≥ 2.05) | Fitted ν → 2 says the sample looks infinite-variance (crisis EM data); estimates remain but unconditional variance and vol targeting lose meaning. Fitted ν is reported so the desk can see it. |
| A6 | Variance stationarity: persistence < 1 (enforced by transform) | Genuine IGARCH data pins at the boundary — fit is still valid QMLE for 1-step, but unconditional variance → ∞ and long-horizon forecasts are meaningless; flagged via `persistence` and documented (VALIDATION §5). |
| A7 | Exogenous x_t known at t−1, γ_x ≥ 0 | Wrong for *unscheduled* events (interventions, surprise hikes) — those belong to the innovation, not the dummy. Negative-effect regressors must be recoded (the constraint keeps σ² > 0). |
| A8 | Squared daily return as realized-variance proxy | r² is unbiased but very noisy; QLIKE (Patton-robust) keeps rankings consistent, but power is limited — with 500 OOS days only large model gaps reach significance (VALIDATION §4). Intraday RV would sharpen this. |
| A9 | No microstructure noise / valid daily marks | ECB reference fixes are once-daily, non-traded; range estimators assume true continuous highs/lows — stale or filtered EM quotes bias range estimators down. |
| A10 | Weekend gap ignored (Friday→Monday treated as one step) | Monday variance is systematically higher; unmodelled, it fattens residual tails slightly. The day-of-week factor estimator quantifies it; a t-distribution absorbs most of the rest. |

## 5. Innovation distribution: Gaussian vs Student-t

Gaussian QMLE is consistent for (ω, α, β) even under fat tails, so *variance
paths* are roughly right — but (i) efficiency is poor when ν is small (the EM
pair fits ν ≈ 3.6), and (ii) anything tail-sensitive (VaR multipliers,
option wings) is badly wrong: on simulated ν = 6 data the Gaussian fit's
standardized residuals have a 1% quantile ≈ 13% beyond −2.326 and excess
kurtosis > 1 (`test_garch.py::TestStudentT`). The t-likelihood also
*downweights* jump days when estimating dynamics, which is why GARCH-t beats
Gaussian GARCH out-of-sample on the EM pair (DM p = 0.017, VALIDATION §4).
On Gaussian data the fitted ν runs to +∞ (tested), so defaulting to t is
safe: it nests the Gaussian in practice.
