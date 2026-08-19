# Methodology — Single-Asset Equity Risk Metrics

This document answers documentation-contract items 1 and 2: **why these
methods** (against alternatives, with trade-offs) and **what assumptions we
make** (numbered register, each with "what breaks if violated").

Conventions used throughout: 252 trading days per year; returns are
**simple daily returns** (`P_t / P_{t-1} - 1`) unless a function name says
`log`; volatility is annualised by `sqrt(252)`; VaR and Expected Shortfall
are reported as **positive numbers representing a loss** at a stated
confidence level and a 1-day horizon (e.g. "VaR 95% = 2.1%" means "on the
worst 5% of days you lose at least 2.1% of position value").

The point of this project is not the arithmetic — every metric here is a
few lines of NumPy/SciPy. The point is that **different, individually
reasonable methods give different answers on the same data**, and knowing
*why* (which assumption each one leans on, and where that assumption
breaks) is the actual skill being demonstrated.

---

## 1. Returns: simple vs log

`simple_returns` (`P_t/P_{t-1} - 1`) and `log_returns` (`ln(P_t/P_{t-1})`)
are both provided. Simple returns aggregate additively **across assets**
at a point in time (portfolio return = weighted sum of constituent simple
returns); log returns aggregate additively **across time** (`T`-day log
return = sum of daily log returns), which is what makes them convenient
for compounding and for GARCH-style variance models.

**Why simple returns as the default here.** This is a single-asset
project, so the cross-sectional aggregation property is unused, and VaR is
conventionally quoted as "how much of position value do I lose", which is
exactly what a simple return measures. For daily horizons the two are
numerically close (`ln(1+r) ≈ r` for small `r`), but the choice is
documented per function rather than left implicit, since it stops
mattering less than people assume once you compound over longer horizons
or aggregate across positions.

**Alternative not implemented:** using log returns throughout (as most
academic GARCH/EWMA literature does) and converting to simple-return VaR
only at the reporting boundary. Trade-off: marginally cleaner theory for
the volatility models, at the cost of an extra conversion step and a
sign/rounding surface a reviewer has to check every time.

## 2. Volatility: three estimators, because "the" volatility does not exist

### 2.1 Full-sample (unconditional) standard deviation

`sigma_annual = std(returns, ddof=1) * sqrt(252)`.

**Why include it.** It is the simplest possible estimator and the natural
baseline everything else is compared against; many finance formulas
(Sharpe, Black-Scholes) are written in terms of "the" volatility and this
is the estimator implicitly meant when no further qualification is given.

**Alternative:** none needed as a baseline — but note it is *not* a good
forecast of tomorrow's volatility on its own; see below.

### 2.2 Rolling window (21-day)

`rolling_volatility(returns, window=21)` — equal-weighted standard
deviation over the trailing month.

**Why include it.** Reflects "the current regime" rather than the
10-year average. Simple, transparent, and exactly what a trader means
when they say "vol's been running at 15 lately".

**Alternative considered: longer/shorter windows.** A shorter window (5d)
reacts faster but is noisier (fewer degrees of freedom); a longer window
(63d, one quarter) is smoother but slower to reflect a regime change.
21 days is a conventional middle ground, not a theoretically derived
optimum — this is a real, documented arbitrariness in the estimator.

### 2.3 EWMA (RiskMetrics, λ = 0.94)

`sigma_t^2 = (1-λ) r_t^2 + λ sigma_{t-1}^2`, seeded at the first
observation, then annualised.

**Why include it.** All of history matters, but recent history matters
exponentially more — this is the simplest model that adapts to a
volatility regime shift without a hard window cutoff, and it is the
industry-standard convention (RiskMetrics 1994, λ=0.94 daily, λ=0.97
monthly) that a risk report is expected to at least mention.

**Assumption:** `λ` is fixed, not estimated from data — a genuine GARCH(1,1)
model would estimate the equivalent persistence parameter by maximum
likelihood. EWMA is GARCH(1,1) with the constant term fixed to zero and
the persistence hand-set to 0.94; that is a *deliberate* simplification
(one less parameter to overfit, no numerical optimizer required) at the
cost of not adapting `λ` itself to how persistent volatility actually is
in this particular asset's data.

**Alternative not implemented: GARCH(1,1).** Estimates `omega`, `alpha`,
`beta` by MLE, with `sigma_t^2 = omega + alpha r_{t-1}^2 + beta
sigma_{t-1}^2` and mean-reversion to a long-run variance `omega/(1-alpha-
beta)`, which EWMA (no `omega`, no mean reversion) lacks entirely. GARCH
is the natural upgrade and is the subject of the companion
`python/equity/02-volatility-modeling` project in this portfolio;
trade-off is a fitted model (parameter and convergence risk, needs a
reasonably long history) vs EWMA's zero-parameters-to-estimate simplicity.

**When they disagree:** if EWMA/rolling vol diverges sharply from the
full-sample figure, the unconditional number is describing a market that
no longer exists — see `docs/VALIDATION.md` for a concrete measured
example on the bundled synthetic data.

## 3. Value at Risk: three methods

All three answer "what daily loss is exceeded with probability
`1 - confidence`?" — they differ only in what they assume about the
*shape* of the return distribution.

### 3.1 Historical (empirical)

`VaR = -percentile(returns, 100*(1-confidence))`.

**Why include it.** Zero distributional assumption: whatever shape the
realised returns actually have (fat tails, skew, whatever) is exactly
what the quantile reflects. This is the most commonly used VaR method on
real trading desks precisely because it makes no model assumption that
can be wrong.

**Assumption A-VaR1: the sample window is representative of the future.**
*What breaks if violated:* a calm sample understates future risk (this is
exactly what happened industry-wide going into 2008 — trailing-window
historical VaR built on the "Great Moderation" understated the risk that
then materialised); a stressed sample overstates it going forward once
the regime normalises. It is also a single order statistic (or an
interpolation between two) at high confidence in a short sample, so it is
a genuinely noisy estimator when `n` is small — quantified in
`docs/VALIDATION.md`.

**Alternative not implemented: filtered historical simulation.** Rescale
historical returns by `current_EWMA_vol / historical_vol_at_that_date`
before taking the empirical quantile, so the *shape* is still empirical
but the *scale* is current. This directly fixes assumption A-VaR1's main
failure mode (stale sample-window volatility) at the cost of needing a
volatility model as an input and losing some of "historical VaR's" appeal
of being assumption-free.

**Alternative not implemented: GARCH-based (conditional) VaR.** Fit a
GARCH model, forecast tomorrow's conditional variance, then apply a
parametric or empirical-innovation quantile to that forecast variance.
Adapts fastest to genuine regime changes and is standard in academic VaR
backtesting literature; trade-off is full model risk (misspecified GARCH
order, wrong innovation distribution) stacked on top of estimation risk,
and it requires enough history to fit reliably.

### 3.2 Gaussian (parametric / variance-covariance)

`VaR = -(mu + sigma * z)`, `z = Phi^{-1}(1 - confidence)`.

**Why include it — deliberately, to demonstrate its failure.** It is the
fastest VaR to compute (closed form, trivially extends to portfolios via
a covariance matrix) and it is what "VaR" meant historically (J.P.
Morgan's original RiskMetrics). Daily equity returns have excess
kurtosis, so this method **systematically understates** tail risk at high
confidence — the whole point of shipping it next to historical VaR is to
make that gap visible and quantified rather than asserted.

**Assumption A-VaR2: returns ~ Normal(mu, sigma).** *What breaks if
violated:* every real daily-return sample used in this project's tests
and examples rejects normality via Jarque-Bera (see §5 below), and the
Gaussian VaR is measurably below historical VaR at 99% on the bundled
synthetic data (quantified in `docs/VALIDATION.md`) — precisely because
the normal distribution assigns too little probability mass to extreme
days.

**Alternative not implemented: Student-t parametric VaR.** Fit a
location-scale Student-t (3 parameters: location, scale, degrees of
freedom) instead of a normal, which captures fat tails parametrically.
Better tail fit than Gaussian, still closed-form-ish (quantile via the
t-distribution), but degrees-of-freedom estimation is itself noisy in
small samples and the model still assumes one fixed tail shape for the
whole sample period (no skew, no regime dependence).

### 3.3 Cornish-Fisher (modified)

`z_cf = z + (z^2-1)s/6 + (z^3-3z)k/24 - (2z^3-5z)s^2/36`, then
`VaR = -(mu + sigma*z_cf)`, where `s` = sample skewness, `k` = sample
excess kurtosis.

**Why include it.** A pragmatic middle ground: still a smooth, closed-form
quantile function (fast, extends to portfolios), but corrects the
Gaussian quantile using the sample's own skew and kurtosis via an
Edgeworth-style expansion. When `s = k = 0` it collapses **exactly** to
the Gaussian formula (tested to 1e-12 in `tests/test_var_es.py`) — a
useful sanity check on the implementation and a clean illustration of
"Cornish-Fisher generalises Gaussian, it doesn't replace it".

**Assumption A-VaR3: the expansion is a valid local correction.** *What
breaks if violated:* the Cornish-Fisher expansion is only a good
approximation for *mild* skew/kurtosis. At extreme confidence levels or
with heavily fat-tailed data, the implied quantile function can become
non-monotonic and the "corrected" VaR can actually be worse than the
plain Gaussian number. It is also unstable when skew/kurtosis are
themselves estimated from a small sample (both are higher-moment
statistics with high sampling variance) — a Cornish-Fisher VaR computed
on 30 days of data is dominated by estimation noise in `s` and `k`, not
signal.

**Alternative not implemented: full non-parametric extreme value theory
(EVT / Peaks-over-Threshold with a fitted Generalized Pareto
Distribution).** Models the tail directly using only the observations
beyond a high threshold, with theoretical justification (Pickands–
Balkema–de Haan theorem) for *why* a GPD is the right tail shape
regardless of the parent distribution. This is the principled way to
extrapolate beyond the observed sample (e.g. 99.9% VaR from 5 years of
daily data, where the empirical quantile is only 1-2 observations).
Trade-off: threshold selection is itself a modelling decision with no
single right answer, and small tail samples make the GPD shape parameter
noisy — exactly the instability Cornish-Fisher has, but with a much
larger machinery cost to fix it properly.

## 4. Expected Shortfall

`ES = -mean(returns | returns <= -VaR_historical(confidence))`.

**Why include it.** VaR answers "where does the tail start"; ES answers
"how bad is it once you're in the tail". Two portfolios can have
identical VaR and very different ES (a cliff-edge loss distribution vs a
long, thin tail) — VaR alone cannot distinguish them, which is precisely
why post-Basel III/FRTB regulatory capital has shifted from VaR to ES.

**Why ES is *coherent* and VaR is not.** A risk measure is coherent if it
is sub-additive: `risk(A+B) <= risk(A) + risk(B)` (diversification never
increases risk). VaR can violate this for non-elliptical distributions
(a classic textbook counterexample combines two assets with a small
probability of a large joint loss); ES is provably sub-additive for any
distribution. This project is single-asset, so sub-additivity across
positions is not directly demonstrated here, but it is the standard
argument for preferring ES over VaR as a portfolio risk *limit* — see
`docs/DESK_GUIDE.md`.

**Assumption:** same representativeness assumption as historical VaR
(§3.1), plus ES additionally *averages over the tail*, which can be very
few observations at 99% confidence in a short sample — a 250-day sample
has only ~2-3 observations beyond the 99% historical VaR threshold, so
the ES estimate itself is noisy even when VaR looks stable. Property-
tested in `tests/test_var_es.py` (`ES >= VaR` always holds by
construction, for any sample).

## 5. Sharpe and Sortino ratios

`Sharpe = mean(excess) / std(excess) * sqrt(252)`,
`Sortino = mean(excess) / std(excess[excess<0]) * sqrt(252)`, where
`excess = returns - rf_daily` and `rf_daily = (1+rf_annual)^(1/252) - 1`.

**Why both.** Sharpe penalises all volatility, upside and downside alike;
Sortino penalises only downside deviation, which better matches how most
investors actually experience risk (nobody complains about volatile
gains). Shipping both, on the same data, shows concretely how much the
choice matters when the return distribution is asymmetric (tested in
`tests/test_performance.py`: Sortino > Sharpe when losses are tighter
than gains).

**Assumption A-Perf1: constant risk-free rate.** *What breaks if
violated:* a single `rf_annual` scalar is a simplification; a real T-bill
series moves, and using a stale constant misprices the excess return,
especially over multi-year samples spanning a hiking/cutting cycle.

**Assumption A-Perf2: `sqrt(252)` annualisation requires i.i.d.
returns.** *What breaks if violated:* positive serial correlation
(momentum, or smoothed/stale marks in illiquid assets) inflates the
annualised ratio above the true risk-adjusted return; negative serial
correlation (mean reversion) deflates it. Neither ratio corrects for
autocorrelation.

**Alternative not implemented: autocorrelation-adjusted (Lo 2002)
Sharpe ratio,** which scales by an effective sample size that accounts
for serial correlation in returns rather than assuming i.i.d. `sqrt(252)`
scaling. More correct for autocorrelated return series (e.g. anything
with stale marks or momentum), at the cost of estimating the
autocorrelation structure itself — another noisy, small-sample-unstable
input.

## 6. Jarque-Bera normality test

`JB = n/6 * (S^2 + K^2/4)`, asymptotically `chi-squared(2)` under the
null of normality, where `S` is sample skewness and `K` is sample excess
kurtosis.

**Why include it.** It is the standard formal test for "should I trust a
Gaussian assumption here", and it directly targets the two moments
(skew, kurtosis) that Gaussian VaR ignores and Cornish-Fisher VaR
corrects for — it is the diagnostic that justifies (or doesn't) using
§3.2 over §3.1/§3.3.

**Assumption A-JB1: the chi-squared(2) reference distribution is
asymptotic.** *What breaks if violated:* in small samples (tens to low
hundreds of observations) the test has poor size (over-rejects the null
more often than the nominal 5%) and low power — a rejection on a 40-day
sample should be treated with real caution, not as decisive evidence.

**Alternative not implemented: Shapiro-Wilk or Anderson-Darling.**
Shapiro-Wilk has better power in small samples and does not rely on an
asymptotic approximation; Anderson-Darling weights the tails more heavily,
which is arguably more relevant for a VaR/ES use case where the tails are
exactly what's being measured. Jarque-Bera is used here because it is the
test conventionally paired with a skewness/kurtosis-based VaR correction
(Cornish-Fisher) — the diagnostic and the correction share the same two
input moments, which makes the story in the report self-consistent.

---

## 7. Backtesting: does the VaR number actually hold up?

Every number above is an *estimate of a quantile*. The only way to know
whether the estimate is any good is to count how often reality fell below
it. At 99% confidence you expect an exception (a realised loss worse than
the VaR) on about 1% of days; materially more means the model understates
risk, materially fewer means it wastes risk budget.

`eq_risk_metrics.backtest` implements **Kupiec's proportion-of-failures
(POF) test**: under the null that the model is correctly calibrated,
exceptions are i.i.d. Bernoulli(`p = 1 - confidence`), and the
likelihood-ratio statistic

`LR_POF = -2 ln[ p^x (1-p)^(n-x) / (x/n)^x (1-x/n)^(n-x) ]`

is asymptotically `chi-squared(1)`. It compares the assumed exception rate
against the rate the data actually delivered.

**Why Kupiec first.** It tests the one property the confidence level
literally promises (unconditional coverage), it needs nothing but a
sequence of returns and the VaR that was quoted against them, and its
statistic has a closed form that can be checked by hand — which is exactly
what `tests/test_backtest.py` does, against an independently computed
value. On the bundled data it delivers the project's sharpest result: the
Gaussian 99% VaR is **rejected** (60 exceptions against 25.2 expected)
while the historical one is not, turning "fat tails make Gaussian VaR
optimistic" from an assertion into a measurement.

**Assumption A-BT1: the chi-squared(1) reference distribution is
asymptotic.** *What breaks if violated:* on the ~250-day window a desk
typically uses, a 99% VaR generates only ~2.5 expected exceptions, and the
test has very low power — it fails to reject models that are badly wrong.
Pinned as a test: at `n=250`, **twice** the nominal exception rate is not
rejected at 5%. Do not read a Kupiec pass on a one-year window as evidence
the model is good; read a Kupiec *failure* as strong evidence it is bad.

**Assumption A-BT2: exceptions are independent.** *What breaks if
violated:* the test counts exceptions and ignores when they happened, so a
model that produces every one of its exceptions in a single stressed
fortnight — the signature of a model blind to volatility regimes — passes
Kupiec with a perfect score. Demonstrated directly in
`test_kupiec_is_blind_to_clustering_by_construction`, which shows five
spread-out and five consecutive exceptions producing an identical
statistic.

**Alternative not implemented: Christoffersen's independence and
conditional-coverage tests.** Christoffersen models the exception sequence
as a two-state Markov chain and tests whether the probability of an
exception depends on whether yesterday was an exception; the conditional
coverage test combines that with Kupiec's into a single `chi-squared(2)`
statistic. This is the natural completion of the backtest and the reason
this module documents itself as covering only half the problem. It is not
implemented here because it wants a *rolling out-of-sample* VaR forecast to
be interesting — which is a backtest engine, i.e. the companion
`python/equity/03-var-es-engine` project, not a metrics kernel.

**Alternative not implemented: Basel's traffic-light approach.** Rather
than a p-value, count exceptions over 250 days and bucket the result into
green (0-4), amber (5-9) or red (10+), with a capital multiplier attached
to the bucket. It is a blunt instrument by design — it exists precisely
because the asymptotic tests have so little power at n=250 — and it is a
supervisory capital rule rather than a statistical test, so it belongs in
a regulatory-reporting layer, not here.

**Alternative not implemented: ES backtests.** Expected Shortfall is
notoriously harder to backtest than VaR because it is not *elicitable*:
there is no scoring function whose minimiser is the ES, so there is no
direct analogue of "count the exceptions". The practical approaches
(Acerbi-Székely's Z-tests, or jointly backtesting the (VaR, ES) pair,
which *is* jointly elicitable) need the full forecast distribution rather
than a single number, and are out of scope for this project even though ES
is computed here. Worth knowing before anyone claims ES is strictly better
than VaR: it is the better *risk measure* and the worse *testable* one.

---

## 8. Assumptions register (summary table)

| # | Assumption | What breaks if violated |
|---|---|---|
| A1 | Returns are i.i.d. for `sqrt(T)` scaling (full-sample vol, Sharpe/Sortino annualisation). | Real returns cluster (autocorrelated squared returns) and can be serially correlated in levels; `sqrt(252)` over/understates true annualised risk/return. Directly motivates rolling/EWMA vol (§2) and is the reason a single unconditional vol number is a blunt instrument. |
| A2 | Historical VaR/ES: the sample window is representative of the future. | Calm sample understates forward risk (pre-2008-style complacency); stressed sample overstates it once the regime normalises. Small samples make the empirical quantile (and especially the ES tail average) noisy — quantified with a 5-day example in `docs/VALIDATION.md`. |
| A3 | Gaussian VaR: returns are exactly Normal(mu, sigma). | Excess kurtosis (fat tails) in real daily equity returns means Gaussian VaR understates 99%+ tail risk — the gap is measured directly against historical VaR in `docs/VALIDATION.md`. |
| A4 | Cornish-Fisher VaR: skew/kurtosis corrections are a valid *local* expansion. | Breaks down (non-monotonic implied quantile, worse than plain Gaussian) at extreme confidence or heavy fat tails; unstable when `s`,`k` are estimated from small samples. |
| A5 | EWMA: `lambda=0.94` is a reasonable fixed decay for this asset — no mean reversion, no fitted persistence. | A true GARCH(1,1) with mean reversion (see companion `02-volatility-modeling` project) would adapt the persistence to the data instead of assuming it; EWMA volatility can drift arbitrarily far from any long-run average since it has no `omega` term pulling it back. |
| A6 | Constant risk-free rate for Sharpe/Sortino. | A real T-bill series moves; a stale constant misprices excess return, particularly over samples spanning a rate-hiking/cutting cycle. |
| A7 | Jarque-Bera's chi-squared(2) reference is asymptotically valid. | Poor size/power in small samples (tens to low hundreds of observations) — a rejection (or non-rejection) on a short sample is weaker evidence than the p-value alone suggests. |
| A8 | Kupiec backtest: chi-squared(1) is asymptotically valid and exceptions are independent. | Very low power at the ~250-day window a desk actually uses (twice the nominal exception rate is not rejected at 5%), and complete blindness to exception clustering — a model that fails only during regime shifts passes. Needs Christoffersen's independence test (§7) to complete. |
| A9 | Single asset, no portfolio effects. | Real books hold multiple correlated positions; portfolio VaR/ES require a covariance (or copula) model and are *not* simply the sum of single-asset VaRs (in fact ES sub-additivity guarantees the portfolio ES is no worse than the sum — see §4). Out of scope here; see `python/equity/03-var-es-engine`. |

## 9. Decision rule: which VaR method to trust, when

- **Reporting/limits at 95%, well-behaved sample:** any of the three
  should roughly agree; use historical as the primary number since it
  makes no shape assumption.
- **Reporting/limits at 99%+ on daily equity returns:** trust historical
  or Cornish-Fisher over Gaussian — the fat-tail gap is largest exactly
  where it matters most for capital.
- **Very short sample (< ~60 days):** treat all VaR numbers as
  provisional; the historical quantile is a handful of points and
  Cornish-Fisher's skew/kurtosis inputs are themselves unreliable at that
  size. Prefer widening the window or falling back to a longer-history
  vol estimate scaled onto a parametric quantile.
- **Known regime change (EWMA vol far from full-sample vol):** none of
  the three VaR methods here adapts the *quantile shape* to the new
  regime automatically — only the Gaussian/Cornish-Fisher `sigma` term
  does, if you feed it a current (EWMA) volatility rather than the
  full-sample one. This project's `var_parametric`/`var_cornish_fisher`
  take `mu`/`sigma` from the same sample as everything else; a
  regime-aware desk would rescale VaR using current EWMA/GARCH vol before
  reporting it (this is exactly what filtered historical simulation and
  GARCH-based VaR, §3.1, do properly).
