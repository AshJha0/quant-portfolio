# Desk Guide: using eq_vol on an equity vol or risk desk

Daily workflow, consumers of the numbers, controls and governance. All
numbers quoted here come from synthetic data via `python examples/run_pipeline.py`.

---

## 1. What the numbers feed

**Option pricing and marking.** The model term structure (`term_structure()`,
annualised average vol to each horizon) is compared with implied vol:
`implied - forecast` is the carry signal for short-vol strategies, and the
model mark is a fallback for illiquid expiries. A GJR term structure after a
sell-off slopes down from an elevated short end, as listed markets do. EWMA's
term structure is flat by construction and cannot price calendar effects.

**VaR / ES.** 1-day 99% VaR = `sigma_{t+1|t} q_{0.01}` from
`rolling_one_step_forecasts` and the fitted innovation distribution. Use
`dist="t"` for the quantile: Gaussian tails understate 1% VaR by about 10%
when returns are t(8) (VALIDATION.md F3). Backtest exceptions monthly;
clustered exceptions point at a structural break (F1).

**Vol targeting.** A vol-target book scales exposure by
`target_vol / forecast_vol`. QLIKE is the right race metric because it
penalises under-forecasting (over-levered into a storm) more than
over-forecasting.

**Dispersion and relative value.** Cross-sectional differences in fitted
leverage (`gamma`) and persistence feed single-name vs index weighting. The
news impact curve gives the extra vol a -2 sigma day buys in each name.

**Risk limits.** Half-life (`extra["halflife_days"]`) says how long an
elevated-vol episode is expected to last, so how long a de-risking flag
should stay on. On the pipeline's synthetic data the fitted half-lives are
17 to 19 days.

## 2. Daily workflow

1. **Every day before the open:** update returns and run the variance
   recursions with the last fitted parameters (`rolling_one_step_forecasts`
   does this between refits). Publish `sigma_{t+1|t}` and the
   1/5/21/63-day term structure.
2. **Every 5 to 25 days:** re-estimate. Parameters move slowly; daily
   refitting adds mark noise without forecast gain. Alert on: a parameter
   moving by more than 3 **robust** standard errors
   (`robust_std_errors`, Bollerslev-Wooldridge) from the previous fit,
   `extra["at_boundary"]`, persistence > 0.99, `nu` < 5, or
   `ConvergenceError`. Classic SEs understate uncertainty by about a third
   on fat-tailed returns (VALIDATION.md section 2) and would fire the alert
   too often.
3. **Monthly:** QLIKE per model vs the EWMA benchmark with a HAC
   Diebold-Mariano test (`forecast_race_table`; Newey-West lag
   `floor(4 (n/100)^(2/9))` by default). A model that cannot beat EWMA(0.94)
   out of sample should not be used.
4. **After a suspected break:** shorten the window (`scheme="rolling"`) or
   refit from the break. EWMA repricing much faster than the fitted GARCH is
   the usual break signature (VALIDATION.md F1).

## 3. Choice of realised proxy

Squared close-to-close returns are unbiased but noisy. They rank models
correctly under QLIKE but with low power. For monitoring, prefer the range
estimators (`parkinson`, `garman_klass`, `rogers_satchell`, `yang_zhang`):
5 to 7 times lower sampling variance from the same bars. Rogers-Satchell is
drift-robust; Yang-Zhang also handles opening gaps. All range estimators read
low under discrete trading and high under bid-ask bounce. A 5-minute
realised-variance feed can be passed to the same evaluation harness as the
proxy; this package does not ship one.

## 4. Model governance

* **Benchmark chain:** rolling historical, EWMA(0.94), GARCH, then GJR or
  EGARCH. Each step up must win a QLIKE DM test out of sample. The
  pipeline runs this chain on synthetic data.
* **Independent implementation check:** likelihoods agree with the `arch`
  package to round-off at identical parameters, and classic and robust SEs
  agree with arch's to about 1e-4 relative (VALIDATION.md section 2).
* **Auditable fits:** `VolatilityFitResult` carries parameters, classic and
  robust SEs, log-likelihood, AIC/BIC, convergence flag, message and the
  boundary flag. Failures raise.
* **Change control:** the recovery tests use absolute tolerances of 0.015 to
  0.03 on alpha, beta and gamma. They catch a broken estimator, not a small
  drift. The arch cross-check (1e-6 likelihood identity) and the forecast
  identities (1e-10) are the tight regression checks.

## 5. Scenarios

Fitted GJR from `run_pipeline.py` section 3: omega 3.18e-6, alpha 0.0388,
gamma 0.113, beta 0.868. Base vol 15% annualised.

| scenario | next-day variance / base | next-day vol / base |
|---|---|---|
| -7% day, GJR | 9.2x | 3.0x |
| -7% day, EWMA(0.94) | 4.2x | 2.1x |

* **Crash day (-7% on an index):** GJR variance rises by
  `(alpha + gamma) r^2`, about 9x the base variance, so 1-day VaR roughly
  triples overnight and the term structure inverts. EWMA puts weight
  `1 - lambda = 0.06` on the shock and rises about 4x. Desks keep both on
  screen for this reason.
* **Grinding rally:** with `gamma > 0` up days add little variance, so GJR
  vol decays toward the long-run level faster than symmetric GARCH implies.
* **Quiet regime:** mean reversion keeps the k-step forecast above spot vol
  (pipeline: 11.6% at 1 day to 14.8% at 252 days). Selling long-dated vol at
  the 1-day rate is the error this prevents.
* **Short history (< 100 obs):** fitters refuse. Use EWMA, or borrow
  persistence from a peer and pin the level with `variance_targeting=True`
  (target = the name's own mean squared return).
* **NaN in the feed:** everything raises `ValueError`. Fix the feed rather
  than mark off a shortened series.

## November 2026: HAR-RV in the daily workflow

Feed it daily RV from 5-minute bars. Use the 1-day and 5-day forecasts as a
benchmark beside GARCH; where intraday data exist, HAR usually does better
on QLIKE.
