# Desk Guide — how a real desk uses this library

Contract item 5: daily workflow, consumers of the numbers, controls and
limits, governance. Written for an equity vol / index derivatives / risk
context.

---

## 1. What the numbers feed

**Option pricing & marking.** The model vol term structure
(`term_structure()`, annualised average vol to each horizon) is the model leg
compared against implied vol: `implied − forecast` is the carry signal for
short-vol/variance-risk-premium strategies, and the model mark is the
fallback for illiquid expiries and names with no options market. A GJR term
structure after a sell-off is upward-sloping from an elevated short end —
exactly the shape the listed market shows — while EWMA's flat structure
(`ewma_forecast`, structurally flat, documented why) cannot price calendar
effects at all.

**VaR / ES.** 1-day 99% VaR = `sigma_{t+1|t} · q_{0.01}` from
`rolling_one_step_forecasts` + the fitted innovation distribution. Use
`dist="t"` for the quantile: Gaussian tails understate 1% VaR by ~10% when
returns are t(8) (VALIDATION.md F3). Backtest VaR exceptions monthly;
clustered exceptions point at F1 (structural break), not bad luck.

**Vol targeting.** A vol-target book scales exposure by
`target_vol / forecast_vol`. Forecast quality maps directly to realised
tracking error and to turnover: the QLIKE race in VALIDATION.md §4 is the
right metric because QLIKE penalises *under*-forecasting vol (the expensive
error — over-levered into a storm) more than over-forecasting.

**Dispersion / relative value.** Cross-sectional differences in fitted
leverage (`gamma`) and persistence feed single-name vs index dispersion
weighting; the news impact curve quantifies how much extra vol a −2σ day
buys you in each name.

**Risk limits.** Persistence and half-life (`extra["halflife_days"]`) tell
the risk manager how long an elevated-vol episode is expected to last —
i.e., how long a de-risking flag should stay on after a shock (fitted
half-lives of ~14–19 days on realistic parameters).

## 2. Daily workflow (recommended cadence)

1. **Every day, before the open:** update return series; run the variance
   recursions with *yesterday's parameters* (`rolling_one_step_forecasts`
   does exactly this between refits — recursion daily, refit sparse).
   Publish `sigma_{t+1|t}` per name plus the 1/5/21/63-day term structure.
2. **Weekly-to-monthly (refit_every ≈ 5–25):** re-estimate parameters.
   Parameters move slowly; daily refitting adds noise and mark instability
   without forecast gain (the pipeline's race uses 25 days). Alert on:
   parameter jump > 3 SE vs previous fit, persistence > 0.99, `nu` < 5,
   `ConvergenceError`.
3. **Monthly:** rolling forecast evaluation — QLIKE per model vs the EWMA
   benchmark with a DM test (`forecast_race_table`). A model that cannot
   beat EWMA(0.94) out of sample should not be in production.
4. **After a suspected regime break:** shorten the window (`scheme="rolling"`)
   or refit from the break; watch the EWMA-vs-GARCH gap — EWMA repricing much
   faster than the fitted GARCH is the classic break signature
   (VALIDATION.md F1 case study).

## 3. Choice of realized proxy

Squared close-to-close returns are unbiased but noisy — fine for QLIKE-based
model ranking (Patton-robust), poor for eyeballing. For monitoring dashboards
prefer the range estimators (`parkinson`, `garman_klass`,
`rogers_satchell`): ~5–7x lower sampling variance from the same daily bars.
Use Rogers–Satchell when names trend hard (drift-robust); be aware all range
estimators read slightly low under discrete/illiquid trading and slightly
high under wide bid–ask bounce. If a clean 5-minute realized-variance feed
exists, plug it into the same evaluation harness as the proxy — nothing else
changes.

## 4. Model governance & benchmarking

* **Benchmark chain, never a single model:** rolling historical (sanity),
  EWMA(0.94) (regulator-recognised, parameter-free), GARCH (mean reversion),
  GJR/EGARCH (asymmetry). Escalation between steps must be justified by a DM
  test on QLIKE, out of sample — this is exactly the committed pipeline, so
  the promotion criterion is reproducible.
* **Independent implementation check:** the from-scratch implementation is
  cross-validated against the `arch` package to machine precision
  (VALIDATION.md §2) — the same procedure model validation teams apply to
  front-office pricers ("re-implement and reconcile").
* **Every fit is auditable:** `VolatilityFitResult` carries parameters, SEs,
  log-likelihood, AIC/BIC, convergence flag and message; failures raise —
  a silent bad fit cannot enter the marks.
* **Change control:** synthetic-data recovery tests (known truth) are the
  regression suite; any code change that shifts a fitted parameter by more
  than optimiser noise fails CI.

## 5. Realistic scenarios

* **Crash day (−7% on an index):** GJR conditional vol jumps by
  `(alpha+gamma)·r²` — with fitted (0.039, 0.113) that is a ~4x variance
  spike from a −7% day at 15% base vol; the term structure inverts (short end
  above long end) and 1-day VaR roughly doubles overnight. EWMA reacts by
  only `(1−lambda) r²` (6% weight) — slower on day one, which is precisely
  why desks keep both on screen.
* **Grinding rally (many small up days):** with `gamma > 0`, GJR vol decays
  toward the unconditional level *faster* than symmetric GARCH would suggest,
  supporting short-vol carry sizing; the sign-bias diagnostic confirms the
  asymmetry is real rather than fitted noise.
* **Quiet regime, vol at multi-year lows:** persistence keeps the k-step
  forecast *above* spot vol (mean reversion upward — the term structure in
  the pipeline rises from 11.6% to 14.8%); selling long-dated vol at the
  1-day rate is the classic error this prevents.
* **New listing / short history (< 100 obs):** fitters refuse; fall back to
  EWMA with a prior-informed initial variance, or borrow parameters from a
  peer name (variance targeting pins the level to the name's own sample
  variance with `variance_targeting=True`).
* **Data gap / NaN in the feed:** everything raises immediately
  (`ValueError`) — the desk fixes the feed rather than shipping marks
  computed off a silently shortened series.
