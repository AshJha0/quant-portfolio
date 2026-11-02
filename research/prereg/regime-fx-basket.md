# Pre-registration: does the high-volatility FX state have a negative mean?

Status: DRAFT, awaiting the coordinator's signature (owner decision 9).
Everything above the Signature heading is hashed.

## Hypothesis

H1: in a two-state Gaussian HMM on the daily return of an equal-weight
basket long the G10 currencies against USD, the next-day mean return in the
predicted high-volatility state is negative.

H0: that mean is zero or positive. If H0 is not rejected on the hold-out,
the regime strategy is renamed a volatility-targeting overlay, and its
README says so.

## Data

- ECB EXR daily reference rates converted to USD per unit: AUD, CAD, CHF,
  EUR, GBP, JPY, NOK, NZD, SEK. Basket return = mean of the nine daily log
  returns.
- Training period: 1999-01-04 to 2021-12-31.
- Hold-out: 2022-01-01 to 2025-12-31, untouched before signing.

## Parameters (fixed)

- Expanding-window refit once a year: the model for year Y uses all days
  before 1 January of Y. First test year in training: 2003. Hold-out years
  are scored with yearly refits on all data before each year.
- Two states, Baum-Welch EM (`fx_regime.hmm.fit_hmm`), seed 0, 5 restarts.
  High-vol state = the state with the larger emission variance.
- Classification: filtered probability of the high-vol state at the close
  of day t above 0.5; the return scored is that of day t+1.

## Costs

None. This is a test of a conditional mean, not a strategy.

## Primary metric

Mean next-day basket return on high-vol days, annualised, with a
Newey-West t-statistic (Bartlett kernel, lag floor(4 (n/100)^(2/9))).
Trial count: 1.

## Decision rule

H0 is rejected if the hold-out t-statistic is below -1.645 (one-sided 5 %).

## Signature

signed_by:
signed_on:
sha256:
