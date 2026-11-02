# Pre-registration: G10 FX pairs, cointegration with a Kalman hedge ratio

Status: DRAFT, awaiting the coordinator's signature (owner decision 9).
Everything above the Signature heading is hashed; editing any of it changes
the registration hash and voids a signature.

## Hypothesis

H1: a pairs strategy on G10 currencies against USD, selected by Engle-Granger
and Johansen tests on a rolling formation window and traded with a Kalman
hedge ratio, earns a positive net Sharpe out of sample.

H0: the net annualised Sharpe on the hold-out is at most zero.

## Data

- ECB euro foreign exchange reference rates (EXR, daily, 14:15 CET fix),
  converted to USD per unit: AUD, CAD, CHF, EUR, GBP, JPY, NOK, NZD, SEK.
  DKK is left out (pegged to EUR). Source: data-api.ecb.europa.eu, fetched
  by `data/fetch/qp_fetch.py`; the file's SHA-256 is stored beside it.
- Training period: 1999-01-04 to 2021-12-31.
- Hold-out: 2022-01-01 to 2025-12-31. No run touches it before this file is
  signed.

## Universe

All 36 pairs of the nine currencies above. No pair is added or removed after
signing.

## Parameters (fixed)

- Walk-forward by calendar year. Formation window: the 3 calendar years
  before the test year, ending 5 business days before it (embargo).
- Selection: Engle-Granger cointegrated at 5 % (MacKinnon 2010 N=2 values)
  AND Johansen trace rank >= 1 at 95 % (constant unrestricted, 1 lagged
  difference). Both on log prices.
- Hedge: Kalman random-walk (alpha, beta), delta = 1e-6, observation
  variance = the variance of the OLS residual of log y on log x over the
  formation window, diffuse prior, run from the start of the formation window.
- Signal: z = one-step prediction error / its forecast standard deviation.
  Enter at |z| > 2.0 against the sign of z; exit at |z| < 0.5; flat outside
  the test year. Decided at the close of day t, earns the return of t+1.
- Weights: pair legs scaled to unit gross notional, 1 / (1 + |beta|).
  Portfolio = equal weight across the pairs selected for that year; zero
  return in a year with no pair selected.

## Costs

1 bp of gross notional traded per leg. No carry (spot only). This is a
known simplification; a carry-inclusive variant would be a new registration.

## Primary metric

Annualised net Sharpe of the daily portfolio return over the test years,
with the Lo (2002) Newey-West standard error and a circular block-bootstrap
95 % interval. Trial count: 3 (training variants tried while drafting:
observation variance 1e-3 with delta 1e-4; estimated variance with delta
1e-4; this one). Deflated Sharpe reported with that count.

## Decision rule

H0 is rejected if the hold-out net Sharpe has a one-sided 95 % lower bound
above zero. The hold-out number is published whatever it is.

## Secondary (reported, not tested)

Number of pairs selected per year, turnover, per-year net return.

## Signature

signed_by:
signed_on:
sha256:
