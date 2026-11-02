# Pre-registration: impact calibration on LOBSTER sample days

Status: DRAFT, blocked on data. The LOBSTER sample files sit behind the
LOBSTER terms of use; the owner must accept them and place the files by hand
(see `data/README.md`). Everything above the Signature heading is hashed.

## Hypothesis

H1: the square-root law fits the temporary impact of marketable volume on
the LOBSTER sample days better than a linear law, and the fitted
coefficient lies in the range the published literature reports
(Almgren, Thum, Hauptmann and Li 2005, Risk 18(7)).

H0: the linear law fits at least as well (out-of-sample mean squared error).

## Data

- LOBSTER free sample files, level 1 order book plus messages, all tickers
  of the 2012-06-21 sample set. These dates are outside the 2022-2025
  hold-out; the hold-out rule does not bind this study, but the signature
  does: no fit is run before signing.

## Parameters (fixed)

- Buckets of 5 minutes, 09:35 to 15:55. Signed marketable volume = executed
  visible volume (message types 4 and 5) by direction.
- Impact = mid-price change over the bucket in units of the bucket's
  average spread. Participation = |signed volume| / bucket volume.
- Fit on the first half of each day's buckets, score on the second half.
- Models: impact = a x participation (linear) and
  impact = b x sign x sqrt(participation) (square root), least squares.

## Costs

Not applicable.

## Primary metric

Out-of-sample MSE ratio, square root / linear, pooled across tickers.

## Decision rule

H0 is rejected if the ratio is below 1 in a paired block bootstrap 95 %
interval. The fitted coefficients feed `python/equity/08-algo-execution`
only after this study's result file passes the registration check.

## Signature

signed_by:
signed_on:
sha256:
