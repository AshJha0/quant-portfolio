# Pre-registration: G10 FX pairs v2, level-based z, forward hold-out

Status: DRAFT, unsigned. The coordinator reviews and signs. Everything above
the Signature heading is hashed; editing any of it changes the registration
hash and voids a signature.

## Why a second pairs registration

The signed pairs registration (`pairs-fx-g10.md`) trades a Kalman one-step
z. Its time-varying intercept follows the spread and absorbs it, so |z|
seldom reaches 2: in training it trades in 3 of 19 test years, and on
simulated pairs that are cointegrated by construction it is above 2 on
0.3 % of test days or fewer, and never at a JPY-like price level (power
check below). A test that cannot see a true edge tells us nothing. This
registration keeps the signed selection rules and replaces the signal with
a level-based z. Its hold-out is forward data, so it does not touch the
2022-2025 window that belongs to the signed registrations.

## Hypothesis

H1: the pairs strategy below earns a positive net Sharpe on ECB fixes dated
from 2026-11-03 onward.

H0: its net annualised Sharpe over the evaluation window is at most zero.

## Data

- ECB euro foreign exchange reference rates (EXR, daily, 14:15 CET fix),
  converted to USD per unit: AUD, CAD, CHF, EUR, GBP, JPY, NOK, NZD, SEK.
  DKK is left out (pegged to EUR). Free, no sign-up: data-api.ecb.europa.eu.
- Training period: 1999-01-04 to 2021-12-31, the only real data this draft
  has scored.
- Hold-out (scored): forward. Every ECB fix dated 2026-11-03 or later, from
  the forward log (`research/forward/SPEC.md`).
- Formation history (never scored): one ECB file covering 1999-01-04 to
  2026-11-02, fetched once at signing, after the signed registrations'
  2022-2025 hold-out has been run, and stored with its SHA-256 recorded in
  the signing commit. It supplies the formation window of fold 1 and of
  any later fold whose window reaches back before 2026-11-03. Formation
  windows are contiguous ECB business days drawn from this file followed by
  the forward log; a fix dated before 2026-11-03 is never scored. Why
  contiguous rather than history to 2025-12-31 plus forward data: a
  10-month hole inside a 3-year window would put a 10-month price jump into
  the cointegration tests and the residual SD. The 2026 fixes before
  signing are not a hold-out of any registration.

## Universe

All 36 pairs of the nine currencies above. No pair is added or removed after
signing.

## Parameters (fixed)

- Folds: fold 1 trades 2026-11-03 to 2027-11-02, fold 2 trades 2027-11-03 to
  2028-11-02. Each fold's formation window is the 756 ECB business days (3
  years) ending 5 business days before its first day (embargo).
- Selection, as in `pairs-fx-g10.md`: Engle-Granger cointegrated at 5 %
  (MacKinnon 2010 N=2 values) AND Johansen trace rank >= 1 at 95 % (constant
  unrestricted, 1 lagged difference), both on log prices over the formation
  window.
- Hedge: OLS log y = a + b log x + u on the formation window, held fixed for
  the fold.
- Signal (changed): z_t = (log y_t - a - b log x_t) / sd(u), with sd(u) the
  population standard deviation of the formation residual. Enter at
  |z| > 2.0 against the sign of z; exit at |z| < 0.5; flat outside the fold.
  Decided at the close of day t, earns the return of t+1; a position at the
  close of a fold's last day earns nothing.
- Weights: legs 1 / (1 + |b|) and -b / (1 + |b|), unit gross. Portfolio =
  equal weight across the pairs selected for the fold; zero return in a fold
  with no pair selected.
- Costs: 1 bp of gross notional traded per leg. Spot only, no carry.
- Code: `research/studies/forward_studies.py: pair_returns_level`,
  `select_pairs`.

## Simulated power check (synthetic data, done before signing)

`research/studies/pairs_power.py`, result `research/power/pairs_v2_power.json`,
200 replications per cell, 3 independent ground-truth pairs per replication,
756 formation days and 504 test days, log y = c + 0.8 log x + s, s an AR(1)
with stationary SD 1 %, log x a random walk (daily SD 0.6 %) at a EUR-like
level (log price 0.1) or a JPY-like level (USD per JPY 0.0067, log price
-5.0), 1 bp per leg, decision rule as below.

| Truth | x level | Selected | Level z: reject H0 | Level z: days with abs(z) > 2 | Kalman v1: reject H0 | Kalman v1: days with abs(z) > 2 |
|---|---|---|---|---|---|---|
| half-life 10 d | EUR-like | 99.8 % | 100 % | 8.6 % | 47 % | 0.28 % |
| half-life 10 d | JPY-like | 99.8 % | 100 % | 8.9 % | 0 % | 0.00 % |
| half-life 30 d | EUR-like | 46.5 % | 54 % | 20.5 % | 0.5 % | 0.01 % |
| half-life 30 d | JPY-like | 40.3 % | 54 % | 19.5 % | 0 % | 0.00 % |
| no cointegration | EUR-like | 2.5 % | 0.5 % | 64 % | 0 % | 0.15 % |
| no cointegration | JPY-like | 4.2 % | 0 % | 65 % | 0 % | 0.00 % |

Days with abs(z) > 2 are counted over the test days of selected pairs. The
level z rejects H0 when a real edge is there and keeps its size under 1 %
when none is; the Kalman signal of the signed design has little power at a
EUR-like level and none at a JPY-like one. These are simulations at chosen
parameters: they show the rule can see an edge of this size, not that the
real pairs have one.

## Training result (1999-2021, not out-of-sample evidence)

Same walk-forward calendar-year folds as the signed design (test years
2003-2021): net annualised Sharpe -0.107, Lo SE 0.234, block-bootstrap 95 %
interval [-0.583, 0.288], trades in 17 of 19 years. Deflated Sharpe 0.036
with N = 7. File: `research/results/pairs_fx_g10_v2_train.json`. The
training period weighs against H1; the registration exists to let forward
data answer it with a signal that can register a result.

## Trials

Trial count 7, every pairs variant scored on this training data: the 3
variants tried while drafting `pairs-fx-g10.md`, 3 further Kalman
observation-variance rules tried for this draft first (0.25 and 0.1 times
the formation residual variance, and the variance of its daily change;
recorded under `superseded_kalman_menu` in the result file), and this
level-z design. The level-z design was adopted on the reviewer's power
finding, not on its training Sharpe, which is lower than three of the
Kalman variants.

## Primary metric

Annualised net Sharpe of the daily portfolio return over the evaluation
window, with the Lo (2002) Newey-West standard error and a circular
block-bootstrap 95 % interval (reported). Deflated Sharpe (Bailey and
Lopez de Prado 2014) reported with N = 7 and V = 1/(T - 1), as in DR-R4-2.

## Evaluation window and date

- Window: every ECB business day from 2026-11-03 to 2028-11-02 inclusive
  (24 months). Why 24 and not 12: with about 252 daily returns the standard
  error of an annualised Sharpe is close to 1, so a 12-month test could
  only detect a true Sharpe above about 1.6; 24 months brings that to about
  1.2. Still low power, and stated as such.
- Evaluation date: on or after 2028-11-13, once the fix for 2028-11-02 is
  in the forward log. An interim 12-month figure may be published as
  descriptive only; it decides nothing and is labelled so.
- Scored once. A second look at the same window is not a test.

## Decision rule

H0 is rejected if Sharpe - 1.645 x Lo SE > 0 over the evaluation window
(the one-sided 95 % bound, as in DR-R4-1). The lower end of the bootstrap
interval is reported beside it and does not decide. If no pair trades in
the window, H0 is not rejected. The number is published whatever it is.

## Secondary (reported, not tested)

Pairs selected per fold, entries, days in the market, turnover, per-fold
net return.

## Signature

signed_by:
signed_on:
sha256:
