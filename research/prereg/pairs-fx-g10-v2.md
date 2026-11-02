# Pre-registration: G10 FX pairs v2, observation variance chosen on training data, forward hold-out

Status: DRAFT, unsigned. The coordinator reviews and signs. Everything above
the Signature heading is hashed; editing any of it changes the registration
hash and voids a signature.

## Why a second pairs registration

The signed pairs registration (`pairs-fx-g10.md`) sets the Kalman
observation variance to the variance of the formation-window OLS residual in
levels. In training that makes the strategy almost dormant: it trades in 3
of 19 test years (`research/README.md`). This registration keeps every other
rule of that design and changes only the observation-variance rule, chosen on
training data (1999-2021) alone. Its hold-out is forward data, so it does not
touch 2022-2025, which belongs to the signed registrations.

## Hypothesis

H1: the pairs strategy below earns a positive net Sharpe on ECB fixes dated
from 2026-11-03 onward.

H0: its net annualised Sharpe over the evaluation window is at most zero.

## Data

- ECB euro foreign exchange reference rates (EXR, daily, 14:15 CET fix),
  converted to USD per unit: AUD, CAD, CHF, EUR, GBP, JPY, NOK, NZD, SEK.
  DKK is left out (pegged to EUR). Free, no sign-up: data-api.ecb.europa.eu.
- Training period: 1999-01-04 to 2021-12-31 (the only data this draft has
  scored).
- Hold-out: forward. Every ECB fix dated 2026-11-03 or later, as recorded in
  the forward log (`research/forward/SPEC.md`), which stores each fix with
  the SHA-256 of the file it came from.
- Formation windows for the forward folds use the 3 years of fixes before
  each fold, which include 2023-2025. Those fixes are part of the signed
  registrations' single-use hold-out, so the forward pairs folds are
  computed only after both `pairs_fx_g10_holdout.json` and
  `regime_fx_basket_holdout.json` exist in `research/results`. Until then
  the forward log records the momentum signal only and pairs fields are
  empty. The forward returns are recomputed from the logged fixes at
  evaluation, so a late start loses no hold-out days.

## Universe

All 36 pairs of the nine currencies above. No pair is added or removed after
signing.

## Parameters (fixed)

- Selection, hedge, signal, weights and costs exactly as in
  `pairs-fx-g10.md`: Engle-Granger at 5 % (MacKinnon 2010 N=2 values) AND
  Johansen trace rank >= 1 at 95 % (constant unrestricted, 1 lagged
  difference), both on log prices over the formation window; Kalman
  random-walk (alpha, beta), delta = 1e-6, diffuse prior, run from the start
  of the formation window; z = one-step prediction error / its forecast
  standard deviation; enter at |z| > 2.0 against the sign of z, exit at
  |z| < 0.5; decided at the close of day t, earns the return of t+1; legs
  scaled to unit gross notional, 1 / (1 + |beta|); equal weight across the
  selected pairs; zero return in a fold with no pair selected; 1 bp of gross
  notional traded per leg; spot only, no carry.
- Changed: observation variance = the variance of the first difference of
  the formation-window OLS residual of log y on log x (rule `diff` below).
- Forward folds: fold 1 trades 2026-11-03 to 2027-11-02, fold 2 trades
  2027-11-03 to 2028-11-02. Each fold's formation window is the 3 years
  before its first day, ending 5 business days before it (embargo).
  Positions are flat outside the fold; a position at the close of a fold's
  last day earns nothing.

## How the rule was chosen (training only)

The menu was fixed before the first run, in
`research/studies/forward_studies.py: VAR_RULES`, with the observation
variance as a function of the formation-window OLS residual u:

| Rule | Variance | Training net Sharpe | Lo SE | Years with trades (of 19) | Entries |
|---|---|---|---|---|---|
| levels (signed v1) | var(u) | 0.241 | 0.154 | 3 | 7 |
| levels_x0.25 | 0.25 var(u) | 0.160 | 0.198 | 15 | 57 |
| levels_x0.1 | 0.1 var(u) | 0.122 | 0.205 | 16 | 178 |
| diff | var(diff(u)) | 0.232 | 0.195 | 17 | 259 |

Walk-forward test years 2003-2021, same folds as the signed design.

The criterion as first coded was the highest training Sharpe. It picked
`levels`, the signed rule, which would make this registration a copy of the
dormant v1 design: on 3 active years in 19, a 24-month forward window would
most likely see no trade and test nothing. After seeing that, an activity
floor was added: a rule must trade in at least 75 % of the training test
years. The floor was chosen after the results were seen, so it is a
researcher choice and is disclosed here rather than hidden. Under it, three
rules qualify (15, 16 and 17 active years) and `diff` has the highest
training Sharpe. The difference between `diff` and `levels` (0.232 against
0.241) is far inside one standard error; the training data does not say
that `diff` is better, only that it trades.

## Primary metric

Annualised net Sharpe of the daily portfolio return over the evaluation
window, with the Lo (2002) Newey-West standard error and a circular
block-bootstrap 95 % interval (reported). Trial count: 6 (the four rules
above plus the two other variants tried while drafting `pairs-fx-g10.md` on
the same training data). Deflated Sharpe (Bailey and Lopez de Prado 2014)
reported with N = 6 and V = 1/(T - 1), as in DR-R4-2.

## Evaluation window and date

- Window: every business day with an ECB fix from 2026-11-03 to 2028-11-02
  inclusive (minimum 24 months). Why 24 and not 12: with about 252 daily
  returns the standard error of an annualised Sharpe is close to 1, so a
  12-month test could only detect a true Sharpe above about 1.6; 24 months
  brings that to about 1.2. Still low power, and stated as such.
- Evaluation date: on or after 2028-11-13, once the fix for 2028-11-02 is in
  the forward log. An interim 12-month figure may be published as
  descriptive only; it decides nothing and is labelled so.
- Scored once. A second look at the same window is not a test.

## Decision rule

H0 is rejected if Sharpe - 1.645 x Lo SE > 0 over the evaluation window
(the one-sided 95 % bound, as in DR-R4-1). The lower end of the bootstrap
interval is reported beside it and does not decide. If no pair trades in the
window, H0 is not rejected. The number is published whatever it is.

## Secondary (reported, not tested)

Pairs selected per fold, entries, days in the market, turnover, per-fold net
return, and the same metric for the `levels` rule on the same window, as a
descriptive comparison only.

## Signature

signed_by:
signed_on:
sha256:
