# Pre-registration: G10 FX cross-sectional spot momentum, forward hold-out

Status: DRAFT, unsigned. The coordinator reviews and signs. Everything above
the Signature heading is hashed; editing any of it changes the registration
hash and voids a signature.

## Hypothesis

H1: a monthly cross-sectional momentum strategy on the nine G10 currencies
against USD (long the three with the highest past 63-day spot return, short
the three with the lowest) earns a positive net Sharpe on ECB fixes dated
from 2026-11-03 onward.

H0: its net annualised Sharpe over the evaluation window is at most zero.

The literature finds currency momentum, with most of the effect in excess
returns and in less liquid currencies (Menkhoff, Sarno, Schmeling and
Schrimpf 2012, "Currency momentum strategies", J. Financial Economics
106(3), 660-684). Nine liquid G10 currencies on spot alone is a hard case
for it. The training result below is negative; this registration exists so
that the forward data, not a re-tuned rule, answers the question.

## Data

- ECB euro foreign exchange reference rates (EXR, daily, 14:15 CET fix),
  converted to USD per unit: AUD, CAD, CHF, EUR, GBP, JPY, NOK, NZD, SEK.
  Free, no sign-up: data-api.ecb.europa.eu. No interest-rate data: the
  strategy is spot only. A carry-inclusive version would be a new
  registration.
- Training period: 1999-01-04 to 2021-12-31 (the only data this draft has
  scored; first scored year 2000, after a full lookback).
- Hold-out: forward. Every ECB fix dated 2026-11-03 or later, as recorded
  in the forward log (`research/forward/SPEC.md`), with the SHA-256 of the
  file each fix came from. The forward computation reads no fix dated
  before 2026-11-03, so it never touches the 2022-2025 hold-out of the
  signed registrations.

## Parameters (fixed, one variant)

- Rebalance at the close of the last ECB business day of each month.
- Score = log spot return (USD per unit) over the previous 63 fixes.
- Long the 3 highest scores, short the 3 lowest, 1/6 of unit gross each;
  the middle 3 get zero. Ties: the
  alphabetically earlier currency code ranks lower.
- Weights at the close of day t earn the log return of day t+1 and hold
  until the next rebalance.
- Cost: 1 bp per unit of gross weight traded, including the first build.
- Forward start: the first rebalance is the first month end with at least
  64 logged fixes dated 2026-11-03 or later (expected 2027-02-26). Before
  it the position is flat and nothing is scored.

Chosen before any data was looked at, from the paper's design (rank on past
returns, long winners, short losers, monthly rebalance) with a 3-month
lookback, its middle formation length. No other variant was run. Trial
count: 1.

## Training result (1999-2021, not out-of-sample evidence)

Net annualised Sharpe -0.200, Lo SE 0.196, block-bootstrap 95 % interval
[-0.589, 0.166], 5,631 days; positive net return in 6 of 22 years. File:
`research/results/momentum_fx_g10_train.json`.

## Primary metric

Annualised net Sharpe of the daily return over the evaluation window, with
the Lo (2002) Newey-West standard error and a circular block-bootstrap 95 %
interval (reported). Deflated Sharpe reported with N = 1.

## Evaluation window and date

- Window: every business day with an ECB fix from the day after the first
  rebalance to 2029-02-28 inclusive (24 months if the first rebalance is
  2027-02-26). With about 500 daily returns the one-sided test can detect a
  true annualised Sharpe above about 1.2 and no smaller; stated as low power.
- Evaluation date: on or after 2029-03-12, once the fix for 2029-02-28 is in
  the forward log. An interim 12-month figure may be published as
  descriptive only.
- Scored once.

## Decision rule

H0 is rejected if Sharpe - 1.645 x Lo SE > 0 over the evaluation window.
The lower end of the bootstrap interval is reported beside it and does not
decide. The number is published whatever it is.

## Secondary (reported, not tested)

Per-month holdings, turnover, per-month net return, and the long-leg and
short-leg returns separately.

## Signature

signed_by:
signed_on:
sha256:
