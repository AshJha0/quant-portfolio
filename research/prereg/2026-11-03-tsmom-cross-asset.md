# Pre-registration: cross-asset time-series momentum, 2016-2026 hold-out

Status: DRAFT, unsigned. Not to be signed until the simulated power result
named under "Power" exists and passes the W6 gate
(`research/tools/prereg.py`). Everything above the Signature heading is
hashed; editing any of it after signing voids the signature.

## Hypothesis

H1: an equal-weight portfolio of time-series momentum positions across the
assets listed under Data earns a positive mean monthly excess return over
the hold-out 2016-01-01 to 2026-12-31.

H0: its mean monthly return over the hold-out is at most zero.

Known premium (Moskowitz, Ooi and Pedersen 2012, JFE 104(2); Hurst, Ooi and
Pedersen 2017, JPM 44(1)). A rejection reproduces the literature on this
asset set; it is not a discovery. Economic prior: under-reaction then
over-reaction to news, and hedgers paying to lay off risk.

## Data

- FX: monthly financed excess returns against USD for the currencies live
  in `research/universe/fx.csv` at each month end, from
  `research/panel/signals.py::monthly_excess_returns` (ECB/FRED spot, the
  panel's point-in-time short rates; CIP, calendar-day accrual).
- US equity: `Mkt-RF` from the Kenneth French library
  (`F-F_Research_Data_Factors`, monthly), file SHA-256 recorded at run time.
- Training period: every month before 2016-01-01 (the only data seen).
- Hold-out (scored): 2016-01-01 to 2026-12-31, opened once through
  `research/panel/holdout_lock.py` with this registration.

## Parameters (fixed, one variant)

`research/tsmom/tsmom.py` at the signing commit: lookback 12 months, sign
of the trailing sum, target volatility 40 % per asset, ex-ante volatility
EWMA of monthly squared excess returns with centre of mass 12 months
(minimum 12 months), portfolio the equal-weight mean over assets with a
weight. No costs in the primary metric; turnover reported. Trial count: 1.

## Primary metric and decision rule

Mean monthly portfolio return over the hold-out with a Newey-West t-statistic
(Bartlett, lag 3). H0 is rejected if t > 1.645. The number is published
whatever it is.

## Power

Required before signing: a simulated power result in `research/power/`
(planted time-series momentum at the literature's size, this rule and this
decision rule, the hold-out's month count). Not yet run.

## Secondary (reported, not tested)

Per-asset returns and Sharpe ratios, turnover, the FX-only and equity-only
sub-portfolios.

## Signature

signed_by:
signed_on:
sha256:
