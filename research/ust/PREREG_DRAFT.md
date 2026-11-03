# Pre-registration: UST 2s10s slope and 2s5s10s fly, carry/roll-down rule

Status: DRAFT, unsigned. Moves to `research/prereg/ust-slope-fly.md` at
integration. Everything above the Signature heading is hashed.

## Classification under the W6 gate

Two single time series, not a 30-currency panel: `power_gate.gate` returns
`monitoring`. This study is therefore **descriptive**: it is scored and
published, and it confirms or refutes nothing. Registering it still fixes
the rule before the data is read, so the numbers cannot be tuned.

## Economic prior: who pays

The term premium: holders of longer Treasuries are paid for duration risk,
and the premium is larger when the curve is steep (Fama and Bliss 1987,
AER 77(4); Campbell and Shiller 1991, RES 58(3); Cochrane and Piazzesi 2005,
AER 95(1)). Carry and roll-down is a direct, model-free estimate of the
part of the premium visible today (Koijen, Moskowitz, Pedersen and Vrugt
2018, JFE 127(2), section on bonds). Known premium, not new alpha.

## Data

Gurkaynak, Sack and Wright (2007) parameters, `feds200628.csv` from
federalreserve.gov (free, no sign-up), SHA-256 recorded at the run.
Scored window: 1990-01 to 2026-10, month ends (Svensson form available
throughout; earlier years use the Nelson-Siegel fallback and are not
scored).

## Rule (fixed, one variant each)

`research/ust/curve_trades.py`, `monthly_study`:

- slope: tenors (2, 10), DV01-neutral, long-end DV01 = 1;
- fly: tenors (2, 5, 10), long belly, wings 50/50 by DV01;
- position = sign of expected carry + roll-down over the next month on an
  unchanged curve; return = realised zero-coupon excess return per unit
  DV01; cost 0.5 bp of DV01 per unit of position change (ASSUMPTION, about
  one tick of bid-offer on on-the-run notes, Fleming 2003, FRBNY EPR).
- Simplifications, stated: zero-coupon legs, financing at the GSW one-month
  zero rate, no repo specialness, no coupon or futures basis.

## Metric

Annualised net Sharpe of the monthly return with Lo (2002) SE, hit rate,
correlation with the 2s10s level. Trial count: 2 (slope, fly).

## Signature

signed_by:
signed_on:
sha256:
