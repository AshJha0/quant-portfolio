# Credit: hazard rates, survival and the CDS (L2)

## Plain idea

A credit default swap is insurance on a borrower. The buyer pays a regular
premium (the spread) until the borrower defaults or the contract ends. The
seller pays the loss, one minus the recovery rate, if default happens. A fair
spread makes the expected value of the two payment streams equal. So the
spread is a price for default risk, and from a market spread you can back out
how likely the market thinks default is.

## Formal statement

Model default as the first jump of a process with intensity (hazard rate)
`lambda(t)`. Survival to time `t` is
`Q(t) = exp(-integral_0^t lambda(u) du)`. With discount factor `D(t)`,
recovery `R`, and payment dates `t_1..t_n` with accrual fractions
`delta_i`:

- Premium leg per unit spread, the risky annuity:
  `A = sum_i delta_i D(t_i) Q(t_i)` plus the accrued premium paid at default.
- Protection leg: `(1 - R) integral_0^T D(t) (-dQ(t))`.
- Par spread: `s = protection leg / A`.

With a flat hazard the "credit triangle" `s ~= lambda (1 - R)` holds
approximately: the spread is the default rate times the loss given default.

Reference: Hull, *Options, Futures, and Other Derivatives* (11th ed.),
ch. 25; O'Kane, *Modelling Single-name and Multi-name Credit Derivatives*
(2008), ch. 6. The code is
`python/equity/06-credit-risk/src/eq_credit/cds.py` (`HazardCurve`,
`cds_legs`, `par_spread`, `bootstrap_hazards`); the ISDA standard model is in
`python/equity/06-credit-risk/src/eq_credit/isda_cds.py`.

## Worked example

Flat hazard 2 % a year, recovery 40 %, flat interest rate 3 %, 5-year
contract with quarterly premiums:

| Quantity | Value |
|---|---|
| Credit triangle `lambda (1 - R)` | 120.0 bp |
| Protection leg (per unit notional) | 0.053088 |
| Risky annuity (years) | 4.4074 |
| Par spread | 120.45 bp |
| 5-year survival `exp(-0.02 x 5)` | 90.48 % |

The triangle is within half a basis point here. The small gap comes from
the premium being paid in arrears on discrete dates while protection is
continuous.

Source: `cds_legs` and `par_spread` in
`python/equity/06-credit-risk/src/eq_credit/cds.py`, checked in
`docs/learn/tools/test_worked_examples.py::test_credit_cds`.

## Where the repository goes further

- Bootstrapping a hazard curve from quoted spreads: `bootstrap_hazards`.
- Default probabilities from borrower data (scorecards, calibration, the
  IFRS 9 stages): the rest of `eq_credit` and appendix Set 10.
- Counterparty credit (CVA, SA-CCR): the
  [counterparty chapter](counterparty-and-capital.md), `qp_core.xva` and
  `qp_core.capital.saccr`, with the FX package versions in `fx_credit.xva`
  and `fx_credit.saccr`.
- Credit indices (factors, defaults, index basis):
  `python/core/src/qp_core/credit_index/index.py`, and the ISDA standard model
  with the half-day accrual bias, on by default since v3.0.0.

## Checks

1. The spread doubles while recovery stays at 40 %. What happens to the
   implied hazard rate?
2. If recovery were 0 %, what spread does the triangle give for a 2 %
   hazard?
3. Why does a CDS seller care about the risky annuity and not just the
   spread?

Answers: it roughly doubles; 200 bp; the mark-to-market of an old contract
is (old spread - new spread) times the risky annuity, so the annuity is the
contract's spread duration.
