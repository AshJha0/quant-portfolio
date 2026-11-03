# Counterparty risk and capital: CVA and SA-CCR (L2)

## Plain idea

When a bank trades a swap with a client, the client might default while the
swap is worth money to the bank. The expected loss from that is the credit
valuation adjustment, CVA: how much the bank could be owed, times the chance
the client defaults at each date, times the share lost in default.

Regulators also want capital held against that exposure, computed the same
way at every bank. The standardised approach for counterparty credit risk,
SA-CCR, does this with fixed formulas: today's replacement cost plus an
add-on for how far the exposure could grow.

## Formal statement

1. **CVA (unilateral).** With expected positive exposure `EE(t)`, discount
   factor `D(t)`, survival `Q(t)` and recovery `R`,
   `CVA = (1 - R) integral_0^T D(t) EE(t) (-dQ(t))`, computed on a time grid
   (Gregory, *The xVA Challenge*, 4th ed., 2020, ch. 14). DVA is the same
   with the bank's own default and the negative exposure; FVA charges a
   funding spread on the exposure. Code: `qp_core.xva.adjustments`
   (`cva`, `dva`, `fva`, `xva`), with exposures from a Hull-White
   simulation in `qp_core.xva.exposure`.
2. **SA-CCR.** Exposure at default is
   `EAD = alpha (RC + PFE)`, `alpha = 1.4`, with replacement cost
   `RC = max(V - C, 0)` and `PFE = multiplier x AddOn`. For interest rates,
   each trade's effective notional is notional x supervisory duration x
   supervisory delta x maturity factor; trades are netted within a currency
   and across three maturity buckets, and the add-on is a supervisory factor
   (0.5 %) times the result (BCBS 279, 2014; Basel framework CRE52). Code:
   `qp_core.capital.saccr` (`ead`).

## Worked example

**CVA.** A discounted expected exposure of 1 at every date out to 5 years,
hazard rate 2 %, recovery 40 %. The default probability over 5 years is
`1 - e^{-0.1} = 0.0952`, so `CVA = 0.6 x 0.0952 = 0.0571` per unit of
exposure.

**SA-CCR, BCBS Example 1.** A netting set of three interest-rate trades
(two USD swaps in opposite directions, 10 and 4 years, and a bought EUR
swaption, modelled as a put, on a 10-year swap starting in a year), with
net mark-to-market 60. The add-on is 347, the
replacement cost 60, and `EAD = 1.4 x (60 + 347) = 569` (thousands), the
figure published by the Basel Committee. The supervisory delta of the
swaption is -0.2694 and the supervisory duration of a 10-year trade
starting today is 7.8694.

Source: recomputed from the package code in
`docs/learn/tools/test_worked_examples.py::test_counterparty_and_capital`.

## Where the repository goes further

- The exposure profile of a swap from a Hull-White simulation, and the
  check that its CVA agrees with a strip of swaptions:
  `python/core/src/qp_core/xva/exposure.py`.
- Capital for market risk under the FRTB sensitivities-based method:
  `python/core/src/qp_core/capital/frtb_sbm.py`.
- Rust and C++ ports of the XVA module with parity tests:
  `rust/core/src/xva.rs` and `cpp/core/include/qp/xva.hpp`.

## Checks

1. Why does SA-CCR take `max(V - C, 0)` and not `V - C`?
2. The client posts collateral of 60 to cover the mark-to-market. What
   happens to RC, and why does EAD not fall to zero?
3. Why is the swaption's supervisory delta negative?

Answers: a negative value means the bank owes the client, which is not a
credit exposure; RC becomes 0, but the PFE add-on remains, because the
exposure can still grow before the next margin call; a bought put
gains when the underlying falls, and CRE52.40 gives a bought put a delta of
`-N(-d)` (CRE52.40, with `d` from the supervisory volatility).
