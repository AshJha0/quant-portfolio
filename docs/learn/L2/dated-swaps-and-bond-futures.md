# Dated swaps, curve building and bond futures (L2)

## Plain idea

The [rates chapter](rates.md) prices a bond off one yield and times measured
in years. A real swap is a list of dates: when each period starts and ends,
when the rate is fixed, when the cash is paid. Getting those dates right is
most of the work, because a coupon's size depends on the number of days it
accrues and on which days the overnight rate is published.

Overnight-rate swaps (SOFR, ESTR, SONIA) pay a rate compounded from the
daily fixings over the period, known only at the end ("in arrears"). Curves
are built by solving for the discount factors that make quoted swaps worth
zero, one maturity at a time ("bootstrapping"), with the overnight curve for
discounting and a separate curve for each term index.

A bond future delivers one of a basket of bonds. A conversion factor puts
the bonds on a common footing, and the seller picks the one that is
cheapest to deliver.

## Formal statement

1. **Compounded in arrears.** For business days `d_1, ..., d_m` in the
   period, fixings `r_i` and day weights `n_i` (a Friday fixing counts for
   three days), the period rate is
   `R = (prod_i (1 + r_i n_i / B) - 1) B / N`, with `B` the day-count basis
   (360 for SOFR and ESTR, 365 for SONIA) and `N` the calendar days in the
   period (ARRC, *SOFR Conventions*, 2020). Lookback, lockout and
   observation shift change which fixings are used, not the formula. Code:
   `fi_rates.legs.compounded_rate`.
2. **Dated bootstrap.** Given a discount curve from OIS quotes, the
   projection curve for a term index is solved so that each quoted swap,
   with its real schedule, prices to zero. Code:
   `fi_rates.multicurve.bootstrap_projection_dated`, with schedules from
   `fi_rates.schedule`; the OIS curve comes from
   `fi_rates.bootstrap.bootstrap_ois_dated`. Both are checked against
   QuantLib to 1e-12 in discount factor
   (`python/equity/04-fixed-income/tests/test_bootstrap_dated.py`).
3. **Conversion factor (CME Treasury futures).** The factor is the price per
   unit of par of the delivered bond at a 6 % yield, with the remaining term
   rounded down to whole months, then quarters. The invoice is
   `futures price x CF + accrued`, and the cheapest to deliver maximises the
   implied repo rate. Code:
   `fi_rates.bond_futures` (`cme_conversion_factor`, `cheapest_to_deliver`,
   `implied_repo`).
4. **Credit index.** A CDS index of `n` names has a factor equal to the
   share of names not yet defaulted. When one name defaults with auction
   recovery `R`, the protection buyer receives `(1 - R) / n` per unit of
   original notional, and the index carries on at the lower factor. Code:
   `python/core/src/qp_core/credit_index/index.py`.

## Worked example

**SOFR in arrears.** A period from Thursday 4 January 2024 to Monday
8 January 2024. Thursday fixes at 5.30 % for one day, Friday at 5.31 % for
three (the weekend). Then
`R = ((1 + 0.0530 / 360)(1 + 0.0531 x 3 / 360) - 1) x 360 / 4 = 5.3081 %`.
A day-weighted average would give 5.3075 %; compounding adds 0.06 basis
points over four days.

**Conversion factor.** CME's own example: the 3 3/4 % note maturing
15 November 2018, delivered into the December 2008 ten-year contract. The
remaining term from 1 December 2008 rounds to 9 years 9 months, and the
factor is 0.8357.

**Credit index.** 125 names, one defaults with 30 % auction recovery. The
factor drops to `124 / 125 = 0.992`, the buyer of protection receives
`0.008 x 0.7 = 0.0056` per unit of original notional, and every later upfront
is the old one times 0.992.

Source: recomputed in
`docs/learn/tools/test_worked_examples.py::test_dated_swaps_and_bond_futures`
(the first two from the package code, the third from the arithmetic above,
which the package docstring states and its tests check).

## Where the repository goes further

- Dated fixed, term and overnight legs, and a swap built from them:
  `fi_rates.legs`; the fixings store refuses to guess a missing past fixing.
- The ISDA standard CDS model with the half-day accrual bias, checked against
  QuantLib's `IsdaCdsEngine`: `python/equity/06-credit-risk/src/eq_credit/isda_cds.py`.
- The switch option in a bond future, in a one-factor yield model priced on
  a grid: `fi_rates.bond_futures.delivery_option`.
- Futures convexity: `fi_rates.legs.futures_convexity_adjustment`.

## Checks

1. Why does a Friday SOFR fixing get a weight of three?
2. Why is the compounded rate above the day-weighted average?
3. A bond has a conversion factor below 1. What does that say about its
   coupon?
4. After the default above, a second name defaults with 40 % recovery. What
   is the factor and what does the buyer receive?

Answers: it is the rate that applies until the next business day, Monday;
interest earned early in the period earns interest itself; its coupon is
below the 6 % notional coupon; `123 / 125 = 0.984`, and
`0.008 x 0.6 = 0.0048` per unit of original notional.
