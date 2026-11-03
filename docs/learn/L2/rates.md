# Rates: bond prices, yield, duration and convexity (L2)

## Plain idea

A fixed-rate bond is a list of promised payments. Its price is what those
payments are worth today. The yield is the single interest rate that, used to
discount every payment, reproduces that price. When yields rise, every
payment is worth less, so the price falls. Duration says by how much, to a
first approximation; convexity says how wrong that first approximation is.

## Formal statement

For a bond paying coupon `c` (annual rate) `f` times a year on face 100, with
`n` remaining payments and yield `y` quoted with the same frequency:

- Price: `P(y) = sum_{i=1..n} (100 c / f) / (1 + y/f)^i + 100 / (1 + y/f)^n`.
- Macaulay duration `D_mac`: the present-value-weighted average time to the
  payments, in years.
- Modified duration `D_mod = D_mac / (1 + y/f) = -(1/P) dP/dy`.
- Convexity `C = (1/P) d^2P/dy^2`.
- DV01, the price change for a one basis point fall in yield,
  `D_mod x P x 0.0001` per 100 face.
- Second-order price change: `dP / P ~= -D_mod dy + (1/2) C dy^2`.

Reference: Hull, *Options, Futures, and Other Derivatives* (11th ed.), ch. 4;
Fabozzi, *Bond Markets, Analysis and Strategies*, ch. 4. The code is
`python/equity/04-fixed-income/src/fi_rates/bond.py` (`price_from_ytm`) and
`python/equity/04-fixed-income/src/fi_rates/risk.py` (`macaulay_duration`,
`modified_duration`, `convexity`, `dv01`).

## Worked example

A 10-year 5 % semi-annual bond, 30/360, settled on its issue date
(15 January 2026, maturing 15 January 2036), at a 5 % yield:

| Quantity | Value |
|---|---|
| Price at 5 % | 100.0000 |
| Price at 6 % | 92.5613 |
| Price at 4 % | 108.1757 |
| Macaulay duration | 7.9894 years |
| Modified duration | 7.7946 |
| Convexity | 73.629 |
| DV01 per 100 face | 0.07795 |

Now move the yield up 100 bp:

- duration only: `-7.7946 x 0.01 x 100 = -7.795`;
- with convexity: `-7.795 + 0.5 x 73.629 x 0.0001 x 100 = -7.426`;
- full repricing: `92.5613 - 100 = -7.439`.

For a 100 bp fall the same arithmetic gives `+8.163` against a true
`+8.176`. Duration alone overstates the loss and understates the gain; the
convexity term closes most of the gap, which is the asymmetry that Set 8 Q6
in the appendix measures at 200 bp on the package's own sample bond.

Source: every figure above is recomputed with the package code in
`docs/learn/tools/test_worked_examples.py::test_rates_bond`.

## Where the repository goes further

- Curves instead of one yield: `fi_rates.curve` and the bootstrap and
  multicurve modules. Those are being reworked in Round 10 (W13b); this
  chapter will gain a bootstrap section once that work is merged.
- Key-rate durations: `fi_rates.keyrates`.
- Short-rate models and rate options: `fi_rates.hull_white` and
  `fi_rates.rate_options`.
- The curve [notebook](../../notebooks/03-yield-curve-bootstrap-exercise.ipynb)
  builds a curve by hand.

## Checks

1. Why is the price exactly 100 at a 5 % yield?
2. Which is larger for this bond, the loss for +100 bp or the gain for
   -100 bp, and why?
3. A zero-coupon 10-year bond has Macaulay duration 10. Why is the coupon
   bond's lower?

Answers: the coupon equals the yield, so every payment is discounted at its
own rate (a par bond); the gain, because convexity is positive; coupons pay
part of the value back earlier.
