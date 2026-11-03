# L2: pricing and hedging

## Plain idea

A derivative's price is what it costs to build the same payoff from things
you can already trade. For a European option under Black-Scholes, that
recipe is a changing amount of stock and cash; for a bond, a set of discounted
cash flows; for a credit default swap, two streams of payments weighted by the
chance of default. The sensitivities (the Greeks, duration, the risky
annuity) say how the price moves when an input moves, which is what a desk
hedges.

## Formal statement

- Black-Scholes call on a non-dividend stock:
  `C = S N(d1) - K e^{-rT} N(d2)`,
  `d1 = (ln(S/K) + (r + sigma^2/2) T) / (sigma sqrt(T))`, `d2 = d1 - sigma sqrt(T)`
  (Hull, *Options, Futures, and Other Derivatives*, 11th ed., ch. 15).
- Delta `N(d1)`, gamma `N'(d1) / (S sigma sqrt(T))`, vega `S N'(d1) sqrt(T)`.
  Vega is quoted per unit of volatility (1.00 = 100 vol points); divide by
  100 for the change per vol point.
- Bonds, duration and convexity: [rates chapter](rates.md).
- Hazard rates and CDS: [credit chapter](credit.md).

## Worked example

`S = 100`, `K = 100`, `r = 5 %`, `sigma = 20 %`, `T = 1`:

| Quantity | Value |
|---|---|
| Call price | 10.4506 |
| Delta | 0.6368 |
| Gamma | 0.018762 |
| Vega (per unit vol) | 37.524 |
| Vega (per vol point) | 0.3752 |
| Theta (per year) | -6.4140 |

A vega of 37.5 next to a price of 10.45 is not a contradiction: it is the
change for a 100-point vol move. A one-point move (20 % to 21 %) changes the
price by about 0.375. An earlier version of the appendix quoted a vega next
to a price without saying which unit it used; Set 3 Q16 now gives both.

Source: `call_price` and `call_greeks` in
`python/foundations/03-black-scholes-replication/src/eq_bs_replication/black_scholes.py`,
checked in `docs/learn/tools/test_worked_examples.py::test_l2_black_scholes`.
Hull's 11th edition gives 10.45 for the same inputs.

## Checks

1. Put-call parity gives the put price from the call above. What is it?
2. Why is at-the-money delta above 0.5 here?
3. A desk is long this call and short 0.6368 shares. Which risk is left?

Answers: `10.4506 - 100 + 100 e^{-0.05} = 5.5735`; the drift term
`r + sigma^2/2` pushes `d1` above zero; gamma (and vega, and the hedge
going stale as the stock moves).

Next: [L2 questions](QUESTIONS.md), the [rates](rates.md) and
[credit](credit.md) chapters, and the
[Black-Scholes notebook](../../notebooks/01-black-scholes-implied-vol-exercise.ipynb).
Then [L3](../L3/README.md).
