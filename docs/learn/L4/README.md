# L4: judgement on a desk and in validation

## Plain idea

At L4 the question is no longer "is the formula right" but "should anyone act
on this number". A capital charge, a backtest, a research result or a latency
figure each comes with a way of being wrong that the number itself does not
show. L4 is knowing those ways, saying them out loud, and naming what evidence
would change your mind.

## Formal statement

At the end of L4 you can:

1. read a VaR backtest result against the exact distribution of exception
   counts for a correct model, not against a rule of thumb;
2. say what a preregistered research test can and cannot conclude, including
   the cost of looking at the data more than once (appendix Set 19,
   optional stopping and power);
3. explain why each desk control (limits, ladders, stress tests,
   champion-challenger) exists to cover a known weakness of a model (appendix
   Set 17);
4. take apart a failure: what happened, why the tests did not catch it, and
   what test would now (appendix Set 18).

## Worked example

A correct 99 % VaR model is backtested over 250 days. The number of
exceptions is `Binomial(250, 0.01)`, with mean 2.5.

| Exceptions | Probability | Cumulative |
|---|---|---|
| 0 | 0.0811 | 0.0811 |
| 1 | 0.2047 | 0.2858 |
| 2 | 0.2574 | 0.5432 |
| 3 | 0.2149 | 0.7581 |
| 4 | 0.1341 | 0.8922 |
| 5 | 0.0666 | 0.9588 |

So a correct model lands in the Basel green zone (0 to 4 exceptions) only
89.2 % of the time; about one correct model in ten is sent to the yellow
zone and pays a higher multiplier. A desk that reads "yellow" as "the model
is broken" will be wrong about one time in ten without anything being
wrong. A desk that reads 5 exceptions as fine will miss a model that is
really at 98 %: under `Binomial(250, 0.02)` the chance of 5 or more is
about 56 %.

Source: the table is computed in
`docs/learn/tools/test_worked_examples.py::test_l4_basel_binomial` and
matches `basel_zone_probabilities` in
`python/equity/03-var-es-engine/src/eq_var/backtesting.py`.

## Checks

1. A model shows 3 exceptions in 250 days, all in one week. Green zone. Is it
   fine?
2. A research result has p = 0.04 after the analyst looked at it every month
   for two years. How should you read it?
3. Why is the peg exposure in the FX desk guide limited by notional and not
   by VaR?

Answers: no, clustering points to a model slow to react, which
Christoffersen's independence test checks; as much weaker than 0.04,
because each look was another chance to cross the line (appendix Primer C);
a pegged currency shows almost no volatility in the window, so its VaR is
near zero while the devaluation risk is large
(`python/fx/03-var-es-engine/docs/DESK_GUIDE.md` §3).

Chapter: [adjoint algorithmic differentiation](aad.md).

Next: [L4 questions](QUESTIONS.md) and appendix Sets 17 and 18.
