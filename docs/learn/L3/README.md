# L3: the models behind the models

## Plain idea

At L2 you used formulas. At L3 you ask where they come from and when they
break: why the Black-Scholes equation has no drift in it, why one implied
volatility per strike is not one model, why a Monte Carlo standard error can
lie, why the same formula in three languages can disagree in the last digit,
and why a large order costs more per share than a small one.

## Formal statement

At the end of L3 you can:

1. derive the Black-Scholes equation from a delta hedge with Ito's lemma
   ([stochastic calculus](stochastic-calculus.md));
2. state the no-arbitrage conditions on a volatility surface (no butterfly,
   no calendar arbitrage) and say which repository check enforces each
   (appendix Set 5);
3. explain the backtests of a VaR model (Kupiec, Christoffersen,
   Acerbi-Szekely) and what each one cannot see (appendix Set 7);
4. reason about market impact with the Roll, Kyle and square-root models
   ([microstructure](microstructure.md));
5. explain why cross-language results are compared with stated tolerances
   and when bit-identity is the right test (appendix Sets 15 and 19).

## Worked example

The Black-Scholes equation checked term by term at one point is worked in
[stochastic-calculus.md](stochastic-calculus.md): theta -6.4140, gamma term
+3.7524, delta term +3.1842, financing -0.5225, sum zero to the printed
precision.

Source: `docs/learn/tools/test_worked_examples.py::test_stochastic_calculus`.

## Checks

1. A trader is long gamma and the stock does not move all day. Did the
   trader make or lose money, and which term of the equation says so?
2. Why does `basel_traffic_light` use the exact binomial distribution
   rather than the chi-squared approximation behind Kupiec's p-value?
3. Why does the repository accept a 1e-15 relative difference between
   `carry_price` with `q = r` and `black76_price`?

Answers: lost, through theta, with no realised variance to pay for it; at
250 days the chi-squared approximation is poor (the 5 % Kupiec test rejects a
correct model about 9.5 % of the time, `python/equity/03-var-es-engine/tests/test_backtesting.py`);
they are the same formula with the terms in a different order, and
floating-point addition is not associative (appendix Set 19 Q5).

Next: [L3 questions](QUESTIONS.md), the two chapters, then
[L4](../L4/README.md).
