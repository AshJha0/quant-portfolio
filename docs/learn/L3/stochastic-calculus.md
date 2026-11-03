# Stochastic calculus: from a hedge to the Black-Scholes equation (L3)

## Plain idea

A stock price wiggles in a way no smooth curve can describe. Over a short
time `dt` the random part of its move is of size `sqrt(dt)`, which is much
bigger than `dt`. That is why the ordinary chain rule fails: when you square
a move to get the curvature term, the random part contributes `dt`, not
something negligible. Ito's lemma is the chain rule with that extra term
kept. Apply it to an option, hold the right amount of stock against it, and
the randomness cancels. What is left is a deterministic equation, and the
option's price is its solution.

## Formal statement

1. **Brownian motion.** `W_t` has `W_0 = 0`, independent increments, and
   `W_t - W_s ~ N(0, t - s)`. Over a step, `(dW)^2 = dt` in the mean-square
   sense; `dt dW` and `(dt)^2` vanish faster than `dt`.
2. **Geometric Brownian motion.** `dS = mu S dt + sigma S dW`.
3. **Ito's lemma.** For `V(S, t)` twice differentiable in `S`:
   `dV = (V_t + mu S V_S + (1/2) sigma^2 S^2 V_SS) dt + sigma S V_S dW`.
4. **Log price.** With `V = ln S`:
   `d ln S = (mu - sigma^2 / 2) dt + sigma dW`, so
   `S_T = S_0 exp((mu - sigma^2/2) T + sigma W_T)`.
5. **The hedge.** Hold the option and short `V_S` shares. The `dW` terms
   cancel. A riskless portfolio must earn the riskless rate, which gives
   `V_t + (1/2) sigma^2 S^2 V_SS + r S V_S - r V = 0`,
   the Black-Scholes equation. `mu` has dropped out.

Reference: Shreve, *Stochastic Calculus for Finance II* (2004), ch. 3-4
(Brownian motion, Ito's lemma) and sec. 4.5 (the Black-Scholes equation);
Hull (11th ed.), ch. 14-15. The repository's replication of the hedge is
`python/foundations/03-black-scholes-replication`.

## Worked example

Two checks you can run with numbers.

**The `-sigma^2/2` drift.** With `mu = 0` and `sigma = 20 %`, the median of
`S_T / S_0` after one year is `exp(-0.02) = 0.9802`, while its mean is
exactly 1. The mean is held up by a long right tail. A Monte Carlo run with
200,000 paths and seed 7 gives a mean of 1.000 to within its standard error
and a median of 0.980.

**The equation holds at a point.** For the at-the-money call in
[L2](../L2/README.md) (`S = K = 100`, `r = 5 %`, `sigma = 20 %`,
`T = 1`): price 10.4506, delta 0.63683, gamma 0.018762, theta -6.4140 a
year. Substitute:

| Term | Value |
|---|---|
| `V_t` (theta) | -6.4140 |
| `(1/2) sigma^2 S^2 V_SS` | +3.7524 |
| `r S V_S` | +3.1842 |
| `-r V` | -0.5225 |
| Sum | 0.0000 |

Read it as a trader would: the long-gamma holder loses 6.41 a year to time
decay and earns 3.75 back from realised variance at the implied level; the
stock hedge and the cash financing make up the rest.

Source: both checks are computed in
`docs/learn/tools/test_worked_examples.py::test_stochastic_calculus` with
`call_price` and `call_greeks` from
`python/foundations/03-black-scholes-replication/src/eq_bs_replication/black_scholes.py`.

## Checks

1. Why does `mu` not appear in the Black-Scholes equation?
2. In point 4, where does `-sigma^2/2` come from?
3. If realised volatility is 25 % while you hedge an option bought at 20 %,
   do you make or lose money on average, and through which term?

Answers: the hedge removes all exposure to the direction of `S`, so its
expected rate of return cannot matter; from the `(1/2) V_SS sigma^2 S^2`
term with `V_SS = -1 / S^2`; you make money, through gamma: the hedge
captures `(1/2) gamma S^2 (sigma_realised^2 - sigma_implied^2) dt`.
