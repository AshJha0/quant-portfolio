# Adjoint algorithmic differentiation (L4)

## Plain idea

A desk wants every sensitivity of a book: to spot, to vol, to each point of
each curve. Bumping each input and repricing costs one full pricing per
input, so a swap on an 18-pillar curve needs 36 repricings for central
differences. Adjoint differentiation records the pricing calculation once,
then walks it backwards, and gets all the first-order sensitivities for a
small multiple of one price, however many inputs there are.

## Formal statement

Write the pricer as a sequence of elementary steps
`v_i = phi_i(v_j : j in parents(i))`, ending in the price `y`. The adjoint
of a node is `vbar_i = dy / dv_i`. Seeding `ybar = 1` and walking the steps
in reverse,

    vbar_j = sum over children i of j:  vbar_i x d phi_i / d v_j.

One reverse sweep gives `dy / dx` for every input `x`. Griewank's "cheap
gradient principle" bounds the arithmetic at a small constant times one
evaluation, about 4 to 5 (Griewank and Walther, *Evaluating Derivatives*,
2nd ed., 2008, ch. 3; Savine, *Modern Computational Finance*, 2018, ch. 9).
In practice the bookkeeping of the tape adds to that constant.

For Monte Carlo, the tape is recorded and swept one path at a time and then
rewound; the adjoints of the inputs accumulate across paths (Savine 2018,
s. 10.3). This gives pathwise Greeks, so it needs a payoff that is
differentiable along the path; a digital payoff needs smoothing first.

Code: `python/core/src/qp_core/aad/tape.py` (the tape),
`python/core/src/qp_core/aad/black_scholes.py`,
`python/core/src/qp_core/aad/monte_carlo.py` and
`python/core/src/qp_core/aad/swap.py`, with Rust and C++ ports in
`rust/aad/` and `cpp/aad/`.

## Worked example

**A tiny tape.** `y = x0 x1 + exp(x0)` at `x0 = 1`, `x1 = 2` has
`y = 4.718282`. The reverse sweep gives `dy/dx0 = x1 + e = 4.718282` and
`dy/dx1 = x0 = 1`.

**Black-Scholes.** `S = K = 100`, `T = 1`, `r = 5 %`, `q = 2 %`,
`sigma = 20 %`. One forward pass and one reverse sweep give the call price
9.2270, delta 0.5869 and vega 37.90 per unit of vol, the same as the
closed-form Greeks in `qp_core.black_scholes` to 1e-12.

**What it costs.** The measured ratio of AAD time to price time, on one
loaded laptop (so a range, not a benchmark), is 1.6 to 2.1 for a European
Monte Carlo and 2.0 to 6.7 for an 18-pillar swap across Rust and C++. For a
single Black-Scholes price in Rust it is 13 to 34, because the price takes
about 20 ns and allocating the tape costs more than the arithmetic.
Source: `rust/aad/README.md`, "Cost ratio".

Source for the first two: recomputed from the package code in
`docs/learn/tools/test_worked_examples.py::test_aad`.

## Where the repository goes further

- Swap quote Greeks: adjoints with respect to the market quotes, not the
  zero rates, through the implicit function theorem (18 residual sweeps and
  one 18 x 18 solve): `rust/aad/README.md`.
- Bump-and-reprice checks of every Greek to 1e-6:
  `python/core/tests/test_aad_python.py`.

## Checks

1. Why does AAD not help when a pricer has one input and many outputs?
2. Why can a pathwise AAD delta of a digital option be zero on every path?
3. When is bumping still the right tool?

Answers: reverse mode costs one sweep per output; with one input, forward
mode (or a single bump) is the cheap direction; the payoff is flat almost
everywhere, so its pathwise derivative is zero and the mass at the strike is
missed; for second-order and cross Greeks when only a few are needed, for
checking the adjoint code, and when the pricer cannot be put on a tape
(a vendor library, say).
