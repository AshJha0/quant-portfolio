# Stochastic calculus: knowledge check (questions only)

Twenty-two problems on Brownian motion, Ito calculus and the processes that
show up in pricing, written fresh for this repository in the style of Zhou
(2008, ch. 5) and Joshi, Denson and Downes (2008). Answers and worked
solutions are in [`keys/answers.md`](keys/answers.md). Work each one with
pen and paper first.

Every answer in the key is a closed form. [`verify.py`](verify.py)
recomputes each one by a different route (quadrature against the Gaussian
density, a numerical ODE or boundary-value solve) and, where a simulation is
cheap, by a seeded Monte Carlo. `career-prep/tests/test_quant_packs.py`
fails if the key and a check disagree.

Notation: `W` is a standard Brownian motion with `W_0 = 0`. Give numbers to
at least 10 significant figures when the answer is not a simple fraction.

Tags: `[bm]` Brownian moments, `[ito]` Ito's lemma and isometry, `[reflect]`
reflection, `[gbm]` geometric Brownian motion, `[ou]` Ornstein-Uhlenbeck,
`[stop]` hitting times, `[girsanov]` change of measure, `[pde]`
Feynman-Kac and pricing.

## Brownian motion

- **S01** `[bm]` What is E[W_3^2]?
- **S02** `[bm]` What is E[W_2 W_5]?
- **S03** `[bm]` What is the variance of the time average, Var(integral from 0 to 1 of W_s ds)?
- **S04** `[bm]` What is E[W_1^4]?
- **S18** `[bm]` What is E[W_1^2 W_2^2]?
- **S22** `[bm]` What is Var(exp(W_1))?

## Ito calculus

- **S05** `[ito]` What is E[(integral from 0 to 2 of W_s dW_s)^2]?
- **S14** `[ito]` What is E[integral from 0 to 1 of W_s^2 ds]?
- **S19** `[ito]` What is E[integral from 0 to 1 of W_s o dW_s], the Stratonovich integral? Compare with the Ito integral.
- **S06** `[ito]` What is E[exp(W_2)]?

## Reflection and hitting times

- **S07** `[reflect]` What is the probability that max over s in [0,1] of W_s is at least 1?
- **S15** `[stop]` X_t = 0.5 t + W_t. What is the expected first time X reaches 2?
- **S16** `[stop]` X_t = -0.5 t + W_t. What is the probability X ever reaches +1?

## Geometric Brownian motion

S follows dS = 0.05 S dt + 0.2 S dW with S_0 = 100.

- **S08** `[gbm]` What is E[S_1]?
- **S09** `[gbm]` What is the median of S_1? Why is it below the mean?
- **S10** `[gbm]` What is Var(S_1)?

## Ornstein-Uhlenbeck

X follows dX = 2 (1 - X) dt + 0.5 dW with X_0 = 0.

- **S11** `[ou]` What is E[X_1]?
- **S12** `[ou]` What is Var(X_1)?
- **S13** `[ou]` What is the stationary variance of X?

## Change of measure and pricing

- **S17** `[girsanov]` What is E[W_1 exp(W_1 - 1/2)]?
- **S20** `[pde]` Under the risk-neutral measure, dS = 0.05 S dt + 0.2 S dW, S_0 = 100, r = 5%. What is today's price of a claim paying S_1^2 at time 1?
- **S21** `[pde]` u(x, t) solves u_t + u_xx / 2 = 0 with u(x, T) = x^2. What is u(1, T - 2)?
