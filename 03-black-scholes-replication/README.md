# Project 3 — Replicate a Known Model: Black-Scholes

The Black-Scholes-Merton European option pricing model, rebuilt from scratch (the pricing core uses only `math.erf` — no scipy), validated three independent ways, and then deliberately pushed to the point where it breaks.

## What did I build?

- `black_scholes.py` — closed-form call/put prices, the five Greeks in analytic form, and implied volatility via Newton-Raphson with a bisection fallback. The derivation logic and the interpretation of `N(d1)`/`N(d2)` are documented in the module docstring.
- `monte_carlo.py` — a Monte Carlo pricer under the *same* risk-neutral GBM dynamics, with antithetic variates. Because the terminal GBM price has a closed form, European payoffs need no path discretisation.
- `test_black_scholes.py` — tests derived from *theory*, not from a reference library: put-call parity, no-arbitrage bounds, monotonicity in volatility, limiting behaviour, Greeks vs central finite differences, and implied-vol round trips.
- `analysis.py` — runs the validation, the Monte Carlo convergence study, and a demonstration of the volatility smile.

## How to run

```
pip install -r requirements.txt
python test_black_scholes.py     # 6 theory-based tests
python analysis.py               # report + figures in output/
```

## Why did I build it this way?

Replicating a model is only convincing if the validation doesn't assume the implementation is right. So the checks are cross-cutting:

1. **Identities that must hold exactly** — put-call parity is a model-free arbitrage relation; if it fails, the code is wrong regardless of the model.
2. **Two independent implementations of the same model** — the Monte Carlo pricer shares no code with the closed form yet must converge to it at O(1/√n). The convergence plot shows exactly that rate.
3. **Analytic Greeks vs numerical derivatives** — the formulas are checked against central finite differences of the price function itself.
4. **Round trips** — price → implied vol → price must return the input across low (8%) and high (120%) vol regimes, which also exercises the Newton/bisection switch.

## What assumptions does the model make?

1. The underlying follows geometric Brownian motion with **constant volatility** — so log-returns are normal.
2. Constant risk-free rate, continuous compounding.
3. **Frictionless markets**: continuous trading, no transaction costs, unlimited shorting — this is what makes the continuous delta-hedge (and hence risk-neutral pricing) possible.
4. European exercise, no dividends (adding a continuous dividend yield is a one-line extension, noted below).

## What did the results tell me?

- The two pricers agree to within Monte Carlo noise at every sample size, and the error shrinks at the theoretical √n rate — evidence both implementations are correct, since they share nothing but the model.
- Parity holds to ~1e-15 and every Greek matches its finite-difference estimate — the calculus is right, not just the headline price.
- The gamma plot makes concrete why hedging near-expiry ATM options is hard: delta changes fastest exactly where gamma peaks.

## Where does the model break down?

The analysis contains a controlled experiment: options are priced by Monte Carlo under a **fat-tailed** return distribution (Student-t, df = 4, variance-matched), then each price is read back through Black-Scholes. If the constant-vol lognormal assumption were true, implied vol would be flat across strikes. Instead it comes out as a **smile** — OTM and ITM options carry higher implied vols because fat tails make extreme outcomes likelier than the lognormal admits. Real option markets show exactly this smile/skew (persistently since 1987), which is the market itself telling you the assumptions fail.

Other known failure points, stated plainly:

- **Volatility is not constant** — it clusters and is itself stochastic (motivating GARCH, Heston, local vol models).
- **Prices jump** — overnight gaps cannot be delta-hedged continuously, breaking the replication argument (motivating jump-diffusion models).
- **Frictions exist** — discrete rebalancing plus transaction costs means the hedge is imperfect, so the "unique arbitrage-free price" is really a band.
- **Early exercise** — American puts need lattice/PDE/LSM methods; the European formula only bounds them.

## What I would improve

- Add a continuous dividend yield *q* (replace `S` with `S·e^(−qT)` throughout) and support American exercise via a binomial tree, converging to the European closed form as a further cross-check.
- Fit the actual smile from real option chain data and calibrate a stochastic-vol model against it.
- Vectorise the pricer (numpy) for whole option chains; the scalar version here favours readability.
