# Cookbook

Copy-pasteable recipes for the most common "how do I..." tasks across every
project area in quant-portfolio. The recipes are written against the
current `main` branch. Outputs are indicative: they come from one run on one
machine, and nothing in CI executes these snippets, so a last digit can move
after a code change. Every data set used here is synthetic.

Install each project into a virtual environment created outside the
repository, as [CONTRIBUTING.md](../CONTRIBUTING.md) describes:

```bash
python -m venv ~/.venvs/qp
. ~/.venvs/qp/bin/activate            # Windows: ~/.venvs/qp/Scripts/activate
python -m pip install -c requirements/ci-constraints.txt -e python/equity/01-options-pricing
```

Each section names the projects its recipes need.

Recipes are grouped to match [ARCHITECTURE.md](ARCHITECTURE.md)'s area map:
options pricing/vol/surfaces, market risk (VaR/ES/backtesting), fixed
income/pairs/credit/portfolio optimization, and execution/regime/building the
compiled engines, plus a final section for what the November 2026 release added. For the theory behind any of these, see the matching set
in [LEARN.md](LEARN.md) Part IV; for the underlying math and assumptions, see
the project's own `docs/METHODOLOGY.md`.


## Options Pricing, Volatility & Surfaces

Recipes for pricing options and Greeks (equity and FX), solving for implied
vol, forecasting volatility, and building/querying volatility surfaces
(including Heston stochastic vol). The setup note before each project's
first recipe names the package to install.

### How do I price a European call/put and get its Greeks (equity)?

`eq_options.bs_price` gives the Black-Scholes-Merton price with a continuous
dividend yield `q`; `bs_greeks` returns the full analytic Greek set (delta,
gamma, vega, theta, rho, vanna, volga) in one call.

Setup once (in your venv): `python -m pip install -e python/equity/01-options-pricing`

```python
from eq_options import bs_price, bs_greeks

price = bs_price(S=100, K=105, T=0.5, r=0.03, sigma=0.22, q=0.01, option_type="call")
g = bs_greeks(S=100, K=105, T=0.5, r=0.03, sigma=0.22, q=0.01, option_type="call")
print(f"call price: {price:.4f}")
print(f"delta={g.delta:.4f} gamma={g.gamma:.5f} vega={g.vega:.4f} "
      f"theta={g.theta:.4f} rho={g.rho:.4f}")
```

Output: `call price: 4.5173`, `delta=0.4297 gamma=0.02514 vega=27.6587 theta=-6.8089 rho=19.2280`. Vega is per unit of annualised vol (divide by 100 for "per vol point"); theta is per year (divide by 365 for per-day).

### How do I price an FX option via Garman-Kohlhagen and interpret its delta convention?

Garman-Kohlhagen is Black-Scholes with the foreign rate `r_f` playing the role
of a dividend yield. `fx_options.delta` reports delta under any of the four
market conventions: spot or forward hedge, premium-adjusted or not.

Setup once (in your venv): `python -m pip install -e python/fx/01-options-pricing`

```python
from fx_options import gk_price, delta

# EURUSD: spot 1.0850 USD per EUR, 3m 1.10-strike call
price = gk_price(S=1.0850, K=1.10, T=0.25, r_d=0.045, r_f=0.030, sigma=0.08, option_type="call")
print(f"EURUSD 3m call premium: {price:.5f} USD per EUR")

d_spot = delta(S=1.0850, K=1.10, T=0.25, r_d=0.045, r_f=0.030, sigma=0.08,
               option_type="call", convention="spot")
d_fwd = delta(S=1.0850, K=1.10, T=0.25, r_d=0.045, r_f=0.030, sigma=0.08,
              option_type="call", convention="forward")
print(f"spot delta: {d_spot:.4f}   forward delta: {d_fwd:.4f}")
```

Output: `premium: 0.01240 USD per EUR`, `spot delta: 0.4062   forward delta: 0.4092`. The forward delta is always the larger (in magnitude) of the two. It excludes the foreign discount factor `e^{-r_f T}` that the spot-hedge delta includes.

### How do I solve for implied volatility given a market price?

`implied_vol` maps the quote to normalised Black form, works on the
out-of-the-money side, and solves by safeguarded Halley iteration from an
asymptotic initial guess (the same algorithm runs in the C++ and Rust
engines). It raises `ValueError` if the quoted price sits outside the
model's no-arbitrage bounds. The old solver controls `tol`, `max_iter`,
`sigma_lo` and `sigma_hi` are still accepted but no longer needed.

```python
from eq_options import bs_price, implied_vol

market_price = bs_price(S=100, K=100, T=0.5, r=0.03, sigma=0.25, option_type="put")
iv = implied_vol(price=market_price, S=100, K=100, T=0.5, r=0.03, option_type="put")
print(f"market put price: {market_price:.4f}")
print(f"recovered implied vol: {iv:.6f}")
```

Output: `market put price: 6.2715`, `recovered implied vol: 0.250000`. An at-the-money, mid-vol quote round-trips to full double precision.

### How do I price an American option via the binomial tree and compare it to the European closed form?

`crr_price` is a vectorised Cox-Ross-Rubinstein tree supporting both
`exercise="european"` and `exercise="american"`; the European tree price
converges to `bs_price` as `n_steps` grows.

```python
from eq_options import crr_price, bs_price

euro_tree = crr_price(S=100, K=95, T=1.0, r=0.03, sigma=0.30, q=0.02,
                       option_type="put", exercise="european", n_steps=500)
euro_bs = bs_price(S=100, K=95, T=1.0, r=0.03, sigma=0.30, q=0.02, option_type="put")
amer_tree = crr_price(S=100, K=95, T=1.0, r=0.03, sigma=0.30, q=0.02,
                       option_type="put", exercise="american", n_steps=500)
print(f"European tree: {euro_tree:.4f}   European BS closed form: {euro_bs:.4f}")
print(f"American tree: {amer_tree:.4f}   early-exercise premium: {amer_tree - euro_tree:.4f}")
```

Output: `European tree: 8.6582` vs `European BS closed form: 8.6602` (500-step tree, about 2bp of discretisation error); `American tree: 8.7527`, early-exercise premium `0.0945`. A dividend-paying (`q=0.02`) put carries real early-exercise value.

### How do I price an American FX option and see the foreign-carry early-exercise premium?

For FX, `binomial_price` treats the foreign rate `r_f` like a dividend yield.
When the foreign currency yields more than the domestic one (`r_f > r_d`), an
American call earns a positive early-exercise premium from capturing that
foreign carry sooner.

```python
from fx_options import binomial_price, gk_price

euro = binomial_price(S=1.10, K=1.05, T=0.5, r_d=0.02, r_f=0.045, sigma=0.09,
                       option_type="call", steps=500, exercise="european")
amer = binomial_price(S=1.10, K=1.05, T=0.5, r_d=0.02, r_f=0.045, sigma=0.09,
                       option_type="call", steps=500, exercise="american")
gk = gk_price(S=1.10, K=1.05, T=0.5, r_d=0.02, r_f=0.045, sigma=0.09, option_type="call")
print(f"GK closed form: {gk:.5f}  European tree: {euro:.5f}  American tree: {amer:.5f}")
print(f"foreign-carry early-exercise premium (r_f > r_d): {amer - euro:.5f}")
```

Output: `GK closed form: 0.04858  European tree: 0.04857  American tree: 0.05314`, premium `0.00457`. The American call is worth strictly more once the foreign rate exceeds the domestic rate.

### How do I cross-check an option price with a Monte Carlo simulation?

`mc_price` simulates the *exact* GBM terminal distribution (no time
discretisation), with antithetic variates and a control variate baked in, and
returns a 95% confidence interval you can check the analytic price against.

```python
from eq_options import bs_price, mc_price

analytic = bs_price(S=100, K=105, T=0.5, r=0.03, sigma=0.22, q=0.01, option_type="call")
mc = mc_price(S=100, K=105, T=0.5, r=0.03, sigma=0.22, q=0.01, option_type="call",
              n_paths=200_000, seed=7)
print(f"analytic price: {analytic:.4f}")
print(f"MC price: {mc.value:.4f}  stderr={mc.std_error:.5f}  "
      f"95% CI=({mc.ci_low:.4f}, {mc.ci_high:.4f})")
print(f"analytic price inside MC 95% CI: {mc.contains(analytic)}")
```

Output: `analytic price: 4.5173`, `MC price: 4.5137  stderr=0.00387  95% CI=(4.5062, 4.5213)`, `inside MC 95% CI: True`.

### How do I forecast next-period volatility with a GARCH(1,1) model?

`eq_vol.fit_garch` estimates GARCH(1,1) by maximum likelihood; `forecast`
dispatches on the fitted model type to produce a multi-step daily-variance
path that mean-reverts to the unconditional variance at rate `(alpha+beta)^k`.

Setup once (in your venv): `python -m pip install -e python/equity/02-volatility-modeling`

```python
import numpy as np
from eq_vol import fit_garch, forecast, term_structure
from eq_vol.data import synthetic

sim = synthetic.simulate_garch(n=2000, omega=5e-6, alpha=0.05, beta=0.90, seed=1)
result = fit_garch(sim.returns, dist="normal")
print("fitted params:", {k: round(v, 6) for k, v in result.params.items()})
print("persistence (alpha+beta):", round(result.extra["persistence"], 4))

fc = forecast(result, horizon=10)
print("10-day variance forecast path (first 5):", np.round(fc[:5], 8))

ts = term_structure(result, horizon=252)
print(ts.loc[[1, 21, 63, 252]])
```

Output (recovers parameters close to the simulation's `alpha=0.05, beta=0.90` truth): `persistence: 0.9295`; the term structure's `forward_vol_annual` rises from `0.1459` (1 day) to `0.1659` (252 days) as the forecast mean-reverts toward the long-run level.

### How do I compare EWMA's flat forecast to GARCH's mean-reverting term structure?

EWMA is an IGARCH(1,1) with zero intercept, so `ewma_forecast` is flat at the
1-step level at every horizon. There is no long-run level to revert to,
unlike stationary GARCH.

```python
import numpy as np
from eq_vol import ewma_forecast, fit_garch, forecast
from eq_vol.data import synthetic

sim = synthetic.simulate_garch(n=2000, omega=5e-6, alpha=0.05, beta=0.90, seed=1)
ewma_fc = ewma_forecast(sim.returns, horizon=10, lam=0.94)
result = fit_garch(sim.returns)
garch_fc = forecast(result, horizon=10)
ewma_vol = np.sqrt(ewma_fc * 252)
garch_vol = np.sqrt(garch_fc * 252)
print("EWMA 10-day annualised-vol forecast (flat):", np.round(ewma_vol, 4))
print("GARCH 10-day annualised-vol forecast (mean-reverting):", np.round(garch_vol, 4))
```

Output: EWMA stays pinned at `0.1404` for all 10 days; GARCH climbs from `0.1459` to `0.1559`, tracking back toward the long-run unconditional vol.

### How do I construct and query an implied volatility surface?

`eq_surface.VolSurface` builds a per-expiry SVI smile grid and interpolates
*total variance* (not vol) across expiries, which is what preserves an
arbitrage-free (non-decreasing) calendar structure between pillars.

Setup once (in your venv): `python -m pip install -e python/equity/09-vol-surface`

```python
import numpy as np
from eq_surface import VolSurface
from eq_surface.data.synthetic import default_svi_slices

expiries = np.array([0.25, 1.0])
slice_map = default_svi_slices(expiries)          # hand-built, arbitrage-free
slices = [slice_map[T] for T in expiries]
surface = VolSurface(expiries=expiries, slices=slices, spot=100.0, rate=0.03, div_yield=0.01)

print("vol at K=100, T=0.5:", round(surface.vol(100.0, 0.5), 5))
print("vol at K=110, T=0.25:", round(surface.vol(110.0, 0.25), 5))
print("vol at K=90, T=1.0 :", round(surface.vol(90.0, 1.0), 5))
print("calendar arbitrage free:", surface.calendar.is_free)
```

Output: `vol(K=100,T=0.5)=0.20148`, `vol(K=110,T=0.25)=0.19106` (skew: an OTM call struck above the forward is quoted at a lower vol), `vol(K=90,T=1.0)=0.21485`, `calendar arbitrage free: True`.

### How do I check a volatility smile for butterfly arbitrage?

`check_butterfly` evaluates the Durrleman density condition `g(k) >= 0` on a
log-moneyness grid for one SVI slice; a negative `g` anywhere means the
implied risk-neutral density would go negative (butterfly arbitrage).

```python
import numpy as np
from eq_surface import check_butterfly
from eq_surface.data.synthetic import default_svi_slices

slices = default_svi_slices(np.array([0.25, 1.0]))
is_free, min_g, violations = check_butterfly(slices[0.25])
print(f"SVI slice (T=0.25) butterfly-arbitrage-free: {is_free}, min Durrleman g = {min_g:.5f}")
```

Output: `butterfly-arbitrage-free: True, min Durrleman g = 0.28607`, positive across the whole grid.

### How do I run a Heston Monte Carlo simulation and cross-check it against the Fourier price?

`heston_mc_price` uses Andersen's QE scheme (exact-ish CIR sampling, handles
`v` touching zero) by default; `heston_call` prices the same contract via
Gauss-Legendre quadrature on the characteristic function. They should agree
within a few Monte Carlo standard errors.

```python
from eq_surface import HestonParams, heston_call, heston_mc_price

p = HestonParams(v0=0.04, kappa=1.5, theta=0.045, rho=-0.6, xi=0.4)
S, K, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.01

fourier = heston_call(S, K, T, r, q, p, method="gl")
mc = heston_mc_price(S, K, T, r, q, p, n_paths=100_000, n_steps=64, scheme="qe", seed=0, kind="call")
print(f"Fourier (Gauss-Legendre) price: {fourier:.4f}")
print(f"QE Monte Carlo price: {mc.price:.4f}  stderr={mc.stderr:.4f}")
print(f"within 3 stderr: {abs(mc.price - fourier) <= 3 * mc.stderr}")
```

Output: `Fourier price: 8.6004`, `QE Monte Carlo price: 8.6328  stderr=0.0355`, `within 3 stderr: True`.

### How do I rebuild Black-Scholes from scratch (foundations F3)?

`eq_bs_replication` (foundations project 03) depends on nothing but
`math.erfc`, with no scipy or numpy in the pricing path, so every piece of the
formula is visible. The normal CDF is `N(x) = 0.5 * erfc(-x / sqrt(2))`,
which keeps full relative precision deep in the left tail. Put Greeks are derived from
call Greeks via put-call parity rather than a second independent formula.

Setup once (in your venv): `python -m pip install -e python/foundations/03-black-scholes-replication`

```python
import math
from eq_bs_replication import call_price, put_price, call_greeks, implied_volatility

S, K, r, sigma, T = 100.0, 100.0, 0.03, 0.20, 1.0
c = call_price(S, K, r, sigma, T)
p = put_price(S, K, r, sigma, T)
print(f"from-scratch call: {c:.6f}  put: {p:.6f}")
print(f"put-call parity check (C-P vs S-K*e^-rT): {c - p:.6f} vs {S - K*math.exp(-r*T):.6f}")

g = call_greeks(S, K, r, sigma, T)
print(f"delta={g.delta:.5f} gamma={g.gamma:.6f} vega={g.vega:.5f} "
      f"theta={g.theta:.5f} rho={g.rho:.5f}")

iv = implied_volatility(price=c, S=S, K=K, r=r, T=T)
print(f"recovered implied vol: {iv:.6f}")
```

Output: `call: 9.413403  put: 6.457957`, parity check `2.955447` vs `2.955447` (exact), `delta=0.59871 gamma=0.019333 vega=38.66681 theta=-5.38040 rho=50.45723`, `recovered implied vol: 0.200000`.

### How do I validate the from-scratch Black-Scholes against an independent Monte Carlo pricer?

`mc_call_price` shares no code with `call_price`. It simulates the exact
terminal GBM distribution and discounts the average payoff, so agreement
between the two is real evidence both are implemented correctly, not just
internally consistent.

```python
from eq_bs_replication import call_price, mc_call_price

S, K, r, sigma, T = 100.0, 100.0, 0.03, 0.20, 1.0
c = call_price(S, K, r, sigma, T)
mc_price, mc_se = mc_call_price(S, K, r, sigma, T, n_paths=500_000, seed=1)
print(f"independent MC call price: {mc_price:.5f} +/- {mc_se:.5f} (analytic: {c:.5f})")
print(f"within 3 stderr: {abs(mc_price - c) <= 3 * mc_se}")
```

Output: `MC call price: 9.39406 +/- 0.01488 (analytic: 9.41340)`, `within 3 stderr: True`.

### What happens at the edges (T→0, sigma→0, deep ITM/OTM) for the equity pricers?

Every pricer in `eq_options` documents and unit-tests these limits explicitly
rather than dividing by zero: `T=0` returns intrinsic value, `sigma=0`
returns the discounted intrinsic on the *forward*, and `implied_vol` raises a
clear `ValueError` (rather than returning a garbage number) when the quoted
price sits at or below the `sigma→0` arbitrage bound.

```python
from eq_options import bs_price, crr_price, implied_vol

# T -> 0: price collapses to intrinsic value
print("T=0 call (deep ITM):", bs_price(S=150, K=100, T=0.0, r=0.03, sigma=0.25, option_type="call"))

# sigma -> 0: discounted intrinsic on the FORWARD, not spot intrinsic
print("sigma=0 call:", bs_price(S=100, K=95, T=1.0, r=0.03, sigma=0.0, option_type="call"))

# deep OTM, short-dated: price underflows toward (but stays above) zero
print("deep OTM short-dated call:", bs_price(S=100, K=200, T=0.05, r=0.03, sigma=0.20, option_type="call"))

# deep ITM: binomial tree and closed form agree to ~5 significant figures
print("deep ITM crr vs bs:",
      crr_price(S=200, K=50, T=1.0, r=0.03, sigma=0.2, option_type="call"),
      bs_price(S=200, K=50, T=1.0, r=0.03, sigma=0.2, option_type="call"))

# implied_vol at/below the sigma->0 arbitrage bound raises, rather than
# returning a meaningless number
try:
    implied_vol(price=4.9, S=100, K=95, T=1.0, r=0.03, option_type="call")
except ValueError as e:
    print("implied_vol below bound raises:", e)
```

Output: `T=0 call: 50`, `sigma=0 call: 7.8077` (the discounted forward intrinsic, *not* `max(S-K,0)=5`), `deep OTM short-dated call: 1.198e-54` (converges cleanly, no error, just numerically tiny), `deep ITM crr vs bs: 151.47772332258143 151.47772332257648` (tree and closed form agree to 1e-10 relative), and the implied-vol call raises `ValueError: price 4.9 is at or below the sigma->0 arbitrage bound 7.807674313; implied vol is undefined`.

### What happens at the edges for Heston: T→0 and vol-of-vol→0?

Heston's Fourier pricer returns exact intrinsic value at `T=0` without going
near the characteristic-function integral, and as vol-of-vol `xi→0` the model
collapses to a Black-Scholes-like price under a deterministic (mean-reverting)
variance path rather than raising or diverging.

```python
from eq_surface import HestonParams, heston_call

p = HestonParams(v0=0.04, kappa=1.5, theta=0.045, rho=-0.6, xi=0.4)
print("T=0 heston call (deep ITM):", heston_call(S=120, K=100, T=0.0, r=0.03, q=0.0, p=p))
print("T=0 heston call (OTM):", heston_call(S=80, K=100, T=0.0, r=0.03, q=0.0, p=p))

# xi -> 0: deterministic variance path, Heston collapses cleanly (no error)
p_det = HestonParams(v0=0.04, kappa=1.5, theta=0.04, rho=0.0, xi=1e-8)
print("near-zero vol-of-vol call:", heston_call(S=100, K=100, T=1.0, r=0.03, q=0.0, p=p_det))
```

Output: `T=0 deep ITM: 20.0`, `T=0 OTM: 0.0` (exact intrinsic, both edges clean), `near-zero vol-of-vol call: 8.288295504662065`. It converges without error rather than a divide-by-`xi` blow-up, because the characteristic function switches to its closed-form deterministic-variance branch below `xi=1e-12`.


## Market Risk: VaR, Expected Shortfall & Backtesting

Setup for every snippet below (once per project, in your venv):

```bash
python -m pip install -e python/equity/03-var-es-engine
python -m pip install -e python/fx/03-var-es-engine
python -m pip install -e python/foundations/01-risk-metrics
```

The equity engine is `eq_var` (multi-asset, `Portfolio` of equity/future/option
positions). The FX engine is `fx_var` (multi-currency `Book` of
cash/spot/forward/option positions). The foundations project is
`eq_risk_metrics` (single-asset, works directly on a returns `Series` with
no portfolio object at all). All three report **VaR/ES as positive numbers
for a loss**. `eq_var` and `fx_var` both take `alpha` as the **tail
probability** (`alpha=0.01` is 99% VaR). `fx_var` rejects `alpha=0.99`
with a `ValueError`; its old `confidence=` keyword is deprecated.

### How do I compute historical VaR and Expected Shortfall on a P&L series?

Plain historical simulation takes the empirical quantile of a portfolio's
historical scenario P&L, with no distributional assumption. `eq_var.historical_var`
and `eq_var.expected_shortfall` both take a raw P&L array (currency units,
loss < 0) and a tail probability `alpha` (0.01 → 99% VaR).

```python
from eq_var.data.synthetic import demo_portfolio, demo_covariance, simulate_returns
from eq_var import historical_var, expected_shortfall

cov = demo_covariance()
port = demo_portfolio()
rets = simulate_returns(1500, cov, dist="t", df=6, seed=1)  # (n_days, n_factors)
pnl = port.pnl(rets)                                        # full revaluation

var99 = historical_var(pnl, alpha=0.01)
es99 = expected_shortfall(pnl, alpha=0.01)
print(f"VaR99 = {var99:,.0f}   ES99 = {es99:,.0f}")
```

Output (synthetic data): `VaR99 = 50,617   ES99 = 67,368`. VaR is an order
statistic of the sorted P&L, not an interpolated quantile. ES exceeds VaR by
construction because it averages the whole tail, not just its boundary.

### How do I compute age-weighted (BRW) historical VaR?

Boudoukh-Richardson-Whitelaw age weighting puts exponentially more weight on
recent scenarios (`w_t ~ lam**age`), so VaR reacts faster after a regime
change than plain equal-weighted historical simulation. `age_weighted_var`
takes the same P&L array plus a decay `lam` (RiskMetrics-style, close to 1).

```python
from eq_var.data.synthetic import demo_portfolio, demo_covariance, simulate_returns
from eq_var import historical_var, age_weighted_var

cov = demo_covariance()
port = demo_portfolio()
rets = simulate_returns(1500, cov, dist="t", df=6, seed=1)
pnl = port.pnl(rets)

plain99 = historical_var(pnl, alpha=0.01)
age99 = age_weighted_var(pnl, alpha=0.01, lam=0.98)
print(f"plain HS VaR99 = {plain99:,.0f}   age-weighted VaR99 = {age99:,.0f}")
```

Output (synthetic data): `plain HS VaR99 = 50,617   age-weighted VaR99 = 45,815`.
Here the recent scenarios happen to be calmer, so the age-weighted figure sits
below plain HS. In a volatility spike the ordering flips.

### How do I compute filtered historical simulation (FHS) VaR?

`factor_filtered_historical_var` devolatilises each risk factor's return
history by its own EWMA volatility, rescales the standardised innovations to
*today's* vol forecast for that factor, and then fully revalues the book under
every filtered scenario. It keeps the empirical (fat-tailed) shape of returns,
makes VaR respond to the current vol regime, and keeps option convexity
because the book is revalued rather than the P&L rescaled. The older
P&L-level `filtered_historical_var(pnl, ...)` is deprecated.

```python
from eq_var.data.synthetic import demo_portfolio, demo_covariance, simulate_returns
from eq_var import historical_var, factor_filtered_historical_var

cov = demo_covariance()
port = demo_portfolio()
rets = simulate_returns(1500, cov, dist="garch", seed=4)  # GARCH -> vol clustering
pnl = port.pnl(rets)

plain99 = historical_var(pnl, alpha=0.01)
fhs99 = factor_filtered_historical_var(port, rets, alpha=0.01, lam=0.94)
print(f"plain HS VaR99 = {plain99:,.0f}   FHS VaR99 = {fhs99:,.0f}")
```

Output (synthetic data): `plain HS VaR99 = 37,975   FHS VaR99 = 30,372`. FHS
rescales to the *current* EWMA vol, which here is calmer than the sample
average, rather than to the unconditional sample tail.

### How do I compute parametric (delta-normal) VaR with a Cornish-Fisher tail correction?

Variance-covariance VaR needs only the portfolio's dollar delta exposures and
a factor covariance matrix (`portfolio_sigma`, `parametric_var`). The
Cornish-Fisher expansion (`cornish_fisher_var`) corrects the normal quantile
for the portfolio's own sample skew/kurtosis, and raises `ValueError` if the
resulting "quantile" would be non-monotone (i.e. not a real quantile function)
for that skew/kurtosis combination.

```python
from scipy.stats import skew, kurtosis
from eq_var.data.synthetic import demo_portfolio, demo_covariance, simulate_returns
from eq_var import parametric_var, cornish_fisher_var, portfolio_sigma

cov = demo_covariance()
port = demo_portfolio()
rets = simulate_returns(1500, cov, dist="t", df=6, seed=1)
pnl = port.pnl(rets)
w = port.delta_exposures()             # dollar delta/vega per factor
sigma = portfolio_sigma(w, cov)

var_normal = parametric_var(w, cov, alpha=0.01, dist="normal")
s, k = skew(pnl), kurtosis(pnl)        # k = excess kurtosis
var_cf = cornish_fisher_var(sigma, alpha=0.01, skew=s, excess_kurt=k)
print(f"delta-normal VaR99 = {var_normal:,.0f}   Cornish-Fisher VaR99 = {var_cf:,.0f}")
```

Output (synthetic data): `delta-normal VaR99 = 46,864   Cornish-Fisher VaR99 = 59,962`.
This portfolio's P&L has positive skew (0.33) and heavy excess kurtosis (4.0) from
its long-gamma option protection, and Cornish-Fisher picks that up where the
plain normal quantile cannot.

### How do I run Monte Carlo VaR with full portfolio revaluation?

`monte_carlo_risk` simulates factor log returns from a covariance matrix,
fully revalues every position scenario by scenario (options via
Black-Scholes, not a Greek approximation), and reports VaR, ES and the
bootstrap standard error of the VaR, all from the same scenario set.
`monte_carlo_var` returns the VaR alone.

```python
from eq_var.data.synthetic import demo_portfolio, demo_covariance
from eq_var import monte_carlo_risk

cov = demo_covariance()
port = demo_portfolio()

risk = monte_carlo_risk(port, cov, alpha=0.01, n_paths=100_000, dist="normal",
                        method="full", seed=1, n_boot=200)
print(f"MC VaR99 = {risk['var']:,.0f}   MC ES99 = {risk['es']:,.0f}   "
      f"bootstrap SE = {risk['var_se']:,.0f}")
```

Output (synthetic data): `MC VaR99 = 45,033   MC ES99 = 51,210   bootstrap SE = 223`.
`risk['pnl']` holds the scenario P&L for further diagnostics.

### How do I build an FX book (spot + forward + option) and run Monte Carlo VaR against it?

`fx_var.Book` holds `Cash`/`Spot`/`Forward`/`Option` positions; `Book.factors`
derives the risk factors it needs (`FX:`, `IR:`, `VOL:` prefixes) directly
from the positions, so you only have to supply a covariance over those
factors. `monte_carlo_var` then simulates and fully revalues (Garman-Kohlhagen
for the option, deposit legs for the forward). Its result carries VaR, ES and
the bootstrap standard error from the same scenarios.

```python
import numpy as np
import pandas as pd
from fx_var import Book, Spot, Forward, Option
from fx_var.data.synthetic import demo_market
from fx_var import monte_carlo_var

market = demo_market()
book = Book([
    Spot("EURUSD", 10_000_000),
    Forward("EURUSD", 5_000_000, 0.5),
    Option("EURUSD", 8_000_000, 1.10, 0.25, "call"),
], base="USD")

factors = book.factors(market)   # ['FX:EUR', 'IR:EUR', 'IR:USD', 'VOL:EURUSD']
daily_vol = {"FX:EUR": 0.075 / np.sqrt(252), "IR:EUR": 8e-5,
             "IR:USD": 8e-5, "VOL:EURUSD": 0.0025}
diag = [daily_vol[f] for f in factors]
cov = pd.DataFrame(np.outer(diag, diag) * np.eye(len(factors)),
                    index=factors, columns=factors)  # uncorrelated toy cov

res = monte_carlo_var(book, market, cov, alpha=0.01, n_scenarios=50_000,
                      dist="normal", seed=1)
print(f"VaR99 = {res.var:,.0f}   ES99 = {res.es:,.0f}   bootstrap SE = {res.se_var:,.0f}")
```

Output (synthetic book and covariance): `VaR99 = 204,226   ES99 = 233,301   bootstrap SE = 1,585`
(base currency, USD). `res.pnl` carries the raw scenario P&L for further diagnostics.

### How do I choose between the normal, Student-t and jump-mixture Monte Carlo distributions?

`simulate_factor_returns` (and `monte_carlo_var`) support three factor
distributions, all matched to the *same* covariance so any VaR difference is
pure tail shape: `"normal"`, `"t"` (fatter tails, covariance-matched), and
`"jump"` (a Bernoulli common-jump overlay via `JumpSpec`, for the peg-break
or devaluation case a covariance matrix alone can't see).

```python
import numpy as np
import pandas as pd
from fx_var import Book, Spot, Forward, Option, JumpSpec, monte_carlo_var
from fx_var.data.synthetic import demo_market

market = demo_market()
book = Book([
    Spot("EURUSD", 10_000_000),
    Forward("EURUSD", 5_000_000, 0.5),
    Option("EURUSD", 8_000_000, 1.10, 0.25, "call"),
], base="USD")
factors = book.factors(market)
daily_vol = {"FX:EUR": 0.075 / np.sqrt(252), "IR:EUR": 8e-5,
             "IR:USD": 8e-5, "VOL:EURUSD": 0.0025}
diag = [daily_vol[f] for f in factors]
cov = pd.DataFrame(np.outer(diag, diag) * np.eye(len(factors)),
                    index=factors, columns=factors)

res_n = monte_carlo_var(book, market, cov, alpha=0.01, n_scenarios=50_000, dist="normal", seed=1)
res_t = monte_carlo_var(book, market, cov, alpha=0.01, n_scenarios=50_000, dist="t", df=5, seed=1)
jumps = JumpSpec(prob=0.01, mean={"FX:EUR": -0.05}, std={"FX:EUR": 0.02})  # 1%/day devaluation risk
res_j = monte_carlo_var(book, market, cov, alpha=0.01, n_scenarios=50_000,
                        dist="jump", jumps=jumps, seed=1)
print(f"normal = {res_n.var:,.0f}   t(5) = {res_t.var:,.0f}   jump = {res_j.var:,.0f}")
```

Output (synthetic data): `normal = 204,226   t(5) = 233,699   jump = 300,930`.
Same covariance, same seed, and VaR climbs as the tail model gets fatter.

### How do I get a bootstrap standard-error cross-check alongside my VaR point estimate?

Every VaR quantile is itself an estimate with sampling noise. The bootstrap
standard error (`var_standard_error_bootstrap`, distribution-free resampling)
is the primary estimate: `monte_carlo_var` in `fx_var` and `monte_carlo_risk`
in `eq_var` report it. `var_standard_error` (asymptotic, Gaussian-KDE density
at the quantile) is kept as a diagnostic. It is fast but understated at deep
tails and small scenario counts.

```python
import numpy as np
import pandas as pd
from fx_var import (Book, Spot, Forward, Option, monte_carlo_var,
                    var_standard_error, var_standard_error_bootstrap)
from fx_var.data.synthetic import demo_market

market = demo_market()
book = Book([
    Spot("EURUSD", 10_000_000),
    Forward("EURUSD", 5_000_000, 0.5),
    Option("EURUSD", 8_000_000, 1.10, 0.25, "call"),
], base="USD")
factors = book.factors(market)
daily_vol = {"FX:EUR": 0.075 / np.sqrt(252), "IR:EUR": 8e-5,
             "IR:USD": 8e-5, "VOL:EURUSD": 0.0025}
diag = [daily_vol[f] for f in factors]
cov = pd.DataFrame(np.outer(diag, diag) * np.eye(len(factors)),
                    index=factors, columns=factors)

res = monte_carlo_var(book, market, cov, alpha=0.01, n_scenarios=50_000, dist="normal", seed=1)
se_boot = var_standard_error_bootstrap(res.pnl, alpha=0.01, n_boot=500, seed=2)
se_kde = var_standard_error(res.pnl, alpha=0.01)
print(f"VaR99 = {res.var:,.0f}   reported SE = {res.se_var:,.0f}   "
      f"bootstrap SE (500) = {se_boot:,.0f}   KDE SE = {se_kde:,.0f}")
```

Output (synthetic data): `VaR99 = 204,226   reported SE = 1,585   bootstrap SE (500) = 1,496   KDE SE = 1,447`.
The reported SE is a 200-resample bootstrap, so it differs from a 500-resample
rerun by resampling noise. The estimators agree closely here (50k scenarios,
99%). They diverge at deeper tails or smaller scenario counts, which is when
the bootstrap number matters.

### How do I compute FX historical VaR directly from a Book (plain, age-weighted, or FHS)?

`fx_var.historical_var` takes the book, a market snapshot, and a factor-return
history, and does the revaluation for you, so there is no scenario matrix to
build by hand. Passing `method="plain"|"age"|"fhs"` switches the weighting
scheme as in the equity engine.

```python
from fx_var import historical_var
from fx_var.data.synthetic import demo_market, demo_book, simulate_history

market = demo_market()
book = demo_book()  # G10 spots, an EURUSD forward+call, EM leg, HKD peg leg
returns = simulate_history(book, market, n_days=1000, seed=5, garch=True)

plain = historical_var(book, market, returns, alpha=0.01, method="plain")
fhs = historical_var(book, market, returns, alpha=0.01, method="fhs")
print(f"plain VaR99 = {plain.var:,.0f}   FHS VaR99 = {fhs.var:,.0f}")
print("flagged peg factors:", plain.flagged_peg_factors)
```

Output (synthetic data): `plain VaR99 = 749,952   FHS VaR99 = 1,733,918`
(base currency, USD), plus a `PegBlindnessWarning` naming `FX:HKD`. The
book's HKD peg leg realises near-zero historical vol, so HS cannot see a peg
break there. The warning tells you to add the peg-break stress add-on
separately (`fx_var.stress_testing.peg_break_scenario`).

### How do I compute FX parametric VaR and compare a normal to a Student-t tail?

The FX variance-covariance engine (`parametric_var`) linearises the book into
factor exposures (`Book.linear_exposures`, options entering via GK delta/vega)
and applies the normal or Student-t quantile to `sigma_p = sqrt(w' Sigma w)`.

```python
from fx_var import parametric_var
from fx_var.data.synthetic import demo_market, demo_book, simulate_history

market = demo_market()
book = demo_book()
returns = simulate_history(book, market, n_days=1000, seed=5, garch=True)

res_n = parametric_var(book, market, returns, alpha=0.01, dist="normal", cov_method="sample")
res_t = parametric_var(book, market, returns, alpha=0.01, dist="t", df=5, cov_method="sample")
print(f"1-day sigma = {res_n.sigma:,.0f}")
print(f"normal VaR/ES = {res_n.var:,.0f} / {res_n.es:,.0f}")
print(f"t(5)   VaR/ES = {res_t.var:,.0f} / {res_t.es:,.0f}")
```

Output (synthetic data): `1-day sigma = 356,859`, `normal VaR/ES = 830,179 / 951,107`,
`t(5) VaR/ES = 930,141 / 1,230,750`. Same sigma; the fatter Student-t tail
moves both VaR and ES up.

### How do I backtest a VaR series with Kupiec's POF test?

`rolling_var_backtest` walk-forwards a VaR estimator over a P&L history
(each day's VaR is forecast from the trailing window only, so it is
out-of-sample), and `kupiec_pof` runs the proportion-of-failures
likelihood-ratio test on the resulting exception count against the nominal
rate `alpha`. `kupiec_pof` takes keyword arguments:
`kupiec_pof(n_obs=..., n_exceptions=..., alpha=...)`. The positional form is
deprecated.

```python
from eq_var.data.synthetic import demo_portfolio, demo_covariance, simulate_returns
from eq_var import historical_var, rolling_var_backtest, kupiec_pof

cov = demo_covariance()
port = demo_portfolio()
rets = simulate_returns(1500, cov, dist="t", df=6, seed=3)
pnl = port.pnl(rets)

var_fn = lambda hist_pnl, alpha: historical_var(hist_pnl, alpha)
bt = rolling_var_backtest(pnl, var_fn, window=250, alpha=0.01, name="historical-99")
print(bt.summary())

kp = kupiec_pof(n_obs=bt.n_obs, n_exceptions=bt.n_exceptions, alpha=0.01)
print(f"LR = {kp['lr']:.3f}   p = {kp['pvalue']:.3f}   expected = {kp['expected']:.1f}")
```

Output (synthetic data): 12 exceptions in 1,250 forecast days (expected 12.5),
`LR = 0.020, p = 0.886`. Not rejected: the exception rate matches the stated
99% level. `bt.summary()` also reports the Christoffersen
independence/conditional-coverage p-values and, at `alpha=0.01`, the 250-day
Basel zone the exception count scales to.

### How do I classify a backtest into a Basel traffic-light zone?

`basel_traffic_light(n_obs=..., n_exceptions=..., alpha=...)` classifies an
exception count for any window and any tail probability. The zone comes from
the cumulative Binomial(`n_obs`, `alpha`) probability: green below 0.95,
yellow below 0.9999, red otherwise. On 250 days at 99% this is the regulatory
table: 0-4 green (multiplier 3.0), 5-9 yellow (3.40 / 3.50 / 3.65 / 3.75 /
3.85), 10+ red (4.0). The positional form `basel_traffic_light(n_exceptions)`
is deprecated. `basel_zone_probabilities` tabulates the boundary.

```python
from eq_var import basel_traffic_light, basel_zone_probabilities

for n_exceptions in (3, 7, 11):
    tl = basel_traffic_light(n_obs=250, n_exceptions=n_exceptions, alpha=0.01)
    print(n_exceptions, tl["zone"], tl["multiplier"], round(tl["cumulative_prob"], 5))

print(basel_zone_probabilities(n_obs=250, alpha=0.01).head(11).to_string(index=False))
```

Output:
```
3 green 3.0 0.75812
7 yellow 3.65 0.99597
11 red 4.0 0.99999
```
The result dict also carries `plus_factor`. The full table shows
`prob_cumulative` crossing the 0.95 (green to yellow) and 0.9999 (yellow to
red) boundaries at 4 to 5 and 9 to 10 exceptions.

### How do I run the single-asset foundations risk-metrics toolkit end to end?

`eq_risk_metrics` is the portfolio's single-asset starting point: it works
directly on a `pandas.Series` of returns (no `Portfolio`/`Book` object), and
its central finding is that Gaussian VaR understates the tail a fat-tailed
historical sample realises.

```python
import eq_risk_metrics as rm
from eq_risk_metrics.data import generate

df = generate(n_days=1500, seed=2)
prices = df.set_index("Date")["Adj Close"]
rets = rm.simple_returns(prices)

vol = rm.annualised_volatility(rets)
var_h = rm.var_historical(rets, 0.99)
var_p = rm.var_parametric(rets, 0.99)
es = rm.expected_shortfall(rets, 0.99)
bt = rm.kupiec_pof_test(rets, var_h, 0.99)
dd = rm.max_drawdown(prices)

print(f"annualised vol = {vol:.2%}")
print(f"VaR99: historical={var_h:.2%} parametric={var_p:.2%}")
print(f"ES99 = {es:.2%}   max drawdown = {dd['max_drawdown']:.2%}")
print(f"Kupiec: {bt['n_exceptions']} exceptions / {bt['n_observations']} days, "
      f"p={bt['p_value']:.3f}, reject={bt['reject_at_5pct']}")

try:
    rm.var_cornish_fisher(rets, 0.99)
except ValueError as e:
    print("Cornish-Fisher refuses:", e)
```

Output (synthetic data): `annualised vol = 16.21%`; `VaR99: historical=3.08%
parametric=2.27%`; `ES99 = 4.10%`, `max drawdown = -18.80%`; Kupiec
`14 exceptions / 1499 days, p=0.795, reject=False`. Historical VaR sits well
above the Gaussian figure because the synthetic sample carries fat tails,
which is this project's headline point (see `docs/METHODOLOGY.md`). The
Cornish-Fisher call raises `ValueError`: with skew -0.66 and excess kurtosis
10.24 the expansion is not monotone at 99%, so it is outside its domain and
the function refuses rather than return a number.

### How do I confirm NaN/Inf inputs raise cleanly instead of silently producing a wrong number?

Every layer of this portfolio treats NaN/Inf as a defect to reject, never to
silently propagate or impute. A single bad value would otherwise poison a
quantile, a covariance, or an entire Monte Carlo run without raising anywhere
on the path. All three engines below raise `ValueError` immediately, with a
message naming the offending input.

```python
import numpy as np
import pandas as pd

# 1. eq_var: a NaN in a raw P&L array
from eq_var import historical_var
pnl = np.random.default_rng(0).normal(0, 1000, 200)
pnl[50] = np.nan
try:
    historical_var(pnl, alpha=0.01)
except ValueError as e:
    print("eq_var:", e)

# 2. eq_risk_metrics: an Inf in a returns Series
from eq_risk_metrics import var_historical
rets = pd.Series(np.random.default_rng(0).normal(0, 0.01, 300))
rets.iloc[10] = np.inf
try:
    var_historical(rets, 0.95)
except ValueError as e:
    print("eq_risk_metrics:", e)

# 3. fx_var: a NaN interest rate at Market construction time
from fx_var import Market
try:
    Market(spot_usd={"EUR": 1.08}, rates={"EUR": np.nan})
except ValueError as e:
    print("fx_var:", e)
```

Output:
```
eq_var: pnl contains NaN or infinite values
eq_risk_metrics: var_historical: returns contains 1 non-finite value(s) (NaN/inf). ...
fx_var: rates['EUR'] must be a finite number, got nan (NaN policy: refuse, never impute)
```
Note the `fx_var` case rejects at `Market` construction, before any VaR
calculation runs, because a NaN rate would otherwise flow silently into
a NaN Cholesky factor and a NaN VaR with no exception raised anywhere.


## Fixed Income, Pairs Trading, Credit Risk & Portfolio Optimization

Setup shared by every snippet below: each project is an installable package
under `python/<asset-class>/<NN>-<project>/`. Install the one you need into
your venv with `python -m pip install -e python/<asset-class>/<NN>-<project>`,
then run the snippet with `python`. Every snippet uses fixed seeds and runs
offline.

### How do I price a fixed-rate bond and compute its yield to maturity?

`fi_rates` prices bonds from a street-convention YTM (`price_from_ytm`) or
solves YTM from a clean price by Brent root-find (`ytm_from_price`); dirty
price minus accrued interest gives the clean price. Treasuries and other
government bonds use the `ACT/ACT-ICMA` day count.

```python
import datetime as dt
from fi_rates import FixedRateBond, price_from_ytm, ytm_from_price, accrued_interest

bond = FixedRateBond(
    effective=dt.date(2023, 5, 15),
    maturity=dt.date(2033, 5, 15),
    coupon=0.045,
    frequency=2,
    daycount="ACT/ACT-ICMA",
)
settlement = dt.date(2025, 8, 15)

dirty = price_from_ytm(bond, settlement, ytm=0.0425)
clean = dirty - accrued_interest(bond, settlement)
print(f"dirty price: {dirty:.4f}")
print(f"clean price: {clean:.4f}")

y = ytm_from_price(bond, settlement, clean_price=clean)
print(f"round-tripped YTM: {y:.6%}")
```

Output: `dirty price: 102.7552`, `clean price: 101.6302`, and the round-tripped
YTM recovers `4.250000%` (price to YTM to price is tested to 1e-10).

### How do I compute a bond's duration and convexity?

Macaulay/modified duration and convexity are closed-form functions of the
bond's cashflows and YTM (`docs/METHODOLOGY.md` derives them); a numerical
central-difference version (`numerical_modified_duration`,
`numerical_convexity`) exists for cross-checking.

```python
import datetime as dt
from fi_rates import FixedRateBond, macaulay_duration, modified_duration, convexity

bond = FixedRateBond(
    effective=dt.date(2023, 5, 15),
    maturity=dt.date(2033, 5, 15),
    coupon=0.045,
    frequency=2,
    daycount="ACT/ACT-ICMA",
)
settlement = dt.date(2025, 8, 15)
ytm = 0.0425

d_mac = macaulay_duration(bond, settlement, ytm)
d_mod = modified_duration(bond, settlement, ytm)
c = convexity(bond, settlement, ytm)

print(f"Macaulay duration: {d_mac:.4f} yr")
print(f"Modified duration: {d_mod:.4f} yr")
print(f"Convexity:         {c:.4f}")
```

Output: Macaulay `6.5681` yr, modified `6.4314` yr, convexity `49.1813`.

### How do I compute DV01 for a bond?

`dv01` gives the analytic (YTM-space) DV01 in $-per-bp; `dv01_curve` gives
the "effective" DV01 by bumping every pillar of a `DiscountCurve` by ±1bp and
repricing. The two agree closely but are not identical, since one holds the
street-convention yield fixed and the other holds the curve shape fixed.

```python
import datetime as dt
from fi_rates import (
    FixedRateBond, Deposit, ParSwap, bootstrap_curve, dv01, dv01_curve,
)

bond = FixedRateBond(
    effective=dt.date(2023, 5, 15), maturity=dt.date(2033, 5, 15),
    coupon=0.045, frequency=2, daycount="ACT/ACT-ICMA",
)
settlement = dt.date(2025, 8, 15)

curve = bootstrap_curve([
    Deposit(maturity=0.25, rate=0.0430),
    Deposit(maturity=0.50, rate=0.0435),
    ParSwap(maturity=2, rate=0.0410, frequency=2),
    ParSwap(maturity=5, rate=0.0395, frequency=2),
    ParSwap(maturity=10, rate=0.0405, frequency=2),
])

print(f"analytic DV01 (per bond, YTM=4.25%): {dv01(bond, settlement, 0.0425):.5f}")
print(f"effective DV01 off the bootstrapped curve: {dv01_curve(bond, settlement, curve):.5f}")
```

Output: analytic `0.06609`, curve-effective `0.06863` per $100 face per 1bp:
close but not identical, because the curve's own shape (not a flat yield)
determines the second number.

### How do I bootstrap and query a yield curve?

`bootstrap_curve` sequentially solves each pillar's discount factor from
deposits/FRAs/par swaps so every quoted instrument reprices exactly (to
~1e-16 for the local interpolations); the resulting `DiscountCurve` exposes
`zero_rate`, `df`, `forward_rate` and `par_rate`.

```python
from fi_rates import Deposit, ParSwap, bootstrap_curve

instruments = [
    Deposit(maturity=0.25, rate=0.0430),
    Deposit(maturity=0.50, rate=0.0435),
    ParSwap(maturity=2, rate=0.0410, frequency=2),
    ParSwap(maturity=5, rate=0.0395, frequency=2),
    ParSwap(maturity=10, rate=0.0405, frequency=2),
]
curve = bootstrap_curve(instruments, interpolation="loglinear_df")

for t in (0.25, 1.0, 2.0, 5.0, 10.0):
    print(f"t={t:>5.2f}y  zero={curve.zero_rate(t):.4%}  df={curve.df(t):.6f}")

print(f"par 5y rate off the curve: {curve.par_rate(5.0, frequency=2):.4%} (quoted 3.95%)")
```

Output shows zeros from `4.28%` (3m) down to `3.90%` (5y) and back up to
`4.02%` (10y), and the par-rate query reprices the 5y swap quote exactly
(`3.9500%`). `interpolation` also accepts `"linear_zero"` (sawtooth forwards)
and `"pchip_zero"` (smooth but not perfectly local; see the edge case below).

### How do I price an FX forward via covered interest parity and see the cross-currency basis?

`fx_rates` builds domestic/foreign discount curves from money-market and par-
swap quotes, then `cip_forward` gives the pure covered-interest-parity
forward while `market_forward` adds the (basis-adjusted) foreign curve; the
gap between the two is the cross-currency basis in price terms.

```python
from fx_rates.data import build_market_state, generate_market_quotes
from fx_rates import cip_forward, market_forward, FXForward, fx_delta, dv01, basis_dv01

quotes = generate_market_quotes("normal", seed=42)
market = build_market_state(quotes)
print(f"EURUSD spot: {market.spot:.4f}")

f_cip = cip_forward(market.spot, market.domestic_curve, market.foreign_curve, 5.0)
f_mkt = market_forward(market, 5.0)
print(f"5y CIP forward (no basis): {f_cip:.4f}")
print(f"5y market forward (basis): {f_mkt:.4f}  ({(f_mkt - f_cip) * 1e4:+.1f} pips vs CIP)")

fwd = FXForward(notional_base=10_000_000.0, strike=f_mkt, expiry=5.0)
print(f"FX delta: {fx_delta(fwd, market):,.0f} USD per 1.00 spot move")
print(f"Domestic (USD) DV01: {dv01(fwd, market, 'domestic'):,.2f}")
print(f"Foreign  (EUR) DV01: {dv01(fwd, market, 'foreign'):,.2f}")
print(f"Basis DV01:          {basis_dv01(fwd, market):,.2f}")
```

Output (synthetic quotes): spot `1.0866`, CIP forward `1.1719` vs market
forward `1.1867` (`+147.9` pips of basis), FX delta `8,914,135` USD, domestic
DV01 `+4,843.18` against foreign DV01 `-4,840.99`, and basis DV01 `-4,810.23`.
The opposite signs of the two rate DV01s are the long-base-forward pattern
the test suite pins.

### How do I see fixed-income risk models break down for a non-parallel move?

A single duration number is a *parallel*-shift approximation. `fi_rates`
ships a scenario engine that full-reprices a portfolio under a steepener and
compares it against the duration+convexity Taylor estimate. This is the
documented failure mode in `docs/VALIDATION.md` §4.3.

```python
import datetime as dt
import pandas as pd
from fi_rates import (
    Deposit, ParSwap, bootstrap_curve, FixedRateBond, Position,
    scenario_pnl_table, steepener_scenario, parallel_scenario,
)

curve = bootstrap_curve([
    Deposit(maturity=0.25, rate=0.0430),
    Deposit(maturity=0.50, rate=0.0435),
    ParSwap(maturity=2, rate=0.0410, frequency=2),
    ParSwap(maturity=5, rate=0.0395, frequency=2),
    ParSwap(maturity=10, rate=0.0405, frequency=2),
    ParSwap(maturity=30, rate=0.0430, frequency=2),
])
bond2y = FixedRateBond(effective=dt.date(2023, 1, 1), maturity=dt.date(2027, 1, 1), coupon=0.04, frequency=2)
bond30y = FixedRateBond(effective=dt.date(2023, 1, 1), maturity=dt.date(2053, 1, 1), coupon=0.045, frequency=2)
settlement = dt.date(2025, 8, 15)
positions = [
    Position(bond2y, quantity=1_000_000 / 100, label="2y"),
    Position(bond30y, quantity=1_000_000 / 100, label="30y"),
]
scenarios = [parallel_scenario(100), steepener_scenario(bp_short=-50, bp_long=50)]
print(scenario_pnl_table(positions, settlement, curve, scenarios))
```

Output (a 2y+30y barbell): a table with `pnl_full`, `pnl_dur_conv`, `error`
and `avg_shift_bp` per scenario. The +100bp parallel shock has a
duration+convexity error of about 2.4% of the full-repricing P&L
(`3,894` on `-162,817`), but the steepener (row `tilt_-50/+50@5y`) has an
error of about 47% (`25,243` on `-53,291`). A single duration number is
nearly useless for a curve twist, the point `docs/VALIDATION.md` makes with
its own barbell example.

### How do I test two price series for cointegration?

`eq_pairs` implements the Engle-Granger two-step test from scratch, including
the MacKinnon (2010) N=2 critical-value surface that the residual-based test
requires (plain ADF critical values badly over-reject cointegration
as real).

```python
from eq_pairs.data.synthetic import cointegrated_pair
from eq_pairs import engle_granger

prices, truth = cointegrated_pair(n=1000, beta=1.5, kappa=0.03, sigma=1.2, seed=7)
print(f"true beta: {truth.beta}, true kappa: {truth.kappa}")

result = engle_granger(prices["Y"], prices["X"])
print(f"estimated hedge ratio (beta): {result.beta:.4f}")
print(f"EG ADF statistic: {result.stat:.4f}")
print(f"5% critical value (N=2): {result.crit['5%']:.4f}")
print(f"cointegrated at 5%: {result.cointegrated('5%')}")
```

Output: true `beta=1.5` recovered as `1.4586`; ADF stat `-3.8883` beats the
N=2 5% critical value `-3.3423`, so `cointegrated=True`.

### How do I estimate a hedge ratio and construct a spread?

The static hedge ratio comes from an OLS cointegrating regression
(`hedge_ratio`); the resulting dollar spread is fit as an Ornstein-Uhlenbeck
process (`fit_ou_ols`), which gives mean-reversion speed, half-life and the
stationary standard deviation used to size z-score bands.

```python
from eq_pairs.data.synthetic import cointegrated_pair
from eq_pairs import hedge_ratio, compute_spread, fit_ou_ols

prices, truth = cointegrated_pair(n=1000, beta=1.5, kappa=0.03, sigma=1.2, seed=7)

beta, alpha, resid = hedge_ratio(prices["Y"], prices["X"])
spread = compute_spread(prices["Y"], prices["X"], beta, alpha)

ou = fit_ou_ols(spread.to_numpy())
print(f"hedge ratio beta={beta:.4f}, alpha={alpha:.4f}")
print(f"OU kappa={ou.kappa:.5f}/day, mu={ou.mu:.4f}, sigma={ou.sigma:.4f}")
print(f"half-life: {ou.half_life:.1f} trading days")
print(f"stationary std: {ou.stationary_std:.4f}")
```

Output: `beta=1.4586`, OU half-life `22.9` trading days, stationary std
`5.0097` dollars, the natural scale for entry/exit z-score thresholds.

### How do I generate a z-score entry/exit signal and backtest a pairs trade?

`generate_signals` runs an entry/exit/stop state machine over a rolling
z-score. `backtest_pair` takes the signal at the close of day t and fills at
the modelled price `p_t + f (p_{t+1} - p_t)` with `f = 0.5` by default
(`ExecutionConfig`; a close-auction mode fills at `p_t` and pays an auction
cost). It charges commission, slippage and borrow, and `summary` rolls the
result up into Sharpe, hit rate, drawdown and so on.

```python
from eq_pairs.data.synthetic import cointegrated_pair
from eq_pairs import (
    hedge_ratio, compute_spread, zscore_rolling, generate_signals,
    SignalRules, backtest_pair, CostModel, summary,
)

prices, truth = cointegrated_pair(n=1000, beta=1.5, kappa=0.03, sigma=1.2, seed=7)
beta, alpha, _ = hedge_ratio(prices["Y"], prices["X"])
spread = compute_spread(prices["Y"], prices["X"], beta, alpha)
z = zscore_rolling(spread, window=60)

rules = SignalRules(entry_z=2.0, exit_z=0.5, stop_z=4.0)
signals = generate_signals(z, rules)

result = backtest_pair(
    prices["Y"], prices["X"], target=signals["position"], beta=beta,
    name="synthetic_pair", costs=CostModel(cost_bps=5, slippage_bps=2, borrow_bps=50),
    gross=1_000_000.0,
)
perf = summary(result.daily, result.trades, result.ledger, capital=1_000_000.0)
print(f"net P&L: ${perf['total_net_pnl']:,.0f}")
print(f"Sharpe: {perf['sharpe']:.3f}   hit rate: {perf['hit_rate']:.1%}   n_trades: {perf['n_trades']}")
```

Output (synthetic pair): 15 round trips, `80.0%` hit rate, Sharpe `0.838`,
net P&L `$546,571` on $1mm gross after 5bp commission, 2bp slippage and 50bp
annualised borrow, with the default t+1 modelled fill.

### How do I test an FX pair for cointegration and backtest it?

`fx_pairs` mirrors the equity pipeline on FX cross rates (AUDUSD/NZDUSD-style
commodity-bloc pairs), but the backtester also decomposes P&L into spot vs
carry, since a mean-reverting-looking FX spread is often carry in disguise.

```python
import numpy as np
from fx_pairs.data.synthetic import make_cointegrated_pair
from fx_pairs import engle_granger, log_spread, zscore, generate_positions, run_backtest, summarize

p1, p2, truth = make_cointegrated_pair(n=1000, beta=1.0, kappa=20.0, sigma_ou=0.05, seed=5)
print(f"true beta={truth['beta']}, half-life~{truth['half_life_days']:.1f} days")

res = engle_granger(np.log(p1), np.log(p2))
print(f"EG stat={res.stat:.4f}  5% crit={res.crit_values['5%']:.4f}  cointegrated={res.cointegrated}")

spread = log_spread(p1, p2, beta=res.beta, alpha=res.alpha)
z = zscore(spread, window=60)
positions, trades = generate_positions(z, entry=2.0, exit_=0.5, stop=4.0)

bt = run_backtest(p1, p2, positions, beta=res.beta, pair1="AUDUSD", pair2="NZDUSD", trades=trades)
perf = summarize(bt)
print(f"total P&L: {perf['total_pnl']:.4f}   Sharpe: {perf['sharpe']:.3f}   n_trades: {perf['n_trades']:.0f}")
```

Output: half-life `~8.7` days, ADF stat `-6.96` well past the 5% critical
value (`cointegrated=True`), and the backtest with no carry input (spot-only)
turns in Sharpe `1.88` over 23 trades on this synthetic pair.

### How do I see the "spurious correlation" trap in pairs trading?

Two independent random walks with correlated *increments* have high return
correlation by construction but no cointegrating relationship. This is the
classic trap `eq_pairs` ships a dedicated generator for.

```python
import numpy as np
from eq_pairs.data.synthetic import correlated_random_walks
from eq_pairs import engle_granger

prices, truth = correlated_random_walks(n=1000, rho=0.9, seed=3)
ret_corr = np.corrcoef(prices["A"].diff().dropna(), prices["B"].diff().dropna())[0, 1]
print(f"return correlation: {ret_corr:.4f}")

result = engle_granger(prices["A"], prices["B"])
print(f"EG ADF statistic: {result.stat:.4f}  vs 5% crit {result.crit['5%']:.4f}")
print(f"cointegrated at 5%: {result.cointegrated('5%')}  (correctly NOT cointegrated)")
```

Output: return correlation `0.9023` (looks like a great pairs candidate on a
correlation screen), but the EG statistic `-2.6148` fails to beat the 5%
critical value `-3.3423`. The cointegration test refuses the pair
that a naive correlation screen would have flagged as tradeable.

### How do I estimate a PD model from borrower features?

`eq_credit` fits a from-scratch IRLS/Newton-Raphson logistic regression
(cross-checked against sklearn in the test suite) on a synthetic loan book
with a known ground-truth PD model, so discrimination can be checked against
truth.

```python
from eq_credit.data.synthetic import generate_loan_book
from eq_credit import fit_logistic, roc_auc, ks_statistic

df = generate_loan_book(n_loans=5000, seed=42)
features = ["leverage", "interest_coverage", "current_ratio", "roa", "log_assets", "behavioral_score"]
df = df.dropna(subset=features)
X = df[features]
y = df["default"].to_numpy()

fit = fit_logistic(X, y, feature_names=features)
print(fit.summary().round(4))

pd_hat = fit.predict_proba(X)
print(f"mean predicted PD: {pd_hat.mean():.4%}  vs realised default rate: {y.mean():.4%}")
print(f"AUC: {roc_auc(y, pd_hat):.4f}   KS: {ks_statistic(y, pd_hat):.4f}")
```

Output: all six features come back significant (p<0.001 except
`current_ratio`, whose true effect is U-shaped and gets washed out by a
linear logit, the case WOE binning exists to fix); AUC `0.766`, KS
`0.437`, mean predicted PD ties the realised default rate exactly (`2.37%`)
by construction of MLE.

### How do I compute expected loss and Basel IRB capital from PD/LGD/EAD?

`expected_loss` sums `PD × LGD × EAD` across the book; `basel_k` implements
the exact Basel II/III corporate IRB formula (asset correlation + maturity
adjustment + the single-factor Vasicek capital term), and RWA/capital follow
from it directly.

```python
from eq_credit.data.synthetic import generate_loan_book
from eq_credit import expected_loss, basel_k, risk_weighted_assets

df = generate_loan_book(n_loans=2000, seed=42)
pd_ = df["true_pd"].to_numpy()
lgd = df["lgd"].to_numpy()
ead = df["ead"].to_numpy()

el_per_loan, portfolio_el = expected_loss(pd_, lgd, ead)
print(f"portfolio EAD: ${ead.sum():,.0f}")
print(f"portfolio EL:  ${portfolio_el:,.0f}  ({portfolio_el / ead.sum():.3%} of EAD)")

k = basel_k(pd_, lgd, maturity=2.5)
rwa = risk_weighted_assets(k, ead)
capital = 0.08 * rwa
print(f"portfolio RWA: ${rwa.sum():,.0f}")
print(f"8% regulatory capital: ${capital.sum():,.0f}  ({capital.sum() / ead.sum():.3%} of EAD)")
```

Output: EAD `$2.89bn`, EL `$41.8m` (`1.446%` of EAD), RWA `$3.64bn`, 8%
regulatory capital `$291.2m` (`10.08%` of EAD), all on a synthetic loan book.
Capital sits well above EL, which is the point of the unexpected-loss buffer.

### How do I compute counterparty exposure and expected loss for an FX forward?

`fx_credit` simulates spot under Garman-Kohlhagen GBM to build an expected
exposure (EE) / potential future exposure (PFE) profile for an FX forward,
then integrates EE against a PD term structure (flat hazard from the 1y PD).
`expected_exposure_loss_for_forward` returns the expected loss on exposure
(real-world PD), not a CVA: a CVA needs risk-neutral default probabilities
implied from market spreads. The old name `cva_for_forward` is a deprecated
alias.

```python
from fx_credit import FXForward, expected_exposure_loss_for_forward

fwd = FXForward(pair="EURUSD", notional_base=10_000_000.0, strike=1.10, maturity=2.0, buy_base=True)

loss, profile = expected_exposure_loss_for_forward(
    fwd, spot=1.0866, vol=0.09, r_d=0.041, r_f=0.026,
    pd_1y=0.015, lgd=0.6, n_steps=24, n_paths=20_000, seed=1,
)
print(f"expected loss on exposure: ${loss:,.0f}")
print(f"peak EE: ${profile.ee.max():,.0f}")
print(f"peak 95% PFE: ${profile.peak_pfe(0.95):,.0f}")
```

Output (simulated paths): expected loss on exposure `$8,059` on a $10m 2y
forward at 1.5% real-world PD and 60% LGD, peak expected exposure `$660,249`
and peak 95% PFE `$2,694,905`. Because a forward
has no intermediate cashflows, exposure grows monotonically to maturity
(`sqrt(t)`-shaped) rather than showing the amortising "mid-life hump" of a
swap.

### How do I see credit models fail with zero defaults?

Both the logistic MLE and every downstream validation statistic are
undefined with zero observed defaults; `eq_credit` detects this explicitly
and raises rather than returning a silently degenerate fit.

```python
import numpy as np
from eq_credit import fit_logistic

X = np.random.default_rng(0).normal(size=(200, 3))
y = np.zeros(200)  # no defaults in this sample

try:
    fit_logistic(X, y)
except ValueError as exc:
    print(f"ValueError: {exc}")
```

Output: `ValueError: zero defaults in sample: logistic MLE is degenerate
(intercept -> -inf). Collect more data or use a low-default-portfolio
calibration approach.`

### How do I run a mean-variance optimization to get efficient-frontier weights?

`eq_port` traces the efficient frontier in closed form via the two-fund
theorem and gives closed-form tangency (max-Sharpe) and minimum-variance
weights directly from the sample covariance.

```python
from eq_port.data.synthetic import generate_panel
from eq_port import sample_cov, efficient_frontier, tangency_weights, min_variance_weights

panel = generate_panel(n_assets=6, n_periods=1000, seed=1)
mu = panel.true_mean
cov = sample_cov(panel.returns)

frontier = efficient_frontier(mu, cov, n_points=10)
print("frontier vols (ann.):", [f"{v * (252 ** 0.5):.2%}" for v in frontier.vols[:5]])
print("frontier rets (ann.):", [f"{r * 252:.2%}" for r in frontier.returns[:5]])

w_tan = tangency_weights(mu, cov)
w_mv = min_variance_weights(cov)
print(f"tangency weights: {[f'{w:.3f}' for w in w_tan]}  sum={w_tan.sum():.4f}")
print(f"min-var weights:  {[f'{w:.3f}' for w in w_mv]}  sum={w_mv.sum():.4f}")
```

Output: the frontier's low-risk end runs from `14.5%` to `15.6%` annualised
vol for `5.6%`–`7.2%` return; both weight vectors sum to 1.0000 exactly
(short positions allowed, since these are the unconstrained closed forms).

### How do I run a minimum-variance / risk-parity allocation and compute risk contributions?

`erc_weights` solves the long-only equal-risk-contribution problem by
cyclical coordinate descent; `risk_contributions` reports each asset's
variance contribution, which by Euler's theorem sums exactly to the
portfolio variance.

```python
from eq_port.data.synthetic import generate_panel
from eq_port import sample_cov, erc_weights, inverse_vol_weights, risk_contributions

panel = generate_panel(n_assets=6, n_periods=1000, seed=1)
cov = sample_cov(panel.returns)

w_erc = erc_weights(cov)
w_ivp = inverse_vol_weights(cov)

rc = risk_contributions(w_erc, cov)
pct_rc = rc / rc.sum()

print("ERC weights:        ", [f"{w:.3f}" for w in w_erc])
print("inverse-vol weights:", [f"{w:.3f}" for w in w_ivp])
print("ERC risk contributions (%):", [f"{p:.3%}" for p in pct_rc])
print(f"sum(RC) == portfolio variance: {rc.sum():.8f} vs {w_erc @ cov @ w_erc:.8f}")
```

Output: ERC weights range `13.2%`–`21.9%` across 6 assets, but every asset's
risk contribution is `16.667%` (`1/6`) by construction. That is the
whole point of ERC versus the naive inverse-vol weights, which only equalize
risk contribution when correlations happen to be equal.

### How do I run portfolio optimization for a basket of currencies?

`fx_port` builds total-return series (spot return + carry accrual) per
currency, shrinks the noisy sample means toward the cross-sectional grand
mean (James-Stein), and feeds the shrunk means/covariance into the same
min-variance / ERC machinery.

```python
from fx_port.data.synthetic import make_panel
from fx_port import total_log_returns, sample_cov, shrunk_means, min_variance_weights, erc_weights

panel = make_panel(seed=1, n_days=1500)
ret = total_log_returns(panel.spots, panel.rates)
mu, lam = shrunk_means(ret.total)
cov = sample_cov(ret.total)

w_mv = min_variance_weights(cov)
w_erc = erc_weights(cov)
print(f"shrinkage intensity: {lam:.3f}")
print("min-variance weights:\n", w_mv.round(3))
print("\nERC weights:\n", w_erc.round(3))
```

Output: shrinkage intensity `0.735` (most of the noisy sample-mean signal is
shrunk away, as expected for daily FX carry/spot means), min-variance piles
into the historically low-vol funders (JPY `20.3%`, CHF `19.2%`) while ERC
spreads risk more evenly across all 12 currencies including higher-vol EM
names (MXN, BRL, TRY each still get several percent).

### How do I see portfolio optimization break down on a near-singular covariance matrix?

A pegged currency (near-zero spot vol by construction) makes the sample
covariance matrix numerically singular. `fx_port`'s own edge-case suite
demonstrates the failure and the fix (`psd_repair`), and shows why a peg is
not a free lunch.

```python
from fx_port.data.synthetic import make_panel
from fx_port import total_log_returns, sample_cov, min_variance_weights, erc_weights, psd_repair
import numpy as np

panel = make_panel(seed=1, n_days=1500, include_peg=True)
ret = total_log_returns(panel.spots, panel.rates)
cov = sample_cov(ret.total)
print(f"PEG variance: {cov.loc['PEG', 'PEG']:.3e}")
print(f"condition number: {np.linalg.cond(cov.to_numpy()):.3e}")

w_mv = min_variance_weights(cov)
print(f"min-variance PEG weight (near-singular cov): {w_mv['PEG']:.4f}")

cov_fixed = psd_repair(cov, min_eig=1e-10)
w_mv_fixed = min_variance_weights(cov_fixed)
w_erc_fixed = erc_weights(cov_fixed)
print(f"after psd_repair, min-var PEG weight:  {w_mv_fixed['PEG']:.4f}")
print(f"after psd_repair, ERC PEG weight:      {w_erc_fixed['PEG']:.4f}")
```

Output: PEG variance `~1.0e-12` drives the covariance matrix's condition
number to `~2.9e8`; unconstrained min-variance responds by piling essentially
100% of the book into the "riskless" peg, and ERC does the same after
`psd_repair` stabilizes the matrix (PEG weight `~98%`). Both optimizers are
mechanically correct and both produce an off-mandate, undiversified book,
which is `docs/VALIDATION.md`'s point: "a peg is a policy option, not
a riskless asset." Peg weights are capped by policy rather than left to the
optimizer.


## Execution, Regime Strategies & Building the Engines

Setup shared by every Python snippet below: each project is an installable
package under `python/<asset-class>/<NN>-<project>/`. Install the one you
need into your venv with `python -m pip install -e python/<asset-class>/<NN>-<project>`,
then run the snippet with `python` from anywhere. Snippets use fixed seeds
and run offline. The shell recipes follow the commands in
[CONTRIBUTING.md](../CONTRIBUTING.md).

### How do I schedule a TWAP / VWAP / participation-rate execution and see the simulated fills?

`eq_algo` implements all three equity schedulers (`twap_schedule`,
`vwap_schedule`, `pov_schedule` in `benchmarks.py`) against a seeded
intraday simulator (`IntradayMarket`) with U-shaped volume, temporary
square-root impact and linear permanent impact. `execute()` returns an
`ExecutionResult` whose `fills` DataFrame has one row per bucket.

```python
import eq_algo as ea

icfg = ea.IntradayConfig(mid0=100.0, day_volume=1_000_000, n_buckets=26,
                          sigma_daily=0.02, spread_bps=5.0, temp_coef=1.0,
                          perm_coef=0.5, vol_noise=0.2)
mkt = ea.IntradayMarket(icfg)
X = 0.05 * icfg.day_volume  # 5% ADV parent order, 50,000 shares

schedules = {
    "TWAP": ea.twap_schedule(X, icfg.n_buckets),
    "VWAP": ea.vwap_schedule(X, icfg.profile),
    "POV 10%": ea.pov_schedule(X, icfg.profile * icfg.day_volume, 0.10),
}
for name, sched in schedules.items():
    res = mkt.execute(sched, side=1, seed=1)
    print(f"{name:8s} filled={res.filled_qty:>10,.0f}  avg_price={res.avg_price:8.4f}  "
          f"arrival={res.arrival_price:8.4f}  buckets_traded={(res.fills['qty']>0).sum()}")
```

Output (synthetic simulator): `TWAP filled=50,000 avg_price=100.3606 ... buckets_traded=26`,
`VWAP avg_price=100.2134 buckets_traded=26`, `POV 10% avg_price=100.4560
buckets_traded=13` (POV finishes early because it participates at a capped
rate of the heaviest-volume buckets first). All three fill the full parent.
POV would instead raise `ValueError` if the day's capacity at that
participation rate were insufficient.

### How do I schedule an FX parent order when there's no consolidated volume tape?

FX has no exchange tape, so `fx_algo` does not implement a VWAP scheduler at
all. `execution/schedulers.py` gives `twap_schedule`,
`liquidity_weighted_schedule` (the VWAP-analog: proportional to *modeled*
session depth, not a print) and `pov_schedule` (participation of modeled
depth), plus `fix_schedule` for targeting the WM/R 4pm London fix window.
`MarketSimulator` is session-aware (Asia/London/overlap/NY/late) over a 24h
grid.

```python
import numpy as np
from fx_algo.sessions import EURUSD
from fx_algo.execution.simulator import MarketSimulator, FirmVenue
from fx_algo.execution.schedulers import twap_schedule, liquidity_weighted_schedule, pov_schedule

sim = MarketSimulator(EURUSD, start_hour=0.0, horizon_hours=24.0, dt_minutes=5.0)
parent = 50.0  # 50mm EUR buy
depths = sim.depth_bucket

twap = twap_schedule(parent, sim.n_buckets)
liq = liquidity_weighted_schedule(parent, depths)
pov = pov_schedule(parent, depths, participation=0.15)

for name, sched in {"TWAP": twap, "LiqWeighted": liq, "POV 15%": pov}.items():
    r = sim.execute(sched, venue=FirmVenue(), seed=1)
    print(f"{name:12s} avg_fill={r.avg_fill:.5f}  arrival={r.arrival_mid:.5f}  "
          f"IS={r.is_pips:+.2f}pips  buckets_traded={(np.abs(r.qty) > 0).sum()}")
```

Output (synthetic simulator): `TWAP avg_fill=1.09686 IS=-31.44pips buckets_traded=288` (all 288
5-minute buckets across the 24h day), `LiqWeighted avg_fill=1.09696
IS=-30.40pips buckets_traded=288`, `POV 15% avg_fill=1.10018 IS=+1.79pips
buckets_traded=4` (POV concentrates in the deep London-NY overlap and
finishes in 4 buckets). The large IS swings here are mostly 24h session
drift, not cost. The TCA decomposition below separates the two.

### How do I compare a firm-liquidity venue to a last-look FX dealer stream?

`LastLookVenue` shows a tighter quote than `FirmVenue` but holds the order
for `hold_seconds` and rejects with a probability that rises with the
adverse price move during the hold window (FX Global Code Principle 17).
`venue_comparison` decomposes the *effective* cost identically across
venues so you can see the tighter-quote/higher-rejection trade-off directly.

```python
from fx_algo.sessions import EURUSD
from fx_algo.execution.simulator import MarketSimulator, FirmVenue, LastLookVenue
from fx_algo.execution.schedulers import twap_schedule
from fx_algo.execution.tca import venue_comparison

sim = MarketSimulator(EURUSD)
sched = twap_schedule(50.0, sim.n_buckets)
r_firm = sim.execute(sched, venue=FirmVenue(), seed=5)
r_ll = sim.execute(sched, venue=LastLookVenue(), seed=5, alpha_pips_per_bucket=0.05)

vc = venue_comparison({"firm-ecn": r_firm, "last-look": r_ll})
for venue, stats in vc.items():
    print(venue, {k: round(v, 3) for k, v in stats.items()})
```

Output: `firm-ecn {'quoted_half_spread_pips': 0.241, 'temp_impact_pips':
0.035, 'effective_cost_pips': 0.276, 'rejection_rate': 0.0,
'rejection_cost_pips': 0.0}` vs `last-look {'quoted_half_spread_pips':
0.144, ..., 'effective_cost_pips': 0.237, 'rejection_rate': 0.122,
'rejection_cost_pips': 0.058}` on the synthetic simulator. The last-look quote is 40% tighter, but a
12.2% reject rate (driven here by 0.05 pips/bucket of client alpha making
the flow "toxic") claws most of that saving back; `effective =
quoted_half_spread + temp_impact + rejection_cost`.

### How do I compute an Almgren-Chriss optimal execution trajectory and see if it beats TWAP/VWAP?

`ac_trades` returns the closed-form discrete AC trajectory
(`x_j = X sinh(kappa(T-t_j)) / sinh(kappa T)`) for a risk-aversion
`lambda`; `evaluate_schedules` runs a common-random-numbers Monte Carlo
horse race of any set of named schedules against the same simulator.

```python
import numpy as np
import eq_algo as ea

icfg = ea.IntradayConfig(mid0=100.0, day_volume=1_000_000, n_buckets=26,
                          sigma_daily=0.02, spread_bps=5.0, temp_coef=1.0,
                          perm_coef=0.5, vol_noise=0.2)
mkt = ea.IntradayMarket(icfg)
X = 0.05 * icfg.day_volume

acp = ea.ACParams(total_shares=X, n_slices=icfg.n_buckets,
                   sigma=icfg.sigma_daily * icfg.mid0, eta=2.0e-6, gamma=1e-6,
                   epsilon=icfg.mid0 * icfg.spread_bps * 1e-4 / 2)
lam = 5e-6
ac_sched = ea.ac_trades(acp, lam)

schedules = {
    "TWAP": ea.twap_schedule(X, icfg.n_buckets),
    "VWAP": ea.vwap_schedule(X, icfg.profile),
    f"AC lam={lam:g}": ac_sched,
}
tab = ea.evaluate_schedules(mkt, schedules, side=1, n_reps=200, seed=42)
print(tab.to_string())
```

Output (200 seeded replications on the synthetic simulator, bps):

```
              expected_is_bps  mean_is_bps  se_mean_is_bps  std_is_bps  mean_vs_vwap_bps  mean_vs_twap_bps
TWAP                14.335673    30.563490        8.076130  114.213719         12.782250         12.061332
VWAP                13.658359    29.107703        7.749112  109.588993         11.391778         10.667148
AC lam=5e-06        15.096783    25.409703        5.416399   76.599451          7.248477          6.326222
```

AC front-loads the schedule and cuts cost dispersion sharply (114 to 77 bps
std) versus TWAP, the risk/cost trade a positive `lambda` is meant to buy.
The gap in mean IS (about 5 bps) is within one standard error of the mean
(`se_mean_is_bps`), so 200 replications do not show a mean-cost win.
`expected_is_bps` is the model's own expected cost, and on that measure AC is
slightly dearer than TWAP.

### How do I run transaction-cost analysis (implementation-shortfall decomposition) on a completed equity execution?

`tca_report` applies the Perold (1988) decomposition into delay (decision to
release), trading (spread + impact + intraday drift) and opportunity
(unfilled tail). The three sum to total IS as an algebraic identity, tested
to 1e-10.

```python
import eq_algo as ea

icfg = ea.IntradayConfig(mid0=100.0, day_volume=1_000_000, n_buckets=26)
mkt = ea.IntradayMarket(icfg)
sched = ea.twap_schedule(50_000, icfg.n_buckets)
res = mkt.execute(sched, side=1, seed=1, decision_price=99.80)

rep = ea.tca_report(res)
print(f"filled {rep.filled_qty:,.0f}/{rep.parent_qty:,.0f} @ avg {rep.avg_fill_price:.4f}")
for k, v in rep.bps().items():
    print(f"  {k:16s} {v:+8.2f} bps")
```

Output (synthetic simulator): `filled 50,000/50,000 @ avg 100.6303`, then `delay_bps +20.04`,
`trading_bps +63.15`, `opportunity_bps -0.00`, `total_is_bps +83.19`. The
83.19 bps of total shortfall is almost entirely the pre-release "decision to
release" drift (delay) plus spread/impact during trading (trading), with
zero opportunity cost because the order fully filled.

### How do I decompose FX implementation shortfall into spread/impact/drift?

FX has no arrival-vs-decision "delay" leg in the Perold sense (no separate
decision price in the simulator); `decompose_implementation_shortfall`
instead splits IS into spread+temporary, permanent impact and market drift,
an exact partition (tested to 1e-10 pips).

```python
from fx_algo.sessions import EURUSD
from fx_algo.execution.simulator import MarketSimulator, FirmVenue
from fx_algo.execution.schedulers import twap_schedule
from fx_algo.execution.tca import decompose_implementation_shortfall

sim = MarketSimulator(EURUSD)
sched = twap_schedule(50.0, sim.n_buckets)
r = sim.execute(sched, venue=FirmVenue(), seed=1)
for k, v in decompose_implementation_shortfall(r).items():
    print(f"  {k:18s} {v:+8.2f} pips")
```

Output (synthetic simulator): `total -31.44`, `spread_temporary +0.28`, `permanent_impact +0.02`,
`market_drift -31.74` (pips). With a passive TWAP over a full 24h session,
market drift (session vol, nothing to do with execution quality) swamps the
controllable spread/impact cost, which is the point of decomposing
IS rather than reading the total in isolation.

### How do I run the regime-switching backtest end to end and inspect regime-conditional stats?

`walk_forward_backtest` fits a 3-state HMM on an expanding/rolling window,
detects the regime online (filtered probabilities only, no lookahead),
sizes a vol-targeted position with hysteresis, and returns ledgers for the
strategy, buy-and-hold, and a 200-day MA rule. `per_regime_stats` then
slices realised P&L by the *detected* regime.

```python
import pandas as pd
from eq_regime import walk_forward_backtest, per_regime_stats, summary_stats
from eq_regime.data import make_regime_panel

panel = make_regime_panel(n_states=3, n_assets=8, n_days=2520, seed=7)

res = walk_forward_backtest(
    panel.prices, n_states=3, min_train=378, refit_every=63,
    cost_bps=5.0, seed=0, enter=0.70, exit_=0.30, target_vol=0.10, n_pca=3,
)
stats = pd.DataFrame({
    "strategy": summary_stats(res.ledger),
    "buy_and_hold": summary_stats(res.benchmark),
    "ma_200d": summary_stats(res.ma_rule),
})
print(stats)

regimes = res.detection["regime"].reindex(res.ledger.index).ffill().bfill()
print(per_regime_stats(res.ledger["net_ret"], regimes))
```

Output (synthetic panel):

```
              strategy  buy_and_hold   ma_200d
cagr          0.103917      0.007695  0.001972
ann_vol       0.084795      0.153366  0.111696
sharpe        1.207943      0.126681  0.073365
max_drawdown  0.101201      0.444398  0.208034
total_cost    0.024259      0.000500  0.026500
final_equity  2.051022      1.057281  1.014419

              days  ann_return   ann_vol    sharpe  max_drawdown  total_pnl
bull         810.0    0.170324  0.096496  1.765090      0.097510   0.547469
transition   440.0    0.109828  0.067273  1.632569      0.036959   0.191763
bear         581.0    0.002168  0.078788  0.027515      0.074976   0.004998
TOTAL       1831.0    0.102428  0.084795  1.207943      0.101201   0.744230
```

The strategy earns most of its return in bull regimes with much lower
drawdown than buy-and-hold, and is nearly flat in bear regimes rather than
losing money. The regime-conditional table shows *why* the headline Sharpe
beats the benchmark on this synthetic panel, not just that it does. The
synthetic generator plants the regime drifts, so this tests the machinery and
is not evidence of an edge.

### How do I check whether my regime detector is trading on lookahead?

`HMMFit.filter` (online, causal) and `HMMFit.smooth` (full-sample,
non-causal) are both exposed so you can see the difference directly around
a regime flip. Trading on smoothed probabilities is a textbook
lookahead bug, and this makes it visible rather than theoretical.

```python
import numpy as np
from eq_regime import fit_hmm
from eq_regime.data import make_regime_panel

panel = make_regime_panel(n_states=3, n_assets=8, n_days=2520, seed=7)
r_idx = panel.returns.mean(axis=1)

hfit = fit_hmm(r_idx.to_numpy(), 3, seed=0, n_init=3, max_iter=200)
bear_state = int(np.argmax(np.sqrt(hfit.covariances[:, 0, 0])))
filt, _ = hfit.filter(r_idx.to_numpy())
smth = hfit.smooth(r_idx.to_numpy())

flips = np.where((panel.states[1:] == 2) & (panel.states[:-1] != 2))[0] + 1
t_flip = int(flips[flips > 300][0])
for t in range(t_flip - 3, t_flip + 3):
    print(f"{t - t_flip:+3d}   filtered={filt[t, bear_state]:.3f}  smoothed={smth[t, bear_state]:.3f}")
```

Output (synthetic panel, true bull-to-bear flip at `t=320`; a sixth row, `+2`, also prints):

```
 -3   filtered=0.016  smoothed=0.137
 -2   filtered=0.014  smoothed=0.318
 -1   filtered=0.103  smoothed=0.743
 +0   filtered=0.367  smoothed=0.946
 +1   filtered=0.973  smoothed=0.998
```

The smoothed probability is already at 0.74 the day *before* the true flip
(it has seen the future); the filtered probability only crosses 0.5 the day
*after*. Only `filter`'s output is legitimate for a live-trading strategy.
`walk_forward_backtest` above uses filtered probabilities exclusively for
this reason.

### How do I run the foundations no-look-ahead backtest, and what breaks if I remove `.shift(1)`?

`eq_signal_backtest.engine.strategy_returns` takes the signal decided at the
close of day *t* and fills it during day *t+1* at the modelled price
`p_t + f (p_{t+1} - p_t)`, with `f = 0.5` by default (`ExecutionConfig`;
`fill_fraction=1.0` is a next-close fill, and a close-auction mode fills at
`p_t` and pays an auction cost). The old position earns the move up to the
fill and the new position earns the rest of the day. The backtest therefore
never trades on a close it has not seen. Removing the shift makes the
strategy hold the same day's signal against that same day's return, which
cannot happen live.

```python
from eq_signal_backtest.data.synthetic import generate
from eq_signal_backtest.signals import ma_crossover_signal
from eq_signal_backtest.engine import run_backtest

prices = generate().set_index("Date")["Adj Close"].astype(float)
signal = ma_crossover_signal(prices, 20, 100)

res = run_backtest(prices, signal, cost_bps=5.0)  # t+1 modelled fill, f = 0.5
print("With .shift(1):", f"sharpe={res.stats['sharpe']:.3f} cagr={res.stats['cagr']:+.2%}")

# A hand-written look-ahead version (the bug): hold today's signal on today's return
rets = prices.pct_change().fillna(0.0)
position_no_lag = signal                      # <-- no .shift(1): the bug
trades = position_no_lag.diff().abs().fillna(0.0)
costs = trades * 5.0 / 10_000
strat_rets_no_lag = (position_no_lag * rets - costs).clip(lower=-1.0)
equity_no_lag = (1 + strat_rets_no_lag).cumprod()
sharpe_no_lag = strat_rets_no_lag.mean() / strat_rets_no_lag.std(ddof=1) * (252 ** 0.5)
cagr_no_lag = equity_no_lag.iloc[-1] ** (252 / len(prices)) - 1
print("Without .shift(1):", f"sharpe={sharpe_no_lag:.3f} cagr={cagr_no_lag:+.2%}")
```

Output (synthetic prices): `With .shift(1): sharpe=0.557 cagr=+7.90%` vs
`Without .shift(1): sharpe=0.621 cagr=+8.98%`. On this slow 20/100-day-MA
signal the inflation is modest because the signal barely moves day to day.
`docs/METHODOLOGY.md` notes the effect scales with signal turnover: a
fast, noisy signal that happens to be right about *today's* close would show
a far larger, fake edge. Removing the shift also breaks the leading-`NaN`
handling the engine tests for (`signal.shift(1)` produces a `NaN` on day 0
that must be filled to `0.0`, flat, not defaulted to long).

### How do I walk-forward-validate a signal instead of trusting one in-sample fit?

`walk_forward_backtest` in `eq_signal_backtest.split` re-selects
`(fast, slow)` by in-sample Sharpe grid search on each formation window,
then trades the frozen parameters on the following out-of-sample trading
window, stitching the OOS legs together. This is what catches
overfitting that a single train/test split can miss.

```python
from eq_signal_backtest.data.synthetic import generate
from eq_signal_backtest.split import walk_forward_backtest

prices = generate().set_index("Date")["Adj Close"].astype(float)
wf = walk_forward_backtest(prices, range(10, 71, 10), range(100, 251, 25),
                            formation=756, trading=252, cost_bps=5.0)
print(wf.windows[["trading_start", "trading_end", "fast", "slow",
                  "n_trades", "window_sharpe"]].to_string(index=False))
print(f"walk-forward stitched OOS: sharpe={wf.stats['sharpe']:.2f} "
      f"cagr={wf.stats['cagr']:+.2%} ({wf.n_trades} trades)")
print(f"walk-forward buy&hold:     sharpe={wf.stats['benchmark']['sharpe']:.2f}")
```

Output (synthetic prices): 7 walk-forward windows with re-selected
`(fast, slow)` each time (window Sharpes ranging `-1.59` to `+1.65`),
stitching to `walk-forward stitched OOS: sharpe=-0.02 cagr=-1.70% (22 trades)`
against `walk-forward buy&hold: sharpe=0.15`. The in-sample edge does not
survive walk-forward validation on this synthetic series, which is the point
of running it rather than reporting one lucky train/test split.

### How do I build and test one C++ engine from a clean build directory?

Every C++ engine is C++23 with `-Wall -Wextra`, GoogleTest and
`gtest_discover_tests`, and needs CMake 3.25. Warnings become errors when
the engine's `*_WERROR` option is on (`EQOPT_WERROR` here), as CI sets it.
Each engine ships a `CMakePresets.json` (`gcc`, `clang`, `gcc16-cxx26`,
`asan`, `tsan`, `msvc`); the presets are the supported way to build. Build
trees go under `build/<preset>` in the engine directory. Build and test
`equity-options-engine` with the `gcc` preset (the commands follow
[CONTRIBUTING.md](../CONTRIBUTING.md)):

```bash
cd "$(git rev-parse --show-toplevel)/cpp/equity-options-engine"
cmake --preset gcc -DEQOPT_WERROR=ON
cmake --build --preset gcc
ctest --preset gcc
```

Configure pulls in the header-only `qp::core` target from `cpp/core`, finds
GoogleTest and `Threads`, and the build compiles the engine
library plus its test and benchmark executables, and `ctest` runs every
discovered test. The per-engine test count is in the README table, filled
from the CI `counts` job.

### How do I build and test one Rust engine with warnings-as-errors?

Rust engines are 2021-edition crates in one Cargo workspace under `rust/`,
with no third-party dependencies outside tests. Their shared code is the
workspace crate `qp-core`. The CI job `rust <crate>` in
`.github/workflows/ci.yml` denies warnings. Reproduce that locally from
`rust/`, naming the package with `-p`:

```bash
cd "$(git rev-parse --show-toplevel)/rust"
RUSTFLAGS="-D warnings" cargo test --release --locked -p eq-options-engine
```

`cargo test` runs the integration-test binaries under `tests/` and then the
rustdoc doc-tests on the public items. The doc-tests are the slow part,
because each one builds a fresh binary. `cargo clean` removes `rust/target/`,
the one target directory of the workspace, afterwards. The per-engine count is in the README table, filled from the CI
`counts` job.

### How do I regenerate golden vectors for an engine, and how does the pipeline work?

The three-language cross-validation pipeline is: a Python
`tests/golden/generate_golden.py` script calls the Python reference
implementation at full `double` precision and writes JSON (doubles via
`repr()`, which round-trips to the same IEEE-754 value); then
`tools/gen_golden_header.py` (C++) and `tools/gen_golden_rs.py` (Rust) each
turn that JSON into a `constexpr`/`const` array committed alongside the
engine's tests. Running all three in sequence for
`equity-options-engine` should reproduce the committed files:

```bash
cd "$(git rev-parse --show-toplevel)"   # the quant-portfolio checkout
PYTHONPATH=python/equity/01-options-pricing/src python \
  python/equity/01-options-pricing/tests/golden/generate_golden.py
python cpp/equity-options-engine/tools/gen_golden_header.py
python rust/equity-options-engine/tools/gen_golden_rs.py
```

The VaR engines use the identical pipeline (`python/{equity,fx}/03-var-es-engine/tests/golden/generate_golden.py`
→ `cpp/*-var-engine/tools/gen_golden_header.py` → `tests/golden_vectors.hpp`,
and `rust/*-var-engine/tools/gen_golden_rs.py` → `tests/golden/mod.rs`).
Every generator also accepts `--check`, which exits 1 if the committed
constants have drifted from the JSON:

```bash
PYTHONPATH=python/equity/03-var-es-engine/src python \
  python/equity/03-var-es-engine/tests/golden/generate_golden.py
python cpp/equity-var-engine/tools/gen_golden_header.py --check
python rust/equity-var-engine/tools/gen_golden_rs.py --check
```

The generator is a pure function of a fixed `CASES` list and the Python
reference, so a regenerated file should match the committed one. The CI
`goldens --check` job runs these checks on every push. Explain every changed
golden value in the commit message.

### How do I run the full portfolio's Python test sweep?

Use the commands in [CONTRIBUTING.md](../CONTRIBUTING.md). CI installs each
project into its own fresh venv against `requirements/ci-constraints.txt` and
runs pytest there, so repeat this per project to reproduce CI. CI installs
`python/core` first, because 16 projects import `qp_core`:

```bash
python -m venv ~/.venvs/qp-eqopt
. ~/.venvs/qp-eqopt/bin/activate      # Windows: ~/.venvs/qp-eqopt/Scripts/activate
python -m pip install -c requirements/ci-constraints.txt -e python/core
cd python/equity/01-options-pricing
python -m pip install -c ../../../requirements/ci-constraints.txt -e ".[test]" pytest hypothesis
python -m pytest -p no:cacheprovider -q
```

A project that declares `dev` instead of `test` takes `".[dev]"`. A shared
environment for all projects is faster but can hide a missing dependency.
The uv workspace gives one: `uv sync --all-packages` from the root installs
`python/core` and all 23 projects from `uv.lock`.
The test counts quoted in the docs come from the CI `counts` job, which sums
the executed pass counts into `counts.json` and then checks the docs against
it.

### How do I add a new project area that's CONVENTIONS.md-compliant?

`CONVENTIONS.md` fixes one directory layout for every Python project. The
minimum skeleton that satisfies it, using a new equity project as an
example (swap the package name and asset-class conventions for FX):

```bash
mkdir -p python/equity/11-my-new-project/{src/eq_newproj,tests,examples,docs}
cd python/equity/11-my-new-project

cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "eq-newproj"
version = "1.0.0"
description = "One-sentence description of what this project does and why."
readme = "README.md"
requires-python = ">=3.10"
authors = [{ name = "Your Name", email = "you@example.com" }]
dependencies = ["numpy", "pandas"]

[project.optional-dependencies]
dev = ["pytest>=7.4"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-q"
EOF

cat > src/eq_newproj/__init__.py << 'EOF'
"""eq_newproj: one-line summary of the pipeline stages."""
__version__ = "1.0.0"
__all__: list[str] = []
EOF

touch tests/test_placeholder.py examples/run_pipeline.py \
      docs/METHODOLOGY.md docs/VALIDATION.md docs/DESK_GUIDE.md README.md

python -m pip install -e ".[dev]" && python -m pytest
```

That gives every file `CONVENTIONS.md` names as required: `README.md`,
`pyproject.toml`, `src/<package_name>/__init__.py` re-exporting a typed
public API with `__all__`, `tests/test_*.py` (pytest, offline, deterministic
seeds), `examples/run_pipeline.py` (data -> model -> validation -> decision,
reproducing the README's numbers), and the three-document
`docs/{METHODOLOGY,VALIDATION,DESK_GUIDE}.md` contract: methodology
(model choice vs. alternatives + numbered assumptions register), validation
(analytic/convergence/statistical checks + documented failure modes) and
desk guide (workflow, controls, P&L attribution). Before it counts as
"done" per the documentation contract, `docs/METHODOLOGY.md` and
`docs/VALIDATION.md` must answer all six numbered questions in
`CONVENTIONS.md` explicitly (why this model, what assumptions, how
validated, where it fails, desk usage, edge cases, with each edge case
both documented *and* unit-tested), and every public function needs a
NumPy-style docstring stating units and conventions (ACT/365F equity day
count; BASE/QUOTE FX pairs with `q = r_f` for the FX version). If the
project pairs with a C++/Rust engine later, its golden vectors live under
`tests/golden/generate_golden.py` in this same Python project, per the
cross-language validation pipeline shown above.


## November 2026 release: Engine Switch, Research Workflow and New Models

Recipes for what the November 2026 release added. Install `python/core` and the package each
recipe names, as above. The new modules are imported by their path (for
example `eq_options.alo`); they are not re-exported from the package root.
Outputs below come from one run on Windows with Python 3.12.

### How do I call the C++ or Rust engine from the Python API?

Needs `python/core` and `python/equity/01-options-pricing`. The compiled
engines need `pip install ./bindings/cpp-nanobind` or
`pip install ./bindings/rust-pyo3` from the repository root.

```python
from eq_options.engine import bs_price
from qp_core.engines import engine_available

p = bs_price(100.0, 105.0, 0.5, 0.03, 0.2, q=0.01, option_type="call")
print(round(p, 6))                           # 3.965696, the Python reference
for eng in ("cpp", "rust"):
    if engine_available(eng):
        print(eng, abs(bs_price(100.0, 105.0, 0.5, 0.03, 0.2, q=0.01, engine=eng) - p))
    else:
        print(eng, "not installed")
```

`engine="python"` is the default and calls the reference unchanged. Asking
for an engine that is not installed raises `ImportError` with the install
command; an unknown name raises `ValueError`. The same switch exists in
`fx_options.engine`, `eq_var.engine` and `fx_var.engine`.

### How do I pre-register a study, sign it and check results?

Needs no install beyond Python for the hash and check steps.

```bash
# 1. draft research/prereg/<name>.md, ending with a "## Signature" section
# 2. print the registration hash of the text above that heading
python research/tools/prereg.py hash research/prereg/pairs-fx-g10.md
# dfca5c7ae8ea...ae953dcc  signed=True  research/prereg/pairs-fx-g10.md
# 3. sign: fill signed_by, signed_on and sha256 under "## Signature"
# 4. check every result file against its registration
python research/tools/prereg.py check     # exit code 0 when all pass
```

The check refuses a result whose `prereg_sha256` does not match the
registration, and a result whose period reaches 2022-01-01 or later unless
the registration is signed with that hash. Run the hold-out only after
signing. Training-period results are not out-of-sample evidence.

### How do I mark an American option with ALO and check it with the Leisen-Reimer tree?

Needs `python/equity/01-options-pricing`.

```python
from eq_options.alo import alo_price
from eq_options.lr_tree import lr_price

alo = alo_price(100.0, 100.0, 1.0, 0.05, 0.25, q=0.0, option_type="put")
lr = lr_price(100.0, 100.0, 1.0, 0.05, 0.25, q=0.0, option_type="put",
              exercise="american", n_steps=2001)
print(round(alo, 5), round(lr, 5), f"{abs(alo - lr):.1e}")
# 7.97448 7.97438 1.1e-04
```

The 1e-4 gap is mostly the tree's own error at 2001 steps; the package
test compares ALO with a Richardson-extrapolated tree (2001 and 4001
steps) and gets within 2e-5. Negative rates work too, including the
`q < r < 0` put (and `r < q < 0` call) where the exercise region is a band
between two boundaries; pass `n_march` to trade speed for accuracy there.

### How do I price an American call on a stock with one discrete dividend?

Needs `python/equity/01-options-pricing`.

```python
from eq_options.dividends import lr_price_discrete_divs, rgw_call

# spot 100, strike 100, 1 year, r 5%, vol 25%, dividend 2.0 paid at t = 0.5
tree = lr_price_discrete_divs(100.0, 100.0, 1.0, 0.05, 0.25, [(0.5, 2.0)],
                              option_type="call", exercise="american")
rgw = rgw_call(100.0, 100.0, 0.5, 1.0, 2.0, 0.05, 0.25)
print(round(tree, 4), round(rgw, 4))          # 11.1414 11.1414
```

The tree uses the escrowed-dividend spot. Roll-Geske-Whaley is the closed
form for one dividend and is the oracle. The tree converges to it only at
O(1/n), because the ex-date is not on a node.

### How do I build an FX smile from ATM, risk reversal and butterfly, and read the 10-delta strike?

Needs `python/fx/01-options-pricing`.

```python
import datetime as dt
from fx_options.smile import smile_from_quotes
from fx_options.cuts import expiry_and_delivery, time_to_expiry

sm = smile_from_quotes(S=1.0850, T=0.25, r_d=0.045, r_f=0.025,
                       atm=0.075, rr25=-0.008, bf25=0.0025)   # spot delta
k10 = sm.strike_from_delta(0.10, "call")
print(round(k10, 5), round(sm.vol(k10), 5))   # 1.14365 0.0735

exp, dlv = expiry_and_delivery(dt.date(2026, 11, 2), "3M", "EURUSD")
print(exp, dlv)                               # 2027-02-02 2027-02-04
now = dt.datetime(2026, 11, 2, 12, 0, tzinfo=dt.timezone.utc)
print(round(time_to_expiry(now, exp, cut="NY"), 6))   # 0.252397, ACT/365F to the cut
```

Pass `holidays="auto"` to use the pair's joint settlement calendar from
`fx_options.holidays` (rules-based, 2010 to 2099; a re-export of
`qp_core.holidays`), or pass your own dates.
The quoted butterfly is read as the smile strangle; if your broker quotes
the market strangle, convert it first with
`fx_options.smile.smile_strangle_from_broker`.

### How do I build OIS and projection curves and price a cap?

Needs `python/equity/04-fixed-income`.

```python
from fi_rates.multicurve import bootstrap_ois, bootstrap_projection
from fi_rates.rate_options import cap_price

mats = [1, 2, 3, 5, 7, 10]
ois = bootstrap_ois(mats, [0.030, 0.031, 0.032, 0.033, 0.034, 0.035])
proj = bootstrap_projection(ois, mats, [0.033, 0.034, 0.035, 0.036, 0.037, 0.038])
print(round(cap_price(ois, proj, strike=0.035, maturity=5.0, vol=0.20), 6))       # 0.020219
print(round(cap_price(ois, proj, strike=0.035, maturity=5.0, vol=0.0075,
                      model="bachelier"), 6))                                      # 0.021254
```

The projection curve reprices the swap quotes while discounting on OIS.
Black vol is lognormal; Bachelier vol is in rate units (75 bp here).

### How do I fit an SSVI surface and get Dupire local vol from it?

Needs `python/equity/09-vol-surface`.

```python
import numpy as np
from eq_surface.ssvi import SSVIParams
from eq_surface.dupire import local_vol_from_total_variance

params = SSVIParams(rho=-0.4, eta=1.0, gamma=0.5)
thetas = 0.04 * np.array([0.25, 0.5, 1.0, 2.0])     # ATM total variance per expiry
print(params.static_arbitrage_free(), all(params.butterfly_free(t) for t in thetas))
# True True

def w(k, T):                                       # total variance w(k, T)
    return params.total_variance(k, 0.04 * T)

print(np.round(local_vol_from_total_variance(w, np.array([-0.2, 0.0, 0.2]), 1.0), 4))
# [0.2781 0.1854 0.1734]
```

To fit to market data use `fit_ssvi(k, w_market, thetas)`. Dupire uses
finite differences, so a noisy surface gives noisy local vol; negative
local variance raises `LocalVolArbitrageError`.

### How do I bootstrap CDS hazard rates and compute bilateral CVA?

Needs `python/equity/06-credit-risk` and `python/fx/06-credit-risk`.

```python
import numpy as np
from eq_credit.cds import bootstrap_hazards, par_spread
from fx_credit.xva import bilateral_cva

cpty = bootstrap_hazards(np.array([1.0, 3.0, 5.0]), np.array([0.0100, 0.0125, 0.0150]),
                         rate=0.03, recovery=0.4)
print(np.round(cpty.hazards, 5), round(par_spread(cpty, 5.0, 0.03, 0.4), 6))
# [0.0166  0.02299 0.03207] 0.015
own = bootstrap_hazards(np.array([5.0]), np.array([0.0060]), rate=0.03, recovery=0.4)

times = np.linspace(0.0, 5.0, 21)                  # grid must start at 0
epe = 1e6 * 0.05 * np.sqrt(times)                 # expected positive exposure
ene = 0.8 * epe                                    # expected negative exposure, as a positive amount
res = bilateral_cva(times, epe, ene, cpty.survival(times), own.survival(times), rate=0.03)
print(round(res.cva, 2), round(res.dva, 2))        # 5121.83 1526.72
```

`eq_credit.cds` uses a flat discount rate and year-fraction maturities.
For dated contracts on the ISDA standard-model conventions (step-in, ACT/360,
exact accrual on default, a real discount curve) use
`eq_credit.isda_cds.bootstrap_isda` with a `LogLinearCurve`. Pass
`calendar="USD"` (or `"EUR"`, `"GBP"`, `"JPY"`, a joint code like
`"EURGBP"`) to roll coupon dates and `T + 3` cash settlement over that
centre's holidays as well as weekends; the default is weekends only. For
a closure the rules do not know yet, wrap the call in
`qp_core.holidays.extra_closures({"USD": [date]})`. For collateral, build the exposure with `collateralised_exposure`.

### How do I calibrate PDs to a central tendency with a margin of conservatism?

Needs `python/equity/06-credit-risk`.

```python
import numpy as np
from eq_credit.calibration import calibrate_with_moc

pd_raw = np.array([0.002, 0.008, 0.02, 0.06])           # model PDs by grade
dr = np.array([0.010, 0.025, 0.018, 0.012, 0.030, 0.015])  # yearly default rates
r = calibrate_with_moc(pd_raw, dr, confidence=0.90, moc_a=0.001)
print(round(r.central_tendency, 5), round(r.moc_c, 5), np.round(r.pd_final, 5))
# 0.01833 0.00468 [0.00729 0.01215 0.02189 0.05471]
```

`moc_c` is the estimation-error margin from the default-rate history;
`moc_a` and `moc_b` are supplied by the modeller. The CRR floor of 3 bp
applies last.

### How do I rebalance a portfolio with transaction costs and a turnover cap?

Needs `python/equity/07-portfolio-optimization` (the same module is in
`fx_port`).

```python
import numpy as np
from eq_port.cost_qp import rebalance

mu = np.array([0.06, 0.05, 0.04])
cov = np.diag([0.04, 0.03, 0.02])
w0 = np.array([0.5, 0.3, 0.2])
rb = rebalance(mu, cov, w0, risk_aversion=4.0, cost=0.002, max_turnover=0.10)
print(np.round(rb.weights, 4), round(rb.turnover, 4))    # [0.45 0.3  0.25] 0.1
```

The turnover cap binds here. The solver is SLSQP, so results agree with
closed forms to about 1e-6, not to the bit.

### How do I split portfolio VaR into Euler components?

Needs `python/equity/03-var-es-engine` (the same module is in `fx_var`).

```python
import numpy as np
from eq_var.risk_attribution import parametric_attribution

a = parametric_attribution([1e6, -4e5],
                           np.array([[4e-4, 1.2e-4], [1.2e-4, 2.25e-4]]), 0.01)
print(round(a.total, 2), np.round(a.component, 2), round(float(np.sum(a.component)), 2))
# 42895.74 [44409.7  -1513.97] 42895.74
```

The components sum to the total because VaR is homogeneous of degree one
in the positions. The short position reduces risk, so its component is
negative. `simulation_attribution` does the same from scenarios.
