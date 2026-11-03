"""Build the learning notebooks from one source of truth.

Each notebook exists twice: ``<name>-exercise.ipynb`` (tasks left as
stubs) and ``<name>-solution.ipynb`` (tasks filled in). Both are written
from the cell lists below, so they can never drift apart.

    python docs/notebooks/build_notebooks.py          # rewrite the .ipynb files
    python docs/notebooks/build_notebooks.py --check  # fail if they are stale

Cell kinds: ``md`` (markdown), ``code`` (same in both files), ``task``
(solution code; the exercise gets the stub instead) and ``check`` (asserts
that compare the library with an independent computation; kept in both,
so a learner can test their own answer).

The notebooks use only library code from this repository, numpy, scipy and
seeded synthetic data, so a CI run is deterministic and needs no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

PATHS = "import _paths  # noqa: F401  (puts the library on sys.path)"


def md(text: str) -> tuple[str, str, str]:
    return ("md", text.strip("\n"), "")


def code(text: str) -> tuple[str, str, str]:
    return ("code", text.strip("\n"), "")


def task(solution: str, stub: str) -> tuple[str, str, str]:
    return ("task", solution.strip("\n"), stub.strip("\n"))


def check(text: str) -> tuple[str, str, str]:
    return ("check", text.strip("\n"), "")


# --------------------------------------------------------------------------
NOTEBOOKS: dict[str, list[tuple[str, str, str]]] = {}

NOTEBOOKS["01-black-scholes-implied-vol"] = [
    md("""
# 1. Black-Scholes prices and implied volatility

**Level:** L1. **Tags:** `options`, `black-scholes`, `implied-vol`.
**Library:** `eq_options.black_scholes` (`python/equity/01-options-pricing`).

**Plain idea.** Under Black-Scholes a European call is worth the discounted
expected payoff when the stock grows at the risk-free rate with constant
volatility. Implied volatility runs the formula backwards: the one sigma that
reproduces a quoted price.

**Formal statement** (Hull, *Options, Futures and Other Derivatives*, 11th
ed., the Black-Scholes-Merton chapter; the dividend yield $q$ is added as in
Hull's chapter on index and currency options):
$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$, with
$d_{1,2} = [\\ln(S/K) + (r - q \\pm \\sigma^2/2)T] / (\\sigma\\sqrt{T})$.
Vega is $\\partial C/\\partial\\sigma = S e^{-qT} \\varphi(d_1)\\sqrt{T}$, per
unit of volatility (so per 1.00, not per 1%).
"""),
    code(PATHS),
    code("""
import math

import numpy as np
from eq_options.black_scholes import bs_price, implied_vol  # noqa: F401
from scipy.stats import norm
"""),
    md("""
## Task 1: price the textbook at-the-money call

Use `bs_price` with $S = K = 100$, $T = 1$, $r = 5\\%$, $\\sigma = 20\\%$, no
dividend. Store it as `call`, and the matching put as `put`.
"""),
    task("""
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
call = bs_price(S, K, T, r, sigma, option_type="call")
put = bs_price(S, K, T, r, sigma, option_type="put")
print(f"call {call:.4f}  put {put:.4f}")
""", """
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
call = ...  # use bs_price
put = ...
"""),
    md("""
**Independent check.** The closed form written out by hand with `scipy`, and
put-call parity $C - P = S e^{-qT} - K e^{-rT}$, which holds for any model.
The value to four places is 10.4506; the checks recompute it rather than
trust a printed figure.
"""),
    check("""
d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
d2 = d1 - sigma * math.sqrt(T)
by_hand = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
assert abs(call - by_hand) < 1e-12
assert abs((call - put) - (S - K * math.exp(-r * T))) < 1e-12
assert round(call, 4) == 10.4506
"""),
    md("""
## Task 2: vega, two ways

Compute `vega_fd`, a central finite difference of `bs_price` in sigma with a
bump of `1e-4`. Then `vega_cf`, the closed form. Note the size: about 37.5
per unit vol, so a one-point move (0.01) is worth about 0.375. A vega that is
bigger than the price is normal for an at-the-money option; it is quoted per
unit of sigma.
"""),
    task("""
h = 1e-4
vega_fd = (bs_price(S, K, T, r, sigma + h) - bs_price(S, K, T, r, sigma - h)) / (2 * h)
vega_cf = S * norm.pdf(d1) * math.sqrt(T)
print(f"vega fd {vega_fd:.6f}  closed form {vega_cf:.6f}  per vol point {vega_cf / 100:.4f}")
""", """
h = 1e-4
vega_fd = ...
vega_cf = ...
"""),
    check("""
assert abs(vega_fd - vega_cf) < 1e-6
assert 37.0 < vega_cf < 38.0
"""),
    md("""
## Task 3: implied volatility round trip

Build a small smile: strikes 80 to 120 with a made-up skew
$\\sigma(K) = 0.20 - 0.10\\ln(K/100)$. Price each strike, then recover sigma
with `implied_vol`. Store the recovered values in `iv`.
"""),
    task("""
strikes = np.arange(80.0, 121.0, 5.0)
true_vol = 0.20 - 0.10 * np.log(strikes / 100.0)
prices = [bs_price(S, k, T, r, v) for k, v in zip(strikes, true_vol, strict=True)]
iv = np.array([implied_vol(p, S, k, T, r) for p, k in zip(prices, strikes, strict=True)])
for k, v in zip(strikes, iv, strict=True):
    print(f"K={k:5.0f}  iv={v:.6f}")
""", """
strikes = np.arange(80.0, 121.0, 5.0)
true_vol = 0.20 - 0.10 * np.log(strikes / 100.0)
iv = ...  # price each strike with bs_price, then invert with implied_vol
"""),
    check("""
assert np.max(np.abs(iv - true_vol)) < 1e-8
"""),
    md("""
**Takeaways.** (1) Parity is the cheapest test of any pricer. (2) Quote vega
with its unit. (3) An inverter is only trusted after a round trip across
strikes, including the wings.
"""),
]

NOTEBOOKS["02-value-at-risk"] = [
    md("""
# 2. Value at risk and expected shortfall

**Level:** L1. **Tags:** `risk`, `var`, `expected-shortfall`.
**Library:** `eq_var` (`python/equity/03-var-es-engine`).

**Plain idea.** VaR at 99% is the loss you exceed on one day in a hundred.
Expected shortfall (ES) is the average loss on those bad days, so it sees
how deep the tail goes.

**Formal statement** (McNeil, Frey and Embrechts, *Quantitative Risk
Management*, 2nd ed., ch. 2, risk measures; they state it for losses, here
it is written for P&L): for P&L $X$ and tail probability $\\alpha$,
$\\mathrm{VaR}_\\alpha = -q_\\alpha(X)$ and
$\\mathrm{ES}_\\alpha = -\\frac{1}{\\alpha}\\int_0^\\alpha q_u(X)\\,du$.
For normal P&L with zero mean, $\\mathrm{VaR} = -z_\\alpha\\sigma$ and
$\\mathrm{ES} = \\sigma\\varphi(z_\\alpha)/\\alpha$.
"""),
    code(PATHS),
    code("""
import math

import numpy as np
from eq_var.expected_shortfall import expected_shortfall, normal_es  # noqa: F401
from eq_var.historical_var import historical_var  # noqa: F401
from eq_var.parametric_var import parametric_var  # noqa: F401
from scipy.stats import norm

rng = np.random.default_rng(20261103)  # fixed seed: the notebook is deterministic
"""),
    md("""
## Task 1: parametric VaR of a two-asset book

Exposures of 1,000,000 and -400,000 in two assets with daily vols 1.2% and
0.9% and correlation 0.6. Build the covariance `cov` and compute the 99%
one-day normal VaR with `parametric_var`. Store it as `var_p`.
"""),
    task("""
w = np.array([1_000_000.0, -400_000.0])
vol = np.array([0.012, 0.009])
corr = np.array([[1.0, 0.6], [0.6, 1.0]])
cov = np.outer(vol, vol) * corr
var_p = parametric_var(w, cov, alpha=0.01)
print(f"99% 1-day parametric VaR: {var_p:,.0f}")
""", """
w = np.array([1_000_000.0, -400_000.0])
vol = np.array([0.012, 0.009])
corr = np.array([[1.0, 0.6], [0.6, 1.0]])
cov = ...
var_p = ...
"""),
    check("""
sigma_p = math.sqrt(w @ cov @ w)
assert abs(var_p - (-norm.ppf(0.01)) * sigma_p) < 1e-6
"""),
    md("""
## Task 2: historical VaR and ES on simulated P&L

Draw 250,000 days of P&L from the same normal model (seeded above), then
compute historical VaR and ES at 99%. With this many days both should sit
close to the closed forms; with 250 days (one year) they would not.
"""),
    task("""
pnl = rng.multivariate_normal(np.zeros(2), cov, size=250_000) @ w
var_h = historical_var(pnl, alpha=0.01)
es_h = expected_shortfall(pnl, alpha=0.01)
es_n = normal_es(sigma_p, alpha=0.01)
print(f"historical VaR {var_h:,.0f}  ES {es_h:,.0f}  normal ES {es_n:,.0f}")
""", """
pnl = ...  # simulate from cov and project on w
var_h = ...
es_h = ...
es_n = ...
"""),
    check("""
z = norm.ppf(0.01)
assert abs(es_n - sigma_p * norm.pdf(z) / 0.01) < 1e-6
assert abs(var_h / var_p - 1) < 0.02
assert abs(es_h / es_n - 1) < 0.02
assert es_h > var_h
"""),
    md("""
## Task 3: why one year of data is noisy

Repeat the historical VaR on 200 independent 250-day samples and measure the
spread. Store the relative standard deviation in `rel_sd`.
"""),
    task("""
samples = rng.multivariate_normal(np.zeros(2), cov, size=(200, 250)) @ w
vars_250 = np.array([historical_var(s, alpha=0.01) for s in samples])
rel_sd = vars_250.std() / var_p
print(f"relative s.d. of 250-day historical VaR: {rel_sd:.1%}")
""", """
rel_sd = ...
"""),
    check("""
# Asymptotic s.e. of a sample quantile: sqrt(a(1-a)/n) / f(q), here relative to VaR.
asym = math.sqrt(0.01 * 0.99 / 250) / norm.pdf(z) / abs(z)
assert 0.5 * asym < rel_sd < 2.0 * asym
"""),
    md("""
**Takeaways.** A one-year 99% VaR has a sampling error of roughly 10%
(the asymptotic variance of a sample quantile, Serfling, *Approximation
Theorems of Mathematical Statistics*, 1980, sec. 2.3). ES is always at least VaR.
"""),
]

NOTEBOOKS["03-yield-curve-bootstrap"] = [
    md("""
# 3. Bootstrapping a discount curve

**Level:** L1. **Tags:** `rates`, `curve`, `bootstrap`.
**Library:** `fi_rates.bootstrap`, `fi_rates.curve` (`python/equity/04-fixed-income`).

**Plain idea.** Each market quote pins one more point of the discount
curve. Short deposits give the front directly; each par swap then adds one
pillar so that the swap is worth zero.

**Formal statement** (Hagan and West, "Interpolation methods for curve
construction", *Applied Mathematical Finance* 13, 2006): a deposit gives
$P(T) = 1/(1 + rT)$; a par swap with annual fixed leg satisfies
$r\\sum_i P(t_i) + P(T) = 1$.
"""),
    code(PATHS),
    code("""
from fi_rates.bootstrap import (  # noqa: F401
    Deposit,
    ParSwap,
    bootstrap_curve,
    reprice_instruments,
)
"""),
    md("""
## Task 1: build the curve

Quotes: deposits at 0.25y 3.00%, 0.5y 3.10%, 1y 3.20%; annual par swaps at
2y 3.35%, 3y 3.45%, 5y 3.60%, 7y 3.70%, 10y 3.80%. Bootstrap with the
default log-linear discount-factor interpolation. Store it as `curve`.
"""),
    task("""
quotes = [
    Deposit(0.25, 0.0300), Deposit(0.5, 0.0310), Deposit(1.0, 0.0320),
    ParSwap(2.0, 0.0335), ParSwap(3.0, 0.0345), ParSwap(5.0, 0.0360),
    ParSwap(7.0, 0.0370), ParSwap(10.0, 0.0380),
]
curve = bootstrap_curve(quotes)
for t in (0.25, 1, 2, 5, 10):
    print(f"t={t:5}: df={float(curve.df(t)):.6f} zero={float(curve.zero_rate(t)):.4%}")
""", """
quotes = [...]  # Deposit(maturity, rate) and ParSwap(maturity, rate)
curve = ...
"""),
    md("**Check 1: every quote reprices.** This is the definition of a bootstrap."),
    check("""
errors = [err for _, err in reprice_instruments(quotes, curve)]
assert max(abs(e) for e in errors) < 1e-10
"""),
    md("""
**Check 2: by hand.** The 1y deposit gives $P(1) = 1/1.032$ exactly. The 2y
swap then needs only $P(1)$ and $P(2)$ (annual fixed leg), so
$P(2) = (1 - 0.0335 P(1)) / 1.0335$. Neither needs interpolation.
"""),
    check("""
p1 = 1 / 1.032
p2 = (1 - 0.0335 * p1) / 1.0335
assert abs(float(curve.df(1.0)) - p1) < 1e-12
assert abs(float(curve.df(2.0)) - p2) < 1e-12
assert abs(curve.par_rate(5.0) - 0.0360) < 1e-10
"""),
    md("""
## Task 2: forwards and a parallel shift

Compute the 1y-into-1y simple forward, and the DV01 of a 5y annual swap
paying fixed at 3.60% on 1,000,000 notional, using a 1bp parallel bump.
"""),
    task("""
fwd_1y1y = curve.simple_forward_rate(1.0, 2.0)

def payer_pv(c, rate=0.0360, n=1_000_000.0):
    annuity = sum(float(c.df(t)) for t in range(1, 6))
    return n * (1 - float(c.df(5.0)) - rate * annuity)

dv01 = payer_pv(curve.bumped_parallel(1e-4)) - payer_pv(curve)
print(f"1y1y forward {fwd_1y1y:.4%}, payer DV01 {dv01:,.2f}")
""", """
fwd_1y1y = ...
dv01 = ...  # PV change of a 5y payer swap for +1bp
"""),
    check("""
assert abs(fwd_1y1y - (p1 / p2 - 1)) < 1e-12
annuity = sum(float(curve.df(t)) for t in range(1, 6))
# A payer gains about notional x annuity x 1bp when rates rise.
assert abs(dv01 / (1_000_000 * annuity * 1e-4) - 1) < 0.05
"""),
    md("""
**Takeaways.** Reprice the inputs, check the first pillars by hand, and
check a risk number against the annuity rule of thumb.
"""),
]

NOTEBOOKS["04-optimal-execution"] = [
    md("""
# 4. Optimal execution (Almgren-Chriss)

**Level:** L2. **Tags:** `execution`, `almgren-chriss`, `market-impact`.
**Library:** `eq_algo.almgren_chriss` (`python/equity/08-algo-execution`).

**Plain idea.** Selling fast costs impact; selling slowly leaves you exposed
to price moves. Almgren and Chriss trade the two off with a risk-aversion
parameter $\\lambda$: $\\lambda = 0$ gives a straight line (TWAP), larger
$\\lambda$ front-loads.

**Formal statement** (Almgren and Chriss, "Optimal execution of portfolio
transactions", *Journal of Risk* 3, 2000): minimise $E + \\lambda V$ with
$E = \\tfrac12\\gamma X^2 + \\tfrac{\\tilde\\eta}{\\tau}\\sum n_k^2$ and
$V = \\sigma^2\\tau\\sum x_k^2$. The optimum is
$x_j = X\\sinh(\\kappa(T - t_j))/\\sinh(\\kappa T)$.
"""),
    code(PATHS),
    code("""
import math  # noqa: F401

import numpy as np
from eq_algo.almgren_chriss import ACParams, ac_cost_moments, ac_trajectory  # noqa: F401
"""),
    md("""
## Task 1: trajectories for three risk aversions

Sell 1,000,000 shares over one day in 20 slices with $\\sigma = 0.5$ per
share per day and $\\eta = 2.5\\times10^{-6}$, $\\gamma = 2.5\\times10^{-7}$.
Store the three trajectories for $\\lambda \\in \\{0, 10^{-6}, 10^{-5}\\}$ in
`traj` (a dict).
"""),
    task("""
p = ACParams(total_shares=1e6, n_slices=20, total_time=1.0, sigma=0.5, eta=2.5e-6, gamma=2.5e-7)
traj = {lam: ac_trajectory(p, lam) for lam in (0.0, 1e-6, 1e-5)}
for lam, x in traj.items():
    e, v = ac_cost_moments(p, x)
    print(f"lambda={lam:g}: half done after slice {int(np.argmax(x <= 5e5))}, E={e:,.0f}, sd={math.sqrt(v):,.0f}")
""", """
p = ACParams(total_shares=1e6, n_slices=20, total_time=1.0, sigma=0.5, eta=2.5e-6, gamma=2.5e-7)
traj = ...
"""),
    md("""
**Independent check.** The objective is a quadratic in the interior
holdings $x_1..x_{N-1}$, so the optimum solves a tridiagonal linear system:
$\\frac{2\\tilde\\eta}{\\tau}(2x_k - x_{k-1} - x_{k+1}) + 2\\lambda\\sigma^2\\tau x_k = 0$.
We solve it with `numpy.linalg.solve`, which shares no code with the
library's sinh formula.
"""),
    check("""
def brute_force(p, lam):
    n, tau, et = p.n_slices, p.tau, p.eta_tilde
    a = np.zeros((n - 1, n - 1))
    b = np.zeros(n - 1)
    for k in range(n - 1):
        a[k, k] = 2 * et / tau * 2 + 2 * lam * p.sigma**2 * tau
        if k > 0:
            a[k, k - 1] = -2 * et / tau
        if k < n - 2:
            a[k, k + 1] = -2 * et / tau
    b[0] = 2 * et / tau * p.total_shares
    return np.concatenate([[p.total_shares], np.linalg.solve(a, b), [0.0]])

for lam, x in traj.items():
    assert np.max(np.abs(x - brute_force(p, lam))) < 1e-6 * p.total_shares
assert np.allclose(traj[0.0], np.linspace(1e6, 0, 21))
"""),
    md("""
## Task 2: the efficient frontier is convex and ordered

For a grid of $\\lambda$ compute (sd, E). Higher $\\lambda$ must give lower
variance and higher expected cost.
"""),
    task("""
lams = np.geomspace(1e-8, 1e-4, 9)
frontier = np.array([ac_cost_moments(p, ac_trajectory(p, l)) for l in lams])
for l, (e, v) in zip(lams, frontier, strict=True):
    print(f"lambda={l:.0e}  E={e:,.0f}  sd={math.sqrt(v):,.0f}")
""", """
lams = np.geomspace(1e-8, 1e-4, 9)
frontier = ...  # rows of (E, V)
"""),
    check("""
assert np.all(np.diff(frontier[:, 0]) > 0)
assert np.all(np.diff(frontier[:, 1]) < 0)
# TWAP has the lowest possible expected cost among all schedules.
assert ac_cost_moments(p, traj[0.0])[0] <= frontier[0, 0] + 1e-6
"""),
    md("""
**Takeaways.** A closed form is trusted once a dumb brute-force solve agrees
with it, and the frontier has the shape the theory says.
"""),
]


NOTEBOOKS["05-aad-greeks"] = [
    md("""
# 5. Greeks by adjoint algorithmic differentiation (AAD)

**Level:** L2. **Tags:** `aad`, `greeks`, `black-scholes`.
**Library:** `qp_core.aad` (`python/core`).

**Plain idea.** A pricer is a long chain of simple operations. Record the
chain once on a "tape" while pricing, then walk it backwards with the chain
rule: one backward pass gives the derivative of the price with respect to
every input at once, for a small constant multiple of the cost of one price.
Bumping each input and repricing costs one extra price per input instead.

**Formal statement** (Griewank and Walther, *Evaluating Derivatives*, 2nd
ed., 2008, ch. 3; Giles and Glasserman, "Smoking adjoints", *Risk*, 2006):
for $y = f(x_1, \\dots, x_n)$ built from elementary operations
$v_j = \\phi_j(v_{i \\prec j})$, set the adjoint $\\bar y = 1$ and run
$\\bar v_i \\mathrel{+}= \\bar v_j\\, \\partial\\phi_j/\\partial v_i$ over the tape in
reverse order; then $\\bar x_i = \\partial y / \\partial x_i$.
"""),
    code(PATHS),
    code("""
import math

from qp_core.aad.black_scholes import bs_greeks_aad, bs_price  # noqa: F401
from scipy.stats import norm
"""),
    md("""
## Task 1: all first-order Greeks from one backward pass

Call `bs_greeks_aad` for a call with $S = 100$, $K = 105$, $T = 0.5$,
$r = 3\\%$, $q = 1\\%$, $\\sigma = 25\\%$. Store the result as `g`.
"""),
    task("""
S, K, T, r, q, sigma = 100.0, 105.0, 0.5, 0.03, 0.01, 0.25
g = bs_greeks_aad(S, K, T, r, q, sigma)
print(f"price {g.price:.6f} delta {g.delta:.6f} vega {g.vega:.6f} rho {g.rho:.6f}")
""", """
S, K, T, r, q, sigma = 100.0, 105.0, 0.5, 0.03, 0.01, 0.25
g = ...  # use bs_greeks_aad
"""),
    md("""
**Independent check.** The textbook closed forms (Hull, 11th ed., the
Greeks chapter): $\\Delta = e^{-qT}N(d_1)$,
$\\mathcal V = S e^{-qT}\\varphi(d_1)\\sqrt T$, $\\rho = K T e^{-rT} N(d_2)$.
They share no code with the tape.
"""),
    check("""
d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
d2 = d1 - sigma * math.sqrt(T)
price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
assert abs(g.price - price) < 1e-12
assert abs(g.delta - math.exp(-q * T) * norm.cdf(d1)) < 1e-12
assert abs(g.vega - S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)) < 1e-10
assert abs(g.rho - K * T * math.exp(-r * T) * norm.cdf(d2)) < 1e-10
"""),
    md("""
## Task 2: the same Greeks by bump and reprice

Central differences with a bump of `1e-5` in $S$ and in $\\sigma$, using the
plain-float `bs_price(S, K, T, r, q, sigma)`. Store `delta_fd` and
`vega_fd`. That is two extra prices per Greek; AAD needed one backward pass
for all seven numbers.
"""),
    task("""
h = 1e-5
delta_fd = (bs_price(S + h, K, T, r, q, sigma) - bs_price(S - h, K, T, r, q, sigma)) / (2 * h)
vega_fd = (bs_price(S, K, T, r, q, sigma + h) - bs_price(S, K, T, r, q, sigma - h)) / (2 * h)
print(f"delta fd {delta_fd:.8f}  vega fd {vega_fd:.6f}")
""", """
h = 1e-5
delta_fd = ...
vega_fd = ...
"""),
    check("""
assert abs(delta_fd - g.delta) < 1e-6
assert abs(vega_fd - g.vega) < 1e-5
"""),
    md("""
**Takeaways.** AAD is exact to rounding, not to a bump size, and its cost
does not grow with the number of inputs. The Rust and C++ versions and their
measured cost ratios are in `rust/aad/README.md` and `cpp/aad/README.md`.
"""),
]

NOTEBOOKS["06-market-making"] = [
    md("""
# 6. Market-making quotes: Avellaneda-Stoikov

**Level:** L3. **Tags:** `market-making`, `inventory`, `microstructure`.
**Library:** `qp_exec.mm` (`python/execution`).

**Plain idea.** A market maker posts a bid and an ask. Holding inventory is
risky, so a long maker shades both quotes down (to sell more and buy less)
and a short maker shades them up. The spread pays for that risk and for the
chance that a quote is not hit at all.

**Formal statement** (Avellaneda and Stoikov, "High-frequency trading in a
limit order book", *Quantitative Finance* 8(3), 2008, eqs. 29-30): with mid
$s$, inventory $q$, risk aversion $\\gamma$, volatility $\\sigma$, order
arrival decaying as $A e^{-k\\delta}$ and time left $\\tau$, the reservation
price is $r = s - q\\gamma\\sigma^2\\tau$ and the total spread is
$\\gamma\\sigma^2\\tau + \\frac{2}{\\gamma}\\ln(1 + \\gamma/k)$, centred on $r$.
"""),
    code(PATHS),
    code("""
import math

from qp_exec.mm.quotes import ASParams, avellaneda_stoikov  # noqa: F401
"""),
    md("""
## Task 1: quotes for flat, long and short inventory

With $\\gamma = 0.1$, $\\sigma = 2$, $k = 1.5$, mid 100 and $\\tau = 1$, store
the quotes for $q = -5, 0, 5$ in a dict `quotes` keyed by $q$.
"""),
    task("""
p = ASParams(gamma=0.1, sigma=2.0, k=1.5)
quotes = {q: avellaneda_stoikov(100.0, q, 1.0, p) for q in (-5, 0, 5)}
for q, x in quotes.items():
    print(f"q={q:+d}  bid {x.bid:.4f}  r {x.reservation:.4f}  ask {x.ask:.4f}")
""", """
p = ASParams(gamma=0.1, sigma=2.0, k=1.5)
quotes = ...
"""),
    md("""
**Independent check.** The two formulas above, written out by hand.
"""),
    check("""
g, s, k, tau = 0.1, 2.0, 1.5, 1.0
spread = g * s**2 * tau + 2 / g * math.log(1 + g / k)
for q, x in quotes.items():
    r = 100.0 - q * g * s**2 * tau
    assert abs(x.reservation - r) < 1e-12
    assert abs(x.bid - (r - spread / 2)) < 1e-12 and abs(x.ask - (r + spread / 2)) < 1e-12
assert quotes[5].ask < quotes[0].ask < quotes[-5].ask
"""),
    md("""
## Task 2: the spread at the close

As $\\tau \\to 0$ the inventory term vanishes and the spread falls to
$\\frac{2}{\\gamma}\\ln(1+\\gamma/k)$, which tends to $2/k$ as $\\gamma \\to 0$.
Store the spread at $\\tau = 0$ for $\\gamma = 10^{-6}$ (inventory 3) as
`close_spread`.
"""),
    task("""
close_spread = avellaneda_stoikov(100.0, 3, 0.0, ASParams(gamma=1e-6, sigma=2.0, k=1.5)).spread
print(f"spread at the close, gamma -> 0: {close_spread:.6f}  (2/k = {2 / 1.5:.6f})")
""", """
close_spread = ...
"""),
    check("""
assert abs(close_spread - 2 / 1.5) < 1e-5
"""),
    md("""
**Takeaways.** Inventory moves the centre of the quotes, not their width;
time left widens them. `qp_exec.mm.sim.run_mm` runs these quotes against a
simulated order flow, and `qp_exec.mm.quotes.GLFT` adds the inventory bound
of Gueant, Lehalle and Fernandez-Tapia (2013).
"""),
]

NOTEBOOKS["07-cva"] = [
    md("""
# 7. Credit valuation adjustment (CVA)

**Level:** L3. **Tags:** `xva`, `cva`, `counterparty-credit`.
**Library:** `qp_core.xva` (`python/core`).

**Plain idea.** If the other side of a trade defaults while it owes you
money, you lose that money less what you recover. CVA is the expected loss:
add up, over future dates, what you would be owed times the chance that the
default happens then.

**Formal statement** (Gregory, *The xVA Challenge*, 4th ed., 2020, the CVA
chapter): with discounted expected exposure $EE^*(t)$, survival $S(t)$ and
recovery $R$,
$\\mathrm{CVA} = (1-R)\\int_0^T EE^*(t)\\,(-dS(t)) \\approx
(1-R)\\sum_k EE^*(t_k)\\,[S(t_{k-1}) - S(t_k)]$.
"""),
    code(PATHS),
    code("""
import math

import numpy as np
from qp_core.xva.adjustments import HazardCurve, cva  # noqa: F401
"""),
    md("""
## Task 1: CVA of a flat exposure

A counterparty with a CDS spread of 120bp and recovery 40% has, by the
credit triangle, hazard $\\lambda = s/(1-R) = 2\\%$. Take a flat discounted
exposure of 1,000,000 on a quarterly grid to 5 years. Store the CVA as `c`.
"""),
    task("""
times = np.linspace(0.0, 5.0, 21)
ee = np.full_like(times, 1_000_000.0)
curve = HazardCurve.from_spread(0.012, 0.40)
c = cva(times, ee, curve, 0.40)
print(f"CVA {c:,.2f}")
""", """
times = np.linspace(0.0, 5.0, 21)
ee = np.full_like(times, 1_000_000.0)
curve = ...
c = ...
"""),
    md("""
**Independent check.** With a flat exposure the sum telescopes:
$\\mathrm{CVA} = (1-R)\\,E\\,(1 - e^{-\\lambda T})$, exactly, on any grid.
"""),
    check("""
lam = 0.012 / 0.60
assert abs(c - 0.60 * 1_000_000.0 * (1 - math.exp(-lam * 5.0))) < 1e-6
"""),
    md("""
## Task 2: an amortising exposure

Exposure that runs down linearly from 1,000,000 to 0 (a loan repaid evenly)
must cost less than the flat one. Store it as `c_amort`.
"""),
    task("""
c_amort = cva(times, 1_000_000.0 * (1 - times / 5.0), curve, 0.40)
print(f"amortising CVA {c_amort:,.2f}")
""", """
c_amort = ...
"""),
    md("""
**Independent check.** The integral of $(1-R)E(1-t/T)\\lambda e^{-\\lambda t}$
in closed form, which the quarterly sum approaches from below (the sum
takes the exposure at the end of each quarter).
"""),
    check("""
exact = 0.60 * 1_000_000.0 * (1 - (1 - math.exp(-lam * 5.0)) / (lam * 5.0))
assert 0 < c_amort < exact < c
assert abs(c_amort - exact) / exact < 0.06
"""),
    md("""
**Takeaways.** CVA is exposure times default probability, priced. The
library computes the exposure profile of a swap under Hull-White
(`qp_core.xva.exposure.swap_exposure`) and adds DVA and FVA.
"""),
]

NOTEBOOKS["08-risk-parity"] = [
    md("""
# 8. Risk parity and covariance shrinkage

**Level:** L2. **Tags:** `portfolio`, `risk-parity`, `covariance`.
**Library:** `qp_bt.portfolio` (`python/backtest`).

**Plain idea.** Equal weights let the most volatile asset dominate the risk.
Risk parity picks the weights so that every asset adds the same amount to
portfolio variance. A sample covariance from a short history is noisy, so it
is shrunk toward a simple target first.

**Formal statement.** Risk contribution $RC_i = w_i(\\Sigma w)_i$, with
$\\sum_i RC_i = w^\\top\\Sigma w$; risk parity solves $RC_i = RC_j$ for all
$i, j$ with $w \\ge 0$ and $\\sum w = 1$ (Maillard, Roncalli and Teiletche,
*Journal of Portfolio Management*, 2010; solved by Newton's method as in
Spinu, 2013). Ledoit and Wolf (*Journal of Multivariate Analysis* 88, 2004)
shrink the sample covariance $S$ to $\\delta\\mu I + (1-\\delta)S$ with
$\\mu = \\mathrm{tr}(S)/N$.
"""),
    code(PATHS),
    code("""
import numpy as np
from qp_bt.portfolio.covariance import ledoit_wolf  # noqa: F401
from qp_bt.portfolio.optimise import risk_contributions, risk_parity  # noqa: F401
"""),
    md("""
## Task 1: risk-parity weights for three assets

Vols 10%, 20% and 30%, correlation 0.3 between every pair. Build `cov` and
store the weights as `w`.
"""),
    task("""
vol = np.array([0.10, 0.20, 0.30])
corr = np.full((3, 3), 0.3) + 0.7 * np.eye(3)
cov = np.outer(vol, vol) * corr
w = risk_parity(cov)
print("weights", np.round(w, 6), " risk contributions", np.round(risk_contributions(w, cov), 8))
""", """
vol = np.array([0.10, 0.20, 0.30])
corr = np.full((3, 3), 0.3) + 0.7 * np.eye(3)
cov = ...
w = ...
"""),
    md("""
**Independent check.** With one common correlation the equal-risk weights
are proportional to $1/\\sigma_i$ (Maillard et al. 2010, section 2), which
needs no solver.
"""),
    check("""
inv = 1 / vol
assert np.allclose(w, inv / inv.sum(), atol=1e-10)
rc = w * (cov @ w)
assert np.allclose(rc, rc.mean(), atol=1e-12)
"""),
    md("""
## Task 2: shrink a noisy covariance

Draw 60 days of returns for 20 assets from a seeded normal with identity
covariance (`np.random.default_rng(7)`). Store the shrunk matrix and its
intensity as `sigma_lw, delta`.
"""),
    task("""
x = np.random.default_rng(7).standard_normal((60, 20))
sigma_lw, delta = ledoit_wolf(x)
print(f"shrinkage intensity {delta:.4f}")
""", """
x = np.random.default_rng(7).standard_normal((60, 20))
sigma_lw, delta = ...
"""),
    md("""
**Independent check.** The definition $\\delta\\mu I + (1-\\delta)S$ rebuilt by
hand, and the shrunk matrix is closer to the true identity than $S$ is.
"""),
    check("""
xc = x - x.mean(axis=0)
s = xc.T @ xc / len(x)
mu = np.trace(s) / 20
assert 0 < delta < 1
assert np.allclose(sigma_lw, delta * mu * np.eye(20) + (1 - delta) * s, atol=1e-14)
assert np.linalg.norm(sigma_lw - np.eye(20)) < np.linalg.norm(s - np.eye(20))
"""),
    md("""
**Takeaways.** Risk parity needs only a covariance, so the quality of that
covariance decides the result; shrinkage is the cheap first fix.
"""),
]


# --------------------------------------------------------------------------
NOQA_UNUSED = "  # noqa: F401"
STUB_FOOTER = "# Exercise: replace the ... above, then run the check cell below."


def _cell(kind: str, source: str) -> dict[str, object]:
    lines = source.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}


def render(name: str, variant: str) -> str:
    cells = []
    for kind, text, stub in NOTEBOOKS[name]:
        if kind == "md":
            cells.append(_cell("md", text))
        elif kind == "task" and variant == "exercise":
            cells.append(_cell("code", stub + "\n" + STUB_FOOTER))
        elif kind == "code" and variant == "solution" and "_paths" not in text:
            # every import is used once the tasks are filled in
            cells.append(_cell("code", text.replace(NOQA_UNUSED, "")))
        else:
            cells.append(_cell("code", text))
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "qp": {"variant": variant, "source": "docs/notebooks/build_notebooks.py"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, c in enumerate(cells):
        c["id"] = f"{name[:2]}-{variant[0]}-{i:02d}"
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def outputs() -> dict[Path, str]:
    return {HERE / f"{name}-{variant}.ipynb": render(name, variant)
            for name in NOTEBOOKS for variant in ("exercise", "solution")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build or check the learning notebooks.")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    stale = []
    for path, text in outputs().items():
        if args.check:
            if not path.exists() or path.read_text("utf-8") != text:
                stale.append(path.name)
        else:
            path.write_text(text, "utf-8", newline="\n")
    if stale:
        print("stale notebooks (run build_notebooks.py):", ", ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
