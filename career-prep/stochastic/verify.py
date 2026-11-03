"""Independent checks for the stochastic calculus pack.

The key in ``keys/answers.md`` uses closed forms (Ito's lemma, the Ito
isometry, reflection, Girsanov). Nothing here reuses those formulas: each
answer is recomputed by numerical integration against a Gaussian density,
by solving the moment or boundary-value ODE numerically, or (as a second
check) by a seeded simulation. ``career-prep/tests/test_quant_packs.py``
compares the three.

Run ``python career-prep/stochastic/verify.py`` to print the table.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy.integrate import dblquad, quad, solve_ivp
from scipy.linalg import solve_banded
from scipy.optimize import brentq
from scipy.stats import invgauss, norm

INF = math.inf


def _gauss(f: Callable[[float], float], var: float) -> float:
    """E[f(X)] for X ~ N(0, var), by quadrature."""
    sd = math.sqrt(var)
    c = 1 / (sd * math.sqrt(2 * math.pi))
    lim = 16 * sd  # the density is below 1e-55 beyond 16 sd
    return float(
        quad(
            lambda x: f(x) * c * math.exp(-x * x / (2 * var)),
            -lim,
            lim,
            epsabs=1e-13,
            epsrel=1e-12,
            limit=200,
        )[0]
    )


def _gauss2(f: Callable[[float, float], float], t1: float, t2: float) -> float:
    """E[f(W_t1, W_t2)] with t1 < t2, integrating over W_t1 and the increment."""
    return _gauss(lambda x: _gauss(lambda z: f(x, x + z), t2 - t1), t1)


def _lognormal(f: Callable[[float], float], s0: float, drift: float, sig: float, t: float) -> float:
    """E[f(S_t)] for S_t = s0 exp((drift - sig^2/2) t + sig W_t)."""
    m = (drift - 0.5 * sig * sig) * t
    return _gauss(lambda x: f(s0 * math.exp(m + sig * x)), t)


GBM = (100.0, 0.05, 0.2, 1.0)


def _gbm_median() -> float:
    """Median of S_1 by root-finding the CDF, the CDF itself by quadrature of the density."""
    s0, mu, sig, t = GBM
    m = (mu - sig * sig / 2) * t

    def cdf(k: float) -> float:
        x_k = (math.log(k / s0) - m) / sig  # S <= k  <=>  W <= x_k
        return float(quad(lambda x: norm.pdf(x, scale=math.sqrt(t)), -INF, x_k, epsabs=1e-14)[0])

    return float(brentq(lambda k: cdf(k) - 0.5, 50.0, 200.0, xtol=1e-12))


def _ou_moments() -> tuple[float, float]:
    """Mean and variance of OU at t=1 from the moment ODEs (kappa=2, theta=1, sigma=0.5)."""
    k, th, s = 2.0, 1.0, 0.5

    def rhs(_t: float, y: np.ndarray) -> list[float]:
        m, v = y
        return [k * (th - m), -2 * k * v + s * s]

    sol = solve_ivp(rhs, (0.0, 1.0), [0.0, 0.0], rtol=1e-12, atol=1e-14)
    return float(sol.y[0, -1]), float(sol.y[1, -1])


def _ou_stationary_var() -> float:
    """Run the variance ODE to t=40, far past the 1/(2 kappa) relaxation time."""
    k, s = 2.0, 0.5
    sol = solve_ivp(lambda _t, v: [-2 * k * v[0] + s * s], (0.0, 40.0), [0.0], rtol=1e-12, atol=1e-14)
    return float(sol.y[0, -1])


def _hit_prob_fd(mu: float, a: float, low: float, n: int) -> float:
    """P(BM with drift mu from 0 hits a before low): solve f''/2 + mu f' = 0 by finite differences."""
    x = np.linspace(low, a, n + 1)
    h = x[1] - x[0]
    lo, di, up = 0.5 / h**2 - mu / (2 * h), -1.0 / h**2, 0.5 / h**2 + mu / (2 * h)
    m = n - 1
    bands = np.zeros((3, m))
    bands[0, 1:] = up
    bands[1, :] = di
    bands[2, :-1] = lo
    rhs = np.zeros(m)
    rhs[-1] = -up  # boundary f(a) = 1; f(low) = 0
    f = solve_banded((1, 1), bands, rhs)
    i0 = int(round((0.0 - low) / h)) - 1  # grid chosen so that 0 is a node
    return float(f[i0])


def _hit_prob() -> float:
    """Richardson extrapolation of the O(h^2) scheme on grids h and h/2."""
    coarse = _hit_prob_fd(-0.5, 1.0, -40.0, 41 * 400)
    fine = _hit_prob_fd(-0.5, 1.0, -40.0, 41 * 800)
    return (4 * fine - coarse) / 3


def _hit_time_mean() -> float:
    """E[first time BM with drift 0.5 reaches 2], integrating t against scipy's inverse Gaussian."""
    mu, a = 0.5, 2.0
    dens = invgauss(mu=1 / (a * mu), scale=a * a)  # mean a/mu, shape a^2
    return float(quad(lambda t: t * dens.pdf(t), 0, INF, limit=500)[0])


def _first_passage_density_mean() -> float:
    """Same quantity from the first-passage density a/sqrt(2 pi t^3) exp(-(a - mu t)^2 / 2t)."""
    mu, a = 0.5, 2.0
    return float(
        quad(
            lambda t: t * a / math.sqrt(2 * math.pi * t**3) * math.exp(-((a - mu * t) ** 2) / (2 * t)),
            0,
            INF,
            limit=500,
        )[0]
    )


def _e_w2(s: float) -> float:
    return _gauss(lambda x: x * x, s) if s > 0 else 0.0


EXACT: dict[str, Callable[[], float]] = {
    "S01": lambda: _gauss(lambda x: x * x, 3.0),
    "S02": lambda: _gauss2(lambda x, y: x * y, 2.0, 5.0),
    # Var = int int min(s, u); split at the diagonal so the integrand is smooth
    "S03": lambda: 2 * float(dblquad(lambda s, u: s, 0, 1, 0, lambda u: u, epsabs=1e-14)[0]),
    "S04": lambda: _gauss(lambda x: x**4, 1.0),
    "S05": lambda: float(quad(_e_w2, 0, 2)[0]),
    "S06": lambda: _gauss(math.exp, 2.0),
    # P(max >= 1) = 1 - P(W_1 < 1, max < 1); the joint density of (W, max<1) by the method of images
    "S07": lambda: 1 - float(quad(lambda y: norm.pdf(y) - norm.pdf(2 - y), -INF, 1)[0]),
    "S08": lambda: _lognormal(lambda s: s, *GBM),
    "S09": _gbm_median,
    "S10": lambda: _lognormal(lambda s: s * s, *GBM) - _lognormal(lambda s: s, *GBM) ** 2,
    "S11": lambda: _ou_moments()[0],
    "S12": lambda: _ou_moments()[1],
    "S13": _ou_stationary_var,
    "S14": lambda: float(quad(_e_w2, 0, 1)[0]),
    "S15": _hit_time_mean,
    "S16": _hit_prob,
    "S17": lambda: _gauss(lambda x: x * math.exp(x - 0.5), 1.0),
    "S18": lambda: _gauss2(lambda x, y: x * x * y * y, 1.0, 2.0),
    "S19": lambda: _gauss(lambda x: x * x / 2, 1.0),
    "S20": lambda: math.exp(-0.05) * _lognormal(lambda s: s * s, 100.0, 0.05, 0.2, 1.0),
    "S21": lambda: _gauss(lambda x: (1 + x) ** 2, 2.0),
    "S22": lambda: _gauss(lambda x: math.exp(2 * x), 1.0) - _gauss(math.exp, 1.0) ** 2,
}

# A second deterministic route where the first leans on a library distribution.
SECOND_ROUTE: dict[str, Callable[[], float]] = {"S15": _first_passage_density_mean}


# ---------------------------------------------------------------- Monte Carlo
N_MC = 200_000


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _mc_cov() -> float:
    rng = _rng(102)
    w2 = rng.standard_normal(N_MC) * math.sqrt(2)
    w5 = w2 + rng.standard_normal(N_MC) * math.sqrt(3)
    return float((w2 * w5).mean())


def _mc_ito_square() -> float:
    """E[(int_0^2 W dW)^2] from a left-point (Ito) sum."""
    steps, paths = 400, 20_000
    dt = 2.0 / steps
    dw = _rng(105).standard_normal((paths, steps)) * math.sqrt(dt)
    w_left = np.cumsum(dw, axis=1) - dw
    return float(((w_left * dw).sum(axis=1) ** 2).mean())


def _mc_strat() -> float:
    """E[int_0^1 W o dW] from a midpoint (Stratonovich) sum."""
    steps, paths = 200, 20_000
    dt = 1.0 / steps
    dw = _rng(119).standard_normal((paths, steps)) * math.sqrt(dt)
    mid = np.cumsum(dw, axis=1) - dw / 2
    return float((mid * dw).sum(axis=1).mean())


def _mc_max() -> float:
    """P(max_{s<=1} W >= 1) on a grid with a Brownian-bridge crossing correction per step."""
    steps, paths = 200, 40_000
    dt = 1.0 / steps
    rng = _rng(107)
    inc = rng.standard_normal((paths, steps)) * math.sqrt(dt)
    w = np.concatenate([np.zeros((paths, 1)), np.cumsum(inc, axis=1)], axis=1)
    a, b = w[:, :-1], w[:, 1:]
    cross = np.exp(-2 * np.clip(1 - a, 0, None) * np.clip(1 - b, 0, None) / dt)
    p_no = np.prod(np.where((a >= 1) | (b >= 1), 0.0, 1 - cross), axis=1)
    return float(1 - p_no.mean())


def _mc_ou(which: int) -> float:
    """Exactly discretised OU at t=1 (kappa=2, theta=1, sigma=0.5, x0=0)."""
    k, th, s = 2.0, 1.0, 0.5
    steps = 50
    dt = 1.0 / steps
    e = math.exp(-k * dt)
    sd = s * math.sqrt((1 - e * e) / (2 * k))
    x = np.zeros(N_MC)
    rng = _rng(111)
    for _ in range(steps):
        x = th + (x - th) * e + sd * rng.standard_normal(N_MC)
    return float(x.mean() if which == 0 else x.var())


def _mc_w1w2() -> float:
    rng = _rng(118)
    w1 = rng.standard_normal(N_MC)
    w2 = w1 + rng.standard_normal(N_MC)
    return float((w1 * w1 * w2 * w2).mean())


MONTE_CARLO: dict[str, tuple[Callable[[], float], float]] = {
    # id: (estimator, absolute tolerance)
    "S02": (_mc_cov, 0.05),
    "S05": (_mc_ito_square, 0.1),
    "S07": (_mc_max, 0.01),
    "S08": (lambda: float((100 * np.exp(0.03 + 0.2 * _rng(108).standard_normal(N_MC))).mean()), 0.2),
    "S11": (lambda: _mc_ou(0), 0.005),
    "S12": (lambda: _mc_ou(1), 0.002),
    "S18": (_mc_w1w2, 0.15),
    "S19": (_mc_strat, 0.03),
}


if __name__ == "__main__":  # pragma: no cover
    for pid, fn in EXACT.items():
        mc = MONTE_CARLO.get(pid)
        extra = f"  MC {mc[0]():.4f}" if mc else ""
        print(f"{pid}  {fn():.10g}{extra}")
