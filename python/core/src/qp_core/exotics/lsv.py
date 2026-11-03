"""Local stochastic volatility calibrated by the particle method.

Plain idea
----------
A local-vol model reprices every vanilla but gives forward smiles that
flatten too fast; a stochastic-vol model has realistic dynamics but cannot
fit the whole surface. LSV multiplies the stochastic vol by a "leverage"
function ``L(t, S)`` chosen so that the model again reprices every vanilla.
The right leverage depends on the model's own conditional variance, which
is unknown until you simulate; the particle method estimates it on the fly
from the cloud of simulated paths, one time step at a time.

Formal statement
----------------
Under the pricing measure, with ``x = ln S``:

    dx = (r - q - L^2 V / 2) dt + L(t, S) sqrt(V) dW_1
    dV = kappa (theta - V) dt + xi sqrt(V) dW_2,   d<W_1, W_2> = rho dt

Gyongy's (1986) mimicking theorem gives the calibration condition
``L(t, S)^2 E[V_t | S_t = S] = sigma_Dup(t, S)^2`` (Guyon and
Henry-Labordere 2012, "Being particular about calibration", Risk,
January, eq. 3). The particle method replaces the conditional expectation
with a Nadaraya-Watson kernel regression over the ``N`` simulated
particles at each step (ibid., s.2), evaluated on a spot grid and
interpolated. Variance uses full-truncation Euler (Lord, Koekkoek and van
Dijk 2010).

The Dupire local vol comes from a total-variance surface ``w(k, T)``,
``k = ln(K / F_T)``, by Gatheral (2006, *The Volatility Surface*, eq. 1.10):

    sigma_Dup^2 = (dw/dT) / (1 - k w_k / w + (w_k^2 / 4)(-1/4 - 1/w + k^2/w^2) + w_kk / 2)

with derivatives by central differences.

After calibration, prices come from a *second, independent* simulation
with the leverage table frozen, so the reported standard error is honest
(the calibration particles are not reused for pricing).

Worked example
--------------
See ``tests/test_exotics_lsv.py``: on a power-law SSVI surface with Heston
``kappa = 1.5, theta = v0 = 0.04, xi = 0.5, rho = -0.6``, the calibrated
LSV reprices one-year vanillas from 80 to 120 within 3 standard errors
and 0.2 vol points of the surface.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "HestonParams",
    "LSVCalibration",
    "calibrate_lsv_particles",
    "dupire_local_vol",
    "lsv_price_vanillas",
]

TotalVariance = Callable[[np.ndarray, float], np.ndarray]


@dataclass(frozen=True)
class HestonParams:
    """Heston variance dynamics: ``v0, kappa, theta, xi, rho``."""

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self) -> None:
        if not (self.v0 > 0 and self.kappa > 0 and self.theta > 0 and self.xi >= 0):
            raise ValueError("v0, kappa, theta must be > 0 and xi >= 0")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must be in (-1, 1)")


def dupire_local_vol(w: TotalVariance, k: np.ndarray, T: float, dk: float = 1e-4, dT: float = 1e-4) -> np.ndarray:
    """Dupire local vol at log-moneyness ``k`` and time ``T`` from total variance ``w(k, T)``."""
    k = np.asarray(k, dtype=float)
    if T <= dT:
        raise ValueError("T must exceed the time bump dT")
    w0 = w(k, T)
    wT = (w(k, T + dT) - w(k, T - dT)) / (2.0 * dT)
    wp, wm = w(k + dk, T), w(k - dk, T)
    wk = (wp - wm) / (2.0 * dk)
    wkk = (wp - 2.0 * w0 + wm) / (dk * dk)
    den = 1.0 - k * wk / w0 + 0.25 * wk * wk * (-0.25 - 1.0 / w0 + k * k / (w0 * w0)) + 0.5 * wkk
    if np.any(den <= 0.0) or np.any(wT < 0.0):
        raise ValueError("surface has butterfly or calendar arbitrage at the requested points")
    return np.sqrt(wT / den)


@dataclass(frozen=True)
class LSVCalibration:
    """Leverage on a spot grid per step: step ``i`` runs ``times[i] -> times[i+1]``
    with leverage ``lev[i]`` on ``x_grids[i]``."""

    times: np.ndarray
    x_grids: tuple[np.ndarray, ...]
    leverage: tuple[np.ndarray, ...]
    S0: float
    r: float
    q: float
    heston: HestonParams

    def lev_at(self, i: int, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self.x_grids[i], self.leverage[i])


def _step(
    rng: np.random.Generator, x: np.ndarray, v: np.ndarray, lev: np.ndarray, dt: float,
    r: float, q: float, h: HestonParams,
) -> tuple[np.ndarray, np.ndarray]:
    z1 = rng.standard_normal(x.size)
    z2 = h.rho * z1 + math.sqrt(1.0 - h.rho * h.rho) * rng.standard_normal(x.size)
    vp = np.maximum(v, 0.0)
    sv = np.sqrt(vp * dt)
    x_new = x + (r - q - 0.5 * lev * lev * vp) * dt + lev * sv * z1
    v_new = v + h.kappa * (h.theta - vp) * dt + h.xi * sv * z2
    return x_new, v_new


def calibrate_lsv_particles(
    w: TotalVariance, S0: float, r: float, q: float, T: float, heston: HestonParams,
    n_particles: int = 50_000, n_steps: int = 100, n_grid: int = 61, bandwidth: float = 1.0,
    grid_power: float = 2.0, seed: int = 0,
) -> LSVCalibration:
    """Particle-method leverage calibration to the surface ``w``.

    The kernel is Gaussian with bandwidth ``bandwidth * sigma_atm sqrt(t) N^(-1/5)``
    in log-spot (Silverman's rate; Guyon and Henry-Labordere use the same
    scaling). The leverage grid spans the 0.1%-99.9% particle quantiles;
    outside it the leverage is held flat.

    Time steps are ``t_i = T (i / n)^grid_power``. Surfaces whose short-end
    wings steepen like ``T^(-1/2)`` (power-law SSVI with ``gamma = 1/2``)
    have a local vol that is singular as ``t -> 0``; a uniform grid leaves
    a bias of 0.1-0.4 vol at one year in the wings, the quadratic default
    removes most of it (``tests/test_exotics_lsv.py`` records both).
    """
    if not (S0 > 0 and T > 0 and n_particles > 100 and n_steps > 0 and n_grid > 2):
        raise ValueError("bad calibration sizes or inputs")
    rng = np.random.default_rng(seed)
    tgrid = T * (np.arange(n_steps + 1) / n_steps) ** grid_power
    x = np.full(n_particles, math.log(S0))
    v = np.full(n_particles, heston.v0)
    grids, levs = [], []
    atm = math.sqrt(max(float(w(np.zeros(1), T)[0]) / T, 1e-8))
    for i in range(n_steps):
        t = float(tgrid[i])
        dt = float(tgrid[i + 1] - tgrid[i])
        if i == 0:
            grid = np.array([x[0] - 1.0, x[0], x[0] + 1.0])
            lv = dupire_local_vol(w, np.zeros(1), 0.5 * dt, dT=min(1e-4, 0.25 * dt))[0]
            lev = np.full(3, lv / math.sqrt(heston.v0))
        else:
            lo, hi = np.quantile(x, [0.001, 0.999])
            grid = np.linspace(lo, hi, n_grid)
            fwd = math.log(S0) + (r - q) * t
            lv = dupire_local_vol(w, grid - fwd, t, dT=min(1e-4, 0.5 * t))
            hbw = bandwidth * atm * math.sqrt(t) * n_particles ** (-0.2)
            vp = np.maximum(v, 0.0)
            # kernel regression on the grid, in chunks to bound memory
            num = np.zeros(n_grid)
            den = np.zeros(n_grid)
            for j0 in range(0, n_particles, 20_000):
                d = (grid[:, None] - x[None, j0:j0 + 20_000]) / hbw
                kern = np.exp(-0.5 * d * d)
                num += kern @ vp[j0:j0 + 20_000]
                den += kern.sum(axis=1)
            cond = np.where(den > 1e-300, num / np.maximum(den, 1e-300), heston.v0)
            lev = lv / np.sqrt(np.maximum(cond, 1e-8))
        grids.append(grid)
        levs.append(lev)
        x, v = _step(rng, x, v, np.interp(x, grid, lev), dt, r, q, heston)
    return LSVCalibration(tgrid, tuple(grids), tuple(levs), S0, r, q, heston)


def lsv_price_vanillas(
    cal: LSVCalibration, strikes: np.ndarray, T: float, n_paths: int = 100_000, seed: int = 1,
    control_variate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Call prices at ``strikes`` and their standard errors from a fresh simulation.

    ``T`` must equal the calibration horizon (the step grid is reused).
    """
    n_steps = cal.times.size - 1
    if abs(cal.times[-1] - T) > 1e-12 * max(T, 1.0):
        raise ValueError("T must equal the calibration horizon")
    rng = np.random.default_rng(seed)
    x = np.full(n_paths, math.log(cal.S0))
    v = np.full(n_paths, cal.heston.v0)
    for i in range(n_steps):
        dt = float(cal.times[i + 1] - cal.times[i])
        x, v = _step(rng, x, v, cal.lev_at(i, x), dt, cal.r, cal.q, cal.heston)
    s = np.exp(x)
    df = math.exp(-cal.r * T)
    K = np.asarray(strikes, dtype=float)
    pay = df * np.maximum(s[:, None] - K[None, :], 0.0)
    if not control_variate:
        return pay.mean(axis=0), pay.std(axis=0, ddof=1) / math.sqrt(n_paths)
    # Control variate: the discounted terminal spot, whose mean is known
    # exactly because each log-Euler step is an exact martingale step given
    # the leverage and variance at its start.
    c = df * s - cal.S0 * math.exp(-cal.q * T)
    cc = c - c.mean()
    beta = (cc @ (pay - pay.mean(axis=0))) / (cc @ cc)
    adj = pay - c[:, None] * beta[None, :]
    return adj.mean(axis=0), adj.std(axis=0, ddof=1) / math.sqrt(n_paths)
