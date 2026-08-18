"""Heston stochastic volatility under Garman-Kohlhagen (FX: q = r_f).

Dynamics under the domestic risk-neutral measure:

    dS/S = (r_d - r_f) dt + sqrt(v) dW_S
    dv   = kappa (theta - v) dt + xi sqrt(v) dW_v,   d<W_S, W_v> = rho dt

Characteristic function uses the "little Heston trap" formulation
(Albrecher et al. 2007): branching in the complex log is avoided by
taking ``g = (beta - d)/(beta + d)`` with the principal square root,
which is continuous in ``u`` for all maturities (the original Heston
1993 form with ``1/g`` winds around the branch cut for long T).

Two independent Fourier pricing methods, cross-validated in tests:

* :func:`price_gil_pelaez` - semi-analytic P1/P2 probabilities via
  adaptive quadrature of the inversion integrals (slow, very accurate).
* :func:`price_cos` - Fang-Oosterlee COS method with cumulant-based
  truncation (fast, vectorised over strikes; used by calibration).

Rates are continuously compounded, ACT/365F year fractions, vols in
decimals; ``cp=+1`` call / ``-1`` put on the base currency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from .garman_kohlhagen import gk_forward

__all__ = [
    "HestonParams",
    "heston_cf",
    "price_gil_pelaez",
    "price_cos",
    "heston_price",
    "heston_digital",
]


@dataclass(frozen=True)
class HestonParams:
    """Heston parameters.

    v0, theta : instantaneous / long-run *variance* (vol squared).
    kappa : mean-reversion speed (1/years).
    xi : vol-of-vol (of variance).
    rho : spot-vol correlation in [-1, 1].
    """

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self) -> None:
        if self.v0 <= 0.0 or self.theta <= 0.0:
            raise ValueError(f"v0 and theta must be positive: v0={self.v0}, theta={self.theta}")
        if self.kappa <= 0.0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if not 1e-8 <= self.xi:
            raise ValueError(f"xi must be >= 1e-8, got {self.xi}")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")

    @property
    def feller_ratio(self) -> float:
        """``2 kappa theta / xi^2``; >= 1 means the origin is unattainable."""
        return 2.0 * self.kappa * self.theta / (self.xi * self.xi)

    @property
    def feller_satisfied(self) -> bool:
        return self.feller_ratio >= 1.0


def heston_cf(
    u: np.ndarray | complex, T: float, params: HestonParams,
    mu: float = 0.0, x0: float = 0.0,
) -> np.ndarray | complex:
    """Characteristic function ``E[exp(i u ln S_T)]``, little-trap form.

    Parameters
    ----------
    u : array-like, real or complex
        Fourier argument (complex supported, needed for the P1 shift).
    T : float
        Year fraction.
    mu : float
        Risk-neutral drift ``r_d - r_f``.
    x0 : float
        ``ln S_0`` (use 0 for the CF of the log-return).
    """
    u = np.asarray(u, dtype=complex)
    p = params
    beta = p.kappa - 1j * p.rho * p.xi * u
    d = np.sqrt(beta * beta + p.xi * p.xi * (1j * u + u * u))
    g = (beta - d) / (beta + d)
    e_dt = np.exp(-d * T)
    D = (beta - d) / (p.xi * p.xi) * (1.0 - e_dt) / (1.0 - g * e_dt)
    C = (p.kappa * p.theta / (p.xi * p.xi)) * (
        (beta - d) * T - 2.0 * np.log((1.0 - g * e_dt) / (1.0 - g))
    )
    return np.exp(1j * u * (x0 + mu * T) + C + D * p.v0)


# ----------------------------------------------------------------------
# Method 1: Gil-Pelaez inversion (adaptive quadrature)
# ----------------------------------------------------------------------

def _gil_pelaez_p1_p2(
    S: float, K: float, T: float, r_d: float, r_f: float, params: HestonParams
) -> tuple[float, float]:
    mu = r_d - r_f
    x0 = math.log(S)
    lnK = math.log(K)
    F = gk_forward(S, T, r_d, r_f)

    def integrand2(u: float) -> float:
        val = np.exp(-1j * u * lnK) * heston_cf(u, T, params, mu, x0) / (1j * u)
        return float(np.real(val))

    def integrand1(u: float) -> float:
        val = (
            np.exp(-1j * u * lnK)
            * heston_cf(u - 1j, T, params, mu, x0)
            / (1j * u * F)
        )
        return float(np.real(val))

    i1 = quad(integrand1, 0.0, np.inf, limit=250, epsabs=1e-12, epsrel=1e-10)[0]
    i2 = quad(integrand2, 0.0, np.inf, limit=250, epsabs=1e-12, epsrel=1e-10)[0]
    return 0.5 + i1 / math.pi, 0.5 + i2 / math.pi


def price_gil_pelaez(
    S: float, K: float, T: float, r_d: float, r_f: float,
    params: HestonParams, cp: int = 1,
) -> float:
    """Heston vanilla via the two Gil-Pelaez inversion probabilities.

    ``Call = e^{-r_d T} (F P1 - K P2)``; puts via exact parity.
    """
    if S <= 0 or K <= 0 or T <= 0:
        raise ValueError("S, K, T must be positive")
    if cp not in (+1, -1):
        raise ValueError(f"cp must be +1 or -1, got {cp}")
    F = gk_forward(S, T, r_d, r_f)
    p1, p2 = _gil_pelaez_p1_p2(S, K, T, r_d, r_f, params)
    call = math.exp(-r_d * T) * (F * p1 - K * p2)
    if cp == 1:
        return call
    return call - math.exp(-r_d * T) * (F - K)


def heston_digital(
    S: float, K: float, T: float, r_d: float, r_f: float,
    params: HestonParams, cp: int = 1,
) -> float:
    """Domestic cash-or-nothing digital: ``e^{-r_d T} P2`` (call side)."""
    _, p2 = _gil_pelaez_p1_p2(S, K, T, r_d, r_f, params)
    if cp == 1:
        return math.exp(-r_d * T) * p2
    return math.exp(-r_d * T) * (1.0 - p2)


# ----------------------------------------------------------------------
# Method 2: COS (Fang-Oosterlee)
# ----------------------------------------------------------------------

def _cumulants(T: float, params: HestonParams, mu: float) -> tuple[float, float]:
    """First two cumulants of ln(S_T/S_0) (Fang & Oosterlee 2008, Table 11)."""
    p = params
    k, th, x, v0 = p.kappa, p.theta, p.xi, p.v0
    r = p.rho
    e = math.exp(-k * T)
    c1 = mu * T + (1.0 - e) * (th - v0) / (2.0 * k) - 0.5 * th * T
    c2 = (
        x * T * k * e * (v0 - th) * (8.0 * k * r - 4.0 * x)
        + k * r * x * (1.0 - e) * (16.0 * th - 8.0 * v0)
        + 2.0 * th * k * T * (-4.0 * k * r * x + x * x + 4.0 * k * k)
        + x * x * ((th - 2.0 * v0) * e * e + th * (6.0 * e - 7.0) + 2.0 * v0)
        + 8.0 * k * k * (v0 - th) * (1.0 - e)
    ) / (8.0 * k ** 3)
    return c1, c2


def price_cos(
    S: float,
    K: np.ndarray | float,
    T: float,
    r_d: float,
    r_f: float,
    params: HestonParams,
    cp: int = 1,
    N: int = 1024,
    L: float = 14.0,
) -> np.ndarray | float:
    """Heston vanilla via the COS method, vectorised over strikes.

    The put is expanded in the coordinate ``y = ln(S_T/K)`` (bounded
    payoff, best conditioning) and the call recovered by exact parity.
    The truncation interval is *per strike*,
    ``[a_K, b_K] = x + c1 -+ L sqrt(|c2|)`` with ``x = ln(S/K)`` (the
    density of y is centred at ``x + c1``); the interval *width* is
    strike-independent, so the CF grid is shared across strikes.

    Returns a scalar for scalar ``K``, else an ndarray.
    """
    if S <= 0 or T <= 0:
        raise ValueError("S and T must be positive")
    if cp not in (+1, -1):
        raise ValueError(f"cp must be +1 or -1, got {cp}")
    scalar = np.ndim(K) == 0
    K = np.atleast_1d(np.asarray(K, dtype=float))
    if np.any(K <= 0.0):
        raise ValueError("strikes must be positive")
    mu = r_d - r_f
    c1, c2 = _cumulants(T, params, mu)
    width = L * math.sqrt(abs(c2))
    x = np.log(S / K)  # (nK,)
    a = x + c1 - width  # per-strike lower truncation, y-coordinate
    span = 2.0 * width

    kk = np.arange(N)
    u = kk * math.pi / span  # shared CF grid (span is strike-independent)
    cf_z = heston_cf(u, T, params, mu=mu, x0=0.0)  # CF of z = ln(S_T/S_0)

    # Put payoff coefficients on [a, d] with d = min(0, b):
    # V_k = 2/span * K * (-chi_k(a, d) + psi_k(a, d))
    d = np.maximum(a, np.minimum(0.0, a + span))  # (nK,) clamp into [a, b]
    da = d - a  # (nK,)
    cos_d = np.cos(np.outer(da, u))  # (nK, N)
    sin_d = np.sin(np.outer(da, u))
    exp_d = np.exp(d)[:, None]
    chi = (cos_d * exp_d - np.exp(a)[:, None] + u[None, :] * sin_d * exp_d) / (
        1.0 + u * u
    )[None, :]
    psi = np.empty((len(K), N))
    psi[:, 0] = da
    psi[:, 1:] = sin_d[:, 1:] / u[None, 1:]
    Vk = (2.0 / span) * (-chi + psi)

    # CF of y = z + x, times e^{-i u a}: e^{i u (x - a)} cf_z(u)
    terms = np.real(np.exp(1j * np.outer(x - a, u)) * cf_z[None, :]) * Vk
    terms[:, 0] *= 0.5  # first term halved
    put = math.exp(-r_d * T) * K * np.sum(terms, axis=1)
    put = np.maximum(put, 0.0)
    if cp == -1:
        out = put
    else:
        F = gk_forward(S, T, r_d, r_f)
        # exact parity, floored at zero (COS noise ~1e-6 at >8-sigma strikes)
        out = np.maximum(put + math.exp(-r_d * T) * (F - K), 0.0)
    return float(out[0]) if scalar else out


def heston_price(
    S: float, K: np.ndarray | float, T: float, r_d: float, r_f: float,
    params: HestonParams, cp: int = 1, method: str = "cos", **kwargs,
) -> np.ndarray | float:
    """Dispatch to a Fourier method: ``method`` in {"cos", "gil_pelaez"}."""
    if method == "cos":
        return price_cos(S, K, T, r_d, r_f, params, cp, **kwargs)
    if method == "gil_pelaez":
        if np.ndim(K) > 0:
            return np.array(
                [price_gil_pelaez(S, float(k), T, r_d, r_f, params, cp) for k in K]
            )
        return price_gil_pelaez(S, float(K), T, r_d, r_f, params, cp)
    raise ValueError(f"unknown method {method!r}; use 'cos' or 'gil_pelaez'")
