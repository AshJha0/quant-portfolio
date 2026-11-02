"""Andersen-Lake-Offengeim (ALO) American option pricer.

Plain idea: an American put is a European put plus the value of being
allowed to exercise early. That premium is an integral over the exercise
boundary ``B(tau)``, and the boundary itself solves a fixed-point equation.
ALO make the fixed point fast: they store a smooth transform of the
boundary at Chebyshev nodes in ``sqrt(tau)``, evaluate the integrals by
Gauss-Legendre after a change of variable that removes the
``1/sqrt(tau - u)`` singularity, and iterate a well-conditioned form of the
equation (their FP-B). A handful of nodes gives prices accurate to 1e-6 or
better, far faster than a tree.

Formal statement (Andersen, Lake and Offengeim 2016, "High-performance
American option pricing", J. Comput. Finance 20(1), 39-87; eq. (2) premium,
eqs. (25)-(27) FP-B, section 4 the collocation). For a put with
``r > 0, q >= 0``, ``X = K min(1, r/q)``, ``H(sqrt tau) = ln(B(tau)/X)**2``
interpolated on ``n + 1`` Chebyshev-Lobatto nodes, and

``B = K exp(-(r - q) tau) N(tau, B) / D(tau, B)`` with
``N = phi(d-(tau, B/K))/(sigma sqrt tau) + r int_0^tau e^{ru} phi(d-(tau-u, B(tau)/B(u)))/(sigma sqrt(tau-u)) du``,
``D = phi(d+)/(sigma sqrt tau) + Phi(d+) + q int_0^tau e^{qu} [phi(d+)/(sigma sqrt(tau-u)) + Phi(d+)] du``.

Price: ``P = p_E + int_0^T [r K e^{-r(T-u)} Phi(-d-(T-u, S/B(u))) - q S e^{-q(T-u)} Phi(-d+(T-u, S/B(u)))] du``.
Calls use McDonald-Schroder symmetry ``C(S, K, r, q) = P(K, S, q, r)``.

Worked example: ``S = K = 100, T = 1, r = 0.05, q = 0, sigma = 0.2``, put:
6.090373 (default nodes), against 6.090371 from Leisen-Reimer at 2001 and
4001 steps with Richardson extrapolation (``2 V_4001 - V_2001``).

Scope: ``r > 0`` and ``q >= 0`` for a put (``q > 0`` and ``r >= 0`` for a
call). Other regions (negative rates, the double boundary of
``r < q < 0``) raise ``ValueError``. Where early exercise has no value (a put with
``r <= 0 <= q``, a call with ``q <= 0 <= r``) the European price is returned. Conventions are those of :mod:`eq_options.black_scholes`.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr

from .black_scholes import OptionType, _check_rates, bs_price, validate_inputs

__all__ = ["alo_boundary", "alo_price"]

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _phi(x: np.ndarray) -> np.ndarray:
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def _d(tau: np.ndarray, z: np.ndarray, r: float, q: float, sigma: float) -> tuple:
    st = sigma * np.sqrt(tau)
    base = (np.log(z) + (r - q) * tau) / st
    return base + 0.5 * st, base - 0.5 * st


class _Cheb:
    """Barycentric interpolant on Chebyshev-Lobatto nodes ``z_i = cos(i pi / n)``."""

    def __init__(self, n: int) -> None:
        self.z = np.cos(np.pi * np.arange(n + 1) / n)
        w = (-1.0) ** np.arange(n + 1)
        w[0] *= 0.5
        w[-1] *= 0.5
        self.w = w

    def __call__(self, values: np.ndarray, z: np.ndarray) -> np.ndarray:
        diff = z[:, None] - self.z[None, :]
        exact = np.isclose(diff, 0.0, atol=1e-15)
        diff[exact] = 1.0
        t = self.w / diff
        out = (t @ values) / t.sum(axis=1)
        rows, cols = np.nonzero(exact)
        out[rows] = values[cols]
        return out


def _put_boundary(K, T, r, q, sigma, n, l, iterations):
    X = K * min(1.0, r / q) if q > 0.0 else K
    cheb = _Cheb(n)
    sqrt_tmax = math.sqrt(T)
    tau = (0.5 * sqrt_tmax * (1.0 + cheb.z)) ** 2  # tau_0 = T ... tau_n = 0
    y, wy = np.polynomial.legendre.leggauss(l)

    # Start from a flat boundary slightly below X; FP-B converges from it.
    H = np.full(n + 1, 0.0)
    H[:-1] = (0.5 * sigma * np.sqrt(tau[:-1])) ** 2

    def boundary_at(u: np.ndarray, Hv: np.ndarray) -> np.ndarray:
        zz = 2.0 * np.sqrt(u) / sqrt_tmax - 1.0
        return X * np.exp(-np.sqrt(np.maximum(cheb(Hv, zz), 0.0)))

    for _ in range(iterations):
        B = X * np.exp(-np.sqrt(H))
        B_new = B.copy()
        for i in range(n):  # node n is tau = 0, B = X
            t_i, b_i = tau[i], B[i]
            # u = tau - tau (1 + y)^2 / 4: tau - u = tau (1 + y)^2/4.
            s = 0.5 * (1.0 + y)
            u = t_i * (1.0 - s * s)
            dtau = t_i * s * s
            jac = 0.5 * t_i * (1.0 + y)  # du = -tau (1 + y)/2 dy
            bu = boundary_at(u, H)
            dp, dm = _d(dtau, b_i / bu, r, q, sigma)
            # Singular parts: du / (sigma sqrt(tau - u)) = sqrt(tau)/sigma dy.
            kn = r * np.sum(wy * np.exp(r * u) * _phi(dm) * (math.sqrt(t_i) / sigma))
            kd = q * np.sum(wy * np.exp(q * u) * (
                _phi(dp) * (math.sqrt(t_i) / sigma) + ndtr(dp) * jac))
            d1, d2 = _d(np.array(t_i), np.array(b_i / K), r, q, sigma)
            st = sigma * math.sqrt(t_i)
            Nv = float(_phi(d2)) / st + kn
            Dv = float(_phi(d1)) / st + float(ndtr(d1)) + kd
            B_new[i] = K * math.exp(-(r - q) * t_i) * Nv / Dv
        B_new = np.minimum(B_new, X)
        H_new = np.log(B_new / X) ** 2
        if np.max(np.abs(H_new - H)) < 1e-14:
            H = H_new
            break
        H = H_new
    return lambda u: boundary_at(np.asarray(u, dtype=float), H)


def alo_boundary(
    K: float, T: float, r: float, sigma: float, q: float = 0.0,
    n_nodes: int = 12, n_quad: int = 32, iterations: int = 60,
):
    """Return a callable ``B(tau)``: the American put exercise boundary."""
    if not (r > 0.0 and q >= 0.0):
        raise ValueError("alo_boundary covers r > 0, q >= 0 (put)")
    return _put_boundary(K, T, r, q, sigma, n_nodes, n_quad, iterations)


def alo_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "put",
    n_nodes: int = 12,
    n_quad: int = 32,
    n_price: int = 64,
    iterations: int = 60,
) -> float:
    """American option price by the Andersen-Lake-Offengeim method.

    Parameters
    ----------
    S, K, T, sigma : float
        Strictly positive spot, strike, expiry in years and volatility.
    r, q : float
        Rate and dividend yield; see the module scope note.
    option_type : {"call", "put"}
    n_nodes, n_quad, n_price : int
        Chebyshev nodes, boundary quadrature points, price quadrature points.
    iterations : int
        Maximum FP-B iterations (stops early on convergence).

    Returns
    -------
    float
        Present value in currency units.
    """
    validate_inputs(S, K, T, sigma, option_type)
    _check_rates(r=r, q=q)
    if min(S, K, T, sigma) <= 0.0:
        raise ValueError("alo_price needs S, K, T and sigma strictly positive")
    if option_type == "call":
        if q <= 0.0 and r >= 0.0:
            return bs_price(S, K, T, r, sigma, q, "call")  # never exercised early
        return alo_price(K, S, T, q, sigma, r, "put", n_nodes, n_quad, n_price, iterations)
    if r <= 0.0 and q >= 0.0:
        return bs_price(S, K, T, r, sigma, q, "put")
    if not (r > 0.0 and q >= 0.0):
        raise ValueError("alo_price covers r > 0, q >= 0 for a put (q > 0, r >= 0 for a call)")
    B = _put_boundary(K, T, r, q, sigma, n_nodes, n_quad, iterations)
    european = bs_price(S, K, T, r, sigma, q, "put")
    if S <= float(B(np.array([T]))[0]):
        return K - S
    # Integrate over v = sqrt(u) on [0, sqrt(T)], du = 2 v dv.
    y, w = np.polynomial.legendre.leggauss(n_price)
    v = 0.5 * math.sqrt(T) * (1.0 + y)
    u = v * v
    bu = B(u)
    rem = T - u
    rem = np.maximum(rem, 1e-300)
    dp, dm = _d(rem, S / bu, r, q, sigma)
    integrand = r * K * np.exp(-r * rem) * ndtr(-dm) - q * S * np.exp(-q * rem) * ndtr(-dp)
    premium = 0.5 * math.sqrt(T) * np.sum(w * integrand * 2.0 * v)
    return float(european + premium)
