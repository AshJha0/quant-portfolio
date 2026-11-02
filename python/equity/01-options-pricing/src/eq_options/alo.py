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

Negative rates and the double boundary. Exercising a put early swaps the
option for ``K - S``; while held in the exercise region, that position
earns carry at rate ``r K - q S``. The sign of that carry sorts every
``(r, q)`` pair into one of three regions (Battauz, De Donno and Sbuelz
2015, "Real options and American derivatives: the double continuation
region", Management Science 61(5), 1094-1107; Andersen and Lake 2021,
"Fast American option pricing: the double-boundary case", Wilmott
2021(116), 30-41):

* single boundary, ``r > 0`` (any ``q``) or ``r = 0 > q``: FP-B as above,
  with ``X = K`` when ``q <= 0``. The ALO derivation does not use the sign
  of ``q``; only ``X`` changes.
* no early exercise, ``r <= 0`` and ``q >= r`` (other than ``r = 0 > q``):
  the carry is never positive in the money, so the European price.
* double boundary, ``q < r < 0``: early exercise pays only inside a band
  ``Y(tau) <= S <= B(tau)`` that starts at ``[K r/q, K]`` at expiry and
  closes at a finite ``tau*``; for longer maturities no new exercise
  region appears. The value is
  ``P = p_E + int_0^{min(T, tau*)} [r K e^{-r(T-u)} (Phi(-d-(S/B(u))) - Phi(-d-(S/Y(u))))
  - q S e^{-q(T-u)} (Phi(-d+(S/B(u))) - Phi(-d+(S/Y(u))))] du``,
  and the two boundaries are pinned by value matching,
  ``K - B(tau) = P(tau, B(tau))`` and ``K - Y(tau) = P(tau, Y(tau))``.
  We solve that pair by marching in ``tau`` on a grid uniform in
  ``sqrt(tau)`` (trapezoid in ``u``, a damped 2x2 Newton per node), and
  stop where the band closes or the monotonicity (``B`` falling, ``Y``
  rising) fails. This is the integral-equation system of the
  double-boundary paper solved by direct marching rather than by its
  Chebyshev collocation, so it is slower (about a second at the default
  ``n_march = 400``) and accurate to about 1e-5 per 100 of strike against
  an extrapolated Leisen-Reimer tree, converging at close to second order.

Worked example (double boundary): ``S = K = 100, T = 1, r = -0.005,
q = -0.02, sigma = 0.2``, put: 7.410691 at the default grid, against
7.410683 from the extrapolated tree; the European put is 7.3291.

Calls use the same symmetry, so a call with ``r < q < 0`` is the
double-boundary put with the roles of ``r`` and ``q`` swapped.
Conventions are those of :mod:`eq_options.black_scholes`.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ndtr

from .black_scholes import OptionType, _check_rates, bs_price, validate_inputs

__all__ = ["alo_boundary", "alo_double_boundary", "alo_price"]

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


def _double_boundary(K, T, r, q, sigma, n):
    """March the band ``[Y(tau), B(tau)]`` for a put with ``q < r < 0``.

    Returns the node times ``u`` (from 0 up to the last tau at which the
    band still exists, capped at ``T``) and the boundary values there.
    """
    u = T * (np.arange(n + 1) / n) ** 2
    B = np.empty(n + 1)
    Y = np.empty(n + 1)
    B[0], Y[0] = K, K * r / q
    last = 0
    for i in range(1, n + 1):
        t = u[i]
        dt = t - u[:i]
        steps = np.diff(u[: i + 1])
        wk = np.empty(i)
        wk[0] = 0.5 * steps[0]
        wk[1:] = 0.5 * (steps[:-1] + steps[1:])
        wi = 0.5 * steps[-1]
        sq = sigma * np.sqrt(dt)
        drift = (r - q) * dt
        er, eq_ = np.exp(-r * dt), np.exp(-q * dt)
        lb, ly = np.log(B[:i]), np.log(Y[:i])
        st = sigma * math.sqrt(t)

        def resid(x, t=t, sq=sq, drift=drift, er=er, eq_=eq_, lb=lb, ly=ly, wk=wk, wi=wi, st=st):
            lx = np.log(x)[:, None]
            dmb = (lx - lb + drift) / sq - 0.5 * sq
            dmy = (lx - ly + drift) / sq - 0.5 * sq
            p_band = ndtr(-dmb) - ndtr(-dmy)
            q_band = ndtr(-(dmb + sq)) - ndtr(-(dmy + sq))
            # The u = tau node: at S = B (or S = Y) the band probability is 1/2.
            integral = (r * K * er * p_band - q * x[:, None] * eq_ * q_band) @ wk \
                + wi * 0.5 * (r * K - q * x)
            d1 = (np.log(x / K) + (r - q + 0.5 * sigma * sigma) * t) / st
            euro = K * math.exp(-r * t) * ndtr(-(d1 - st)) - x * math.exp(-q * t) * ndtr(-d1)
            return K - x - euro - integral

        prev = np.array([B[i - 1], Y[i - 1]])
        x = prev if i < 2 else 2.0 * prev - np.array([B[i - 2], Y[i - 2]])
        x = np.array([min(x[0], B[i - 1]), max(x[1], Y[i - 1])])
        converged = False
        for _ in range(50):
            f = resid(x)
            fnorm = float(np.max(np.abs(f)))
            if fnorm < 1e-11 * K:
                converged = True
                break
            jac = np.empty((2, 2))
            for j in range(2):
                xp = x.copy()
                xp[j] += 1e-7 * x[j]
                jac[:, j] = (resid(xp) - f) / (1e-7 * x[j])
            try:
                dx = np.linalg.solve(jac, -f)
            except np.linalg.LinAlgError:
                break
            lam, step_ok = 1.0, False
            while lam > 1e-4:
                xn = x + lam * dx
                if xn[1] > 0.0 and xn[0] > xn[1] and np.max(np.abs(resid(xn))) < fnorm:
                    step_ok = True
                    break
                lam *= 0.5
            if not step_ok:
                break
            x = xn
        if (not converged or x[0] - x[1] < 1e-9 * K
                or x[0] > B[i - 1] * (1.0 + 1e-12) or x[1] < Y[i - 1] * (1.0 - 1e-12)):
            break  # the band has closed: tau* lies in (u[last], u[i])
        B[i], Y[i] = x
        last = i
    return u[: last + 1], B[: last + 1], Y[: last + 1]


def _double_boundary_put(S, K, T, r, q, sigma, n):
    u, B, Y = _double_boundary(K, T, r, q, sigma, n)
    if u[-1] >= T and Y[-1] <= S <= B[-1]:
        return K - S
    dt = T - u
    w = np.zeros(u.size)
    steps = np.diff(u)
    w[:-1] += 0.5 * steps
    w[1:] += 0.5 * steps
    sq = sigma * np.sqrt(np.maximum(dt, 1e-300))
    drift = (r - q) * dt
    dmb = (math.log(S) - np.log(B) + drift) / sq - 0.5 * sq
    dmy = (math.log(S) - np.log(Y) + drift) / sq - 0.5 * sq
    p_band = ndtr(-dmb) - ndtr(-dmy)
    q_band = ndtr(-(dmb + sq)) - ndtr(-(dmy + sq))
    premium = np.sum(w * (r * K * np.exp(-r * dt) * p_band - q * S * np.exp(-q * dt) * q_band))
    return float(bs_price(S, K, T, r, sigma, q, "put") + premium)


def alo_double_boundary(K: float, T: float, r: float, sigma: float, q: float,
                        n_march: int = 400):
    """Return ``(tau, lower, upper)``: the put exercise band for ``q < r < 0``.

    ``tau`` runs from 0 to ``T`` or to the last grid time before the band
    closes, whichever comes first (see the module note).
    """
    if not q < r < 0.0:
        raise ValueError("alo_double_boundary covers q < r < 0 (put)")
    if min(K, T, sigma) <= 0.0 or n_march < 2:
        raise ValueError("need K, T, sigma > 0 and n_march >= 2")
    u, B, Y = _double_boundary(K, T, r, q, sigma, n_march)
    return u, Y, B


def alo_boundary(
    K: float, T: float, r: float, sigma: float, q: float = 0.0,
    n_nodes: int = 12, n_quad: int = 32, iterations: int = 60,
):
    """Return a callable ``B(tau)``: the American put exercise boundary."""
    if not (r > 0.0 or (r == 0.0 and q < 0.0)):
        raise ValueError("alo_boundary covers the single-boundary put region r > 0 or r = 0 > q")
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
    n_march: int = 400,
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
    n_march : int
        Time steps for the double-boundary march (``q < r < 0`` put,
        ``r < q < 0`` call); ignored elsewhere.

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
        return alo_price(K, S, T, q, sigma, r, "put", n_nodes, n_quad, n_price, iterations,
                         n_march)
    if r <= 0.0 and q >= r and not (r == 0.0 and q < 0.0):
        return bs_price(S, K, T, r, sigma, q, "put")  # carry r K - q S never positive in the money
    if r < 0.0:  # q < r < 0: the double boundary
        if n_march < 2:
            raise ValueError("n_march must be >= 2")
        return _double_boundary_put(S, K, T, r, q, sigma, n_march)
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
