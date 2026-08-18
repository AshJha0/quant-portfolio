"""Heston stochastic-volatility model: characteristic function and Fourier pricing.

Model (risk-neutral)
--------------------
    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW^S_t
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW^v_t,   d<W^S, W^v> = rho dt

Parameters: ``v0`` (initial variance), ``kappa`` (mean-reversion speed),
``theta`` (long-run variance), ``rho`` (spot-vol correlation), ``xi``
(vol-of-vol).  All variances are annualised; rates continuously compounded,
ACT/365F.

The 'little Heston trap' characteristic function
------------------------------------------------
The original Heston (1993) formulation writes the log-price characteristic
function with ``g1 = (b + d)/(b - d)`` and the term ``ln((1 - g1 e^{d T})/(1 - g1))``.
Because ``e^{dT}`` grows without bound, the complex logarithm's argument
spirals across the negative real axis as ``u`` or ``T`` grows, and the
principal-branch log then jumps by ``2*pi*i`` -- producing discontinuous,
wrong prices unless the branch is tracked manually.  Albrecher et al. (2007,
"The little Heston trap") showed that the algebraically equivalent form using
the *decaying* exponential,

    g2 = (b - d)/(b + d),
    C  = (kappa*theta/xi^2) * ( (b - d) T - 2 ln((1 - g2 e^{-dT})/(1 - g2)) ),
    D  = ((b - d)/xi^2) * (1 - e^{-dT}) / (1 - g2 e^{-dT}),

with ``b = kappa - i rho xi u`` and ``d = sqrt(b^2 + xi^2 (i u + u^2))``
(principal root), keeps the log argument in the right half-plane for all
practically relevant parameters: ``e^{-dT} -> 0``, so the argument never
winds around the origin and the principal branch is continuous in ``u``.
This is the formulation implemented here (as in Gatheral, *The Volatility
Surface*).

Two independent pricing routes (cross-validated in tests)
---------------------------------------------------------
1. ``heston_call_p1p2`` -- Heston's semi-analytic form
   ``C = S e^{-qT} P1 - K e^{-rT} P2`` with each probability computed by
   adaptive quadrature (``scipy.integrate.quad``) of
   ``1/2 + (1/pi) Int_0^inf Re[e^{-iu ln K} f_j(u) / (iu)] du``.
2. ``heston_call_damped`` -- Carr-Madan style damped direct integration:
   multiply the call by ``e^{alpha k}`` so its Fourier transform exists,
   integrate the damped transform by adaptive quadrature.  (Direct
   integration rather than FFT: we need prices at arbitrary strikes, not a
   log-strike grid, and adaptive quadrature is more accurate per evaluation.)
3. ``heston_call_gl`` -- same damped integrand on a fixed Gauss-Legendre rule,
   vectorised over strikes (the characteristic function is strike-independent,
   so one CF evaluation per node prices a whole strike column).  This is the
   fast path used by calibration, Greeks and MC benchmarks; it is validated
   against routes 1 and 2 to 1e-6 in the test suite.

Feller condition
----------------
``2 kappa theta >= xi^2`` keeps the variance process strictly positive.
Market calibrations of equity index smiles *routinely violate* Feller: short-
dated skew demands high ``xi`` and low ``kappa*theta``, so the fitted process
touches zero variance.  This is not a model error -- the CIR process remains
well-defined (zero is attainable but instantaneously reflecting) and the
characteristic function stays valid.  We therefore *warn* rather than raise,
and the Monte Carlo schemes are chosen to handle v = 0 (full truncation, QE).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import quad

__all__ = [
    "HestonParams",
    "FellerWarning",
    "feller_condition",
    "heston_cf",
    "heston_call_p1p2",
    "heston_call_damped",
    "heston_call_gl",
    "heston_call",
    "heston_put",
]


class FellerWarning(UserWarning):
    """Warns when 2*kappa*theta < xi^2 (variance can touch zero)."""


@dataclass(frozen=True)
class HestonParams:
    """Heston model parameters (annualised variance units).

    Attributes
    ----------
    v0 : float
        Initial instantaneous variance (> 0).
    kappa : float
        Mean-reversion speed of variance (> 0).
    theta : float
        Long-run variance level (> 0).
    rho : float
        Spot-variance correlation, in [-1, 1] (boundaries allowed).
    xi : float
        Volatility of variance (>= 0; 0 gives deterministic variance).
    """

    v0: float
    kappa: float
    theta: float
    rho: float
    xi: float

    def __post_init__(self) -> None:
        if self.v0 <= 0.0:
            raise ValueError(f"v0 must be positive, got {self.v0}")
        if self.kappa <= 0.0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.theta <= 0.0:
            raise ValueError(f"theta must be positive, got {self.theta}")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
        if self.xi < 0.0:
            raise ValueError(f"xi must be non-negative, got {self.xi}")

    def as_array(self) -> np.ndarray:
        """Return ``[v0, kappa, theta, rho, xi]``."""
        return np.array([self.v0, self.kappa, self.theta, self.rho, self.xi])


def feller_condition(p: HestonParams, warn: bool = True) -> float:
    """Feller ratio ``2*kappa*theta / xi^2`` (>= 1 means condition holds).

    Emits :class:`FellerWarning` (does not raise) when violated -- see module
    docstring for why market calibrations often live below 1.

    Returns
    -------
    float
        The Feller ratio (``inf`` when xi == 0).
    """
    if p.xi == 0.0:
        return np.inf
    ratio = 2.0 * p.kappa * p.theta / (p.xi**2)
    if ratio < 1.0 and warn:
        warnings.warn(
            f"Feller condition violated: 2*kappa*theta/xi^2 = {ratio:.3f} < 1; "
            "variance process can touch zero (common in equity calibrations; "
            "pricing remains valid, MC schemes handle v=0)",
            FellerWarning,
        )
    return float(ratio)


def heston_cf(
    u: np.ndarray | complex,
    T: float,
    S: float,
    r: float,
    q: float,
    p: HestonParams,
) -> np.ndarray | complex:
    """Characteristic function of ``ln S_T`` in the little-trap formulation.

    ``phi(u) = E[exp(i u ln S_T)]`` under the risk-neutral measure.  Accepts
    real or complex ``u`` (complex needed for damped pricing), scalar or array.

    For ``xi = 0`` the variance path is deterministic,
    ``v(t) = theta + (v0 - theta) e^{-kappa t}``, and the CF degenerates to
    the Gaussian CF with integrated variance
    ``IV = theta T + (v0 - theta)(1 - e^{-kappa T})/kappa``.
    """
    if T < 0.0:
        raise ValueError(f"T must be non-negative, got {T}")
    u = np.asarray(u, dtype=complex)
    x0 = np.log(S) + (r - q) * T  # log-forward

    if p.xi < 1e-12:
        iv = p.theta * T + (p.v0 - p.theta) * (1.0 - np.exp(-p.kappa * T)) / p.kappa
        out = np.exp(1j * u * (x0 - 0.5 * iv) - 0.5 * u * u * iv)
        return out if out.ndim else complex(out)

    xi2 = p.xi * p.xi
    b = p.kappa - 1j * p.rho * p.xi * u
    d = np.sqrt(b * b + xi2 * (1j * u + u * u))
    g2 = (b - d) / (b + d)
    e_dT = np.exp(-d * T)
    C = (p.kappa * p.theta / xi2) * ((b - d) * T - 2.0 * np.log((1.0 - g2 * e_dT) / (1.0 - g2)))
    D = ((b - d) / xi2) * (1.0 - e_dT) / (1.0 - g2 * e_dT)
    out = np.exp(1j * u * x0 + C + D * p.v0)
    return out if out.ndim else complex(out)


def _u_max(
    T: float,
    p: HestonParams,
    S: float = 100.0,
    r: float = 0.0,
    q: float = 0.0,
    alpha: float = 1.5,
) -> float:
    """Integration cutoff chosen by probing the CF envelope.

    For large ``u`` the Heston CF decays like
    ``exp(-u (v0 + kappa*theta*T) sqrt(1-rho^2) / xi)`` -- *slowly* when
    vol-of-vol is high, so a fixed cutoff silently drops tail mass.  Instead
    we probe ``max(|phi(u)|, |phi(u-i)|, |phi(u-(alpha+1)i)|) / (1 + u^2)``
    (the envelope common to the P1/P2 and damped integrands) at geometrically
    growing ``u`` until it falls below 1e-15, capped at 40000.
    """
    F = S * np.exp((r - q) * T)
    u = 100.0
    while u < 40000.0:
        m = max(
            abs(heston_cf(u, T, S, r, q, p)),
            abs(heston_cf(u - 1j, T, S, r, q, p)) / F,
            abs(heston_cf(u - 1j * (alpha + 1.0), T, S, r, q, p)) / F ** (alpha + 1.0),
        ) / (1.0 + u * u)
        if m < 1e-15:
            break
        u *= 1.4
    return float(min(u, 40000.0))


def heston_call_p1p2(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    p: HestonParams,
    check_feller: bool = False,
) -> float:
    """Heston European call via the semi-analytic P1/P2 probabilities.

    ``C = S e^{-qT} P1 - K e^{-rT} P2`` with
    ``P_j = 1/2 + (1/pi) Int_0^inf Re[e^{-iu ln K} f_j(u)/(iu)] du``,
    ``f2(u) = phi(u)``, ``f1(u) = phi(u - i)/phi(-i)``.  Each integral is
    evaluated with adaptive quadrature.

    Returns the *undamped, adaptive-quadrature reference price*; slower than
    :func:`heston_call_gl` but with rigorous error control.
    """
    _validate_option(S, K, T)
    if check_feller:
        feller_condition(p)
    if T == 0.0:
        return max(S - K, 0.0)

    lnK = np.log(K)
    phi_mi = heston_cf(-1j, T, S, r, q, p)  # = F(T) = S e^{(r-q)T}

    def integrand_p1(u: float) -> float:
        f1 = heston_cf(u - 1j, T, S, r, q, p) / phi_mi
        return float(np.real(np.exp(-1j * u * lnK) * f1 / (1j * u)))

    def integrand_p2(u: float) -> float:
        f2 = heston_cf(u, T, S, r, q, p)
        return float(np.real(np.exp(-1j * u * lnK) * f2 / (1j * u)))

    umax = _u_max(T, p, S, r, q)
    i1, _ = quad(integrand_p1, 1e-10, umax, limit=400)
    i2, _ = quad(integrand_p2, 1e-10, umax, limit=400)
    P1 = 0.5 + i1 / np.pi
    P2 = 0.5 + i2 / np.pi
    price = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    return float(max(price, 0.0))


def _damped_integrand_factory(S, K, T, r, q, p, alpha):
    lnK = np.log(K)
    df = np.exp(-r * T)

    def integrand(u: float) -> float:
        z = u - 1j * (alpha + 1.0)
        num = heston_cf(z, T, S, r, q, p)
        den = alpha * alpha + alpha - u * u + 1j * (2.0 * alpha + 1.0) * u
        return float(np.real(np.exp(-1j * u * lnK) * df * num / den))

    return integrand


def heston_call_damped(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    p: HestonParams,
    alpha: float = 1.5,
) -> float:
    """Heston European call via Carr-Madan damped direct integration.

    The call price is damped by ``e^{alpha ln K}`` (``alpha > 0``) so its
    Fourier transform in log-strike exists; the damped transform is

        psi(u) = e^{-rT} phi(u - (alpha+1) i) /
                 (alpha^2 + alpha - u^2 + i (2 alpha + 1) u)

    and ``C(K) = (e^{-alpha ln K}/pi) Int_0^inf Re[e^{-iu ln K} psi(u)] du``,
    evaluated by adaptive quadrature.  Independent of the P1/P2 route (single
    integral, damped integrand, no 1/(iu) singularity) -- used for
    cross-validation.
    """
    _validate_option(S, K, T)
    if alpha <= 0.0:
        raise ValueError(f"damping alpha must be positive, got {alpha}")
    if T == 0.0:
        return max(S - K, 0.0)
    integrand = _damped_integrand_factory(S, K, T, r, q, p, alpha)
    umax = _u_max(T, p, S, r, q, alpha)
    integral, _ = quad(integrand, 0.0, umax, limit=400)
    price = np.exp(-alpha * np.log(K)) / np.pi * integral
    return float(max(price, 0.0))


# Cache Gauss-Legendre nodes by size (they are expensive to regenerate).
_GL_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gl_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n not in _GL_CACHE:
        x, w = np.polynomial.legendre.leggauss(n)
        _GL_CACHE[n] = (x, w)
    return _GL_CACHE[n]


def heston_call_gl(
    S: float,
    K: np.ndarray | float,
    T: float,
    r: float,
    q: float,
    p: HestonParams,
    alpha: float = 1.5,
    n_nodes: int = 384,
) -> np.ndarray | float:
    """Fast damped-integral Heston call, Gauss-Legendre, vectorised in strike.

    Same integrand as :func:`heston_call_damped` but on a fixed Gauss-Legendre
    rule.  The characteristic function does not depend on the strike, so it is
    evaluated once per node and reused across the whole strike vector -- this
    is the fast path for calibration and Greeks.  Accuracy vs the adaptive
    reference is validated to 1e-6 in the test suite.

    Parameters
    ----------
    K : float or array
        Strike(s).
    n_nodes : int
        Minimum Gauss-Legendre node count (default 384).  The integration
        domain is parameter-adaptive (CF-envelope probe) and the node count
        is scaled up with the domain length so oscillations stay resolved.
    """
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))
    _validate_option(S, float(K_arr.min()), T)
    if T == 0.0:
        out = np.maximum(S - K_arr, 0.0)
        return out if np.ndim(K) else float(out[0])

    umax = _u_max(T, p, S, r, q, alpha)
    # >= ~1 node per unit of u keeps e^{-iuk} resolved for |k| <~ 2;
    # quantised to multiples of 128 so the node cache stays small.
    n_nodes = max(n_nodes, 128 * int(np.ceil(umax / 128.0)))
    x, wts = _gl_nodes(n_nodes)
    u = 0.5 * umax * (x + 1.0)
    w_scaled = 0.5 * umax * wts

    z = u - 1j * (alpha + 1.0)
    phi_z = heston_cf(z, T, S, r, q, p)
    den = alpha * alpha + alpha - u * u + 1j * (2.0 * alpha + 1.0) * u
    psi = np.exp(-r * T) * phi_z / den  # (n_nodes,)

    lnK = np.log(K_arr)  # (nK,)
    kernel = np.exp(-1j * np.outer(lnK, u))  # (nK, n_nodes)
    integral = kernel @ (psi * w_scaled)
    prices = np.exp(-alpha * lnK) / np.pi * np.real(integral)
    prices = np.maximum(prices, 0.0)
    return prices if np.ndim(K) else float(prices[0])


def heston_call(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    p: HestonParams,
    method: Literal["p1p2", "damped", "gl"] = "gl",
) -> float:
    """Heston European call price; dispatches to the chosen Fourier route."""
    if method == "p1p2":
        return heston_call_p1p2(S, K, T, r, q, p)
    if method == "damped":
        return heston_call_damped(S, K, T, r, q, p)
    if method == "gl":
        return float(heston_call_gl(S, K, T, r, q, p))
    raise ValueError(f"unknown method {method!r}; use 'p1p2', 'damped' or 'gl'")


def heston_put(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    p: HestonParams,
    method: Literal["p1p2", "damped", "gl"] = "gl",
) -> float:
    """Heston European put via put-call parity.

    ``P = C - S e^{-qT} + K e^{-rT}`` -- parity holds exactly under any
    arbitrage-free model, so the put inherits the call's accuracy.
    """
    call = heston_call(S, K, T, r, q, p, method=method)
    put = call - S * np.exp(-q * T) + K * np.exp(-r * T)
    return float(max(put, 0.0))


def _validate_option(S: float, K: float, T: float) -> None:
    if S <= 0.0:
        raise ValueError(f"spot must be positive, got S={S}")
    if K <= 0.0:
        raise ValueError(f"strike must be positive, got K={K}")
    if T < 0.0:
        raise ValueError(f"time to expiry must be non-negative, got T={T}")
