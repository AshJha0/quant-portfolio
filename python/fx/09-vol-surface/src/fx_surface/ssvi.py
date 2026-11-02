"""SSVI and eSSVI total-variance surfaces for FX smiles.

Plain idea
----------
SSVI describes a whole surface with three numbers plus the ATM term
structure, and comes with simple conditions that rule out static arbitrage.
FX adds two wrinkles: the smile is quoted in delta, and "ATM" means the
delta-neutral straddle (DNS) strike, which sits at ``k = +-sigma^2 T / 2``
rather than at the forward. So ``theta`` (the total variance at the forward)
is not the quoted ATM vol squared times T; we read it off the pillar points
at ``k = 0``.

Formal statement
----------------
SSVI (Gatheral and Jacquier 2014, "Arbitrage-free SVI volatility surfaces",
Quantitative Finance 14(1), eq. 4.1), ``k = ln(K/F_T)``,
``F_T = S exp((r_d - r_f) T)``:

    w(k, theta) = theta/2 * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1 - rho^2))

with power-law ``phi(theta) = eta / (theta^gamma (1 + theta)^(1 - gamma))``
(ibid., Remark 4.5). Each slice is raw SVI with ``a = theta(1-rho^2)/2``,
``b = theta phi/2``, ``m = -rho/phi``, ``s = sqrt(1-rho^2)/phi`` (ibid.,
s.4). No butterfly arbitrage when ``theta phi (1+|rho|) < 4`` and
``theta phi^2 (1+|rho|) <= 4`` (Theorem 4.2); the power law with
``eta (1+|rho|) <= 2`` is free of static arbitrage (Remark 4.4).

eSSVI (Hendriks and Martini 2019, "The extended SSVI volatility surface",
J. Computational Finance 22(5)) lets ``rho`` depend on ``theta``; here it is
linear in ``theta`` between two anchors, and calendar arbitrage is checked
numerically.

FX additions: :func:`fit_ssvi_from_pillars` takes pillar strikes and vols
(as solved by :func:`fx_surface.solve_pillar_strikes` under the pair's delta
convention) and estimates ``theta`` by interpolating the pillar total
variances at ``k = 0``; :func:`ssvi_strike_at_delta` returns the
smile-consistent strike for a quoted delta under any of the four
conventions, premium-adjusted included.

Equity twin: ``eq_surface.ssvi``. The surface maths is identical; the tests
check the two packages agree to machine precision.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ._smile_delta import strike_at_smile_delta
from .smile import SVIParams

__all__ = [
    "ESSVIParams",
    "SSVIFitResult",
    "SSVIParams",
    "calendar_arbitrage_free",
    "fit_ssvi",
    "fit_ssvi_from_pillars",
    "ssvi_slice_as_svi",
    "ssvi_strike_at_delta",
    "ssvi_total_variance",
]


def _phi_power(theta: np.ndarray | float, eta: float, gamma: float) -> np.ndarray | float:
    theta = np.asarray(theta, dtype=float)
    return eta / (theta**gamma * (1.0 + theta) ** (1.0 - gamma))


def ssvi_total_variance(
    k: np.ndarray | float, theta: float, rho: float, phi: float
) -> np.ndarray | float:
    """SSVI total variance for one slice (Gatheral-Jacquier 2014, eq. 4.1)."""
    k = np.asarray(k, dtype=float)
    pk = phi * k
    return 0.5 * theta * (1.0 + rho * pk + np.sqrt((pk + rho) ** 2 + 1.0 - rho * rho))


def ssvi_slice_as_svi(theta: float, rho: float, phi: float) -> SVIParams:
    """The raw SVI parameters of one SSVI slice (``fx_surface.smile.SVIParams``)."""
    return SVIParams(
        a=0.5 * theta * (1.0 - rho * rho),
        b=0.5 * theta * phi,
        rho=rho,
        m=-rho / phi,
        s=float(np.sqrt(1.0 - rho * rho) / phi),
    )


@dataclass(frozen=True)
class SSVIParams:
    """Power-law SSVI: one ``rho`` plus ``(eta, gamma)``; ``theta(T)`` given per expiry."""

    rho: float
    eta: float
    gamma: float

    def __post_init__(self) -> None:
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")
        if not self.eta > 0.0:
            raise ValueError(f"eta must be > 0, got {self.eta}")
        if not 0.0 < self.gamma <= 0.5:
            raise ValueError(f"gamma must be in (0, 0.5], got {self.gamma}")

    def phi(self, theta: np.ndarray | float) -> np.ndarray | float:
        return _phi_power(theta, self.eta, self.gamma)

    def rho_of(self, theta: np.ndarray | float) -> np.ndarray | float:
        return np.full_like(np.asarray(theta, dtype=float), self.rho)

    def total_variance(self, k: np.ndarray | float, theta: float) -> np.ndarray | float:
        return ssvi_total_variance(k, theta, self.rho, float(self.phi(theta)))

    def butterfly_free(self, theta: float) -> bool:
        """Sufficient condition of Gatheral-Jacquier Theorem 4.2."""
        ph = float(self.phi(theta))
        c = 1.0 + abs(self.rho)
        return theta * ph * c < 4.0 and theta * ph * ph * c <= 4.0

    def static_arbitrage_free(self) -> bool:
        """Power-law sufficient condition ``eta (1 + |rho|) <= 2`` (Remark 4.4)."""
        return self.eta * (1.0 + abs(self.rho)) <= 2.0


@dataclass(frozen=True)
class ESSVIParams:
    """eSSVI: ``rho(theta)`` linear between ``(theta_lo, rho_lo)`` and ``(theta_hi, rho_hi)``.

    ``rho`` is held flat outside the anchors.
    """

    rho_lo: float
    rho_hi: float
    theta_lo: float
    theta_hi: float
    eta: float
    gamma: float

    def __post_init__(self) -> None:
        for r in (self.rho_lo, self.rho_hi):
            if not -1.0 < r < 1.0:
                raise ValueError(f"rho anchors must be in (-1, 1), got {r}")
        if not 0.0 < self.theta_lo < self.theta_hi:
            raise ValueError("need 0 < theta_lo < theta_hi")
        if not self.eta > 0.0 or not 0.0 < self.gamma <= 0.5:
            raise ValueError("need eta > 0 and gamma in (0, 0.5]")

    def phi(self, theta: np.ndarray | float) -> np.ndarray | float:
        return _phi_power(theta, self.eta, self.gamma)

    def rho_of(self, theta: np.ndarray | float) -> np.ndarray | float:
        return np.interp(theta, [self.theta_lo, self.theta_hi], [self.rho_lo, self.rho_hi])

    def total_variance(self, k: np.ndarray | float, theta: float) -> np.ndarray | float:
        return ssvi_total_variance(k, theta, float(self.rho_of(theta)), float(self.phi(theta)))

    def butterfly_free(self, theta: float) -> bool:
        ph = float(self.phi(theta))
        c = 1.0 + abs(float(self.rho_of(theta)))
        return theta * ph * c < 4.0 and theta * ph * ph * c <= 4.0


def calendar_arbitrage_free(
    params: SSVIParams | ESSVIParams,
    thetas: np.ndarray,
    k_grid: np.ndarray | None = None,
    tol: float = 1e-12,
) -> bool:
    """True when ``w(k, theta_i)`` is non-decreasing in ``i`` on ``k_grid``."""
    thetas = np.asarray(thetas, dtype=float)
    if np.any(np.diff(thetas) < 0.0):
        return False
    k = np.linspace(-3.0, 3.0, 241) if k_grid is None else np.asarray(k_grid, dtype=float)
    w = np.array([params.total_variance(k, th) for th in thetas])
    return bool(np.all(np.diff(w, axis=0) >= -tol))


@dataclass(frozen=True)
class SSVIFitResult:
    params: SSVIParams | ESSVIParams
    thetas: np.ndarray
    rmse_total_variance: float


def fit_ssvi(
    k: Sequence[np.ndarray],
    w_market: Sequence[np.ndarray],
    thetas: np.ndarray,
    extended: bool = False,
) -> SSVIFitResult:
    """Fit SSVI (or eSSVI) to market total variances with ``theta`` per expiry fixed.

    Least squares on total variance with ``eta (1+|rho|) <= 2`` imposed by a
    penalty, so an SSVI result is free of static arbitrage.
    """
    thetas = np.asarray(thetas, dtype=float)
    if len(k) != len(w_market) or len(k) != thetas.size:
        raise ValueError("k, w_market and thetas must have one entry per expiry")
    if thetas.size == 0 or np.any(~np.isfinite(thetas)) or np.any(thetas <= 0.0):
        raise ValueError("thetas must be finite and positive")

    def build(x: np.ndarray) -> SSVIParams | ESSVIParams:
        if extended:
            return ESSVIParams(np.tanh(x[0]), np.tanh(x[1]), float(thetas.min()),
                               float(thetas.max()) + 1e-12, np.exp(x[2]),
                               0.5 / (1.0 + np.exp(-x[3])))
        return SSVIParams(np.tanh(x[0]), np.exp(x[1]), 0.5 / (1.0 + np.exp(-x[2])))

    def resid(x: np.ndarray) -> np.ndarray:
        p = build(x)
        out = [p.total_variance(ki, th) - wi
               for ki, wi, th in zip(k, w_market, thetas, strict=True)]
        rho_max = max(abs(float(p.rho_of(thetas.min()))), abs(float(p.rho_of(thetas.max()))))
        pen = max(0.0, p.eta * (1.0 + rho_max) - 2.0)
        return np.concatenate([*out, np.array([10.0 * pen])])

    x0 = np.array([-0.3, -0.3, 0.0, 0.0]) if extended else np.array([-0.3, 0.0, 0.0])
    sol = least_squares(resid, x0, method="trf")
    p = build(sol.x)
    r = np.concatenate([p.total_variance(ki, th) - wi
                        for ki, wi, th in zip(k, w_market, thetas, strict=True)])
    return SSVIFitResult(p, thetas, float(np.sqrt(np.mean(r * r))))


def fit_ssvi_from_pillars(
    strikes: Sequence[Sequence[float]],
    vols: Sequence[Sequence[float]],
    forwards: Sequence[float],
    expiries: Sequence[float],
    extended: bool = False,
) -> SSVIFitResult:
    """Fit SSVI to FX pillar quotes given as solved strikes and vols per expiry.

    ``theta_i`` is the pillar total variance linearly interpolated at
    ``k = 0`` (the forward), because the quoted DNS ATM strike is not the
    forward. Each expiry needs at least three pillars bracketing the forward.
    """
    if not (len(strikes) == len(vols) == len(forwards) == len(expiries)):
        raise ValueError("strikes, vols, forwards and expiries need one entry per expiry")
    ks, ws, thetas = [], [], []
    for K_in, v_in, F, T in zip(strikes, vols, forwards, expiries, strict=True):
        K = np.asarray(K_in, dtype=float)
        v = np.asarray(v_in, dtype=float)
        if K.shape != v.shape or K.size < 3:
            raise ValueError("each expiry needs at least three (strike, vol) pillars")
        if not (F > 0.0 and T > 0.0) or np.any(~(K > 0.0)) or np.any(~(v > 0.0)):
            raise ValueError("forwards, expiries, strikes and vols must be positive")
        k = np.log(K / F)
        order = np.argsort(k)
        k, w = k[order], (v[order] ** 2) * T
        if not k[0] < 0.0 < k[-1]:
            raise ValueError("pillars must bracket the forward (k = 0)")
        ks.append(k)
        ws.append(w)
        thetas.append(float(np.interp(0.0, k, w)))
    return fit_ssvi(ks, ws, np.array(thetas), extended=extended)


def ssvi_strike_at_delta(
    params: SSVIParams | ESSVIParams,
    theta: float,
    delta: float,
    cp: int,
    S: float,
    T: float,
    r_d: float,
    r_f: float,
    convention: str = "spot",
) -> tuple[float, float]:
    """Smile-consistent ``(strike, vol)`` of a quoted delta on one SSVI slice."""
    if not T > 0.0:
        raise ValueError("T must be positive")

    def vol_of_k(k: float) -> float:
        return float(np.sqrt(params.total_variance(k, theta) / T))

    K = strike_at_smile_delta(vol_of_k, delta, cp, S, T, r_d, r_f, convention)
    F = S * np.exp((r_d - r_f) * T)
    return K, vol_of_k(float(np.log(K / F)))
