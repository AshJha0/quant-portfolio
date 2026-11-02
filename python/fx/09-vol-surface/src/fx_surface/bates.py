"""Bates (1996) stochastic volatility with lognormal jumps, for an FX pair.

Plain idea
----------
Heston explains the smile's skew and term structure with a random variance;
it struggles with the steep short-dated wings that come from sudden moves
(devaluations, central-bank surprises). Bates adds Poisson jumps in the log
spot. The model was built for currencies: Bates fitted it to Deutsche Mark
options.

Formal statement
----------------
Bates, D. (1996), "Jumps and stochastic volatility: exchange rate processes
implicit in Deutsche Mark options", Review of Financial Studies 9(1),
69-107. Under the domestic risk-neutral measure the spot drifts at
``r_d - r_f - lam*kbar``; the CF of ``X = ln(S_T/S) - (r_d - r_f)T`` is the
Heston CF times an independent compound-Poisson factor (Gatheral 2006, eq.
5.6):

    phi_X(u) = phi_Heston,X(u) * exp(lam T (exp(i u mu_j - delta^2 u^2/2) - 1)
                                     - i u lam kbar T),
    kbar = exp(mu_j + delta^2/2) - 1.

The call price (domestic currency per unit of foreign notional) is the
Lewis (2001) single integral, "A simple option formula for general
jump-diffusion and other exponential Levy processes", eq. 2.10, with
``q -> r_f``:

    C = S e^{-r_f T} - sqrt(S K) e^{-(r_d + r_f)T/2} / pi
        * int_0^inf Re[e^{i u x} phi_X(u - i/2)] / (u^2 + 1/4) du,
    x = ln(S/K) + (r_d - r_f)T.

Oracles in the tests: ``lam = 0`` gives ``fx_surface.heston_price``;
``xi -> 0`` (handled exactly, see :func:`bates_cf_x`) with ``v0 = theta`` gives Merton (1976) jump-diffusion priced as a
Poisson series of Garman-Kohlhagen prices; the equity twin
``eq_surface.bates`` agrees with ``q = r_f``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from .garman_kohlhagen import gk_price
from .heston import HestonParams, heston_cf

#: Below this vol-of-vol the Heston factor is replaced by its xi -> 0 limit.
_XI_DETERMINISTIC = 1e-6

__all__ = ["JumpParams", "bates_call", "bates_cf_x", "bates_put", "merton_call"]


@dataclass(frozen=True)
class JumpParams:
    """Poisson intensity ``lam`` (per year), mean ``mu_j`` and sd ``delta`` of the log-jump."""

    lam: float
    mu_j: float
    delta: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.lam) and self.lam >= 0.0):
            raise ValueError(f"lam must be finite and >= 0, got {self.lam}")
        if not (math.isfinite(self.delta) and self.delta >= 0.0):
            raise ValueError(f"delta must be finite and >= 0, got {self.delta}")
        if not math.isfinite(self.mu_j):
            raise ValueError(f"mu_j must be finite, got {self.mu_j}")

    @property
    def kbar(self) -> float:
        """Mean relative jump size ``E[e^J] - 1``."""
        return float(np.exp(self.mu_j + 0.5 * self.delta**2) - 1.0)


def bates_cf_x(u: np.ndarray | complex, T: float, p: HestonParams, j: JumpParams):
    """CF of ``X = ln(S_T/S) - (r_d - r_f)T`` (independent of spot and rates)."""
    u = np.asarray(u, dtype=complex)
    if p.xi < _XI_DETERMINISTIC:
        # Heston's little-trap CF divides by xi^2 and loses every digit as
        # xi -> 0. In that limit variance is deterministic,
        # v_t = theta + (v0 - theta) e^{-kappa t}, so ln S_T is Gaussian with
        # integrated variance V (Gatheral 2006, s.2.2); use that exactly.
        V = p.theta * T + (p.v0 - p.theta) * (1.0 - np.exp(-p.kappa * T)) / p.kappa
        heston = np.exp(-0.5 * V * (1j * u + u * u))
    else:
        heston = heston_cf(u, T, p, mu=0.0, x0=0.0)
    jump = np.exp(j.lam * T * (np.exp(1j * u * j.mu_j - 0.5 * j.delta**2 * u * u) - 1.0)
                  - 1j * u * j.lam * j.kbar * T)
    return heston * jump


def _check(S: float, K: float, T: float, r_d: float, r_f: float) -> None:
    if not all(math.isfinite(v) for v in (S, K, T, r_d, r_f)):
        raise ValueError("S, K, T, r_d and r_f must be finite")
    if S <= 0.0 or K <= 0.0 or T <= 0.0:
        raise ValueError("S, K and T must be > 0")


def bates_call(S: float, K: float, T: float, r_d: float, r_f: float,
               p: HestonParams, j: JumpParams) -> float:
    """Bates FX call on the base currency, by the Lewis (2001) integral."""
    _check(S, K, T, r_d, r_f)
    x = np.log(S / K) + (r_d - r_f) * T

    def integrand(u: float) -> float:
        return float(np.real(np.exp(1j * u * x) * bates_cf_x(u - 0.5j, T, p, j))
                     / (u * u + 0.25))

    integral, _ = quad(integrand, 0.0, np.inf, limit=500, epsabs=1e-12, epsrel=1e-10)
    price = (S * np.exp(-r_f * T)
             - np.sqrt(S * K) * np.exp(-0.5 * (r_d + r_f) * T) / np.pi * integral)
    intrinsic = max(S * np.exp(-r_f * T) - K * np.exp(-r_d * T), 0.0)
    return float(max(price, intrinsic))


def bates_put(S: float, K: float, T: float, r_d: float, r_f: float,
              p: HestonParams, j: JumpParams) -> float:
    """Put by FX put-call parity ``C - P = S e^{-r_f T} - K e^{-r_d T}``."""
    return bates_call(S, K, T, r_d, r_f, p, j) - S * np.exp(-r_f * T) + K * np.exp(-r_d * T)


def merton_call(S: float, K: float, T: float, r_d: float, r_f: float, sigma: float,
                j: JumpParams, n_terms: int = 80) -> float:
    """Merton (1976) jump-diffusion FX call as a Poisson-weighted sum of GK prices.

    Merton, R. (1976), "Option pricing when underlying stock returns are
    discontinuous", J. Financial Economics 3, 125-144, eq. (19), with the
    dividend yield replaced by ``r_f``.
    """
    _check(S, K, T, r_d, r_f)
    if j.lam == 0.0:
        return float(gk_price(S, K, T, r_d, r_f, sigma, 1))
    total = 0.0
    log_w = -j.lam * T
    for n in range(n_terms):
        if n > 0:
            log_w += np.log(j.lam * T) - np.log(n)
        sig_n = math.sqrt(sigma * sigma + n * j.delta**2 / T)
        # given n jumps the forward grows at (r_d - r_f) - lam kbar + n ln(1+kbar)/T;
        # express that as a shifted domestic rate and discount at r_d
        r_n = r_d - j.lam * j.kbar + n * math.log1p(j.kbar) / T
        total += math.exp(log_w) * gk_price(S, K, T, r_n, r_f, sig_n, 1) * math.exp((r_n - r_d) * T)
    return float(total)
