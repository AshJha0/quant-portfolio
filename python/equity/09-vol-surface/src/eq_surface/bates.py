"""Bates (1996) stochastic volatility with lognormal jumps.

Bates, D. (1996), "Jumps and stochastic volatility: exchange rate processes
implicit in Deutsche Mark options", Review of Financial Studies 9(1),
69-107. The log-price characteristic function factorises into the Heston
part and an independent compound-Poisson part (Gatheral 2006, eq. 5.6):

    phi_Bates(u) = phi_Heston(u) * exp(lam*T*(exp(i*u*mu_j - delta^2 u^2 / 2) - 1)
                                       - i*u*lam*kbar*T),
    kbar = exp(mu_j + delta^2/2) - 1,

the last term being the drift compensator that keeps the forward a
martingale. Pricing uses the Lewis (2001) single-integral formula
("A simple option formula for general jump-diffusion and other
exponential Levy processes", eq. 2.10):

    C = S e^{-qT} - sqrt(S K) e^{-(r+q)T/2} / pi
        * int_0^inf Re[e^{i u x} phi_X(u - i/2)] / (u^2 + 1/4) du,

with ``X = ln(S_T/S) - (r-q)T`` and ``x = ln(S/K) + (r-q)T``.

Oracles in the tests: ``lam = 0`` gives Heston; ``xi = 0`` with
``v0 = theta`` gives Merton (1976) jump-diffusion, priced by its Poisson
series of Black-Scholes prices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from .black_scholes import bs_price
from .heston import HestonParams, heston_cf


@dataclass(frozen=True)
class JumpParams:
    """Poisson intensity ``lam`` (per year), mean ``mu_j`` and sd ``delta`` of log-jump."""

    lam: float
    mu_j: float
    delta: float

    def __post_init__(self) -> None:
        if self.lam < 0.0:
            raise ValueError(f"lam must be >= 0, got {self.lam}")
        if self.delta < 0.0:
            raise ValueError(f"delta must be >= 0, got {self.delta}")

    @property
    def kbar(self) -> float:
        return float(np.exp(self.mu_j + 0.5 * self.delta**2) - 1.0)


def bates_cf_x(u: np.ndarray | complex, T: float, p: HestonParams, j: JumpParams):
    """CF of ``X = ln(S_T/S) - (r-q)T`` (independent of S, r, q)."""
    u = np.asarray(u, dtype=complex)
    heston = heston_cf(u, T, 1.0, 0.0, 0.0, p)  # S=1, r=q=0 -> CF of X
    jump = np.exp(j.lam * T * (np.exp(1j * u * j.mu_j - 0.5 * j.delta**2 * u * u) - 1.0)
                  - 1j * u * j.lam * j.kbar * T)
    return heston * jump


def bates_call(S: float, K: float, T: float, r: float, q: float,
               p: HestonParams, j: JumpParams) -> float:
    """Bates European call by the Lewis (2001) integral."""
    if S <= 0.0 or K <= 0.0 or T <= 0.0:
        raise ValueError("S, K and T must be > 0")
    x = np.log(S / K) + (r - q) * T

    def integrand(u: float) -> float:
        return float(np.real(np.exp(1j * u * x) * bates_cf_x(u - 0.5j, T, p, j))
                     / (u * u + 0.25))

    integral, _ = quad(integrand, 0.0, np.inf, limit=500, epsabs=1e-12, epsrel=1e-10)
    price = S * np.exp(-q * T) - np.sqrt(S * K) * np.exp(-0.5 * (r + q) * T) / np.pi * integral
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    return float(max(price, intrinsic))


def bates_put(S: float, K: float, T: float, r: float, q: float,
              p: HestonParams, j: JumpParams) -> float:
    """Put by parity."""
    return bates_call(S, K, T, r, q, p, j) - S * np.exp(-q * T) + K * np.exp(-r * T)


def merton_call(S: float, K: float, T: float, r: float, q: float, sigma: float,
                j: JumpParams, n_terms: int = 80) -> float:
    """Merton (1976) jump-diffusion call as a Poisson-weighted sum of BS prices.

    Merton, R. (1976), "Option pricing when underlying stock returns are
    discontinuous", J. Financial Economics 3, 125-144, eq. (19).
    """
    total = 0.0
    log_w = -j.lam * T  # Poisson(lam T) weight of n jumps
    if j.lam == 0.0:
        return float(bs_price(S, K, T, r, q, sigma))
    for n in range(n_terms):
        if n > 0:
            log_w += np.log(j.lam * T) - np.log(n)
        sig_n = np.sqrt(sigma * sigma + n * j.delta**2 / T)
        r_n = r - j.lam * j.kbar + n * np.log1p(j.kbar) / T
        # given n jumps the forward grows at r_n; discount at r, not r_n
        total += np.exp(log_w) * bs_price(S, K, T, r_n, q, sig_n) * np.exp((r_n - r) * T)
    return float(total)
