"""Heston Monte Carlo: full-truncation Euler and Andersen QE.

Discretisation bias (documented, measured in tests and docs/VALIDATION.md)
--------------------------------------------------------------------------
The Euler scheme applied to the CIR variance process produces negative
variances with positive probability at any finite step size; *full
truncation* (Lord-Koekkoek-van Dijk 2010) replaces ``v`` by ``v+ = max(v,0)``
in **both** the drift and the diffusion, which is the least-biased of the
simple Euler fixes.  Even so, Euler carries an O(dt) weak bias that becomes
material at coarse steps whenever the Feller condition is violated or
vol-of-vol is high: truncation systematically distorts the variance
distribution near zero.

Andersen's QE (quadratic-exponential) scheme instead samples the *exact* CIR
conditional distribution's moment-matched approximation: a squared Gaussian
when the conditional distribution is far from zero (psi <= 1.5), and a mixed
exponential/point-mass-at-zero when it is close to zero (psi > 1.5).  The
log-spot is advanced with Andersen's correlation-preserving scheme (the
Broadie-Kaya decomposition with central weights gamma1 = gamma2 = 1/2).  QE's
weak bias at coarse steps is typically an order of magnitude below Euler's at
the same step count -- the comparative bias test in the suite demonstrates
this on a Feller-violating parameter set.

QE applies Andersen's martingale correction by default (section 4.4.3 of
Andersen 2008), so the discounted simulated forward has mean ``S e^{-qT}``
up to Monte Carlo error at any step size.

Variance reduction: ``antithetic=True`` mirrors every Gaussian (and every
uniform as ``1-u``) and the price SE is computed on pair averages;
``control_variate=True`` regresses the discounted payoff on the discounted
terminal spot, whose mean ``S e^{-qT}`` is known exactly.

Both schemes store only non-negative variances (Euler stores ``v+``), are
driven by an explicit ``numpy.random.Generator`` seed, and share the same
random-number layout per scheme so results are reproducible bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .heston import HestonParams

__all__ = ["MCResult", "heston_mc_price", "simulate_heston_terminal"]

Scheme = Literal["euler_ft", "qe"]


@dataclass(frozen=True)
class MCResult:
    """Monte Carlo pricing result.

    Attributes
    ----------
    price : float
        Discounted mean payoff.
    stderr : float
        Standard error of the price estimate.
    n_paths, n_steps : int
        Simulation size.
    scheme : str
        Discretisation scheme used.
    """

    price: float
    stderr: float
    n_paths: int
    n_steps: int
    scheme: str


def _validate_mc(S: float, T: float, n_paths: int, n_steps: int) -> None:
    if S <= 0.0:
        raise ValueError(f"spot must be positive, got {S}")
    if T <= 0.0:
        raise ValueError(f"T must be positive for simulation, got {T}")
    if n_paths < 2:
        raise ValueError(f"need at least 2 paths, got {n_paths}")
    if n_steps < 1:
        raise ValueError(f"need at least 1 time step, got {n_steps}")


def simulate_heston_terminal(
    S: float,
    T: float,
    r: float,
    q: float,
    p: HestonParams,
    n_paths: int,
    n_steps: int,
    scheme: Scheme = "qe",
    seed: int | np.random.Generator = 0,
    antithetic: bool = False,
    martingale_correction: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate terminal spot and variance under Heston.

    Parameters
    ----------
    S : float
        Initial spot (> 0).
    T : float
        Horizon in years (> 0).
    r, q : float
        Continuously compounded rate and dividend yield (annualised).
    p : HestonParams
        Model parameters.
    n_paths, n_steps : int
        Number of paths and uniform time steps.
    scheme : {"euler_ft", "qe"}
        ``"euler_ft"`` = full-truncation Euler; ``"qe"`` = Andersen QE with
        central log-spot weights.
    seed : int or numpy.random.Generator
        Seed for reproducibility (explicit, per the testing contract).
    antithetic : bool
        Mirror every random draw; path ``i + n_paths/2`` is the antithetic
        partner of path ``i``.  Requires an even ``n_paths``.
    martingale_correction : bool
        QE only: Andersen's drift correction (see module docstring).  Where
        the required moment generating function does not exist on a path
        (very coarse steps), that path keeps the uncorrected drift.

    Returns
    -------
    (S_T, v_T) : tuple of arrays, each shape (n_paths,)
        Terminal spot and terminal (non-negative) variance.
    """
    _validate_mc(S, T, n_paths, n_steps)
    if antithetic and n_paths % 2:
        raise ValueError(
            f"n_paths must be even when antithetic=True (an odd count leaves one "
            f"draw without its mirror), got {n_paths}"
        )
    rng = np.random.default_rng(seed) if not isinstance(seed, np.random.Generator) else seed
    dt = T / n_steps
    sq_dt = np.sqrt(dt)
    v0, kappa, theta, rho, xi = p.v0, p.kappa, p.theta, p.rho, p.xi
    rho_bar = np.sqrt(max(1.0 - rho * rho, 0.0))

    x = np.full(n_paths, np.log(S))
    v = np.full(n_paths, v0)
    drift = (r - q) * dt

    def normal() -> np.ndarray:
        if antithetic:
            z = rng.standard_normal(n_paths // 2)
            return np.concatenate([z, -z])
        return rng.standard_normal(n_paths)

    def uniform() -> np.ndarray:
        if antithetic:
            uu = rng.uniform(size=n_paths // 2)
            return np.concatenate([uu, 1.0 - uu])
        return rng.uniform(size=n_paths)

    if scheme == "euler_ft":
        for _ in range(n_steps):
            z_v = normal()
            z_s = rho * z_v + rho_bar * normal()
            v_plus = np.maximum(v, 0.0)
            sqv = np.sqrt(v_plus)
            x = x + drift - 0.5 * v_plus * dt + sqv * sq_dt * z_s
            v = v + kappa * (theta - v_plus) * dt + xi * sqv * sq_dt * z_v
        return np.exp(x), np.maximum(v, 0.0)

    if scheme != "qe":
        raise ValueError(f"unknown scheme {scheme!r}; use 'euler_ft' or 'qe'")

    # ---------------- Andersen QE ---------------- #
    if xi < 1e-12:
        # Deterministic variance path: exact integration per step.
        t_grid = np.linspace(0.0, T, n_steps + 1)
        v_det = theta + (v0 - theta) * np.exp(-kappa * t_grid)
        for i in range(n_steps):
            v_mid = 0.5 * (v_det[i] + v_det[i + 1])
            z = normal()
            x = x + drift - 0.5 * v_mid * dt + np.sqrt(v_mid * dt) * z
        return np.exp(x), np.full(n_paths, v_det[-1])

    psi_c = 1.5
    e_kdt = np.exp(-kappa * dt)
    g1 = g2 = 0.5
    k0 = -rho * kappa * theta * dt / xi
    k1 = g1 * dt * (kappa * rho / xi - 0.5) - rho / xi
    k2 = g2 * dt * (kappa * rho / xi - 0.5) + rho / xi
    k3 = g1 * dt * (1.0 - rho * rho)
    k4 = g2 * dt * (1.0 - rho * rho)

    for _ in range(n_steps):
        m = theta + (v - theta) * e_kdt
        s2 = (
            v * xi * xi * e_kdt * (1.0 - e_kdt) / kappa
            + theta * xi * xi * (1.0 - e_kdt) ** 2 / (2.0 * kappa)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            psi = np.where(m > 0.0, s2 / np.maximum(m * m, 1e-300), np.inf)

        v_next = np.empty_like(v)
        z_v = normal()
        u = uniform()
        step_drift = k0 + k1 * v  # uncorrected (central discretisation)
        A = k2 + 0.5 * k4

        low = psi <= psi_c
        # Quadratic branch: v' = a (b + Z)^2
        if low.any():
            psi_l = np.maximum(psi[low], 1e-12)
            inv2 = 2.0 / psi_l
            b2 = inv2 - 1.0 + np.sqrt(inv2 * np.maximum(inv2 - 1.0, 0.0))
            a = m[low] / (1.0 + b2)
            v_next[low] = a * (np.sqrt(b2) + z_v[low]) ** 2
            if martingale_correction:
                # E[exp(A v')] = exp(A b^2 a / (1 - 2 A a)) / sqrt(1 - 2 A a)
                ok = A < 0.5 / a
                one_m = np.where(ok, 1.0 - 2.0 * A * a, 1.0)
                log_m = A * b2 * a / one_m - 0.5 * np.log(one_m)
                step_drift[low] = np.where(ok, -log_m - 0.5 * k3 * v[low], step_drift[low])
        # Exponential branch: point mass at zero + exponential tail
        high = ~low
        if high.any():
            psi_h = psi[high]
            pp = (psi_h - 1.0) / (psi_h + 1.0)
            beta = (1.0 - pp) / np.maximum(m[high], 1e-300)
            uh = u[high]
            v_h = np.where(uh <= pp, 0.0, np.log((1.0 - pp) / np.maximum(1.0 - uh, 1e-300)) / beta)
            v_next[high] = v_h
            if martingale_correction:
                # E[exp(A v')] = p + beta (1 - p) / (beta - A)
                ok = A < beta
                mgf = pp + beta * (1.0 - pp) / np.where(ok, beta - A, 1.0)
                step_drift[high] = np.where(ok, -np.log(mgf) - 0.5 * k3 * v[high],
                                            step_drift[high])

        z_x = normal()
        x = (
            x
            + drift
            + step_drift
            + k2 * v_next
            + np.sqrt(np.maximum(k3 * v + k4 * v_next, 0.0)) * z_x
        )
        v = v_next

    return np.exp(x), v


def heston_mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    p: HestonParams,
    n_paths: int = 100_000,
    n_steps: int = 64,
    scheme: Scheme = "qe",
    seed: int | np.random.Generator = 0,
    kind: Literal["call", "put"] = "call",
    antithetic: bool = False,
    control_variate: bool = False,
    martingale_correction: bool = True,
) -> MCResult:
    """Price a European option by Heston Monte Carlo.

    Returns
    -------
    MCResult
        Price, standard error and simulation metadata.  The estimator is the
        discounted sample mean of the payoff (on antithetic pair averages
        when ``antithetic=True``).  With ``control_variate=True`` the
        discounted terminal spot, whose mean ``S e^{-qT}`` is known, is used
        as a control with the regression coefficient estimated from the
        same paths (an O(1/n) bias, negligible at these path counts).
        ``stderr`` is the standard deviation of the per-sample estimator
        divided by the square root of the number of independent samples.
    """
    if K <= 0.0:
        raise ValueError(f"strike must be positive, got {K}")
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    s_T, _ = simulate_heston_terminal(S, T, r, q, p, n_paths, n_steps, scheme, seed,
                                      antithetic=antithetic,
                                      martingale_correction=martingale_correction)
    payoff = np.maximum(s_T - K, 0.0) if kind == "call" else np.maximum(K - s_T, 0.0)
    df = np.exp(-r * T)
    y = df * payoff
    c = df * s_T  # control: E[c] = S exp(-qT)
    if antithetic:
        h = n_paths // 2
        y = 0.5 * (y[:h] + y[h:])
        c = 0.5 * (c[:h] + c[h:])
    if control_variate:
        cc = c - c.mean()
        beta = float(np.dot(cc, y - y.mean()) / np.dot(cc, cc))
        y = y - beta * (c - S * np.exp(-q * T))
    price = float(y.mean())
    stderr = float(y.std(ddof=1) / np.sqrt(y.size))
    return MCResult(price=price, stderr=stderr, n_paths=n_paths, n_steps=n_steps, scheme=scheme)
