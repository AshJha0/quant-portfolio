"""Heston Monte Carlo: full-truncation Euler and Andersen's QE scheme.

Both schemes simulate under the domestic risk-neutral measure with drift
``r_d - r_f`` and are cross-validated against the Fourier prices to
within 3 Monte Carlo standard errors (tests).  Full-truncation Euler
(Lord et al. 2010) is the robust baseline but needs fine time steps;
QE (Andersen 2008) matches the first two conditional moments of the
CIR transition and is accurate even on coarse grids - the standard
production choice when the Feller condition fails (as it usually does
for calibrated FX smiles).

Every function takes an explicit ``seed`` (or Generator) and is fully
deterministic given it.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

from .heston import HestonParams

__all__ = ["simulate_terminal", "mc_price"]

_PSI_C = 1.5  # Andersen's switching threshold


def _rng(seed: Union[int, np.random.Generator]) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def simulate_terminal(
    S: float,
    T: float,
    r_d: float,
    r_f: float,
    params: HestonParams,
    n_paths: int = 50_000,
    n_steps: int = 100,
    scheme: str = "qe",
    seed: Union[int, np.random.Generator] = 0,
    antithetic: bool = True,
) -> np.ndarray:
    """Simulate terminal spots ``S_T`` under Heston.

    Parameters
    ----------
    scheme : {"euler_ft", "qe"}
        Full-truncation Euler or Andersen quadratic-exponential.
    antithetic : bool
        Antithetic Gaussians (Euler and the QE log-spot shock; QE's
        variance draw uses the same antithetic normals via inverse
        transform on the quadratic branch and independent uniforms on
        the exponential branch).

    Returns
    -------
    ndarray, shape (n_paths,)
    """
    if S <= 0 or T <= 0:
        raise ValueError("S and T must be positive")
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive")
    if scheme not in ("euler_ft", "qe"):
        raise ValueError(f"unknown scheme {scheme!r}; use 'euler_ft' or 'qe'")
    if antithetic and n_paths % 2 != 0:
        # An odd n_paths silently breaks pairing: n_base = (n_paths+1)//2
        # antithetic normals get concatenated [z, -z] and truncated back to
        # n_paths, so the *last* base draw loses its antithetic mirror while
        # every other draw keeps one. mc_price's pairwise averaging (which
        # is what removes the antithetic correlation from the standard-error
        # estimate) then silently degrades to treating correlated pairs as
        # independent samples, understating the reported standard error.
        raise ValueError(
            f"n_paths must be even when antithetic=True (odd n_paths breaks "
            f"pairing and silently understates the standard error), got {n_paths}"
        )
    gen = _rng(seed)
    p = params
    dt = T / n_steps
    mu = r_d - r_f
    n_base = (n_paths + 1) // 2 if antithetic else n_paths

    x = np.full(n_paths, math.log(S))
    v = np.full(n_paths, p.v0)

    if scheme == "euler_ft":
        sq_dt = math.sqrt(dt)
        for _ in range(n_steps):
            z1 = gen.standard_normal(n_base)
            z2 = gen.standard_normal(n_base)
            if antithetic:
                z1 = np.concatenate([z1, -z1])[:n_paths]
                z2 = np.concatenate([z2, -z2])[:n_paths]
            zs = z1
            zv = p.rho * z1 + math.sqrt(1.0 - p.rho * p.rho) * z2
            vp = np.maximum(v, 0.0)
            x = x + (mu - 0.5 * vp) * dt + np.sqrt(vp) * sq_dt * zs
            v = v + p.kappa * (p.theta - vp) * dt + p.xi * np.sqrt(vp) * sq_dt * zv
        return np.exp(x)

    # --- QE (Andersen 2008), central discretisation gamma1 = gamma2 = 1/2
    g1 = g2 = 0.5
    e = math.exp(-p.kappa * dt)
    k0 = -p.rho * p.kappa * p.theta * dt / p.xi
    k1 = g1 * dt * (p.kappa * p.rho / p.xi - 0.5) - p.rho / p.xi
    k2 = g2 * dt * (p.kappa * p.rho / p.xi - 0.5) + p.rho / p.xi
    k3 = g1 * dt * (1.0 - p.rho * p.rho)
    k4 = g2 * dt * (1.0 - p.rho * p.rho)

    for _ in range(n_steps):
        m = p.theta + (v - p.theta) * e
        s2 = (
            v * p.xi * p.xi * e * (1.0 - e) / p.kappa
            + p.theta * p.xi * p.xi * (1.0 - e) ** 2 / (2.0 * p.kappa)
        )
        psi = s2 / np.maximum(m * m, 1e-300)

        zv = gen.standard_normal(n_base)
        u = gen.random(n_base)
        zx = gen.standard_normal(n_base)
        if antithetic:
            zv = np.concatenate([zv, -zv])[:n_paths]
            u = np.concatenate([u, 1.0 - u])[:n_paths]
            zx = np.concatenate([zx, -zx])[:n_paths]

        v_next = np.empty_like(v)
        quad_branch = psi <= _PSI_C
        # Quadratic branch: v' = a (b + Z)^2
        if np.any(quad_branch):
            psi_q = psi[quad_branch]
            inv2 = 2.0 / psi_q
            b2 = inv2 - 1.0 + np.sqrt(inv2 * np.maximum(inv2 - 1.0, 0.0))
            a = m[quad_branch] / (1.0 + b2)
            v_next[quad_branch] = a * (np.sqrt(b2) + zv[quad_branch]) ** 2
        # Exponential branch: mass at zero + exponential tail
        if np.any(~quad_branch):
            psi_e = psi[~quad_branch]
            pmass = (psi_e - 1.0) / (psi_e + 1.0)
            beta = (1.0 - pmass) / np.maximum(m[~quad_branch], 1e-300)
            ue = u[~quad_branch]
            ve = np.zeros_like(ue)
            tail = ue > pmass
            ve[tail] = np.log((1.0 - pmass[tail]) / (1.0 - ue[tail])) / beta[tail]
            v_next[~quad_branch] = ve

        x = (
            x + mu * dt + k0 + k1 * v + k2 * v_next
            + np.sqrt(np.maximum(k3 * v + k4 * v_next, 0.0)) * zx
        )
        v = v_next
    return np.exp(x)


def mc_price(
    S: float,
    K: float,
    T: float,
    r_d: float,
    r_f: float,
    params: HestonParams,
    cp: int = 1,
    n_paths: int = 50_000,
    n_steps: int = 100,
    scheme: str = "qe",
    seed: Union[int, np.random.Generator] = 0,
    antithetic: bool = True,
) -> tuple[float, float]:
    """Monte Carlo vanilla price.

    Returns
    -------
    (price, stderr) : tuple of float
        Discounted mean payoff and its Monte Carlo standard error
        (sample std / sqrt(n); antithetic pairs are averaged first so
        the SE is not understated by pair correlation).
    """
    if cp not in (+1, -1):
        raise ValueError(f"cp must be +1 or -1, got {cp}")
    if K <= 0:
        raise ValueError("strike must be positive")
    ST = simulate_terminal(
        S, T, r_d, r_f, params, n_paths, n_steps, scheme, seed, antithetic
    )
    payoff = np.maximum(cp * (ST - K), 0.0) * math.exp(-r_d * T)
    if antithetic:
        # simulate_terminal enforces n_paths even under antithetic=True, so
        # this split is always a clean pairing of [z, -z] halves.
        half = len(payoff) // 2
        payoff = 0.5 * (payoff[:half] + payoff[half:])
    price = float(np.mean(payoff))
    se = float(np.std(payoff, ddof=1) / math.sqrt(len(payoff)))
    return price, se
