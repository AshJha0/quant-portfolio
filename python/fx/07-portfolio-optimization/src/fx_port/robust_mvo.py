"""Resampled and robust mean-variance optimisation.

Both attack the same problem: MVO treats the estimated mean as exact, so it
loads on estimation error (Michaud 1989, "The Markowitz optimization
enigma", Financial Analysts Journal 45(1)).

Classic problem (budget only)
-----------------------------
max_w mu' w - (lambda / 2) w' Sigma w  s.t. 1' w = 1 has the closed form

    w = Sigma^{-1} (mu - gamma 1) / lambda,
    gamma = (1' Sigma^{-1} mu - lambda) / (1' Sigma^{-1} 1).

Resampled efficiency
--------------------
Michaud (1998), "Efficient Asset Management", chapter 6: draw B samples of
T returns from N(mu, Sigma), re-estimate (mu_b, Sigma_b), solve, and average
the weights. As T grows the average converges to the classic solution.

Robust (ellipsoidal uncertainty in the mean)
--------------------------------------------
With mu in {mu_hat + Omega^{1/2} u : |u| <= kappa}, the worst case gives
(Ceria and Stubbs 2006, "Incorporating estimation errors into portfolio
selection", J. Asset Management 7(2), eq. (6); Goldfarb and Iyengar 2003):

    max_w mu_hat' w - kappa sqrt(w' Omega w) - (lambda / 2) w' Sigma w,  1' w = 1.

With Omega = Sigma / T (the standard error of the sample mean), the first-
order condition shows the robust portfolio is the classic one at the higher
risk aversion lambda + kappa / (T sqrt(w' Omega w)); the tests check this.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _check(mu: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(mu, dtype=float)
    c = np.asarray(cov, dtype=float)
    if m.ndim != 1 or c.shape != (m.size, m.size):
        raise ValueError("mu must be 1-D and cov square of matching size")
    if not np.allclose(c, c.T, atol=1e-12):
        raise ValueError("cov must be symmetric")
    return m, c


def mean_variance_weights(mu: np.ndarray, cov: np.ndarray, risk_aversion: float) -> np.ndarray:
    """Closed-form budget-constrained mean-variance weights (module docstring)."""
    m, c = _check(mu, cov)
    if risk_aversion <= 0.0:
        raise ValueError("risk_aversion must be positive")
    ones = np.ones(m.size)
    inv_mu = np.linalg.solve(c, m)
    inv_1 = np.linalg.solve(c, ones)
    gamma = (ones @ inv_mu - risk_aversion) / (ones @ inv_1)
    return (inv_mu - gamma * inv_1) / risk_aversion


def _long_only_mv(mu: np.ndarray, cov: np.ndarray, lam: float) -> np.ndarray:
    n = mu.size
    res = minimize(  # type: ignore[call-overload]
        lambda w: -(mu @ w) + 0.5 * lam * w @ cov @ w,
        np.full(n, 1.0 / n),
        jac=lambda w: -mu + lam * cov @ w,
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"long-only MV solve failed: {res.message}")
    return np.clip(res.x, 0.0, None) / np.clip(res.x, 0.0, None).sum()


def resampled_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float,
    n_obs: int,
    n_resamples: int = 200,
    long_only: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Michaud resampled weights: the average of MV solutions over resampled inputs."""
    m, c = _check(mu, cov)
    if n_obs <= m.size or n_resamples < 1:
        raise ValueError("need n_obs > number of assets and n_resamples >= 1")
    rng = np.random.default_rng(seed)
    acc = np.zeros(m.size)
    for _ in range(n_resamples):
        x = rng.multivariate_normal(m, c, size=n_obs)
        mu_b, cov_b = x.mean(axis=0), np.cov(x, rowvar=False)
        acc += (
            _long_only_mv(mu_b, cov_b, risk_aversion)
            if long_only
            else mean_variance_weights(mu_b, cov_b, risk_aversion)
        )
    return acc / n_resamples


def robust_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float,
    kappa: float,
    omega: np.ndarray | None = None,
    n_obs: int | None = None,
) -> np.ndarray:
    """Robust MVO with an ellipsoidal mean-uncertainty set (module docstring).

    Give either ``omega`` or ``n_obs`` (then Omega = Sigma / n_obs).
    """
    m, c = _check(mu, cov)
    if kappa < 0.0 or risk_aversion <= 0.0:
        raise ValueError("kappa must be non-negative and risk_aversion positive")
    if omega is None:
        if n_obs is None or n_obs < 1:
            raise ValueError("give omega or a positive n_obs")
        om = c / n_obs
    else:
        om = np.asarray(omega, dtype=float)
    w0 = mean_variance_weights(m, c, risk_aversion)
    if kappa == 0.0:
        return w0
    n = m.size

    def f(w: np.ndarray) -> float:
        return float(-(m @ w) + kappa * np.sqrt(w @ om @ w) + 0.5 * risk_aversion * w @ c @ w)

    def g(w: np.ndarray) -> np.ndarray:
        s = np.sqrt(w @ om @ w)
        return -m + kappa * (om @ w) / s + risk_aversion * c @ w

    res = minimize(  # type: ignore[call-overload]
        f,
        w0,
        jac=g,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
        method="SLSQP",
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    if not res.success:
        raise RuntimeError(f"robust MV solve failed: {res.message}")
    return res.x
