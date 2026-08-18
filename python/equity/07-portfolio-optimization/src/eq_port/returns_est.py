"""Expected-return estimation: sample, EWMA, James-Stein shrinkage,
CAPM reverse optimization and simplified Black-Litterman.

Conventions
-----------
* Inputs are (T, N) panels of simple per-period returns (numpy array or
  pandas DataFrame); outputs are per-period quantities (NOT annualised).
* "Excess" means in excess of the per-period risk-free rate.

The estimation-error problem
----------------------------
Merton (1980): over T periods the standard error of a mean estimate is
sigma/sqrt(T) *regardless of sampling frequency* — sampling daily instead
of monthly improves variance estimates but not mean estimates. With
sigma ~ 16%/yr and 10 years of data the SE of the annual mean is ~5%,
the same order as the equity risk premium itself. Mean-variance weights
are roughly linear in Sigma^{-1} mu, so this noise is amplified into
extreme, unstable portfolios ("error maximization", Michaud 1989).
The estimators below attack that problem:

* James-Stein / Bayes-Stein: shrink the sample mean vector toward the
  grand mean, trading a little bias for a large variance reduction.
* Reverse optimization: discard the sample mean entirely and back out
  the returns implied by market-cap weights (the Black-Litterman prior).
* Black-Litterman: blend that equilibrium prior with explicit views in a
  Bayesian posterior, with view confidence controlling the blend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "sample_mean",
    "ewma_mean",
    "james_stein_mean",
    "JamesSteinResult",
    "implied_equilibrium_returns",
    "black_litterman",
    "BlackLittermanResult",
]


def _as_matrix(returns: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Coerce a (T, N) return panel to a 2-D float ndarray with validation."""
    x = np.asarray(returns, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"returns must be (T, N), got shape {x.shape}")
    if x.shape[0] < 1:
        raise ValueError("returns must contain at least one observation")
    if not np.all(np.isfinite(x)):
        raise ValueError("returns contain NaN or infinite values")
    return x


def sample_mean(returns: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Per-period sample mean return, one entry per asset.

    Parameters
    ----------
    returns : (T, N) array-like
        Simple per-period returns.

    Returns
    -------
    np.ndarray
        (N,) sample means (matches ``np.mean(returns, axis=0)``).
    """
    return _as_matrix(returns).mean(axis=0)


def ewma_mean(returns: np.ndarray | pd.DataFrame, halflife: float = 60.0) -> np.ndarray:
    """Exponentially weighted mean with weights normalised to sum to 1.

    Weight on the observation lagged k periods is proportional to
    ``lambda**k`` with ``lambda = 0.5**(1/halflife)``; the most recent
    observation gets the largest weight.

    Parameters
    ----------
    returns : (T, N) array-like
        Simple per-period returns, oldest first.
    halflife : float
        Half-life in periods (> 0).

    Returns
    -------
    np.ndarray
        (N,) exponentially weighted per-period means.
    """
    if halflife <= 0:
        raise ValueError(f"halflife must be > 0, got {halflife}")
    x = _as_matrix(returns)
    t = x.shape[0]
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(t - 1, -1, -1, dtype=float)  # oldest gets lam^(T-1)
    w /= w.sum()
    return w @ x


@dataclass(frozen=True)
class JamesSteinResult:
    """James-Stein shrinkage output.

    Attributes
    ----------
    mean : np.ndarray
        (N,) shrunk per-period mean vector.
    intensity : float
        Shrinkage intensity phi in [0, 1]; 0 = sample mean, 1 = grand mean.
    grand_mean : float
        The shrinkage target (cross-sectional average of sample means).
    """

    mean: np.ndarray
    intensity: float
    grand_mean: float


def james_stein_mean(returns: np.ndarray | pd.DataFrame) -> JamesSteinResult:
    """James-Stein estimator: shrink sample means toward the grand mean.

    The estimator is the convex combination

        mu_JS = phi * m * 1 + (1 - phi) * mu_hat,

    with grand mean ``m = mean(mu_hat)`` and intensity

        phi = min(1, max(0, (N - 3) * sbar2 / T / sum_i (mu_hat_i - m)^2)),

    where ``sbar2`` is the average sample variance across assets, so
    ``sbar2 / T`` proxies the sampling variance of each mean estimate
    (Merton's point: this shrinks harder when means are noisy relative to
    their cross-sectional dispersion). ``N - 3`` is floored at 1 so the
    estimator still shrinks (mildly) for N <= 3.

    Parameters
    ----------
    returns : (T, N) array-like
        Simple per-period returns.

    Returns
    -------
    JamesSteinResult
        Shrunk mean, intensity in [0, 1], and the grand mean target.
    """
    x = _as_matrix(returns)
    t, n = x.shape
    mu = x.mean(axis=0)
    m = float(mu.mean())
    if t < 2:
        # cannot estimate sampling variance: return the grand mean (full shrink)
        return JamesSteinResult(np.full(n, m), 1.0, m)
    sbar2 = float(x.var(axis=0, ddof=1).mean())
    dispersion = float(np.sum((mu - m) ** 2))
    if dispersion <= 0.0 or sbar2 == 0.0:
        phi = 1.0 if dispersion <= 0.0 else 0.0
    else:
        phi = max(0.0, min(1.0, max(n - 3, 1) * sbar2 / t / dispersion))
    shrunk = phi * m + (1.0 - phi) * mu
    return JamesSteinResult(shrunk, phi, m)


def implied_equilibrium_returns(
    cov: np.ndarray,
    market_weights: np.ndarray,
    risk_aversion: float = 2.5,
) -> np.ndarray:
    """CAPM-implied equilibrium EXCESS returns via reverse optimization.

    If the market holds weights ``w_mkt`` and those weights are the
    solution of an unconstrained mean-variance problem with risk aversion
    ``delta`` (max w'pi - delta/2 w'Sigma w), then the first-order
    condition inverts to

        pi = delta * Sigma @ w_mkt.

    Feeding ``pi`` back into the tangency portfolio reproduces
    ``w_mkt`` exactly (round-trip identity) — this is the
    Black-Litterman prior.

    Parameters
    ----------
    cov : (N, N) array-like
        Per-period covariance matrix.
    market_weights : (N,) array-like
        Market-cap weights; should sum to 1.
    risk_aversion : float
        delta > 0. Scales the level (not the direction) of pi.

    Returns
    -------
    np.ndarray
        (N,) implied per-period excess returns.
    """
    sigma = np.asarray(cov, dtype=float)
    w = np.asarray(market_weights, dtype=float).ravel()
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"cov must be square, got shape {sigma.shape}")
    if w.shape[0] != sigma.shape[0]:
        raise ValueError(
            f"dimension mismatch: cov is {sigma.shape}, weights have {w.shape[0]} entries"
        )
    if risk_aversion <= 0:
        raise ValueError(f"risk_aversion must be > 0, got {risk_aversion}")
    return risk_aversion * sigma @ w


@dataclass(frozen=True)
class BlackLittermanResult:
    """Black-Litterman posterior.

    Attributes
    ----------
    mean : np.ndarray
        (N,) posterior expected excess returns.
    cov : np.ndarray
        (N, N) posterior return covariance ``Sigma + M`` where M is the
        posterior uncertainty about the mean (use for MVO).
    mean_uncertainty : np.ndarray
        (N, N) M = posterior covariance of the mean estimate itself.
    """

    mean: np.ndarray
    cov: np.ndarray
    mean_uncertainty: np.ndarray


def black_litterman(
    prior_mean: np.ndarray,
    cov: np.ndarray,
    view_matrix: np.ndarray | None = None,
    view_returns: np.ndarray | None = None,
    tau: float = 0.05,
    omega: np.ndarray | None = None,
) -> BlackLittermanResult:
    """Simplified Black-Litterman posterior mean and covariance.

    Model: the unknown mean mu ~ N(pi, tau * Sigma) (equilibrium prior),
    and K views  P mu = Q + eta,  eta ~ N(0, Omega). The posterior mean is

        mu_BL = pi + tau Sigma P' (P tau Sigma P' + Omega)^{-1} (Q - P pi)

    and the posterior uncertainty about the mean is

        M = tau Sigma - tau Sigma P' (P tau Sigma P' + Omega)^{-1} P tau Sigma,

    so the return distribution for portfolio construction is
    N(mu_BL, Sigma + M). This form never inverts Omega, so Omega = 0
    (perfect confidence) is handled exactly: the posterior then satisfies
    P mu_BL = Q identically. With no views the posterior mean equals the
    prior exactly and M = tau Sigma.

    Parameters
    ----------
    prior_mean : (N,) array-like
        Equilibrium prior pi (per-period excess returns), typically from
        :func:`implied_equilibrium_returns`.
    cov : (N, N) array-like
        Per-period return covariance Sigma.
    view_matrix : (K, N) array-like or None
        Pick matrix P; each row is one view portfolio. None/empty = no views.
    view_returns : (K,) array-like or None
        View targets Q (per-period excess returns).
    tau : float
        Prior scale in (0, inf); small tau = tight prior.
    omega : (K, K) array-like or None
        View error covariance. Default (He-Litterman): diag(P tau Sigma P').

    Returns
    -------
    BlackLittermanResult
        Posterior mean, posterior return covariance Sigma + M, and M.
    """
    pi = np.asarray(prior_mean, dtype=float).ravel()
    sigma = np.asarray(cov, dtype=float)
    n = pi.shape[0]
    if sigma.shape != (n, n):
        raise ValueError(
            f"dimension mismatch: prior_mean has {n} entries, cov is {sigma.shape}"
        )
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    tau_sigma = tau * sigma
    if view_matrix is None or np.size(view_matrix) == 0:
        return BlackLittermanResult(pi.copy(), sigma + tau_sigma, tau_sigma)

    p = np.atleast_2d(np.asarray(view_matrix, dtype=float))
    if p.shape[1] != n:
        raise ValueError(f"view_matrix must have {n} columns, got {p.shape[1]}")
    if view_returns is None:
        raise ValueError("view_returns must be provided when view_matrix is given")
    q = np.atleast_1d(np.asarray(view_returns, dtype=float))
    k = p.shape[0]
    if q.shape[0] != k:
        raise ValueError(
            f"view_returns has {q.shape[0]} entries but view_matrix has {k} rows"
        )
    if omega is None:
        omega_m = np.diag(np.diag(p @ tau_sigma @ p.T))
    else:
        omega_m = np.atleast_2d(np.asarray(omega, dtype=float))
        if omega_m.shape != (k, k):
            raise ValueError(f"omega must be ({k}, {k}), got {omega_m.shape}")

    a = p @ tau_sigma @ p.T + omega_m  # (K, K)
    b = np.linalg.solve(a, np.column_stack([q - p @ pi, p @ tau_sigma]))
    adj = b[:, 0]  # A^{-1} (Q - P pi)
    proj = b[:, 1:]  # A^{-1} P tau Sigma
    mu_bl = pi + tau_sigma @ p.T @ adj
    m = tau_sigma - tau_sigma @ p.T @ proj
    m = 0.5 * (m + m.T)
    return BlackLittermanResult(mu_bl, sigma + m, m)
