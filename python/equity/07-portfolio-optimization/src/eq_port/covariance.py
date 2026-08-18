"""Covariance estimation: sample, EWMA, Ledoit-Wolf shrinkage (from
scratch), single-factor model, PSD repair and conditioning diagnostics.

Conventions: inputs are (T, N) simple per-period returns; all outputs are
per-period covariance matrices. Multiply by 252 to annualise daily.

Why shrinkage: with N assets and T observations the sample covariance has
N(N+1)/2 free parameters; when T is not >> N it is ill-conditioned (or
singular for T <= N), and Sigma^{-1} — the object MVO actually uses —
amplifies the smallest, worst-estimated eigenvalues. Ledoit-Wolf (2004,
"Honey, I Shrunk the Sample Covariance Matrix") shrinks the sample matrix
toward a constant-correlation target with a closed-form, data-driven
intensity that is optimal under Frobenius loss asymptotics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "sample_cov",
    "ewma_cov",
    "ledoit_wolf_cc",
    "LedoitWolfResult",
    "single_factor_cov",
    "psd_repair",
    "condition_number",
    "is_psd",
]


def _as_matrix(returns: np.ndarray | pd.DataFrame) -> np.ndarray:
    x = np.asarray(returns, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"returns must be (T, N), got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("returns contain NaN or infinite values")
    return x


def sample_cov(returns: np.ndarray | pd.DataFrame, ddof: int = 1) -> np.ndarray:
    """Sample covariance matrix (matches ``np.cov(returns.T, ddof=ddof)``).

    Parameters
    ----------
    returns : (T, N) array-like
        Simple per-period returns.
    ddof : int
        Delta degrees of freedom (1 = unbiased, 0 = maximum likelihood).

    Returns
    -------
    np.ndarray
        (N, N) per-period covariance.
    """
    x = _as_matrix(returns)
    t = x.shape[0]
    if t - ddof < 1:
        raise ValueError(f"need at least {ddof + 1} observations, got {t}")
    xc = x - x.mean(axis=0)
    return xc.T @ xc / (t - ddof)


def ewma_cov(
    returns: np.ndarray | pd.DataFrame,
    lam: float = 0.94,
    demean: bool = False,
) -> np.ndarray:
    """RiskMetrics EWMA covariance via the recursion
    ``S_t = lam * S_{t-1} + (1 - lam) * r_t r_t'`` with ``S_1 = r_1 r_1'``.

    The RiskMetrics convention treats returns as zero-mean at daily
    frequency (the mean is negligible relative to the vol; ``demean=True``
    subtracts the sample mean first).

    Parameters
    ----------
    returns : (T, N) array-like
        Simple per-period returns, oldest first.
    lam : float
        Decay in (0, 1); 0.94 is the RiskMetrics daily standard.
    demean : bool
        Subtract the full-sample mean before the recursion.

    Returns
    -------
    np.ndarray
        (N, N) final EWMA covariance S_T.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must be in (0, 1), got {lam}")
    x = _as_matrix(returns)
    if demean:
        x = x - x.mean(axis=0)
    s = np.outer(x[0], x[0])
    for r in x[1:]:
        s = lam * s + (1.0 - lam) * np.outer(r, r)
    return s


@dataclass(frozen=True)
class LedoitWolfResult:
    """Ledoit-Wolf shrinkage output.

    Attributes
    ----------
    cov : np.ndarray
        (N, N) shrunk covariance ``delta * F + (1 - delta) * S``.
    intensity : float
        Optimal shrinkage intensity delta in [0, 1].
    target : np.ndarray
        (N, N) constant-correlation target F.
    sample : np.ndarray
        (N, N) sample covariance S (1/T normalisation, per the paper).
    """

    cov: np.ndarray
    intensity: float
    target: np.ndarray
    sample: np.ndarray


def ledoit_wolf_cc(returns: np.ndarray | pd.DataFrame) -> LedoitWolfResult:
    """Ledoit-Wolf (2004) shrinkage to the constant-correlation target,
    implemented from scratch with the paper's closed-form intensity.

    Steps (notation of the paper; X is the demeaned (T, N) panel):

    1. S = X'X / T (MLE sample covariance).
    2. Target F: f_ii = s_ii, f_ij = rbar * sqrt(s_ii s_jj) with rbar the
       average off-diagonal sample correlation.
    3. pi_hat = sum_ij (1/T) sum_t (x_ti x_tj - s_ij)^2  — asymptotic
       variance of the sample covariance entries.
    4. rho_hat = sum_i pi_hat_ii + sum_{i != j} (rbar / 2) *
       [ sqrt(s_jj / s_ii) theta_ii,ij + sqrt(s_ii / s_jj) theta_jj,ij ]
       with theta_ii,ij = (1/T) sum_t (x_ti^2 - s_ii)(x_ti x_tj - s_ij)
       — asymptotic covariance between target and sample.
    5. gamma_hat = ||F - S||_F^2 (squared Frobenius distance).
    6. delta = clip( (pi_hat - rho_hat) / gamma_hat / T, 0, 1 ).

    Degenerate case: if S already equals F (gamma ~ 0, e.g. N = 1), the
    estimator returns S with intensity 0 — shrinkage is a no-op.

    Parameters
    ----------
    returns : (T, N) array-like
        Simple per-period returns; T >= 2. Works (and is the point) even
        when T <= N, where S is singular but the shrunk matrix is not.

    Returns
    -------
    LedoitWolfResult
        Shrunk matrix, intensity in [0, 1], target and sample matrices.
    """
    x = _as_matrix(returns)
    t, n = x.shape
    if t < 2:
        raise ValueError(f"need at least 2 observations, got {t}")
    x = x - x.mean(axis=0)
    s = x.T @ x / t

    std = np.sqrt(np.diag(s))
    if np.any(std == 0.0):
        # zero-variance asset: correlations undefined; treat its corr as 0
        std_safe = np.where(std == 0.0, 1.0, std)
    else:
        std_safe = std
    corr = s / np.outer(std_safe, std_safe)
    if n > 1:
        rbar = (corr.sum() - np.trace(corr)) / (n * (n - 1))
    else:
        rbar = 0.0
    f = rbar * np.outer(std, std)
    np.fill_diagonal(f, np.diag(s))

    gamma = float(np.sum((f - s) ** 2))
    if gamma < 1e-20:
        return LedoitWolfResult(s.copy(), 0.0, f, s)

    # pi_hat: (1/T) sum_t (x_ti x_tj - s_ij)^2 summed over i, j
    y = x**2
    phi_mat = (y.T @ y) / t - s**2
    pi_hat = float(phi_mat.sum())

    # theta terms: theta_ii,ij = (1/T) sum_t x_ti^3 x_tj - s_ii * s_ij
    term = (x**3).T @ x / t  # term[i, j] = (1/T) sum_t x_ti^3 x_tj
    theta_ii = term - np.diag(s)[:, None] * s  # theta_{ii,ij}
    theta_jj = theta_ii.T  # theta_{jj,ij} by symmetry of roles
    np.fill_diagonal(theta_ii, 0.0)
    np.fill_diagonal(theta_jj, 0.0)
    diag_safe = np.where(np.diag(s) == 0.0, 1.0, np.diag(s))  # zero-vol guard:
    # theta terms for a zero-variance asset are identically 0, so the
    # substitute denominator never touches a nonzero summand.
    ratio = np.sqrt(np.outer(1.0 / diag_safe, diag_safe))  # sqrt(s_jj/s_ii)
    rho_hat = float(np.trace(phi_mat)) + rbar * 0.5 * float(
        np.sum(ratio * theta_ii + ratio.T * theta_jj)
    )

    kappa = (pi_hat - rho_hat) / gamma
    delta = float(np.clip(kappa / t, 0.0, 1.0))
    shrunk = delta * f + (1.0 - delta) * s
    return LedoitWolfResult(shrunk, delta, f, s)


def single_factor_cov(
    returns: np.ndarray | pd.DataFrame,
    market: np.ndarray | None = None,
) -> np.ndarray:
    """Single-factor (market model) covariance: Sigma = var_m * b b' + D.

    Betas are OLS slopes of each asset on the market factor; D is the
    diagonal of residual variances. If no market series is supplied, the
    equal-weighted cross-sectional average return is used as the proxy.

    Parameters
    ----------
    returns : (T, N) array-like
        Simple per-period returns.
    market : (T,) array-like, optional
        Market factor returns; defaults to the equal-weighted asset average.

    Returns
    -------
    np.ndarray
        (N, N) structured covariance (always PSD by construction).
    """
    x = _as_matrix(returns)
    t = x.shape[0]
    if t < 3:
        raise ValueError(f"need at least 3 observations, got {t}")
    m = x.mean(axis=1) if market is None else np.asarray(market, dtype=float).ravel()
    if m.shape[0] != t:
        raise ValueError(f"market has {m.shape[0]} observations, returns have {t}")
    mc = m - m.mean()
    var_m = float(mc @ mc / (t - 1))
    if var_m <= 0.0:
        raise ValueError("market factor has zero variance; cannot fit factor model")
    xc = x - x.mean(axis=0)
    beta = xc.T @ mc / (mc @ mc)
    resid = xc - np.outer(mc, beta)
    d = resid.var(axis=0, ddof=2) if t > 2 else resid.var(axis=0, ddof=0)
    return var_m * np.outer(beta, beta) + np.diag(np.maximum(d, 0.0))


def psd_repair(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Repair a symmetric matrix to be positive semi-definite by
    eigenvalue clipping: eigenvalues below ``eps * max_eigenvalue`` are
    floored at that level and the matrix is rebuilt and re-symmetrised.

    Parameters
    ----------
    cov : (N, N) array-like
        Symmetric (or nearly symmetric) matrix.
    eps : float
        Relative eigenvalue floor (>= 0). ``eps > 0`` yields a strictly
        positive-definite result, making downstream inversion safe.

    Returns
    -------
    np.ndarray
        (N, N) symmetric PSD matrix.
    """
    a = np.asarray(cov, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"cov must be square, got shape {a.shape}")
    a = 0.5 * (a + a.T)
    vals, vecs = np.linalg.eigh(a)
    floor = eps * max(float(vals.max()), 0.0)
    if floor == 0.0:
        floor = eps  # all-zero / negative matrix: fall back to absolute floor
    clipped = np.maximum(vals, floor)
    out = (vecs * clipped) @ vecs.T
    return 0.5 * (out + out.T)


def condition_number(cov: np.ndarray) -> float:
    """2-norm condition number (max/min eigenvalue for a symmetric PSD
    matrix); ``np.inf`` when the smallest eigenvalue is <= 0."""
    a = np.asarray(cov, dtype=float)
    vals = np.linalg.eigvalsh(0.5 * (a + a.T))
    lo, hi = float(vals.min()), float(vals.max())
    if lo <= 0.0:
        return float("inf")
    return hi / lo


def is_psd(cov: np.ndarray, tol: float = 1e-12) -> bool:
    """True if all eigenvalues exceed ``-tol * max(|eigenvalue|, 1)``."""
    a = np.asarray(cov, dtype=float)
    vals = np.linalg.eigvalsh(0.5 * (a + a.T))
    scale = max(float(np.abs(vals).max()), 1.0)
    return bool(vals.min() >= -tol * scale)
