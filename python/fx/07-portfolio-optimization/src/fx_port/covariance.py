"""Covariance estimators for currency panels.

Implemented from scratch (numpy only):

* :func:`sample_cov` — unbiased sample covariance.
* :func:`ewma_cov` — RiskMetrics exponentially weighted covariance.
* :func:`lw_shrinkage` — Ledoit-Wolf (2004) "well-conditioned" shrinkage to a
  scaled-identity target, with the analytic optimal intensity in [0, 1].
* :func:`one_factor_cov` — risk-on/off single-factor model: every currency
  loads on one global risk factor (estimated as the first principal
  component, sign-oriented so risk-on currencies load positively) plus
  idiosyncratic variance.
* :func:`psd_repair` — eigenvalue clipping to restore positive
  (semi-)definiteness.

All returns are daily log returns; covariances are per-day (multiply by 252
to annualise).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _validate_returns(returns: pd.DataFrame, min_rows: int = 2) -> None:
    if returns.shape[0] < min_rows:
        raise ValueError(
            f"need at least {min_rows} return rows, got {returns.shape[0]}"
        )
    if returns.isna().any().any():
        raise ValueError("returns contain NaNs; clean or trim the panel first")
    if not np.all(np.isfinite(returns.to_numpy(dtype=float))):
        raise ValueError("returns contain Inf; clean or trim the panel first")


def sample_cov(returns: pd.DataFrame, ddof: int = 1) -> pd.DataFrame:
    """Unbiased sample covariance of a daily return panel."""
    _validate_returns(returns, min_rows=ddof + 1)
    x = returns.to_numpy()
    xc = x - x.mean(axis=0)
    cov = xc.T @ xc / (len(x) - ddof)
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def ewma_cov(
    returns: pd.DataFrame, lam: float = 0.94, init_window: int = 30
) -> pd.DataFrame:
    """RiskMetrics EWMA covariance ``S_t = lam*S_{t-1} + (1-lam)*r_t r_t'``.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns.
    lam : float
        Decay in (0, 1); 0.94 is the RiskMetrics daily standard.
    init_window : int
        The recursion is seeded with the sample covariance (ddof=0, no mean
        subtraction — RiskMetrics assumes zero mean) of the first
        ``min(init_window, T)`` rows, then run over the remaining rows.

    Returns
    -------
    pd.DataFrame
        Covariance at the final date.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must be in (0, 1), got {lam}")
    _validate_returns(returns)
    x = returns.to_numpy()
    w = min(init_window, len(x))
    s = x[:w].T @ x[:w] / w
    for t in range(w, len(x)):
        s = lam * s + (1 - lam) * np.outer(x[t], x[t])
    return pd.DataFrame(s, index=returns.columns, columns=returns.columns)


def lw_shrinkage(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf (2004) shrinkage to a scaled-identity target, from scratch.

    Sigma_LW = delta * m * I + (1 - delta) * S, where S is the (ddof=0)
    sample covariance, ``m = tr(S)/N`` and the optimal intensity is
    ``delta = min(b^2, d^2) / d^2`` with

    * ``d^2 = ||S - m I||_F^2``  (dispersion of S around the target),
    * ``b^2 = (1/T^2) sum_t ||x_t x_t' - S||_F^2``  (estimation noise in S).

    Returns
    -------
    (pd.DataFrame, float)
        Shrunk covariance and intensity delta, guaranteed in [0, 1].
    """
    _validate_returns(returns)
    x = returns.to_numpy()
    t_obs, n = x.shape
    xc = x - x.mean(axis=0)
    s = xc.T @ xc / t_obs
    m = np.trace(s) / n
    d2 = float(np.sum((s - m * np.eye(n)) ** 2))
    if d2 <= 1e-300:  # S already equals the target (e.g. iid scaled identity)
        return (
            pd.DataFrame(s, index=returns.columns, columns=returns.columns),
            0.0,
        )
    b2_bar = float(sum(np.sum((np.outer(r, r) - s) ** 2) for r in xc)) / t_obs**2
    b2 = min(b2_bar, d2)
    delta = b2 / d2
    sigma = delta * m * np.eye(n) + (1 - delta) * s
    return (
        pd.DataFrame(sigma, index=returns.columns, columns=returns.columns),
        float(delta),
    )


@dataclass
class OneFactorModel:
    """Risk-on/off single-factor covariance model.

    Attributes
    ----------
    loadings : pd.Series
        Regression loading of each currency on the risk factor.  Positive =
        risk-on currency (AUD/NZD/EM), negative = safe haven (JPY/CHF).
    factor : pd.Series
        Realised factor time series (first PC scores, sign-oriented so the
        factor correlates POSITIVELY with the cross-sectional mean currency
        return — 'risk-on' means the dollar falls vs the average currency).
    factor_var : float
        Factor variance (per day).
    resid_var : pd.Series
        Idiosyncratic variance per currency (floored at 0).
    cov : pd.DataFrame
        Implied covariance ``factor_var * b b' + diag(resid_var)``.
    """

    loadings: pd.Series
    factor: pd.Series
    factor_var: float
    resid_var: pd.Series
    cov: pd.DataFrame


def one_factor_cov(returns: pd.DataFrame) -> OneFactorModel:
    """Estimate the risk-on/off single-factor model from a return panel.

    The factor is the first principal component of the sample covariance;
    its sign is oriented so that it correlates positively with the
    equal-weighted currency return vs USD (dollar-down = risk-on).  Loadings
    and residual variances come from per-currency OLS on the factor.
    """
    _validate_returns(returns, min_rows=3)
    x = returns.to_numpy()
    xc = x - x.mean(axis=0)
    s = xc.T @ xc / len(x)
    eigval, eigvec = np.linalg.eigh(s)
    v1 = eigvec[:, -1]
    f = xc @ v1
    ew = xc.mean(axis=1)
    if float(f @ ew) < 0:
        f = -f
    fvar = float(f @ f) / len(f)
    if fvar <= 0:
        raise ValueError("degenerate factor (zero variance); check the panel")
    beta = xc.T @ f / (f @ f)
    resid = xc - np.outer(f, beta)
    rvar = np.maximum((resid**2).mean(axis=0), 0.0)
    cov = fvar * np.outer(beta, beta) + np.diag(rvar)
    cols = returns.columns
    return OneFactorModel(
        loadings=pd.Series(beta, index=cols, name="loading"),
        factor=pd.Series(f, index=returns.index, name="risk_factor"),
        factor_var=fvar,
        resid_var=pd.Series(rvar, index=cols, name="resid_var"),
        cov=pd.DataFrame(cov, index=cols, columns=cols),
    )


def is_psd(sigma: pd.DataFrame | np.ndarray, tol: float = 1e-10) -> bool:
    """True if the (symmetrised) matrix has all eigenvalues >= -tol."""
    a = np.asarray(sigma, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"sigma must be square, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("sigma contains NaN/Inf; eigenvalues are undefined")
    a = 0.5 * (a + a.T)
    return bool(np.linalg.eigvalsh(a).min() >= -tol)


def psd_repair(
    sigma: pd.DataFrame, min_eig: float = 0.0
) -> pd.DataFrame:
    """Repair a covariance matrix by symmetrising and clipping eigenvalues.

    Eigenvalues below ``min_eig`` are raised to ``min_eig``.  With
    ``min_eig = 0`` this is the Frobenius-nearest PSD matrix (Higham
    projection for symmetric input); a small positive floor (e.g. 1e-10)
    additionally makes the matrix invertible — needed when the universe
    contains a pegged (zero-vol) currency.

    Parameters
    ----------
    sigma : pd.DataFrame
        Square symmetric-ish matrix.
    min_eig : float
        Eigenvalue floor, must be >= 0.

    Returns
    -------
    pd.DataFrame
        Repaired matrix with the original labels.
    """
    if min_eig < 0:
        raise ValueError(f"min_eig must be >= 0, got {min_eig}")
    raw = sigma.to_numpy(dtype=float)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError(f"sigma must be square, got shape {raw.shape}")
    if not np.all(np.isfinite(raw)):
        raise ValueError(
            "sigma contains NaN/Inf; repair the estimation inputs first "
            "(a NaN eigen-decomposition returns an all-NaN matrix silently)"
        )
    a = 0.5 * (raw + raw.T)
    eigval, eigvec = np.linalg.eigh(a)
    clipped = np.maximum(eigval, min_eig)
    out = (eigvec * clipped) @ eigvec.T
    out = 0.5 * (out + out.T)
    return pd.DataFrame(out, index=sigma.index, columns=sigma.columns)
