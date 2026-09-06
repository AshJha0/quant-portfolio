"""Monte Carlo VaR: simulate factor returns, fully revalue, take the quantile.

Factor models: multivariate normal, or multivariate Student-t (a t copula
with t margins of common df, built as normal / sqrt(chi2/df)), both matched
to a target covariance.  Cholesky with a jitter fallback handles
near-singular covariance matrices (e.g. perfectly correlated factors).

All simulation takes an explicit seed or ``numpy.random.Generator``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import beta as beta_dist

from .portfolio import Portfolio

__all__ = [
    "safe_cholesky",
    "simulate_factor_returns",
    "monte_carlo_pnl",
    "monte_carlo_var",
    "var_standard_error_bootstrap",
    "var_confidence_interval",
]


def _as_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def safe_cholesky(cov: np.ndarray, jitter: float = 1e-10, max_tries: int = 12) -> np.ndarray:
    """Cholesky factor of a covariance matrix with diagonal-jitter fallback.

    For exactly or numerically singular matrices (perfectly correlated
    factors, zero-variance factors) plain Cholesky fails; we add
    ``jitter * mean(diag)`` to the diagonal, escalating by x10 up to
    ``max_tries`` times.  The perturbation is tiny relative to the variances,
    so simulated moments are unchanged to within MC noise.

    Returns lower-triangular ``L`` with ``L L' ~= cov``.
    """
    sig = np.atleast_2d(np.asarray(cov, dtype=float))
    if sig.shape[0] != sig.shape[1]:
        raise ValueError(f"covariance must be square, got shape {sig.shape}")
    # Check finiteness *before* symmetry: allclose(NaN, NaN) is False, so a
    # NaN matrix would otherwise be reported as "not symmetric".
    if not np.isfinite(sig).all():
        raise ValueError("covariance contains NaN or infinite values")
    if not np.allclose(sig, sig.T, atol=1e-12 * max(1.0, np.abs(sig).max())):
        raise ValueError("covariance matrix must be symmetric")
    try:
        return np.linalg.cholesky(sig)
    except np.linalg.LinAlgError:
        pass
    scale = float(np.mean(np.diag(sig)))
    if scale <= 0.0:
        scale = 1.0
    eps = jitter * scale
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(sig + eps * np.eye(sig.shape[0]))
        except np.linalg.LinAlgError:
            eps *= 10.0
    raise np.linalg.LinAlgError(
        "Cholesky failed even with jitter; covariance matrix is badly indefinite"
    )


def simulate_factor_returns(
    cov: np.ndarray,
    n_paths: int,
    dist: Literal["normal", "t"] = "normal",
    df: float = 6.0,
    mean: np.ndarray | None = None,
    seed: int | np.random.Generator | None = 0,
) -> np.ndarray:
    """Simulate factor-return scenarios with a target covariance.

    Parameters
    ----------
    cov : (n, n) daily factor-return covariance (target of the simulation).
    n_paths : number of scenarios.
    dist : {"normal", "t"}
        ``"t"``: multivariate Student-t via ``Z / sqrt(W/df)`` with the scale
        matrix set to ``cov * (df-2)/df`` so the *covariance* matches ``cov``
        exactly while the tails fatten.
    df : degrees of freedom for ``dist="t"`` (> 2 required).
    mean : optional (n,) mean vector, default zero.
    seed : int or Generator — simulation is fully reproducible.

    Returns
    -------
    ndarray (n_paths, n) of factor returns.
    """
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    rng = _as_rng(seed)
    sig = np.atleast_2d(np.asarray(cov, dtype=float))
    n = sig.shape[0]
    mu = np.zeros(n) if mean is None else np.asarray(mean, dtype=float).ravel()
    if mu.size != n:
        raise ValueError(f"mean has {mu.size} entries, covariance is {n}x{n}")
    if dist == "normal":
        chol = safe_cholesky(sig)
        z = rng.standard_normal((n_paths, n))
        return mu + z @ chol.T
    if dist == "t":
        if df <= 2:
            raise ValueError(f"Student-t df must be > 2 for finite variance, got {df}")
        chol = safe_cholesky(sig * (df - 2.0) / df)
        z = rng.standard_normal((n_paths, n))
        w = rng.chisquare(df, size=n_paths) / df
        return mu + (z @ chol.T) / np.sqrt(w)[:, None]
    raise ValueError(f"dist must be 'normal' or 't', got {dist!r}")


def monte_carlo_pnl(
    portfolio: Portfolio,
    cov: np.ndarray,
    n_paths: int = 100_000,
    dist: Literal["normal", "t"] = "normal",
    df: float = 6.0,
    method: Literal["full", "delta_gamma"] = "full",
    seed: int | np.random.Generator | None = 0,
) -> np.ndarray:
    """Simulate factor scenarios and revalue the portfolio (full reval default)."""
    scen = simulate_factor_returns(cov, n_paths, dist=dist, df=df, seed=seed)
    return portfolio.pnl(scen, method=method)


def monte_carlo_var(
    portfolio: Portfolio,
    cov: np.ndarray,
    alpha: float = 0.01,
    n_paths: int = 100_000,
    dist: Literal["normal", "t"] = "normal",
    df: float = 6.0,
    method: Literal["full", "delta_gamma"] = "full",
    seed: int | np.random.Generator | None = 0,
) -> float:
    """Monte Carlo VaR (positive for a loss) with full portfolio revaluation."""
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
    pnl = monte_carlo_pnl(portfolio, cov, n_paths, dist, df, method, seed)
    return float(-np.quantile(pnl, alpha, method="linear"))


def var_standard_error_bootstrap(
    pnl: np.ndarray,
    alpha: float = 0.01,
    n_boot: int = 500,
    seed: int | np.random.Generator | None = 0,
) -> float:
    """Bootstrap standard error of the empirical VaR quantile estimate.

    Resamples the scenario P&L with replacement ``n_boot`` times and returns
    the standard deviation of the re-estimated VaR.  Distribution-free and
    the desk-standard way to attach error bars to an MC or historical VaR.
    """
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
    if n_boot < 2:
        raise ValueError(f"n_boot must be >= 2, got {n_boot}")
    rng = _as_rng(seed)
    arr = np.asarray(pnl, dtype=float).ravel()
    if arr.size < 10:
        raise ValueError(f"need at least 10 observations to bootstrap, got {arr.size}")
    if not np.isfinite(arr).all():
        raise ValueError("pnl contains NaN or infinite values")
    # Resample in chunks so memory stays O(chunk * n) instead of materialising
    # an (n_boot, n) index matrix plus the gathered floats (~800 MB at the
    # default 100k paths x 500 resamples).  Cost is identical.
    chunk = max(1, min(n_boot, 50))
    boot_vars = np.empty(n_boot)
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rng.integers(0, arr.size, size=(m, arr.size))
        boot_vars[done : done + m] = -np.quantile(arr[idx], alpha, axis=1, method="linear")
        done += m
    return float(np.std(boot_vars, ddof=1))


def var_confidence_interval(
    pnl: np.ndarray, alpha: float = 0.01, conf: float = 0.95
) -> tuple[float, float]:
    """Order-statistics (distribution-free) confidence interval for VaR.

    The rank of the true ``alpha`` quantile among ``n`` i.i.d. draws is
    Binomial(n, alpha); we invert it via the beta distribution of order
    statistics to get ranks ``(lo, hi)`` such that
    ``P(X_(lo) <= q_alpha <= X_(hi)) >= conf``, and return the corresponding
    VaR bracket ``(-X_(hi), -X_(lo))`` as (lower, upper) positive losses.
    """
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
    if not 0.5 < conf < 1.0:
        raise ValueError(f"conf must be in (0.5, 1), got {conf}")
    arr = np.sort(np.asarray(pnl, dtype=float).ravel())
    n = arr.size
    if n < 20:
        raise ValueError(f"need at least 20 observations, got {n}")
    tail = (1.0 - conf) / 2.0
    ranks = np.arange(1, n + 1)
    # P(X_(k) <= q_alpha) = P(Beta(k, n-k+1) <= alpha) is decreasing in k
    cdf_at_alpha = beta_dist.cdf(alpha, ranks, n - ranks + 1)
    lo_candidates = np.where(cdf_at_alpha >= 1.0 - tail)[0]
    hi_candidates = np.where(cdf_at_alpha <= tail)[0]
    lo_rank = int(lo_candidates[-1]) if lo_candidates.size else 0
    hi_rank = int(hi_candidates[0]) if hi_candidates.size else n - 1
    return float(-arr[hi_rank]), float(-arr[lo_rank])
