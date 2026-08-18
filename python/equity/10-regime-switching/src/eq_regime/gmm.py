"""Gaussian mixture model from scratch (EM, full covariance).

Implements the classic EM algorithm with:

* full covariance matrices per component, ridge-regularised with
  ``reg_covar`` on the diagonal so degenerate (singleton / collapsed)
  clusters remain positive-definite;
* multiple seeded initialisations (k-means-lite + random restarts), keeping
  the best final log-likelihood;
* per-iteration log-likelihood history — EM guarantees monotone
  non-decreasing log-likelihood, and the test suite asserts this for EVERY
  iteration;
* BIC / AIC for selecting the number of components.

Cross-check: on identical data the converged log-likelihood agrees with
``sklearn.mixture.GaussianMixture`` within a documented tolerance (tests).
Label-permutation-invariant comparison utilities are provided because
mixture labels are arbitrary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np
from scipy.special import logsumexp

__all__ = [
    "GMMFit",
    "fit_gmm",
    "gmm_log_likelihood",
    "bic",
    "aic",
    "select_k_bic",
    "match_permutation",
    "kmeans_lite",
]

_LOG_2PI = np.log(2.0 * np.pi)


def _log_gaussian_pdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log density of N(mean, cov) at each row of x, via Cholesky.

    Parameters
    ----------
    x : (T x D) observations.
    mean : (D,) mean vector.
    cov : (D x D) positive-definite covariance.

    Returns
    -------
    (T,) log densities.
    """
    d = len(mean)
    chol = np.linalg.cholesky(cov)
    dev = x - mean
    sol = np.linalg.solve(chol, dev.T)  # (D x T)
    maha = np.sum(sol**2, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    return -0.5 * (d * _LOG_2PI + log_det + maha)


def kmeans_lite(
    x: np.ndarray, k: int, rng: np.random.Generator, n_iter: int = 20
) -> np.ndarray:
    """Minimal Lloyd's k-means, returning cluster means (k x D).

    Initialises with k distinct random observations; empty clusters are
    re-seeded from random points.  Used only to initialise EM — a handful of
    iterations is enough.
    """
    t_len = x.shape[0]
    if k > t_len:
        raise ValueError(f"k ({k}) exceeds number of observations ({t_len})")
    idx = rng.choice(t_len, size=k, replace=False)
    centers = x[idx].copy()
    for _ in range(n_iter):
        d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = np.argmin(d2, axis=1)
        new_centers = centers.copy()
        for j in range(k):
            mask = assign == j
            if mask.any():
                new_centers[j] = x[mask].mean(axis=0)
            else:
                new_centers[j] = x[rng.integers(t_len)]
        if np.allclose(new_centers, centers, atol=1e-12):
            centers = new_centers
            break
        centers = new_centers
    return centers


@dataclass
class GMMFit:
    """Converged GMM parameters and diagnostics.

    Attributes
    ----------
    weights : (K,) mixing proportions, sum to 1.
    means : (K x D) component means.
    covariances : (K x D x D) full covariance matrices.
    log_likelihood : float
        Total log-likelihood of the training data at convergence.
    log_likelihood_history : list of float
        Log-likelihood after each EM iteration of the winning
        initialisation (monotone non-decreasing).
    n_iter : int
        Iterations run by the winning initialisation.
    converged : bool
        Whether the tolerance was met before ``max_iter``.
    n_params : int
        Free-parameter count used by BIC/AIC.
    """

    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    log_likelihood: float
    log_likelihood_history: list[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False
    n_params: int = 0

    @property
    def n_components(self) -> int:
        return len(self.weights)

    def log_responsibilities(self, x: np.ndarray) -> np.ndarray:
        """(T x K) log posterior responsibilities for observations x."""
        log_prob = np.stack(
            [
                np.log(self.weights[k]) + _log_gaussian_pdf(x, self.means[k], self.covariances[k])
                for k in range(self.n_components)
            ],
            axis=1,
        )
        return log_prob - logsumexp(log_prob, axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """(T,) hard component assignments (argmax responsibility)."""
        return np.argmax(self.log_responsibilities(x), axis=1)


def gmm_log_likelihood(
    x: np.ndarray, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray
) -> float:
    """Total log-likelihood of x under the given mixture parameters."""
    log_prob = np.stack(
        [
            np.log(weights[k]) + _log_gaussian_pdf(x, means[k], covariances[k])
            for k in range(len(weights))
        ],
        axis=1,
    )
    return float(logsumexp(log_prob, axis=1).sum())


def _em_run(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_iter: int,
    tol: float,
    reg_covar: float,
    init: str,
) -> GMMFit:
    """One EM run from one initialisation."""
    t_len, d = x.shape
    if init == "kmeans":
        means = kmeans_lite(x, k, rng)
    else:
        means = x[rng.choice(t_len, size=k, replace=False)].copy()
    global_cov = np.cov(x.T, ddof=1).reshape(d, d) + reg_covar * np.eye(d)
    covs = np.array([global_cov.copy() for _ in range(k)])
    weights = np.full(k, 1.0 / k)

    history: list[float] = []
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        # E-step
        log_prob = np.stack(
            [np.log(weights[j]) + _log_gaussian_pdf(x, means[j], covs[j]) for j in range(k)],
            axis=1,
        )
        log_norm = logsumexp(log_prob, axis=1)
        ll = float(log_norm.sum())
        resp = np.exp(log_prob - log_norm[:, None])
        history.append(ll)
        if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
            converged = True
            break
        # M-step
        nk = resp.sum(axis=0)
        nk = np.maximum(nk, 10.0 * np.finfo(float).tiny)
        weights = nk / t_len
        means = (resp.T @ x) / nk[:, None]
        for j in range(k):
            dev = x - means[j]
            covs[j] = (resp[:, j][:, None] * dev).T @ dev / nk[j]
            covs[j] += reg_covar * np.eye(d)

    n_params = (k - 1) + k * d + k * d * (d + 1) // 2
    return GMMFit(
        weights=weights,
        means=means,
        covariances=covs,
        log_likelihood=history[-1],
        log_likelihood_history=history,
        n_iter=it,
        converged=converged,
        n_params=n_params,
    )


def fit_gmm(
    x: np.ndarray,
    n_components: int,
    seed: int = 0,
    n_init: int = 4,
    max_iter: int = 300,
    tol: float = 1e-7,
    reg_covar: float = 1e-6,
) -> GMMFit:
    """Fit a full-covariance Gaussian mixture by EM with restarts.

    Runs ``n_init`` initialisations (first from k-means-lite, remainder from
    random observations) and returns the fit with the best final
    log-likelihood.

    Parameters
    ----------
    x : (T x D) observations (a 1-D array is treated as (T x 1)).
    n_components : int
        Number of mixture components, >= 1.
    seed : int
        Master seed; each restart derives its own generator.
    n_init : int
        Number of initialisations, >= 1.
    max_iter, tol : EM stopping rule (absolute log-likelihood change).
    reg_covar : float
        Ridge added to every covariance diagonal each M-step; keeps
        singleton/degenerate clusters positive-definite.

    Returns
    -------
    GMMFit

    Raises
    ------
    ValueError
        For invalid shapes, ``n_components < 1``, or T < n_components.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("x must be 1-D or 2-D")
    if np.isnan(x).any():
        raise ValueError("x contains NaN")
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    if x.shape[0] < max(2 * n_components, x.shape[1] + 1):
        raise ValueError(
            f"too few observations ({x.shape[0]}) for {n_components} components "
            f"in {x.shape[1]} dimensions"
        )
    if n_init < 1:
        raise ValueError(f"n_init must be >= 1, got {n_init}")

    best: GMMFit | None = None
    for i in range(n_init):
        rng = np.random.default_rng(seed + 1000 * i)
        init = "kmeans" if i == 0 else "random"
        fit = _em_run(x, n_components, rng, max_iter, tol, reg_covar, init)
        if best is None or fit.log_likelihood > best.log_likelihood:
            best = fit
    assert best is not None
    return best


def bic(fit: GMMFit, n_obs: int) -> float:
    """Bayesian information criterion: ``-2 ll + n_params ln T`` (lower = better)."""
    return -2.0 * fit.log_likelihood + fit.n_params * np.log(n_obs)


def aic(fit: GMMFit, n_obs: int) -> float:
    """Akaike information criterion: ``-2 ll + 2 n_params`` (lower = better)."""
    return -2.0 * fit.log_likelihood + 2.0 * fit.n_params


def select_k_bic(
    x: np.ndarray,
    k_range: tuple[int, ...] = (1, 2, 3, 4, 5),
    seed: int = 0,
    **fit_kwargs,
) -> tuple[int, dict[int, float]]:
    """Select the number of components by BIC.

    Parameters
    ----------
    x : (T x D) observations.
    k_range : candidate component counts.
    seed : master seed passed to each fit.
    **fit_kwargs : forwarded to :func:`fit_gmm`.

    Returns
    -------
    (best_k, {k: bic_value})
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    scores: dict[int, float] = {}
    for k in k_range:
        fit = fit_gmm(x, k, seed=seed, **fit_kwargs)
        scores[k] = bic(fit, x.shape[0])
    best_k = min(scores, key=scores.get)
    return best_k, scores


def match_permutation(true_means: np.ndarray, est_means: np.ndarray) -> tuple[int, ...]:
    """Best label permutation aligning estimated components with truth.

    Minimises the total Euclidean distance between ``true_means[i]`` and
    ``est_means[perm[i]]`` over all permutations (K! search — fine for the
    small K used in regime models).

    Returns
    -------
    tuple
        ``perm`` such that estimated component ``perm[i]`` corresponds to
        true component ``i``.
    """
    true_means = np.atleast_2d(np.asarray(true_means, dtype=float))
    est_means = np.atleast_2d(np.asarray(est_means, dtype=float))
    if true_means.shape != est_means.shape:
        raise ValueError("mean arrays must have identical shapes")
    k = true_means.shape[0]
    best_perm: tuple[int, ...] | None = None
    best_cost = np.inf
    for perm in permutations(range(k)):
        cost = sum(
            float(np.linalg.norm(true_means[i] - est_means[perm[i]])) for i in range(k)
        )
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    assert best_perm is not None
    return best_perm
