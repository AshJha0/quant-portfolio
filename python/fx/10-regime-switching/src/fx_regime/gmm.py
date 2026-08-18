"""Gaussian mixture model fitted by EM, from scratch.

Full-covariance GMM with log-space responsibilities (logsumexp), BIC/AIC
model selection, multiple restarts.  ``sklearn.mixture.GaussianMixture``
is used only as a cross-check in the test suite.

The GMM serves two purposes in the pipeline: (i) a static (no-dynamics)
regime clustering baseline, and (ii) BIC-based selection of the number
of regimes k before fitting the HMM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp


def gaussian_logpdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Multivariate normal log-density via Cholesky.

    Parameters
    ----------
    X : (n, p), mean : (p,), cov : (p, p) SPD.

    Returns
    -------
    (n,) log-densities.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    p = X.shape[1]
    L = np.linalg.cholesky(cov)
    dev = X - mean
    sol = np.linalg.solve(L, dev.T)  # (p, n)
    maha = np.sum(sol**2, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (p * np.log(2.0 * np.pi) + logdet + maha)


@dataclass
class GMMResult:
    """Fitted Gaussian mixture.

    Attributes
    ----------
    weights : (k,) mixing proportions.
    means : (k, p) component means.
    covs : (k, p, p) component covariances.
    log_likelihood : float — total log-likelihood of the training data.
    log_likelihood_path : list of per-iteration log-likelihoods
        (monotone non-decreasing up to tolerance).
    converged : bool
    n_iter : int
    """

    weights: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    log_likelihood: float
    log_likelihood_path: list[float] = field(default_factory=list)
    converged: bool = False
    n_iter: int = 0

    @property
    def k(self) -> int:
        return len(self.weights)

    def n_parameters(self, p: int | None = None) -> int:
        """Free parameters: (k-1) weights + k*p means + k*p(p+1)/2 covs."""
        if p is None:
            p = self.means.shape[1]
        k = self.k
        return (k - 1) + k * p + k * p * (p + 1) // 2

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Per-observation log-likelihood under the mixture."""
        return logsumexp(self._weighted_logpdf(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Responsibilities (n, k), rows sum to 1."""
        wl = self._weighted_logpdf(X)
        return np.exp(wl - logsumexp(wl, axis=1, keepdims=True))

    def _weighted_logpdf(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.empty((X.shape[0], self.k))
        for j in range(self.k):
            out[:, j] = np.log(self.weights[j]) + gaussian_logpdf(
                X, self.means[j], self.covs[j]
            )
        return out

    def bic(self, X: np.ndarray) -> float:
        """Bayesian information criterion (lower is better)."""
        n, p = np.atleast_2d(X).shape
        return -2.0 * float(self.score_samples(X).sum()) + self.n_parameters(
            p
        ) * np.log(n)

    def aic(self, X: np.ndarray) -> float:
        """Akaike information criterion (lower is better)."""
        p = np.atleast_2d(X).shape[1]
        return -2.0 * float(self.score_samples(X).sum()) + 2.0 * self.n_parameters(p)


def _init_params(
    X: np.ndarray, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K-means-style init: Lloyd iterations from random seed points.

    Hard-assignment means break the symmetric centroid saddle that a
    shared broad covariance init falls into; per-cluster covariances
    (with a pooled fallback for tiny clusters) start EM close to a
    sensible local optimum.
    """
    n, p = X.shape
    idx = rng.choice(n, size=k, replace=False)
    means = X[idx].copy()
    assign = np.zeros(n, dtype=int)
    for _ in range(10):
        d2 = ((X[:, None, :] - means[None, :, :]) ** 2).sum(axis=2)
        assign = np.argmin(d2, axis=1)
        for j in range(k):
            if (assign == j).any():
                means[j] = X[assign == j].mean(axis=0)
    pooled = np.cov(X.T, ddof=1).reshape(p, p) + 1e-6 * np.eye(p)
    covs = np.empty((k, p, p))
    weights = np.empty(k)
    for j in range(k):
        members = X[assign == j]
        weights[j] = max(len(members), 1) / n
        if len(members) > p + 1:
            covs[j] = np.cov(members.T, ddof=1).reshape(p, p) + 1e-6 * np.eye(p)
        else:
            covs[j] = pooled
    weights /= weights.sum()
    return weights, means, covs


def fit_gmm(
    X: np.ndarray,
    k: int,
    seed: int = 0,
    n_init: int = 3,
    max_iter: int = 200,
    tol: float = 1e-6,
    reg_covar: float = 1e-6,
) -> GMMResult:
    """Fit a full-covariance GMM by EM with restarts.

    Parameters
    ----------
    X : (n, p) data.
    k : number of components (>= 1).
    seed : RNG seed for initialisations.
    n_init : number of random restarts; the best log-likelihood wins.
    max_iter : EM iteration cap per restart.
    tol : absolute log-likelihood improvement for convergence.
    reg_covar : ridge added to covariance diagonals each M-step.

    Returns
    -------
    GMMResult

    Raises
    ------
    ValueError
        If k < 1 or there are fewer observations than components.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, p = X.shape
    if k < 1:
        raise ValueError("k must be >= 1")
    if n < max(2, k):
        raise ValueError(f"need at least {max(2, k)} observations for k={k}")
    rng = np.random.default_rng(seed)

    best: GMMResult | None = None
    for _ in range(max(1, n_init)):
        weights, means, covs = _init_params(X, k, rng)
        path: list[float] = []
        prev_ll = -np.inf
        converged = False
        it = 0
        for it in range(1, max_iter + 1):
            # E-step (log space)
            wl = np.empty((n, k))
            for j in range(k):
                wl[:, j] = np.log(weights[j]) + gaussian_logpdf(
                    X, means[j], covs[j]
                )
            norm = logsumexp(wl, axis=1)
            ll = float(norm.sum())
            path.append(ll)
            resp = np.exp(wl - norm[:, None])
            if ll - prev_ll < tol and it > 1:
                converged = True
                break
            prev_ll = ll
            # M-step
            nk = resp.sum(axis=0) + 1e-300
            weights = nk / n
            means = (resp.T @ X) / nk[:, None]
            for j in range(k):
                dev = X - means[j]
                covs[j] = (resp[:, j][:, None] * dev).T @ dev / nk[j]
                covs[j] += reg_covar * np.eye(p)
        result = GMMResult(
            weights=weights,
            means=means,
            covs=covs,
            log_likelihood=path[-1],
            log_likelihood_path=path,
            converged=converged,
            n_iter=it,
        )
        if best is None or result.log_likelihood > best.log_likelihood:
            best = result
    assert best is not None
    return best


def select_k_bic(
    X: np.ndarray,
    k_max: int = 4,
    seed: int = 0,
    n_init: int = 3,
    max_iter: int = 200,
) -> tuple[int, dict[int, float]]:
    """Select the number of mixture components by BIC.

    Returns
    -------
    (best_k, {k: bic}) — best_k has the lowest BIC.
    """
    if k_max < 1:
        raise ValueError("k_max must be >= 1")
    bics: dict[int, float] = {}
    for k in range(1, k_max + 1):
        model = fit_gmm(X, k, seed=seed, n_init=n_init, max_iter=max_iter)
        bics[k] = model.bic(X)
    best_k = min(bics, key=bics.get)
    return best_k, bics
