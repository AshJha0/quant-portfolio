"""Gaussian hidden Markov model from scratch, in log space.

Implements the full classical toolkit:

* log-space forward/backward recursions (no underflow at any T),
* filtered probabilities P(s_t | x_{1..t}) — the ONLY probabilities a
  live trading system may use,
* smoothed posteriors P(s_t | x_{1..T}) for research,
* Baum-Welch EM with monotone log-likelihood and multiple restarts,
* Viterbi decoding,
* expected state durations 1/(1-A_ii) and the stationary distribution.

``hmmlearn`` is used only as a cross-check in the test suite.

Convention: rows of X are time-ordered observations; ``transmat[i, j]``
is P(s_{t+1}=j | s_t=i).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from .gmm import fit_gmm, gaussian_logpdf


@dataclass
class HMMModel:
    """Parameters of a fitted Gaussian HMM.

    Attributes
    ----------
    startprob : (k,) initial state distribution.
    transmat : (k, k) row-stochastic transition matrix.
    means : (k, p) emission means.
    covs : (k, p, p) emission covariances.
    log_likelihood : float — training log-likelihood at the optimum.
    log_likelihood_path : per-EM-iteration log-likelihoods (monotone).
    converged : bool
    n_iter : int
    """

    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    log_likelihood: float = np.nan
    log_likelihood_path: list[float] = field(default_factory=list)
    converged: bool = False
    n_iter: int = 0

    @property
    def k(self) -> int:
        return len(self.startprob)

    def emission_logprob(self, X: np.ndarray) -> np.ndarray:
        """(T, k) log emission densities."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.empty((X.shape[0], self.k))
        for j in range(self.k):
            out[:, j] = gaussian_logpdf(X, self.means[j], self.covs[j])
        return out


def _validate_model(model: HMMModel) -> None:
    if not np.allclose(model.transmat.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("transmat rows must sum to 1")
    if not np.isclose(model.startprob.sum(), 1.0, atol=1e-8):
        raise ValueError("startprob must sum to 1")


def log_forward(model: HMMModel, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Log-space forward recursion.

    Returns
    -------
    log_alpha : (T, k) with log_alpha[t, j] = log P(x_{1..t}, s_t=j).
    log_likelihood : float = logsumexp(log_alpha[-1]).
    """
    _validate_model(model)
    logb = model.emission_logprob(X)
    T, k = logb.shape
    with np.errstate(divide="ignore"):
        log_pi = np.log(model.startprob)
    A = model.transmat
    log_alpha = np.empty((T, k))
    log_alpha[0] = log_pi + logb[0]
    for t in range(1, T):
        c = log_alpha[t - 1].max()
        with np.errstate(divide="ignore"):
            log_alpha[t] = (
                c + np.log(np.exp(log_alpha[t - 1] - c) @ A) + logb[t]
            )
    return log_alpha, float(logsumexp(log_alpha[-1]))


def log_backward(model: HMMModel, X: np.ndarray) -> np.ndarray:
    """Log-space backward recursion.

    Returns
    -------
    log_beta : (T, k) with log_beta[t, i] = log P(x_{t+1..T} | s_t=i).
    """
    _validate_model(model)
    logb = model.emission_logprob(X)
    T, k = logb.shape
    A = model.transmat
    log_beta = np.zeros((T, k))
    for t in range(T - 2, -1, -1):
        u = logb[t + 1] + log_beta[t + 1]
        c = u.max()
        with np.errstate(divide="ignore"):
            log_beta[t] = c + np.log(A @ np.exp(u - c))
    return log_beta


def filtered_probabilities(model: HMMModel, X: np.ndarray) -> np.ndarray:
    """Filtered state probabilities P(s_t = j | x_{1..t}).

    These are the causal, tradeable probabilities: the value in row t
    depends ONLY on observations up to and including t (enforced by a
    mutation test in the suite).

    Returns
    -------
    (T, k) array, rows sum to 1.
    """
    log_alpha, _ = log_forward(model, X)
    return np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))


def smoothed_probabilities(model: HMMModel, X: np.ndarray) -> np.ndarray:
    """Smoothed posteriors P(s_t = j | x_{1..T}) — research only.

    Uses the full sample (future included); must never drive live
    positions.  Returns (T, k), rows sum to 1.
    """
    log_alpha, ll = log_forward(model, X)
    log_beta = log_backward(model, X)
    return np.exp(log_alpha + log_beta - ll)


def viterbi(model: HMMModel, X: np.ndarray) -> np.ndarray:
    """Most likely state path (Viterbi decoding).

    Returns
    -------
    (T,) int array of state indices.
    """
    _validate_model(model)
    logb = model.emission_logprob(X)
    T, k = logb.shape
    with np.errstate(divide="ignore"):
        log_pi = np.log(model.startprob)
        log_A = np.log(model.transmat)
    delta = np.empty((T, k))
    back = np.zeros((T, k), dtype=int)
    delta[0] = log_pi + logb[0]
    for t in range(1, T):
        cand = delta[t - 1][:, None] + log_A
        back[t] = np.argmax(cand, axis=0)
        delta[t] = cand[back[t], np.arange(k)] + logb[t]
    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def expected_durations(transmat: np.ndarray) -> np.ndarray:
    """Expected sojourn time per state: 1 / (1 - A_ii) days."""
    diag = np.clip(np.diag(np.asarray(transmat, dtype=float)), 0.0, 1.0 - 1e-12)
    return 1.0 / (1.0 - diag)


def stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """Stationary distribution pi with pi @ P = pi, pi >= 0, sum = 1.

    Computed from the left eigenvector of P for eigenvalue 1.
    """
    P = np.asarray(transmat, dtype=float)
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("transmat rows must sum to 1")
    w, v = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(w - 1.0)))
    pi = np.real(v[:, idx])
    pi = np.abs(pi)
    return pi / pi.sum()


def fit_hmm(
    X: np.ndarray,
    k: int,
    seed: int = 0,
    n_init: int = 3,
    max_iter: int = 100,
    tol: float = 1e-6,
    reg_covar: float = 1e-6,
    init_model: HMMModel | None = None,
    self_bias: float = 0.9,
) -> HMMModel:
    """Fit a Gaussian HMM by Baum-Welch EM.

    Initialisation: a short from-scratch GMM fit provides means /
    covariances; the transition matrix starts sticky (``self_bias`` on
    the diagonal).  ``init_model`` warm-starts from a previous fit
    (used by the expanding-window detector), in which case restarts are
    skipped.

    Parameters
    ----------
    X : (T, p) time-ordered observations.
    k : number of hidden states (>= 1).
    seed, n_init, max_iter, tol, reg_covar : EM controls.
    init_model : optional HMMModel warm start.
    self_bias : initial diagonal weight of the transition matrix.

    Returns
    -------
    HMMModel

    Raises
    ------
    ValueError
        If k < 1 or the series is shorter than max(10, 2k) observations.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    T, p = X.shape
    if k < 1:
        raise ValueError("k must be >= 1")
    if T < max(10, 2 * k):
        raise ValueError(f"series too short for HMM: T={T}, k={k}")

    inits: list[HMMModel] = []
    if init_model is not None:
        if init_model.k != k:
            raise ValueError("init_model has wrong number of states")
        inits.append(init_model)
    else:
        for i in range(max(1, n_init)):
            g = fit_gmm(X, k, seed=seed + i, n_init=1, max_iter=25,
                        reg_covar=reg_covar)
            A = np.full((k, k), (1.0 - self_bias) / max(k - 1, 1))
            np.fill_diagonal(A, self_bias if k > 1 else 1.0)
            inits.append(
                HMMModel(
                    startprob=np.full(k, 1.0 / k),
                    transmat=A,
                    means=g.means.copy(),
                    covs=g.covs.copy(),
                )
            )

    best: HMMModel | None = None
    for init in inits:
        model = HMMModel(
            startprob=init.startprob.copy(),
            transmat=init.transmat.copy(),
            means=init.means.copy(),
            covs=init.covs.copy(),
        )
        path: list[float] = []
        prev_ll = -np.inf
        converged = False
        it = 0
        for it in range(1, max_iter + 1):
            logb = model.emission_logprob(X)
            log_alpha, ll = log_forward(model, X)
            log_beta = log_backward(model, X)
            path.append(ll)
            if ll - prev_ll < tol and it > 1:
                converged = True
                break
            prev_ll = ll
            gamma = np.exp(log_alpha + log_beta - ll)  # (T, k)
            with np.errstate(divide="ignore"):
                log_A = np.log(model.transmat)
            # xi summed over t, in log space per (i, j)
            if k > 1 and T > 1:
                # (T-1, k, k) log xi_t, summed over t via logsumexp
                m = (
                    log_alpha[:-1, :, None]
                    + log_A[None, :, :]
                    + (logb[1:] + log_beta[1:])[:, None, :]
                    - ll
                )
                xi_sum = np.exp(logsumexp(m, axis=0))
                denom = gamma[:-1].sum(axis=0)[:, None] + 1e-300
                model.transmat = xi_sum / denom
                model.transmat /= model.transmat.sum(axis=1, keepdims=True)
            model.startprob = gamma[0] / gamma[0].sum()
            nk = gamma.sum(axis=0) + 1e-300
            model.means = (gamma.T @ X) / nk[:, None]
            for j in range(k):
                dev = X - model.means[j]
                model.covs[j] = (gamma[:, j][:, None] * dev).T @ dev / nk[j]
                model.covs[j] += reg_covar * np.eye(p)
        model.log_likelihood = path[-1]
        model.log_likelihood_path = path
        model.converged = converged
        model.n_iter = it
        if best is None or model.log_likelihood > best.log_likelihood:
            best = model
    assert best is not None
    return best


def hmm_bic(model: HMMModel, X: np.ndarray) -> float:
    """BIC for a fitted HMM (lower is better).

    Free parameters: (k-1) start + k(k-1) transition + k p means
    + k p(p+1)/2 covariances.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    T, p = X.shape
    k = model.k
    n_params = (k - 1) + k * (k - 1) + k * p + k * p * (p + 1) // 2
    _, ll = log_forward(model, X)
    return -2.0 * ll + n_params * np.log(T)


def match_states(
    true_means: np.ndarray,
    est_means: np.ndarray,
    true_covs: np.ndarray | None = None,
    est_covs: np.ndarray | None = None,
) -> np.ndarray:
    """Optimal permutation mapping estimated states to true states.

    Solves the assignment problem on squared mean distances; if
    covariances are supplied, the cost is the squared Frobenius
    distance between covariances INSTEAD.  For daily FX returns,
    drifts are tiny relative to estimation noise, so matching on
    covariances (where regimes really differ — vol and correlation)
    is far more robust.

    Returns
    -------
    perm : (k,) int array with perm[est_state] = true_state.
    """
    from scipy.optimize import linear_sum_assignment

    true_means = np.atleast_2d(true_means)
    est_means = np.atleast_2d(est_means)
    if true_means.shape != est_means.shape:
        raise ValueError("mean arrays must have the same shape")
    if true_covs is not None and est_covs is not None:
        k = true_means.shape[0]
        cost = np.empty((k, k))
        for i in range(k):
            for j in range(k):
                cost[i, j] = ((est_covs[i] - true_covs[j]) ** 2).sum()
    else:
        cost = ((est_means[:, None, :] - true_means[None, :, :]) ** 2).sum(
            axis=2
        )
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(len(rows), dtype=int)
    perm[rows] = cols
    return perm
