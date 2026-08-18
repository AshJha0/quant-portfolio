"""Gaussian hidden Markov model from scratch.

Implements the full HMM toolkit used by the regime pipeline:

* scaled forward–backward (Rabiner scaling with per-step log-emission
  shifts, so nothing under/overflows even for extreme outliers);
* Baum–Welch EM for the transition matrix, state means and full
  covariances (log-likelihood monotone non-decreasing — test-enforced at
  every iteration);
* Viterbi decoding in log space;
* the stationary distribution of the fitted chain (``pi P = pi`` to 1e-12);
* expected regime durations ``1 / (1 - p_ii)``;
* FILTERED state probabilities ``P(s_t | x_{1..t})`` — the only quantity a
  live trading system may use (see :mod:`eq_regime.detection` for the
  filtered-vs-smoothed distinction);
* initialisation by k-means-lite plus random restarts.

Cross-check: loading the fitted parameters into ``hmmlearn.hmm.GaussianHMM``
and scoring the same data reproduces our log-likelihood within tolerance
(tests/test_hmm.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gmm import _log_gaussian_pdf, kmeans_lite

__all__ = [
    "HMMFit",
    "fit_hmm",
    "forward_filter",
    "forward_backward",
    "viterbi",
    "stationary_distribution",
    "expected_durations",
]


def _log_emissions(x: np.ndarray, means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """(T x K) log emission densities."""
    return np.stack(
        [_log_gaussian_pdf(x, means[k], covs[k]) for k in range(len(means))], axis=1
    )


def _scaled_forward(
    log_b: np.ndarray, startprob: np.ndarray, transmat: np.ndarray
) -> tuple[np.ndarray, float]:
    """Scaled forward pass.

    Returns
    -------
    (alpha_hat, log_likelihood)
        ``alpha_hat[t, k] = P(s_t = k | x_{1..t})`` — the FILTERED
        probabilities — and the total log-likelihood ``log P(x_{1..T})``.
    """
    t_len, k = log_b.shape
    shift = log_b.max(axis=1)
    b = np.exp(log_b - shift[:, None])
    alpha_hat = np.empty((t_len, k))
    log_c = np.empty(t_len)
    a = startprob * b[0]
    c = a.sum()
    if c <= 0.0:
        raise FloatingPointError("forward pass underflow at t=0")
    alpha_hat[0] = a / c
    log_c[0] = np.log(c)
    for t in range(1, t_len):
        a = (alpha_hat[t - 1] @ transmat) * b[t]
        c = a.sum()
        if c <= 0.0:
            raise FloatingPointError(f"forward pass underflow at t={t}")
        alpha_hat[t] = a / c
        log_c[t] = np.log(c)
    return alpha_hat, float(log_c.sum() + shift.sum())


def _scaled_backward(
    log_b: np.ndarray, transmat: np.ndarray, alpha_hat: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scaled backward pass and posteriors.

    Returns
    -------
    (gamma, xi_sum, beta_hat)
        ``gamma[t, k] = P(s_t = k | x_{1..T})`` (SMOOTHED posteriors),
        ``xi_sum[i, j] = sum_t P(s_t = i, s_{t+1} = j | x_{1..T})``.
    """
    t_len, k = log_b.shape
    shift = log_b.max(axis=1)
    b = np.exp(log_b - shift[:, None])
    # Reconstruct scaling constants from alpha_hat recursion for consistency.
    beta_hat = np.empty((t_len, k))
    beta_hat[-1] = 1.0
    xi_sum = np.zeros((k, k))
    for t in range(t_len - 2, -1, -1):
        m = b[t + 1] * beta_hat[t + 1]
        pred = alpha_hat[t] @ transmat  # proportional to P(s_{t+1} | x_{1..t})
        c_next = float((pred * b[t + 1]).sum())
        beta_hat[t] = (transmat @ m) / c_next
        xi = (alpha_hat[t][:, None] * transmat) * m[None, :] / c_next
        xi_sum += xi
    gamma = alpha_hat * beta_hat
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma, xi_sum, beta_hat


@dataclass
class HMMFit:
    """Fitted Gaussian HMM parameters and diagnostics.

    Attributes
    ----------
    startprob : (K,) initial state distribution.
    transmat : (K x K) transition matrix, rows sum to 1.
    means : (K x D) state emission means.
    covariances : (K x D x D) state emission covariances.
    log_likelihood : float
        Training log-likelihood at convergence.
    log_likelihood_history : list of float
        Per-EM-iteration log-likelihood of the winning restart (monotone).
    n_iter : int, converged : bool
        EM diagnostics of the winning restart.
    """

    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    log_likelihood: float
    log_likelihood_history: list[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False

    @property
    def n_states(self) -> int:
        return len(self.startprob)

    def log_emissions(self, x: np.ndarray) -> np.ndarray:
        return _log_emissions(_as_2d(x), self.means, self.covariances)

    def filter(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """FILTERED probabilities ``P(s_t | x_{1..t})`` and log-likelihood.

        This is the causal, online quantity: the value at ``t`` depends only
        on observations up to and including ``t``.  Trading signals MUST use
        this — never the smoothed posteriors.
        """
        return forward_filter(_as_2d(x), self.startprob, self.transmat, self.means, self.covariances)

    def smooth(self, x: np.ndarray) -> np.ndarray:
        """SMOOTHED posteriors ``P(s_t | x_{1..T})`` — uses the FULL sample.

        For diagnostics and historical labelling only: the value at ``t``
        depends on future observations, so it is NOT tradeable.
        """
        gamma, _, _ = forward_backward(
            _as_2d(x), self.startprob, self.transmat, self.means, self.covariances
        )
        return gamma

    def viterbi(self, x: np.ndarray) -> np.ndarray:
        """Most likely state path (uses the full sample; diagnostic only)."""
        return viterbi(_as_2d(x), self.startprob, self.transmat, self.means, self.covariances)

    def score(self, x: np.ndarray) -> float:
        """Log-likelihood of a (new) observation sequence."""
        _, ll = self.filter(x)
        return ll

    def stationary_distribution(self) -> np.ndarray:
        return stationary_distribution(self.transmat)

    def expected_durations(self) -> np.ndarray:
        return expected_durations(self.transmat)


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("observations must be 1-D or 2-D")
    return x


def forward_filter(
    x: np.ndarray,
    startprob: np.ndarray,
    transmat: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Forward (filtering) pass: ``P(s_t | x_{1..t})`` and log-likelihood.

    Purely causal — the returned row ``t`` depends only on ``x[:t+1]``.
    """
    x = _as_2d(x)
    log_b = _log_emissions(x, means, covs)
    return _scaled_forward(log_b, np.asarray(startprob, float), np.asarray(transmat, float))


def forward_backward(
    x: np.ndarray,
    startprob: np.ndarray,
    transmat: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Full forward–backward pass.

    Returns
    -------
    (gamma, xi_sum, log_likelihood)
        ``gamma[t]`` are the SMOOTHED posteriors ``P(s_t | x_{1..T})``
        (each row sums to 1); ``xi_sum`` the expected transition counts.
    """
    x = _as_2d(x)
    log_b = _log_emissions(x, means, covs)
    alpha_hat, ll = _scaled_forward(log_b, np.asarray(startprob, float), np.asarray(transmat, float))
    gamma, xi_sum, _ = _scaled_backward(log_b, np.asarray(transmat, float), alpha_hat)
    return gamma, xi_sum, ll


def viterbi(
    x: np.ndarray,
    startprob: np.ndarray,
    transmat: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
) -> np.ndarray:
    """Most likely hidden-state path (log-space Viterbi).

    Returns
    -------
    (T,) integer state path.
    """
    x = _as_2d(x)
    log_b = _log_emissions(x, means, covs)
    t_len, k = log_b.shape
    with np.errstate(divide="ignore"):
        log_pi = np.log(np.asarray(startprob, float))
        log_a = np.log(np.asarray(transmat, float))
    delta = log_pi + log_b[0]
    psi = np.zeros((t_len, k), dtype=int)
    for t in range(1, t_len):
        cand = delta[:, None] + log_a
        psi[t] = np.argmax(cand, axis=0)
        delta = cand[psi[t], np.arange(k)] + log_b[t]
    path = np.empty(t_len, dtype=int)
    path[-1] = int(np.argmax(delta))
    for t in range(t_len - 2, -1, -1):
        path[t] = psi[t + 1][path[t + 1]]
    return path


def stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """Stationary distribution ``pi`` with ``pi P = pi``, ``sum(pi) = 1``.

    Solved as an overdetermined linear system (``P^T - I`` stacked with the
    normalisation row) by least squares — accurate to ~1e-14 for
    well-conditioned chains; tests require ``pi P = pi`` to 1e-12.
    """
    p = np.asarray(transmat, dtype=float)
    k = p.shape[0]
    if p.shape != (k, k):
        raise ValueError("transmat must be square")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("transmat rows must sum to 1")
    a = np.vstack([p.T - np.eye(k), np.ones((1, k))])
    b = np.concatenate([np.zeros(k), [1.0]])
    pi, *_ = np.linalg.lstsq(a, b, rcond=None)
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


def expected_durations(transmat: np.ndarray) -> np.ndarray:
    """Expected sojourn time per state: ``1 / (1 - p_ii)`` days.

    A geometric-duration identity: while in state ``i`` the chain stays with
    probability ``p_ii`` each day, so the expected run length is
    ``1 / (1 - p_ii)``.
    """
    p = np.asarray(transmat, dtype=float)
    diag = np.diag(p)
    with np.errstate(divide="ignore"):
        return 1.0 / (1.0 - diag)


def _init_params(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    reg_covar: float,
    restart: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initial (startprob, transmat, means, covs)."""
    t_len, d = x.shape
    if restart == 0:
        means = kmeans_lite(x, k, rng)
    else:
        means = x[rng.choice(t_len, size=k, replace=False)].copy()
        means += 0.1 * x.std(axis=0, ddof=0) * rng.standard_normal((k, d))
    cov = np.cov(x.T, ddof=1).reshape(d, d) + reg_covar * np.eye(d)
    covs = np.array([cov.copy() for _ in range(k)])
    if k == 1:
        transmat = np.ones((1, 1))
    else:
        transmat = np.full((k, k), 0.10 / (k - 1))
        np.fill_diagonal(transmat, 0.90)
    startprob = np.full(k, 1.0 / k)
    return startprob, transmat, means, covs


def fit_hmm(
    x: np.ndarray,
    n_states: int,
    seed: int = 0,
    n_init: int = 3,
    max_iter: int = 200,
    tol: float = 1e-6,
    reg_covar: float = 1e-6,
    min_startprob: float = 1e-12,
) -> HMMFit:
    """Fit a Gaussian HMM by Baum–Welch EM with restarts.

    The first initialisation uses k-means-lite state means; subsequent
    restarts use perturbed random observations.  The restart with the best
    final log-likelihood wins.

    Parameters
    ----------
    x : (T x D) observation sequence (1-D treated as (T x 1)).
    n_states : int
        Number of hidden states, >= 1.
    seed : int
        Master seed; restart ``i`` uses generator ``default_rng(seed + 1000 i)``.
    n_init : int
        Number of restarts, >= 1.
    max_iter, tol : EM stopping rule (absolute log-likelihood change).
    reg_covar : float
        Ridge on covariance diagonals — guards against state collapse on
        (near-)degenerate data.
    min_startprob : float
        Floor on start probabilities (keeps logs finite).

    Returns
    -------
    HMMFit

    Raises
    ------
    ValueError
        For invalid ``n_states`` or a series too short to fit
        (fewer than ``max(10, 2 K)`` observations, or T <= D).
    """
    x = _as_2d(x)
    t_len, d = x.shape
    if np.isnan(x).any():
        raise ValueError("x contains NaN")
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1, got {n_states}")
    if t_len < max(10, 2 * n_states) or t_len <= d:
        raise ValueError(
            f"series too short to fit an HMM: T={t_len}, D={d}, K={n_states}"
        )
    if n_init < 1:
        raise ValueError(f"n_init must be >= 1, got {n_init}")

    best: HMMFit | None = None
    for restart in range(n_init):
        rng = np.random.default_rng(seed + 1000 * restart)
        startprob, transmat, means, covs = _init_params(x, n_states, rng, reg_covar, restart)
        history: list[float] = []
        converged = False
        it = 0
        for it in range(1, max_iter + 1):
            log_b = _log_emissions(x, means, covs)
            alpha_hat, ll = _scaled_forward(log_b, startprob, transmat)
            history.append(ll)
            if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
                converged = True
                break
            gamma, xi_sum, _ = _scaled_backward(log_b, transmat, alpha_hat)
            # M-step
            startprob = np.clip(gamma[0], min_startprob, None)
            startprob /= startprob.sum()
            if n_states > 1:
                denom = xi_sum.sum(axis=1, keepdims=True)
                denom = np.maximum(denom, 10.0 * np.finfo(float).tiny)
                transmat = xi_sum / denom
                transmat = np.clip(transmat, 1e-12, None)
                transmat /= transmat.sum(axis=1, keepdims=True)
            nk = gamma.sum(axis=0)
            nk = np.maximum(nk, 10.0 * np.finfo(float).tiny)
            means = (gamma.T @ x) / nk[:, None]
            for j in range(n_states):
                dev = x - means[j]
                covs[j] = (gamma[:, j][:, None] * dev).T @ dev / nk[j]
                covs[j] += reg_covar * np.eye(d)
        fit = HMMFit(
            startprob=startprob,
            transmat=transmat,
            means=means,
            covariances=covs,
            log_likelihood=history[-1],
            log_likelihood_history=history,
            n_iter=it,
            converged=converged,
        )
        if best is None or fit.log_likelihood > best.log_likelihood:
            best = fit
    assert best is not None
    return best
