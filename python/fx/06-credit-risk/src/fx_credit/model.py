"""From-scratch IRLS logistic regression, scorecard scaling, and rating mapping.

The logistic model is fitted by iteratively reweighted least squares
(Newton-Raphson on the log-likelihood), with standard errors from the
observed information matrix — no sklearn in the core path.  sklearn is used
only as an external cross-check (see tests: coefficients match to 1e-6).

Separation: with a low-default portfolio and strong features, (quasi-)
complete separation is a real hazard — the MLE diverges.  ``fit_logistic_irls``
detects diverging coefficients and raises ``ValueError`` unless a small ridge
penalty is supplied (``ridge > 0``), which restores a finite maximiser.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "LogisticFit",
    "fit_logistic_irls",
    "predict_pd",
    "score_from_pd",
    "pd_from_score",
    "RATING_BANDS",
    "RATING_ORDER",
    "assign_rating",
    "rating_midpoint_pd",
]


@dataclass(frozen=True)
class LogisticFit:
    """Fitted logistic regression.

    Attributes
    ----------
    intercept : float
    coef : ndarray, shape (k,)
    se : ndarray, shape (k+1,)
        Standard errors, intercept first, from the inverse observed
        information (X'WX)^-1.
    cov : ndarray, shape (k+1, k+1)
        Full covariance matrix of [intercept, coef].
    loglik : float
        Maximised log-likelihood (excluding any ridge term).
    n_iter : int
    converged : bool
    ridge : float
        Ridge penalty used (0 = pure MLE).
    """

    intercept: float
    coef: np.ndarray
    se: np.ndarray
    cov: np.ndarray
    loglik: float
    n_iter: int
    converged: bool
    ridge: float


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def fit_logistic_irls(
    X: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 100,
    ridge: float = 0.0,
    separation_bound: float = 30.0,
) -> LogisticFit:
    """Fit logistic regression by IRLS (Newton-Raphson) from scratch.

    Parameters
    ----------
    X : ndarray, shape (n, k)
        Design matrix WITHOUT intercept column (added internally).
    y : ndarray, shape (n,)
        Binary outcomes in {0, 1}.
    tol : float
        Convergence tolerance on the max absolute coefficient update.
    max_iter : int
        Maximum Newton iterations.
    ridge : float
        L2 penalty ``ridge/2 * ||beta||^2`` on all coefficients including the
        intercept (0 = MLE).  Use a small value (e.g. 1e-6) to stabilise
        separated data.
    separation_bound : float
        If any |coefficient| exceeds this during iteration with ``ridge=0``,
        (quasi-)separation is declared and ``ValueError`` is raised: a
        log-odds of 30 corresponds to a PD below 1e-13, far outside any
        estimable sovereign PD.

    Returns
    -------
    LogisticFit

    Raises
    ------
    ValueError
        On non-binary y, shape mismatch, or detected separation.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError("X must be (n, k) with n matching y")
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("y must be binary {0,1}")
    n, k = X.shape
    Xd = np.column_stack([np.ones(n), X])
    beta = np.zeros(k + 1)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        eta = Xd @ beta
        p = _sigmoid(eta)
        w = p * (1.0 - p)
        w = np.maximum(w, 1e-12)
        # Newton step: (X'WX + ridge I) delta = X'(y - p) - ridge*beta
        H = Xd.T @ (Xd * w[:, None]) + ridge * np.eye(k + 1)
        g = Xd.T @ (y - p) - ridge * beta
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "singular information matrix — collinear features or separation; "
                "drop collinear columns or set ridge>0"
            ) from exc
        beta = beta + delta
        if ridge == 0.0 and np.max(np.abs(beta)) > separation_bound:
            raise ValueError(
                "coefficients diverging (|beta| > "
                f"{separation_bound}): (quasi-)complete separation detected. "
                "With a low-default portfolio this happens when a bin or dummy "
                "perfectly predicts the outcome; coarsen bins or refit with "
                "ridge>0 (e.g. 1e-6)."
            )
        if np.max(np.abs(delta)) < tol:
            converged = True
            break
    eta = Xd @ beta
    p = _sigmoid(eta)
    w = np.maximum(p * (1.0 - p), 1e-12)
    H = Xd.T @ (Xd * w[:, None]) + ridge * np.eye(k + 1)
    cov = np.linalg.inv(H)
    se = np.sqrt(np.diag(cov))
    eps = 1e-300
    loglik = float(np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    return LogisticFit(
        intercept=float(beta[0]),
        coef=beta[1:].copy(),
        se=se,
        cov=cov,
        loglik=loglik,
        n_iter=it,
        converged=converged,
        ridge=ridge,
    )


def predict_pd(fit: LogisticFit, X: np.ndarray) -> np.ndarray:
    """Predicted 1-year PD for each row of ``X`` (same columns as the fit)."""
    X = np.asarray(X, dtype=float)
    return _sigmoid(fit.intercept + X @ fit.coef)


# ---------------------------------------------------------------------------
# Scorecard scaling (points-to-double-odds) and rating-agency-style bands
# ---------------------------------------------------------------------------

def score_from_pd(
    pd_1y: np.ndarray | float,
    base_score: float = 600.0,
    base_odds: float = 50.0,
    pdo: float = 20.0,
) -> np.ndarray | float:
    """Map PD to a scorecard score with points-to-double-odds (PDO) scaling.

    ``score = base_score + pdo/ln 2 * ln(odds_good / base_odds)`` where
    ``odds_good = (1 - PD)/PD``.  At ``odds_good = base_odds`` the score is
    exactly ``base_score``; every doubling of good odds adds exactly ``pdo``
    points (the PDO identity, unit-tested).  Higher score = safer.
    """
    p = np.clip(np.asarray(pd_1y, dtype=float), 1e-12, 1 - 1e-12)
    odds_good = (1.0 - p) / p
    score = base_score + pdo / np.log(2.0) * np.log(odds_good / base_odds)
    return float(score) if np.isscalar(pd_1y) else score


def pd_from_score(
    score: np.ndarray | float,
    base_score: float = 600.0,
    base_odds: float = 50.0,
    pdo: float = 20.0,
) -> np.ndarray | float:
    """Inverse of ``score_from_pd`` (exact round-trip, unit-tested)."""
    s = np.asarray(score, dtype=float)
    odds_good = base_odds * np.power(2.0, (s - base_score) / pdo)
    p = 1.0 / (1.0 + odds_good)
    return float(p) if np.isscalar(score) else p


#: Rating bands: letter -> (upper 1y-PD bound, PD midpoint).  Bands partition
#: (0, 1]; midpoints are monotone increasing (unit-tested), loosely anchored
#: to long-run rating-agency sovereign default studies.
RATING_BANDS: dict[str, tuple[float, float]] = {
    "AAA": (0.0003, 0.0001),
    "AA": (0.0010, 0.0005),
    "A": (0.0030, 0.0020),
    "BBB": (0.0100, 0.0050),
    "BB": (0.0300, 0.0200),
    "B": (0.1000, 0.0600),
    "CCC": (0.2500, 0.1500),
    "C": (1.0001, 0.4000),
}

#: Best-to-worst letter order.
RATING_ORDER: tuple[str, ...] = tuple(RATING_BANDS)


def assign_rating(pd_1y: float) -> str:
    """Letter rating for a 1-year PD (first band whose upper bound covers it)."""
    if not 0.0 <= pd_1y <= 1.0:
        raise ValueError(f"pd_1y must be in [0,1], got {pd_1y}")
    for letter, (upper, _) in RATING_BANDS.items():
        if pd_1y < upper:
            return letter
    return "C"


def rating_midpoint_pd(rating: str) -> float:
    """PD midpoint assigned to a letter band (used for pricing/CVA/limits)."""
    try:
        return RATING_BANDS[rating][1]
    except KeyError as exc:
        raise ValueError(f"unknown rating {rating!r}; known: {RATING_ORDER}") from exc
