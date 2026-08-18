"""Principal component analysis from scratch (SVD-based).

Used to compress the FX feature block and to demonstrate that the first
principal component of a RORO panel IS the risk-on/risk-off axis: carry
and EM currencies load with one sign, safe havens (JPY, CHF) with the
opposite sign.  ``sklearn.decomposition.PCA`` is used only as a
cross-check in the test suite.

Convention: rows are observations, columns are variables.  Components
are returned as rows of ``components`` (like sklearn), with a
deterministic sign convention: the entry of largest absolute value in
each component is positive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCAResult:
    """Fitted PCA.

    Attributes
    ----------
    components : (k, p) ndarray
        Orthonormal loading vectors (rows).
    explained_variance : (k,) ndarray
        Eigenvalues of the sample covariance (ddof=1).
    explained_variance_ratio : (k,) ndarray
    mean : (p,) ndarray
        Column means removed before decomposition.
    scores : (n, k) ndarray
        Projections of the (centred) training data.
    total_variance : float
        Sum of ALL sample variances (not just kept components).
    """

    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    mean: np.ndarray
    scores: np.ndarray
    total_variance: float

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new data onto the fitted components."""
        X = np.asarray(X, dtype=float)
        return (X - self.mean) @ self.components.T

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        """Reconstruct data from scores (exact iff k = rank)."""
        return np.asarray(scores, dtype=float) @ self.components + self.mean


def fit_pca(X: np.ndarray, n_components: int | None = None) -> PCAResult:
    """Fit PCA by SVD of the centred data matrix.

    Parameters
    ----------
    X : (n, p) array
        Observations in rows.  n must be >= 2.
    n_components : int, optional
        Number of components to keep (default: min(n, p)).

    Returns
    -------
    PCAResult

    Raises
    ------
    ValueError
        On invalid shapes or n_components out of range.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    n, p = X.shape
    if n < 2:
        raise ValueError("need at least 2 observations")
    max_k = min(n, p)
    k = max_k if n_components is None else int(n_components)
    if not 1 <= k <= max_k:
        raise ValueError(f"n_components must be in [1, {max_k}]")

    mean = X.mean(axis=0)
    Xc = X - mean
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    ev_all = s**2 / (n - 1)
    total_var = float(ev_all.sum())

    comps = Vt[:k]
    scores = U[:, :k] * s[:k]
    # deterministic sign: largest-|loading| entry of each component > 0
    for i in range(k):
        j = int(np.argmax(np.abs(comps[i])))
        if comps[i, j] < 0:
            comps[i] = -comps[i]
            scores[:, i] = -scores[:, i]

    ev = ev_all[:k]
    ratio = ev / total_var if total_var > 0 else np.zeros(k)
    return PCAResult(
        components=comps,
        explained_variance=ev,
        explained_variance_ratio=ratio,
        mean=mean,
        scores=scores,
        total_variance=total_var,
    )


def roro_axis_check(
    components: np.ndarray,
    columns: list[str],
    risk_cols: list[str],
    haven_cols: list[str],
) -> bool:
    """Check that PC1 separates the risk block from the havens.

    True iff every risk-block loading shares one sign and every haven
    loading has the opposite sign on the first component.

    Parameters
    ----------
    components : (k, p) loadings (rows), columns aligned to ``columns``.
    columns : names of the p variables.
    risk_cols, haven_cols : subsets of ``columns``.
    """
    pc1 = dict(zip(columns, components[0]))
    risk = np.array([pc1[c] for c in risk_cols])
    haven = np.array([pc1[c] for c in haven_cols])
    if len(risk) == 0 or len(haven) == 0:
        raise ValueError("risk_cols and haven_cols must be non-empty")
    risk_sign = np.sign(risk.mean())
    return bool(np.all(np.sign(risk) == risk_sign) and np.all(
        np.sign(haven) == -risk_sign
    ))
