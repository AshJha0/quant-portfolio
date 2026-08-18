"""Principal component analysis from scratch (eigendecomposition).

PCA is computed on the CORRELATION matrix: inputs are standardised
column-wise (ddof=1), then the symmetric eigenproblem is solved with
``numpy.linalg.eigh``.  This is numerically identical to sklearn's
covariance PCA applied to the standardised data, which is exactly the
cross-check enforced in tests (agreement to 1e-8, up to sign).

Sign convention: for each component, the loading with the largest absolute
value is made positive.  This removes the eigenvector sign ambiguity and is
applied consistently in :func:`fit_pca` and :func:`rolling_pca` (the latter
additionally enforces sign CONTINUITY across windows so projected factors do
not flip sign from one refit to the next).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["PCAModel", "fit_pca", "project", "scree_table", "rolling_pca"]


@dataclass(frozen=True)
class PCAModel:
    """Fitted PCA on the correlation matrix.

    Attributes
    ----------
    components : np.ndarray
        (n_features x n_components) loading matrix; column j is the j-th
        eigenvector (unit norm), sign-fixed so its largest-|.| entry is > 0.
    explained_variance : np.ndarray
        (n_components,) eigenvalues in descending order.
    explained_variance_ratio : np.ndarray
        Eigenvalues divided by the TOTAL variance (sum over all features,
        = n_features for a correlation matrix), so the full set sums to 1.
    mean, std : np.ndarray
        (n_features,) standardisation parameters (ddof=1 std).
    feature_names : tuple of str
        Column names of the fitted data.
    """

    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    feature_names: tuple[str, ...]

    @property
    def n_components(self) -> int:
        return self.components.shape[1]


def _fix_signs(components: np.ndarray) -> np.ndarray:
    """Flip each column so its largest-absolute loading is positive."""
    comp = components.copy()
    for j in range(comp.shape[1]):
        i = int(np.argmax(np.abs(comp[:, j])))
        if comp[i, j] < 0:
            comp[:, j] = -comp[:, j]
    return comp


def fit_pca(X: pd.DataFrame | np.ndarray, n_components: int | None = None) -> PCAModel:
    """Fit PCA via eigendecomposition of the correlation matrix.

    Parameters
    ----------
    X : (T x F) array or DataFrame
        Observations in rows.  T must exceed F and every column must have
        non-zero variance.
    n_components : int, optional
        Number of components to keep (default: all F).

    Returns
    -------
    PCAModel

    Raises
    ------
    ValueError
        If T <= F, if any column is constant, or n_components out of range.
    """
    if isinstance(X, pd.DataFrame):
        names = tuple(str(c) for c in X.columns)
        arr = X.to_numpy(dtype=float)
    else:
        arr = np.asarray(X, dtype=float)
        names = tuple(f"f{i}" for i in range(arr.shape[1]))
    if arr.ndim != 2:
        raise ValueError("X must be 2-D (T x F)")
    t_len, n_feat = arr.shape
    if t_len <= n_feat:
        raise ValueError(f"need more observations ({t_len}) than features ({n_feat})")
    if np.isnan(arr).any():
        raise ValueError("X contains NaN — drop warmup rows before PCA")
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1)
    tol = 1e-13 * (np.abs(mean) + 1.0)
    if np.any(std <= tol):
        bad = [names[i] for i in np.where(std <= tol)[0]]
        raise ValueError(f"constant (zero-variance) columns cannot be standardised: {bad}")
    z = (arr - mean) / std
    corr = (z.T @ z) / (t_len - 1)
    eigval, eigvec = np.linalg.eigh(corr)
    order = np.argsort(eigval)[::-1]
    eigval = np.clip(eigval[order], 0.0, None)
    eigvec = _fix_signs(eigvec[:, order])

    k = n_feat if n_components is None else int(n_components)
    if not 1 <= k <= n_feat:
        raise ValueError(f"n_components must be in [1, {n_feat}], got {k}")
    total = eigval.sum()
    return PCAModel(
        components=eigvec[:, :k],
        explained_variance=eigval[:k],
        explained_variance_ratio=eigval[:k] / total,
        mean=mean,
        std=std,
        feature_names=names,
    )


def project(X: pd.DataFrame | np.ndarray, model: PCAModel) -> pd.DataFrame | np.ndarray:
    """Project observations onto the fitted components.

    Standardises with the model's (training) mean/std then multiplies by the
    loading matrix.  DataFrames come back as DataFrames with PC column names.

    Parameters
    ----------
    X : (T x F) array or DataFrame with the same features as the fit.
    model : PCAModel

    Returns
    -------
    (T x n_components) scores.
    """
    is_df = isinstance(X, pd.DataFrame)
    arr = X.to_numpy(dtype=float) if is_df else np.asarray(X, dtype=float)
    if arr.shape[1] != len(model.mean):
        raise ValueError(
            f"feature count mismatch: X has {arr.shape[1]}, model fitted on {len(model.mean)}"
        )
    scores = ((arr - model.mean) / model.std) @ model.components
    if is_df:
        cols = [f"PC{i + 1}" for i in range(model.n_components)]
        return pd.DataFrame(scores, index=X.index, columns=cols)
    return scores


def scree_table(model: PCAModel) -> pd.DataFrame:
    """Scree summary: eigenvalue, variance ratio and cumulative ratio per PC."""
    k = model.n_components
    return pd.DataFrame(
        {
            "eigenvalue": model.explained_variance,
            "var_ratio": model.explained_variance_ratio,
            "cum_ratio": np.cumsum(model.explained_variance_ratio),
        },
        index=[f"PC{i + 1}" for i in range(k)],
    )


def rolling_pca(
    X: pd.DataFrame,
    window: int,
    n_components: int = 2,
    step: int = 21,
) -> pd.DataFrame:
    """Rolling-window PCA with sign continuity across refits.

    Refits PCA every ``step`` days on the trailing ``window`` observations and
    projects the CURRENT block of observations with the latest model.  Sign
    continuity: each new component is aligned with its predecessor — if the
    dot product with the previous window's component is negative, the sign is
    flipped — so factor scores do not jump sign at refit dates.

    Parameters
    ----------
    X : pd.DataFrame
        (T x F) feature table, NaN-free.
    window : int
        Trailing fit window (must exceed the feature count).
    n_components : int
        Components to keep.
    step : int
        Refit frequency in days.

    Returns
    -------
    pd.DataFrame
        Scores from ``window - 1`` onwards (each block scored point-in-time
        with the model fitted on data available at the block start).
    """
    if window <= X.shape[1]:
        raise ValueError(f"window ({window}) must exceed feature count ({X.shape[1]})")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    t_len = len(X)
    if t_len < window:
        raise ValueError(f"series length {t_len} shorter than window {window}")

    prev_comp: np.ndarray | None = None
    frames: list[pd.DataFrame] = []
    for start in range(window - 1, t_len, step):
        fit_block = X.iloc[start - window + 1 : start + 1]
        model = fit_pca(fit_block, n_components)
        comp = model.components.copy()
        if prev_comp is not None:
            for j in range(comp.shape[1]):
                if float(comp[:, j] @ prev_comp[:, j]) < 0.0:
                    comp[:, j] = -comp[:, j]
        prev_comp = comp
        model = PCAModel(
            components=comp,
            explained_variance=model.explained_variance,
            explained_variance_ratio=model.explained_variance_ratio,
            mean=model.mean,
            std=model.std,
            feature_names=model.feature_names,
        )
        score_block = X.iloc[start : min(start + step, t_len)]
        frames.append(project(score_block, model))
    return pd.concat(frames)
