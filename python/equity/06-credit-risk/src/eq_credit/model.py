"""Logistic regression from scratch (Newton-Raphson / IRLS) and scorecard scaling.

Model: ``P(default=1 | x) = sigmoid(b0 + x' b)``, fitted by iteratively
reweighted least squares (exact Newton steps on the log-likelihood).  With
``ridge > 0`` an L2 penalty ``(ridge/2)||b||^2`` (intercept unpenalised) is
added — used for separation control, NOT for the production scorecard.

Standard errors come from the inverse Fisher information ``(X'WX)^{-1}`` at
the MLE (for ridge > 0, ``(X'WX + λI)^{-1}`` — a common, slightly
anti-conservative approximation, documented as such).

Scorecard scaling: ``score = offset + factor * ln(odds_good)`` with
``factor = PDO / ln 2`` and ``offset = base_score - factor * ln(base_odds)``,
so an obligor at good:bad odds of ``base_odds`` scores ``base_score`` and
every doubling of odds adds exactly PDO points.  Equivalently
``score = offset - factor * logit(PD)``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "SeparationWarning",
    "LogisticFit",
    "fit_logistic",
    "crosscheck_sklearn",
    "stepwise_select",
    "ScorecardScaling",
    "scorecard_points_table",
]


class SeparationWarning(UserWarning):
    """(Quasi-)complete separation detected: MLE diverges; use ridge > 0."""


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))


@dataclass
class LogisticFit:
    """Fitted logistic regression.

    Attributes
    ----------
    coef : np.ndarray
        Slope coefficients (log-odds units per unit of feature).
    intercept : float
        Intercept ``b0``.
    se : np.ndarray
        Standard errors, ordered ``[intercept, coef...]``.
    z, p_values : np.ndarray
        Wald z-statistics and two-sided p-values, same ordering as ``se``.
    cov : np.ndarray
        Coefficient covariance ``(X'WX)^{-1}`` (incl. intercept row/col).
    loglik : float
        Log-likelihood at the optimum (unpenalised part).
    converged : bool
        Newton iteration converged.
    n_iter : int
        Newton iterations used.
    feature_names : list[str]
        Names for the slope coefficients.
    ridge : float
        L2 penalty used (0 = MLE).
    """

    coef: np.ndarray
    intercept: float
    se: np.ndarray
    z: np.ndarray
    p_values: np.ndarray
    cov: np.ndarray
    loglik: float
    converged: bool
    n_iter: int
    feature_names: list[str] = field(default_factory=list)
    ridge: float = 0.0

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predicted PDs for a feature matrix (same column order as fit)."""
        Xa = np.asarray(X, dtype=float)
        return _sigmoid(self.intercept + Xa @ self.coef)

    def predict_log_odds(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predicted log-odds (logit of PD)."""
        Xa = np.asarray(X, dtype=float)
        return self.intercept + Xa @ self.coef

    def summary(self) -> pd.DataFrame:
        """Coefficient table: estimate, SE, z, p-value."""
        names = ["intercept"] + (
            self.feature_names or [f"x{i}" for i in range(len(self.coef))]
        )
        est = np.concatenate([[self.intercept], self.coef])
        return pd.DataFrame(
            {"term": names, "coef": est, "se": self.se, "z": self.z, "p_value": self.p_values}
        )


def fit_logistic(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    ridge: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-10,
    feature_names: list[str] | None = None,
) -> LogisticFit:
    """Fit logistic regression by Newton-Raphson / IRLS.

    Parameters
    ----------
    X : array-like, shape (n, k)
        Feature matrix WITHOUT an intercept column (added internally).
    y : array-like of {0, 1}
        Default indicator.
    ridge : float
        L2 penalty λ on slopes (intercept unpenalised).  0 = plain MLE.
    max_iter, tol : int, float
        Newton iteration controls; convergence on max |step| < tol.
    feature_names : list of str, optional
        Names for reporting (taken from DataFrame columns if available).

    Returns
    -------
    LogisticFit

    Raises
    ------
    ValueError
        On empty input, all-0 or all-1 targets (informative message), or
        non-binary targets.

    Warns
    -----
    SeparationWarning
        If the MLE appears to diverge (perfect separation) at ridge = 0.
    """
    if isinstance(X, pd.DataFrame) and feature_names is None:
        feature_names = [str(c) for c in X.columns]
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float).ravel()
    if Xa.ndim == 1:
        Xa = Xa[:, None]
    n, k = Xa.shape
    if n == 0:
        raise ValueError("empty design matrix")
    if len(ya) != n:
        raise ValueError("X and y length mismatch")
    if not np.isin(ya, [0.0, 1.0]).all():
        raise ValueError("y must be binary 0/1")
    n_def = int(ya.sum())
    if n_def == 0:
        raise ValueError(
            "zero defaults in sample: logistic MLE is degenerate (intercept -> -inf). "
            "Collect more data or use a low-default-portfolio calibration approach."
        )
    if n_def == n:
        raise ValueError("sample contains no goods (all defaults): MLE degenerate")

    Xd = np.column_stack([np.ones(n), Xa])
    beta = np.zeros(k + 1)
    beta[0] = np.log(n_def / (n - n_def))  # start at the base-rate intercept
    pen = np.zeros(k + 1)
    pen[1:] = ridge

    converged = False
    separated = False
    it = 0
    for it in range(1, max_iter + 1):
        eta = Xd @ beta
        p = _sigmoid(eta)
        w = p * (1.0 - p)
        grad = Xd.T @ (ya - p) - pen * beta
        H = (Xd * w[:, None]).T @ Xd + np.diag(pen)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            H = H + 1e-10 * np.eye(k + 1)
            step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
        if ridge == 0.0 and beta[1:].size and np.max(np.abs(beta[1:])) > 40.0:
            separated = True
            break

    if separated:
        warnings.warn(
            "possible complete separation: coefficients diverging under the "
            "unpenalised MLE. Returning the current (large) iterate; refit "
            "with ridge > 0 for a finite, stable solution.",
            SeparationWarning,
            stacklevel=2,
        )

    eta = Xd @ beta
    p = _sigmoid(eta)
    w = np.maximum(p * (1.0 - p), 1e-12)
    H = (Xd * w[:, None]).T @ Xd + np.diag(pen)
    cov = np.linalg.inv(H)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    p_values = 2.0 * stats.norm.sf(np.abs(z))
    eps = 1e-12
    loglik = float(np.sum(ya * np.log(p + eps) + (1 - ya) * np.log(1 - p + eps)))

    return LogisticFit(
        coef=beta[1:].copy(),
        intercept=float(beta[0]),
        se=se,
        z=z,
        p_values=p_values,
        cov=cov,
        loglik=loglik,
        converged=converged,
        n_iter=it,
        feature_names=feature_names or [f"x{i}" for i in range(k)],
        ridge=ridge,
    )


def crosscheck_sklearn(
    X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, fit: LogisticFit
) -> float:
    """Max abs difference between our MLE and sklearn's (no penalty).

    Uses ``LogisticRegression(penalty=None, solver="newton-cg", tol=1e-12)``
    as an independent implementation of the same estimator.  Returns the max
    absolute coefficient difference (incl. intercept).
    """
    from sklearn.linear_model import LogisticRegression  # benchmark only

    # penalty=None == unpenalised MLE on all supported sklearn versions;
    # sklearn 1.8 emits a deprecation FutureWarning for it, silenced here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", UserWarning)
        sk = LogisticRegression(
            penalty=None, solver="newton-cg", tol=1e-12, max_iter=500
        )
        sk.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int).ravel())
    ours = np.concatenate([[fit.intercept], fit.coef])
    theirs = np.concatenate([sk.intercept_, sk.coef_.ravel()])
    return float(np.max(np.abs(ours - theirs)))


def stepwise_select(
    woe_df: pd.DataFrame,
    y: np.ndarray | pd.Series,
    ivs: dict[str, float],
    *,
    iv_min: float = 0.02,
    iv_max: float = 0.5,
    p_max: float = 0.05,
) -> list[str]:
    """Simple scorecard feature selection: rank by IV, forward-add, prune.

    1. Drop features with IV < ``iv_min`` (useless) or IV > ``iv_max``
       (suspicious / potential leakage).
    2. Forward pass in descending-IV order: keep a candidate if its Wald
       p-value in the joint fit is < ``p_max``.
    3. Backward prune: refit and drop the worst-p feature until all < ``p_max``.

    Returns the selected column names of ``woe_df``.
    """
    ranked = [
        f for f, iv in sorted(ivs.items(), key=lambda kv: -kv[1])
        if iv_min <= iv <= iv_max and f in woe_df.columns
    ]
    selected: list[str] = []
    for f in ranked:
        trial = selected + [f]
        fit = fit_logistic(woe_df[trial], y, feature_names=trial)
        if fit.p_values[-1] < p_max:
            selected = trial
    # Backward prune.
    while selected:
        fit = fit_logistic(woe_df[selected], y, feature_names=selected)
        pv = fit.p_values[1:]
        worst = int(np.argmax(pv))
        if pv[worst] < p_max:
            break
        selected.pop(worst)
    if not selected:
        raise ValueError("stepwise selection eliminated all features")
    return selected


@dataclass(frozen=True)
class ScorecardScaling:
    """Points / PDO scorecard scaling.

    Attributes
    ----------
    base_score : float
        Score assigned at ``base_odds`` (default 600 points).
    base_odds : float
        Good:bad odds at the base score (default 50, i.e. 50:1).
    pdo : float
        Points to Double the Odds (default 20).
    """

    base_score: float = 600.0
    base_odds: float = 50.0
    pdo: float = 20.0

    @property
    def factor(self) -> float:
        """Points per unit of ln(odds): PDO / ln 2."""
        return self.pdo / np.log(2.0)

    @property
    def offset(self) -> float:
        """Score at odds 1:1: base_score - factor * ln(base_odds)."""
        return self.base_score - self.factor * np.log(self.base_odds)

    def score_from_pd(self, pd_: np.ndarray | float) -> np.ndarray:
        """Score from PD; PD clamped to [1e-9, 1 - 1e-9] before the logit."""
        p = np.clip(np.asarray(pd_, dtype=float), 1e-9, 1.0 - 1e-9)
        odds_good = (1.0 - p) / p
        return self.offset + self.factor * np.log(odds_good)

    def pd_from_score(self, score: np.ndarray | float) -> np.ndarray:
        """Invert :meth:`score_from_pd`."""
        s = np.asarray(score, dtype=float)
        log_odds_good = (s - self.offset) / self.factor
        return 1.0 / (1.0 + np.exp(log_odds_good))


def scorecard_points_table(
    fit: LogisticFit,
    binnings: dict[str, "object"],
    scaling: ScorecardScaling,
) -> pd.DataFrame:
    """Per-bin points allocation for the scorecard.

    Points for feature j in bin i:
    ``pts = (offset - factor*b0)/k - factor * b_j * WOE_ij`` so the points of
    an obligor's bins sum exactly to their scaled score.

    Parameters
    ----------
    fit : LogisticFit
        Fitted on WOE columns named ``woe_<feature>``.
    binnings : dict[str, FeatureBinning]
        Fitted binnings keyed by raw feature name.
    scaling : ScorecardScaling

    Returns
    -------
    pd.DataFrame with columns feature, bin, woe, points.
    """
    k = len(fit.coef)
    base_alloc = (scaling.offset - scaling.factor * fit.intercept) / k
    rows = []
    for j, col in enumerate(fit.feature_names):
        feat = col[4:] if col.startswith("woe_") else col
        fb = binnings.get(feat)
        if fb is None:
            continue
        table = fb.table
        for _, r in table.iterrows():
            rows.append(
                {
                    "feature": feat,
                    "bin": r["bin"],
                    "woe": r["woe"],
                    "points": base_alloc - scaling.factor * fit.coef[j] * r["woe"],
                }
            )
    return pd.DataFrame(rows)
