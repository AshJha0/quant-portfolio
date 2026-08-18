"""Engle-Granger two-step cointegration test, implemented from scratch.

Step 1 — cointegrating regression (OLS):  y_t = alpha + beta x_t + u_t.
Step 2 — ADF unit-root test on the fitted residuals u_hat_t.

The augmented Dickey-Fuller regression, lag selection by AIC, and the
MacKinnon critical-value response surfaces are all implemented here from
first principles; the test suite cross-validates the t-statistic against
``statsmodels.tsa.stattools.adfuller`` to 1e-8 on the same lag specification.

THE CLASSIC MISTAKE — which critical values?
--------------------------------------------
The step-2 ADF statistic must NOT be compared against ordinary ADF critical
values. The residuals u_hat are not an observed series: OLS chose
(alpha, beta) to make u_hat look as stationary as possible, so the test
statistic is biased towards rejection. Using the plain ADF 5% value (-2.86,
constant case) instead of the Engle-Granger value for two variables (-3.34)
makes the test badly oversized — you "find" cointegration that isn't there.
MacKinnon (1991, 2010) tabulates response surfaces indexed by N, the number
of I(1) series in the cointegrating regression: N=1 reproduces plain ADF,
N=2 is the correct table for a two-leg pairs test. :func:`mackinnon_crit`
implements both so the difference is testable.

Intercept choice (documented per CONVENTIONS): the cointegrating regression
includes an intercept by default. Two stocks trade at arbitrary per-share
price levels (a 5:1 split changes the level, not the economics), so forcing
the line through the origin misspecifies the long-run relation unless the
legs are already on a comparable scale. With an intercept, step-2 runs the
ADF regression WITHOUT a constant (residuals are exactly mean-zero) but uses
the N=2 "constant" critical-value surface — matching the deterministic terms
of step 1, which is what determines the null distribution. MacKinnon does
not tabulate a no-constant surface for N >= 2 (he considers the case
unrealistic); ``engle_granger(intercept=False)`` therefore reuses the N=2
'c' surface and flags the approximation in the result.

Johansen's ML procedure is the standard alternative for systems of 3+ series
(it estimates cointegration rank and avoids EG's arbitrary choice of
dependent variable). For two-leg pairs, EG is standard, transparent, and
directly yields the tradeable hedge ratio; Johansen is out of scope here —
see docs/METHODOLOGY.md for the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

__all__ = [
    "OLSFit",
    "ols",
    "hedge_ratio",
    "mackinnon_crit",
    "ADFResult",
    "adf_test",
    "EGResult",
    "engle_granger",
]

ArrayLike = Union[np.ndarray, pd.Series, list]


# --------------------------------------------------------------------------
# OLS building block
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class OLSFit:
    """OLS fit y = X b + e with classical (non-robust) standard errors.

    Attributes
    ----------
    params : ndarray (k,)
    stderr : ndarray (k,)
    tvalues : ndarray (k,)
    resid : ndarray (n,)
    ssr : float
        Sum of squared residuals.
    llf : float
        Gaussian log-likelihood at the MLE variance ssr/n.
    aic : float
        -2 llf + 2 k (matches statsmodels OLS ``aic``).
    nobs : int
    """

    params: np.ndarray
    stderr: np.ndarray
    tvalues: np.ndarray
    resid: np.ndarray
    ssr: float
    llf: float
    aic: float
    nobs: int


def ols(y: np.ndarray, X: np.ndarray) -> OLSFit:
    """Ordinary least squares via QR-backed lstsq, classical standard errors.

    Parameters
    ----------
    y : ndarray (n,)
    X : ndarray (n, k)
        Design matrix (include a column of ones for an intercept).

    Raises
    ------
    ValueError
        If the design matrix is rank-deficient (e.g. a zero-variance
        regressor alongside a constant) or n <= k.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, k = X.shape
    if len(y) != n:
        raise ValueError(f"y has {len(y)} rows, X has {n}")
    if n <= k:
        raise ValueError(f"need nobs > nparams, got nobs={n}, nparams={k}")
    rank = np.linalg.matrix_rank(X)
    if rank < k:
        raise ValueError(
            f"design matrix is rank-deficient (rank {rank} < {k}); "
            "check for a zero-variance or collinear regressor"
        )
    params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ params
    ssr = float(resid @ resid)
    dof = n - k
    s2 = ssr / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    stderr = np.sqrt(s2 * np.diag(xtx_inv))
    tvalues = params / stderr
    # Gaussian log-likelihood with MLE variance ssr/n (statsmodels convention)
    llf = -0.5 * n * (np.log(2.0 * np.pi) + np.log(ssr / n) + 1.0)
    aic = -2.0 * llf + 2.0 * k
    return OLSFit(params, stderr, tvalues, resid, ssr, llf, aic, n)


def hedge_ratio(
    y: ArrayLike, x: ArrayLike, intercept: bool = True
) -> tuple[float, float, np.ndarray]:
    """Static OLS hedge ratio: regress y on x (levels).

    Parameters
    ----------
    y, x : array-like (n,)
        Price levels in dollars.
    intercept : bool
        Include an intercept (default True; see module docstring).

    Returns
    -------
    (beta, alpha, resid) — hedge ratio, intercept (0.0 when excluded),
    residual spread in dollars.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(y) != len(x):
        raise ValueError(f"length mismatch: {len(y)} vs {len(x)}")
    if np.std(x) == 0.0:
        raise ValueError("x has zero variance; hedge ratio undefined")
    if intercept:
        X = np.column_stack([np.ones_like(x), x])
        fit = ols(y, X)
        alpha, beta = float(fit.params[0]), float(fit.params[1])
    else:
        fit = ols(y, x[:, None])
        alpha, beta = 0.0, float(fit.params[0])
    return beta, alpha, fit.resid


# --------------------------------------------------------------------------
# MacKinnon (2010) critical-value response surfaces
# --------------------------------------------------------------------------
# crit(T) = b0 + b1/T + b2/T^2 + b3/T^3, rows = (1%, 5%, 10%).
# N = number of I(1) series in the cointegrating regression:
#   N=1 -> plain ADF unit-root test; N=2 -> Engle-Granger residual test
#   for a two-variable regression. Source: MacKinnon (2010), "Critical
#   Values for Cointegration Tests", QED working paper 1227, Tables 2-4;
#   the N=1 no-constant row is from MacKinnon (1996).
_MACKINNON_SURFACES: dict[tuple[int, str], np.ndarray] = {
    (1, "n"): np.array(
        [
            [-2.56574, -2.2358, -3.627, 0.0],
            [-1.94100, -0.2686, -3.365, 31.223],
            [-1.61682, 0.2656, -2.714, 25.364],
        ]
    ),
    (1, "c"): np.array(
        [
            [-3.43035, -6.5393, -16.786, -79.433],
            [-2.86154, -2.8903, -4.234, -40.040],
            [-2.56677, -1.5384, -2.809, 0.0],
        ]
    ),
    (1, "ct"): np.array(
        [
            [-3.95877, -9.0531, -28.428, -134.155],
            [-3.41049, -4.3904, -9.036, -45.374],
            [-3.12705, -2.5856, -3.925, -22.380],
        ]
    ),
    (2, "c"): np.array(
        [
            [-3.89644, -10.9519, -33.527, 0.0],
            [-3.33613, -6.1101, -6.823, 0.0],
            [-3.04445, -4.2412, -2.720, 0.0],
        ]
    ),
    (2, "ct"): np.array(
        [
            [-4.32762, -15.4387, -35.679, 0.0],
            [-3.78057, -9.5106, -12.074, 0.0],
            [-3.49631, -7.0815, -7.538, 21.892],
        ]
    ),
}


def mackinnon_crit(
    n_series: int = 1, regression: str = "c", nobs: float = np.inf
) -> dict[str, float]:
    """MacKinnon critical values for ADF (N=1) and Engle-Granger (N=2) tests.

    Parameters
    ----------
    n_series : {1, 2}
        Number of I(1) series. 1 = plain ADF on an observed series;
        2 = residual-based Engle-Granger test for a two-leg pair. Using the
        N=1 table for a residual-based test is the classic error this module
        exists to prevent.
    regression : {"n", "c", "ct"}
        Deterministic terms of the (cointegrating) regression. "n" is only
        tabulated for N=1.
    nobs : int or inf
        Effective sample size for the finite-sample response surface;
        ``inf`` returns asymptotic values.

    Returns
    -------
    dict with keys "1%", "5%", "10%".
    """
    key = (n_series, regression)
    if key not in _MACKINNON_SURFACES:
        raise ValueError(
            f"no MacKinnon surface for N={n_series}, regression={regression!r}; "
            f"available: {sorted(_MACKINNON_SURFACES)}"
        )
    b = _MACKINNON_SURFACES[key]
    if np.isinf(nobs):
        vals = b[:, 0]
    else:
        if nobs <= 0:
            raise ValueError(f"nobs must be positive, got {nobs}")
        t = float(nobs)
        vals = b[:, 0] + b[:, 1] / t + b[:, 2] / t**2 + b[:, 3] / t**3
    return {"1%": float(vals[0]), "5%": float(vals[1]), "10%": float(vals[2])}


# --------------------------------------------------------------------------
# Augmented Dickey-Fuller test (from scratch)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ADFResult:
    """Result of an augmented Dickey-Fuller regression.

    Attributes
    ----------
    stat : float
        t-statistic on the level coefficient rho in
        dy_t = [deterministics] + rho * y_{t-1} + sum_i phi_i dy_{t-i} + e_t.
    lags : int
        Number of lagged differences included.
    nobs : int
        Effective regression sample size.
    crit : dict
        Critical values used ("1%", "5%", "10%").
    n_series : int
        MacKinnon N used for the critical values (1 = plain ADF).
    regression : str
        Deterministic terms in the ADF regression ("n", "c", "ct").
    """

    stat: float
    lags: int
    nobs: int
    crit: dict[str, float]
    n_series: int
    regression: str

    def reject(self, level: str = "5%") -> bool:
        """True if the unit-root null is rejected at ``level``."""
        return self.stat < self.crit[level]


def _adf_design(
    y: np.ndarray, lags: int, regression: str, start: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build (dependent, design) for the ADF regression using rows from
    ``start`` onward of the differenced series (start >= lags)."""
    dy = np.diff(y)
    n = len(dy)
    rows = np.arange(start, n)
    dep = dy[rows]
    cols = [y[rows]]  # y_{t-1}: level aligned one step before each dy row
    for i in range(1, lags + 1):
        cols.append(dy[rows - i])
    if regression == "c":
        cols.append(np.ones(len(rows)))
    elif regression == "ct":
        cols.append(np.ones(len(rows)))
        cols.append(np.arange(1, len(rows) + 1, dtype=float))
    elif regression != "n":
        raise ValueError(f"regression must be 'n', 'c' or 'ct', got {regression!r}")
    return dep, np.column_stack(cols)


def adf_test(
    y: ArrayLike,
    regression: str = "c",
    lags: Optional[int] = None,
    max_lags: Optional[int] = None,
    n_series: int = 1,
) -> ADFResult:
    """Augmented Dickey-Fuller test with AIC lag selection, from scratch.

    dy_t = deterministics + rho y_{t-1} + sum_{i=1}^{p} phi_i dy_{t-i} + e_t;
    the statistic is the OLS t-value of rho. H0: unit root (rho = 0).

    Lag selection follows the statsmodels/Ng-Perron convention: candidate
    regressions for p = 0..max_lags are all fitted on the SAME sample (rows
    max_lags.. of the differenced series) so their AICs are comparable; the
    winner is then refitted on the longest usable sample. Default
    ``max_lags`` is Schwert's rule 12*(nobs/100)^0.25.

    Parameters
    ----------
    y : array-like (n,)
        Series to test (a price series, or EG residuals via n_series=2).
    regression : {"n", "c", "ct"}
        Deterministic terms in the ADF regression.
    lags : int, optional
        Fixed lag order p; overrides AIC selection.
    max_lags : int, optional
        Upper bound for AIC search (ignored when ``lags`` given).
    n_series : {1, 2}
        Which MacKinnon table to use for critical values. Use 2 ONLY via
        :func:`engle_granger`, which also passes the step-1 deterministic
        spec; plain series tests must use 1.

    Returns
    -------
    ADFResult
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be 1-D")
    if np.any(~np.isfinite(y)):
        raise ValueError("y contains NaN/inf; clean the series first")
    nobs = len(y)
    ntrend = {"n": 0, "c": 1, "ct": 2}.get(regression)
    if ntrend is None:
        raise ValueError(f"regression must be 'n', 'c' or 'ct', got {regression!r}")
    if np.std(y) == 0.0:
        raise ValueError("y has zero variance; ADF test undefined")

    if lags is not None:
        if lags < 0:
            raise ValueError(f"lags must be >= 0, got {lags}")
        best_lag = int(lags)
    else:
        if max_lags is None:
            max_lags = int(np.ceil(12.0 * (nobs / 100.0) ** 0.25))
            max_lags = min(max_lags, nobs // 2 - ntrend - 1)
        if max_lags < 0:
            raise ValueError(f"max_lags must be >= 0, got {max_lags}")
        best_aic, best_lag = np.inf, 0
        for p in range(0, max_lags + 1):
            dep, X = _adf_design(y, p, regression, start=max_lags)
            fit = ols(dep, X)
            if fit.aic < best_aic:
                best_aic, best_lag = fit.aic, p

    if len(y) - 1 - best_lag <= best_lag + 1 + ntrend:
        raise ValueError(
            f"series too short (n={nobs}) for ADF with {best_lag} lags"
        )
    dep, X = _adf_design(y, best_lag, regression, start=best_lag)
    fit = ols(dep, X)
    stat = float(fit.tvalues[0])
    eff_nobs = len(dep)
    crit_reg = regression if n_series == 1 else "c"
    crit = mackinnon_crit(n_series=n_series, regression=crit_reg, nobs=eff_nobs)
    return ADFResult(
        stat=stat,
        lags=best_lag,
        nobs=eff_nobs,
        crit=crit,
        n_series=n_series,
        regression=regression,
    )


# --------------------------------------------------------------------------
# Engle-Granger two-step
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EGResult:
    """Engle-Granger two-step test result for a pair (y, x).

    Attributes
    ----------
    beta, alpha : float
        Cointegrating (hedge) ratio and intercept from step 1.
    resid : ndarray
        Step-1 residual spread (dollars).
    adf : ADFResult
        Step-2 ADF on the residuals; ``adf.crit`` holds the MacKinnon
        Engle-Granger (N=2) critical values, NOT plain ADF values.
    intercept : bool
        Whether step 1 included an intercept.
    crit_approx : bool
        True when intercept=False (no tabulated N=2 no-constant surface;
        the N=2 'c' surface is used as a conservative stand-in).
    """

    beta: float
    alpha: float
    resid: np.ndarray
    adf: ADFResult
    intercept: bool
    crit_approx: bool

    @property
    def stat(self) -> float:
        return self.adf.stat

    @property
    def crit(self) -> dict[str, float]:
        return self.adf.crit

    def cointegrated(self, level: str = "5%") -> bool:
        """True if no-cointegration is rejected at ``level``."""
        return self.adf.reject(level)


def engle_granger(
    y: ArrayLike,
    x: ArrayLike,
    intercept: bool = True,
    lags: Optional[int] = None,
    max_lags: Optional[int] = None,
) -> EGResult:
    """Engle-Granger two-step cointegration test for a pair.

    Step 1: OLS y = alpha + beta x + u (intercept optional, default on).
    Step 2: ADF on u_hat with NO constant in the ADF regression (residuals
    are exactly mean-zero when step 1 has an intercept), compared against
    MacKinnon N=2 Engle-Granger critical values.

    Parameters
    ----------
    y, x : array-like (n,)
        Price levels of the two legs (y = dependent leg).
    intercept : bool
        Include intercept in the cointegrating regression (default True).
    lags, max_lags : int, optional
        Passed to :func:`adf_test` for the residual test.

    Returns
    -------
    EGResult
    """
    beta, alpha, resid = hedge_ratio(y, x, intercept=intercept)
    adf = adf_test(
        resid, regression="n", lags=lags, max_lags=max_lags, n_series=2
    )
    return EGResult(
        beta=beta,
        alpha=alpha,
        resid=resid,
        adf=adf,
        intercept=intercept,
        crit_approx=not intercept,
    )
