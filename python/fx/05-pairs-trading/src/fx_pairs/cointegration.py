"""Engle-Granger cointegration and ADF unit-root testing, implemented from scratch.

Everything here is built on plain OLS (``numpy.linalg.lstsq``) so it can be
cross-validated against ``statsmodels`` in the test suite:

* :func:`adf_test` — Augmented Dickey-Fuller regression with fixed lags or
  AIC auto-lag selection, replicating the ``statsmodels.tsa.stattools.adfuller``
  construction (same sample trimming, same AIC), so t-statistics match to
  ~1e-8.
* :func:`engle_granger` — two-step Engle-Granger: OLS cointegrating
  regression with intercept, then ADF (no deterministic terms) on the
  residuals, judged against MacKinnon (2010) **N=2** critical values.  Using
  plain N=1 ADF critical values on estimated residuals over-rejects — the
  test suite asserts the two tables differ.
* Degenerate-spread detection: an exact-identity spread (e.g. the triangular
  spread ``log EURUSD + log USDJPY - log EURJPY`` which is identically zero
  under no-arbitrage) has ~zero variance.  The ADF regression is then
  numerically meaningless, and economically there is nothing to trade (any
  deviation is inside transaction costs).  Such spreads are flagged
  ``degenerate=True`` and never reported as cointegrated.

Critical values use the MacKinnon (2010) response surface
``cv = b0 + b1/T + b2/T^2 + b3/T^3`` with the published coefficients for a
constant-only cointegrating regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "ADFResult",
    "EngleGrangerResult",
    "adf_test",
    "engle_granger",
    "mackinnon_crit",
    "is_degenerate_spread",
]

# MacKinnon (2010) response-surface coefficients, constant ('c') case.
# Rows: 1%, 5%, 10%.  Columns: b0, b1, b2, b3 in cv = b0 + b1/T + b2/T^2 + b3/T^3.
_TAU_C_2010: dict[int, np.ndarray] = {
    # N = number of I(1) variables (1 = plain ADF, 2 = Engle-Granger on 2 series)
    1: np.array([
        [-3.43035, -6.5393, -16.786, -79.433],
        [-2.86154, -2.8903, -4.234, -40.040],
        [-2.56677, -1.5384, -2.809, 0.0],
    ]),
    2: np.array([
        [-3.89644, -10.9519, -33.527, 0.0],
        [-3.33613, -6.1101, -6.823, 0.0],
        [-3.04445, -4.2412, -2.720, 0.0],
    ]),
    3: np.array([
        [-4.29374, -14.4354, -33.195, 47.433],
        [-3.74066, -8.5632, -10.852, 27.982],
        [-3.45218, -6.2143, -3.718, 0.0],
    ]),
}

_LEVELS = ("1%", "5%", "10%")


def mackinnon_crit(n_vars: int, nobs: float | int = np.inf) -> dict[str, float]:
    """MacKinnon (2010) finite-sample critical values, constant-only case.

    Parameters
    ----------
    n_vars : int
        Number of I(1) series in the cointegrating relation: 1 for a plain
        ADF test, 2 for Engle-Granger on a pair, 3 for a triple.
    nobs : float
        Effective sample size T; ``numpy.inf`` gives asymptotic values.

    Returns
    -------
    dict
        ``{"1%": cv, "5%": cv, "10%": cv}``.
    """
    if n_vars not in _TAU_C_2010:
        raise ValueError(f"n_vars must be one of {sorted(_TAU_C_2010)}, got {n_vars}")
    b = _TAU_C_2010[n_vars]
    if np.isinf(nobs):
        cvs = b[:, 0]
    else:
        T = float(nobs)
        cvs = b[:, 0] + b[:, 1] / T + b[:, 2] / T**2 + b[:, 3] / T**3
    return {lvl: float(cv) for lvl, cv in zip(_LEVELS, cvs)}


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """OLS with classical standard errors.

    Returns
    -------
    (beta, se, rss, resid)
    """
    n, k = X.shape
    if n <= k:
        raise ValueError(f"not enough observations ({n}) for {k} regressors")
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    sigma2 = rss / (n - k)
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    return beta, se, rss, resid


def _gaussian_llf(rss: float, nobs: int) -> float:
    """Gaussian log-likelihood at the OLS optimum (matches statsmodels OLS)."""
    return -0.5 * nobs * (np.log(2.0 * np.pi) + np.log(rss / nobs) + 1.0)


@dataclass
class ADFResult:
    """Result of an Augmented Dickey-Fuller test.

    Attributes
    ----------
    stat : float
        t-statistic on the lagged level (the Dickey-Fuller tau statistic).
    crit_values : dict
        MacKinnon (2010) critical values at 1/5/10% for the effective sample.
    pvalue : float
        Approximate p-value by monotone log-interpolation across the critical
        values (documented as approximate; decisions should use
        ``crit_values``).
    usedlag : int
        Number of lagged differences included.
    nobs : int
        Effective number of observations in the ADF regression.
    regression : str
        ``"c"`` (constant) or ``"n"`` (no deterministic terms).
    """

    stat: float
    crit_values: dict[str, float]
    pvalue: float
    usedlag: int
    nobs: int
    regression: str

    def reject(self, level: str = "5%") -> bool:
        """True if the unit root is rejected at ``level``."""
        return self.stat < self.crit_values[level]


def _approx_pvalue(stat: float, crit: dict[str, float]) -> float:
    """Crude monotone p-value from the three critical values.

    Piecewise-linear in the statistic through (cv1%, .01), (cv5%, .05),
    (cv10%, .10), with flat-ish extrapolation clipped to [1e-4, 0.999].
    Documented as approximate — use critical values for decisions.
    """
    xs = np.array([crit["1%"], crit["5%"], crit["10%"]])
    ps = np.array([0.01, 0.05, 0.10])
    if stat <= xs[0]:
        # linear extrapolation on the left, clipped
        slope = (ps[1] - ps[0]) / (xs[1] - xs[0])
        p = ps[0] + slope * (stat - xs[0])
    elif stat >= xs[2]:
        slope = (0.50 - ps[2]) / 1.0  # ~0.4 probability mass per unit of tau
        p = ps[2] + slope * (stat - xs[2])
    else:
        p = float(np.interp(stat, xs, ps))
    return float(np.clip(p, 1e-4, 0.999))


def _build_adf_arrays(
    y: np.ndarray, maxlag: int, regression: str
) -> tuple[np.ndarray, np.ndarray]:
    """Dependent variable and full regressor matrix trimmed at ``maxlag``.

    Rows are t = maxlag+1 .. n-1 (0-based).  Columns of X:
    ``[y_{t-1}, dy_{t-1}, ..., dy_{t-maxlag}, (const)]``.
    """
    dy = np.diff(y)
    n = len(y)
    nobs = n - maxlag - 1
    if nobs < 5:
        raise ValueError(
            f"series too short for ADF with maxlag={maxlag}: effective nobs={nobs}"
        )
    dep = dy[maxlag:]
    cols = [y[maxlag:-1]]
    for j in range(1, maxlag + 1):
        cols.append(dy[maxlag - j : n - 1 - j])
    if regression == "c":
        cols.append(np.ones(nobs))
    X = np.column_stack(cols)
    return dep, X


def adf_test(
    y: pd.Series | np.ndarray,
    regression: str = "c",
    lags: int | None = None,
    max_lags: int | None = None,
    n_vars: int = 1,
) -> ADFResult:
    """Augmented Dickey-Fuller unit-root test (from scratch).

    Regression: ``dy_t = (a) + rho * y_{t-1} + sum_j phi_j dy_{t-j} + e_t``;
    the statistic is the OLS t-ratio on ``rho``.  With ``lags=None`` the lag
    order is chosen by AIC over ``0..max_lags`` on a common trimmed sample and
    the model is then refit on the longest sample for that lag — the same
    protocol as ``statsmodels.tsa.stattools.adfuller(autolag="AIC")``, so the
    statistics agree to numerical precision.

    Parameters
    ----------
    y : array-like
        Level series.
    regression : {"c", "n"}
        Constant, or no deterministic terms (used on Engle-Granger residuals,
        which are mean-zero by construction).
    lags : int, optional
        Fixed number of lagged differences.  ``None`` selects by AIC.
    max_lags : int, optional
        Upper bound for the AIC search; default is Schwert's
        ``ceil(12 * (n/100)^0.25)`` capped by the sample.
    n_vars : int
        Which MacKinnon critical-value family to report (1 = plain ADF,
        2 = Engle-Granger residuals of a pair).

    Returns
    -------
    ADFResult
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if np.isnan(y).any():
        raise ValueError("y contains NaNs; clean the series first")
    if regression not in ("c", "n"):
        raise ValueError(f"regression must be 'c' or 'n', got {regression!r}")
    n = len(y)
    if n < 15:
        raise ValueError(f"series too short for ADF: n={n}")

    if lags is None:
        if max_lags is None:
            ntrend = 1 if regression == "c" else 0
            max_lags = int(np.ceil(12.0 * (n / 100.0) ** 0.25))
            max_lags = min(max_lags, n // 2 - ntrend - 1)
        max_lags = max(int(max_lags), 0)
        dep, Xfull = _build_adf_arrays(y, max_lags, regression)
        nobs_sel = len(dep)
        best = (np.inf, 0)
        for k in range(0, max_lags + 1):
            keep = list(range(0, k + 1))
            if regression == "c":
                keep.append(Xfull.shape[1] - 1)
            Xk = Xfull[:, keep]
            _, _, rss, _ = _ols(Xk, dep)
            llf = _gaussian_llf(rss, nobs_sel)
            aic = -2.0 * llf + 2.0 * Xk.shape[1]
            if aic < best[0]:
                best = (aic, k)
        usedlag = best[1]
    else:
        usedlag = int(lags)
        if usedlag < 0:
            raise ValueError("lags must be >= 0")

    dep, X = _build_adf_arrays(y, usedlag, regression)
    beta, se, _, _ = _ols(X, dep)
    stat = float(beta[0] / se[0])
    nobs = len(dep)
    crit = mackinnon_crit(n_vars, nobs)
    return ADFResult(
        stat=stat,
        crit_values=crit,
        pvalue=_approx_pvalue(stat, crit),
        usedlag=usedlag,
        nobs=nobs,
        regression=regression,
    )


def is_degenerate_spread(
    spread: pd.Series | np.ndarray,
    abs_tol: float = 1e-7,
    rel_tol: float = 1e-5,
    level: pd.Series | np.ndarray | None = None,
) -> bool:
    """Detect an (economically untradable) exact-identity spread.

    A spread is degenerate when its standard deviation is below ``abs_tol``
    in log units, or below ``rel_tol`` times the standard deviation of the
    level series it was built from.  The canonical example is the triangular
    spread ``log EURUSD + log USDJPY - log EURJPY``, identically zero under
    no-arbitrage: cointegration machinery must flag it rather than declare a
    'perfectly mean-reverting' trade.  Any real deviation lives inside the
    bid-ask spread at daily frequency, so there is no true arbitrage to trade.
    """
    s = np.asarray(spread, dtype=float)
    sd = float(np.nanstd(s))
    if sd < abs_tol:
        return True
    if level is not None:
        lv = float(np.nanstd(np.asarray(level, dtype=float)))
        if lv > 0 and sd < rel_tol * lv:
            return True
    return False


@dataclass
class EngleGrangerResult:
    """Result of the two-step Engle-Granger procedure on (log) levels.

    Attributes
    ----------
    alpha, beta : float
        Cointegrating regression ``y = alpha + beta x + u`` (OLS).
    stat : float
        ADF tau statistic on the residuals (regression ``"n"``), NaN when
        degenerate.
    crit_values : dict
        MacKinnon (2010) N=2 critical values (constant case).
    cointegrated : bool
        ``stat < 5% critical value`` and not degenerate.
    degenerate : bool
        Residual spread has ~zero variance (exact identity, e.g. triangular).
    resid : numpy.ndarray
        Cointegrating residuals (the spread).
    """

    alpha: float
    beta: float
    stat: float
    crit_values: dict[str, float]
    pvalue: float
    cointegrated: bool
    degenerate: bool
    usedlag: int
    nobs: int
    resid: np.ndarray = field(repr=False)


def engle_granger(
    y: pd.Series | np.ndarray,
    x: pd.Series | np.ndarray,
    lags: int | None = None,
    max_lags: int | None = None,
    degenerate_abs_tol: float = 1e-7,
) -> EngleGrangerResult:
    """Two-step Engle-Granger cointegration test (from scratch).

    Step 1: OLS ``y_t = alpha + beta x_t + u_t`` on (log) levels.
    Step 2: ADF on ``u_t`` with **no** deterministic terms, judged against
    MacKinnon (2010) N=2 critical values — the residuals are estimated, so
    plain N=1 ADF critical values would over-reject.

    Degenerate spreads (std below ``degenerate_abs_tol`` in log units, e.g. a
    triangular identity) are flagged and never reported as cointegrated: the
    ADF regression on a numerically-zero series is meaningless and there is
    nothing tradable — deviations are inside transaction costs.

    Parameters
    ----------
    y, x : array-like
        Log price levels of the two currency pairs.
    lags, max_lags : see :func:`adf_test`.
    degenerate_abs_tol : float
        Threshold on the residual standard deviation (log units).

    Returns
    -------
    EngleGrangerResult
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.shape != x.shape:
        raise ValueError(f"y and x must have equal length, got {y.shape} vs {x.shape}")
    if y.ndim != 1:
        raise ValueError("y and x must be one-dimensional")
    if len(y) < 30:
        raise ValueError(f"series too short for Engle-Granger: n={len(y)}")
    if np.isnan(y).any() or np.isnan(x).any():
        raise ValueError("inputs contain NaNs; clean the series first")

    X = np.column_stack([np.ones(len(x)), x])
    coefs, _, _, resid = _ols(X, y)
    alpha, beta = float(coefs[0]), float(coefs[1])

    if is_degenerate_spread(resid, abs_tol=degenerate_abs_tol):
        crit = mackinnon_crit(2, len(resid))
        return EngleGrangerResult(
            alpha=alpha, beta=beta, stat=float("nan"), crit_values=crit,
            pvalue=float("nan"), cointegrated=False, degenerate=True,
            usedlag=0, nobs=len(resid), resid=resid,
        )

    adf = adf_test(resid, regression="n", lags=lags, max_lags=max_lags, n_vars=2)
    crit = mackinnon_crit(2, adf.nobs)
    stat = adf.stat
    return EngleGrangerResult(
        alpha=alpha, beta=beta, stat=stat, crit_values=crit,
        pvalue=_approx_pvalue(stat, crit),
        cointegrated=bool(stat < crit["5%"]), degenerate=False,
        usedlag=adf.usedlag, nobs=adf.nobs, resid=resid,
    )
