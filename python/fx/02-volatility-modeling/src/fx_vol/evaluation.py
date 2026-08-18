"""Forecast evaluation and residual diagnostics.

Loss functions
--------------
* **QLIKE**: ``L = log f + r/f`` for forecast variance f and realized proxy r
  (here r_t^2). QLIKE is *robust* in the Patton (2011) sense: its expected
  loss is minimized at the true conditional variance even when the realized
  proxy is noisy -- the reason it is the standard loss for variance forecast
  races. It also penalizes *under*-prediction more than over-prediction,
  matching desk risk preferences (an under-forecast blows the VaR).
* **MSE**: ``(f - r)^2`` -- robust but heavily driven by high-variance days;
  reported alongside QLIKE.

Tests
-----
* Mincer-Zarnowitz calibration regression (HAC/Newey-West covariance),
* Diebold-Mariano equal-predictive-accuracy test with a from-scratch
  Newey-West long-run variance,
* Ljung-Box, ARCH-LM and Engle-Ng sign-bias diagnostics on standardized
  residuals (statsmodels used for the auxiliary OLS regressions).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

__all__ = [
    "qlike_loss",
    "mse_loss",
    "mincer_zarnowitz",
    "newey_west_variance",
    "diebold_mariano",
    "ljung_box",
    "arch_lm",
    "sign_bias_test",
]


def _pair(forecast, realized) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(forecast, dtype=float)
    r = np.asarray(realized, dtype=float)
    if f.shape != r.shape or f.ndim != 1:
        raise ValueError(f"forecast and realized must be 1-D and aligned, got {f.shape} vs {r.shape}")
    if not (np.isfinite(f).all() and np.isfinite(r).all()):
        raise ValueError("forecast/realized contain NaN or infinite values")
    return f, r


def qlike_loss(forecast_var: Sequence[float], realized_var: Sequence[float]) -> np.ndarray:
    """Per-observation QLIKE loss ``log f + r/f`` (variance units in, unitless out).

    Requires strictly positive forecasts; zero realized variance is allowed
    (a legitimately quiet day contributes ``log f``).
    """
    f, r = _pair(forecast_var, realized_var)
    if (f <= 0).any():
        raise ValueError("QLIKE requires strictly positive variance forecasts")
    if (r < 0).any():
        raise ValueError("realized variance proxy must be non-negative")
    return np.log(f) + r / f


def mse_loss(forecast_var: Sequence[float], realized_var: Sequence[float]) -> np.ndarray:
    """Per-observation squared error on variances ``(f - r)^2``."""
    f, r = _pair(forecast_var, realized_var)
    return (f - r) ** 2


def mincer_zarnowitz(
    forecast_var: Sequence[float],
    realized_var: Sequence[float],
    hac_lags: int = 5,
) -> dict:
    """Mincer-Zarnowitz calibration regression ``r_t = a + b f_t + e_t``.

    A well-calibrated variance forecast has (a, b) = (0, 1); ``b < 1`` with
    ``a > 0`` is the classic over-responsive-forecast signature. Inference
    uses HAC (Newey-West) standard errors because squared-return proxies are
    serially dependent.

    Returns
    -------
    dict
        ``intercept``, ``slope``, their HAC standard errors, ``r2``, and
        ``p_joint`` for the Wald test of H0: (a, b) = (0, 1).
    """
    f, r = _pair(forecast_var, realized_var)
    X = sm.add_constant(f)
    fit = sm.OLS(r, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    # Wald test of (a, b) = (0, 1) under the HAC covariance:
    q = np.array([0.0, 1.0])
    diff = fit.params - q
    cov = fit.cov_params()
    stat = float(diff @ np.linalg.solve(cov, diff))
    p_joint = float(stats.chi2.sf(stat, df=2))
    return {
        "intercept": float(fit.params[0]),
        "slope": float(fit.params[1]),
        "se_intercept": float(fit.bse[0]),
        "se_slope": float(fit.bse[1]),
        "r2": float(fit.rsquared),
        "wald_stat": stat,
        "p_joint": p_joint,
        "n": int(f.size),
    }


def newey_west_variance(d: Sequence[float], lags: int) -> float:
    """Newey-West (Bartlett-kernel) long-run variance of a series, from scratch.

    ``lrv = gamma_0 + 2 * sum_{j=1..L} (1 - j/(L+1)) * gamma_j`` with
    autocovariances ``gamma_j`` computed about the sample mean.
    """
    x = np.asarray(d, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("d must be a 1-D series with at least 2 observations")
    if lags < 0 or lags >= x.size:
        raise ValueError(f"lags must be in [0, {x.size - 1}], got {lags}")
    xc = x - x.mean()
    n = x.size
    lrv = float(xc @ xc) / n
    for j in range(1, lags + 1):
        gamma_j = float(xc[j:] @ xc[:-j]) / n
        lrv += 2.0 * (1.0 - j / (lags + 1.0)) * gamma_j
    return lrv


def diebold_mariano(
    loss1: Sequence[float],
    loss2: Sequence[float],
    h: int = 1,
    lags: int | None = None,
) -> dict:
    """Diebold-Mariano test of equal predictive accuracy.

    Tests H0: ``E[loss1 - loss2] = 0`` using the Newey-West long-run variance
    of the loss differential; the statistic is asymptotically N(0,1). A
    *negative* statistic favours model 1 (lower loss).

    Parameters
    ----------
    loss1, loss2 : array-like
        Aligned per-period losses (e.g. QLIKE series) of the two models.
    h : int
        Forecast horizon; sets the default truncation ``lags = h - 1``
        (h-step optimal forecast errors are MA(h-1)).
    lags : int, optional
        Override the Newey-West truncation.

    Returns
    -------
    dict
        ``stat``, ``pvalue`` (two-sided), ``mean_diff``, ``n``, ``lags``.
    """
    l1, l2 = _pair(loss1, loss2)
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")
    d = l1 - l2
    n = d.size
    if n < 10:
        raise ValueError(f"need at least 10 loss observations, got {n}")
    L = (h - 1) if lags is None else lags
    lrv = newey_west_variance(d, L)
    if lrv <= 0:
        raise ValueError("non-positive long-run variance -- losses may be identical")
    stat = float(d.mean() / np.sqrt(lrv / n))
    pvalue = float(2.0 * stats.norm.sf(abs(stat)))
    return {"stat": stat, "pvalue": pvalue, "mean_diff": float(d.mean()), "n": n, "lags": L}


def ljung_box(x: Sequence[float], lags: int = 10) -> dict:
    """Ljung-Box test for serial correlation (use on z_t and z_t^2).

    On squared standardized residuals this checks for *remaining* ARCH
    effects after the fit.
    """
    arr = np.asarray(x, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("input contains NaN or infinite values")
    out = acorr_ljungbox(arr, lags=[lags], return_df=True)
    return {"stat": float(out["lb_stat"].iloc[0]), "pvalue": float(out["lb_pvalue"].iloc[0]), "lags": lags}


def arch_lm(x: Sequence[float], lags: int = 10) -> dict:
    """Engle's ARCH-LM test on a (residual) series.

    Small p-value = conditional heteroskedasticity present. Run on raw
    returns to justify GARCH; run on standardized residuals to verify the
    fitted model has absorbed it.
    """
    arr = np.asarray(x, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("input contains NaN or infinite values")
    lm, lm_pval, f, f_pval = het_arch(arr, nlags=lags)
    return {"stat": float(lm), "pvalue": float(lm_pval), "f_stat": float(f), "f_pvalue": float(f_pval), "lags": lags}


def sign_bias_test(returns: Sequence[float], sigma2: Sequence[float]) -> dict:
    """Engle-Ng (1993) sign and size bias tests.

    Auxiliary regression on standardized residuals:

        z_t^2 = c + b1 * S-_{t-1} + b2 * S-_{t-1} e_{t-1} + b3 * S+_{t-1} e_{t-1} + u_t

    with ``S- = 1[e_{t-1} < 0]``, ``e`` the (raw) residual. Significant b1 =
    sign bias; b2/b3 = negative/positive size bias. The joint F-test is the
    standard pre-test for whether an asymmetric model (GJR/EGARCH) is needed
    -- for G10 FX it typically fails to reject, for EM pairs it rejects
    strongly (see docs/METHODOLOGY.md).

    Returns
    -------
    dict
        t-statistics and p-values for each bias term plus the joint F p-value.
    """
    r, s2 = _pair(returns, sigma2)
    if (s2 <= 0).any():
        raise ValueError("sigma2 must be strictly positive")
    z = r / np.sqrt(s2)
    z2 = z[1:] ** 2
    e_lag = r[:-1]
    s_neg = (e_lag < 0).astype(float)
    X = np.column_stack([s_neg, s_neg * e_lag, (1.0 - s_neg) * e_lag])
    X = sm.add_constant(X)
    fit = sm.OLS(z2, X).fit()
    R = np.zeros((3, 4)); R[0, 1] = R[1, 2] = R[2, 3] = 1.0
    joint = fit.f_test(R)
    return {
        "sign_bias_t": float(fit.tvalues[1]),
        "sign_bias_p": float(fit.pvalues[1]),
        "neg_size_t": float(fit.tvalues[2]),
        "neg_size_p": float(fit.pvalues[2]),
        "pos_size_t": float(fit.tvalues[3]),
        "pos_size_p": float(fit.pvalues[3]),
        "joint_f_p": float(joint.pvalue),
        "n": int(z2.size),
    }
