"""Forecast evaluation and in-sample diagnostics.

Loss functions and the noisy-proxy problem
------------------------------------------
Realised conditional variance is unobservable; we evaluate forecasts against a
noisy *proxy* (squared returns, or a realized-variance estimate). Patton
(2011, "Volatility forecast comparison using imperfect volatility proxies",
Journal of Econometrics 160) shows that only a specific class of losses is
**robust** to this noise, meaning the expected-loss ranking of forecasts under
the proxy equals the ranking under the true conditional variance. MSE and
QLIKE are the two canonical members:

* ``QLIKE(f, p) = ln f + p / f`` — minimised in expectation at f = E[p]
  (proved by first-order condition; empirically verified in tests). QLIKE
  weights errors multiplicatively, so it does not let a few high-variance days
  dominate, and it penalises *under*-prediction more than over-prediction —
  the asymmetry a risk desk wants.
* ``MSE(f, p) = (f - p)^2`` — also robust, but its expectation is dominated
  by the (proxy-noise) level in high-variance episodes, making rankings far
  noisier in practice. We report both; QLIKE is the headline metric.

Pairwise significance uses the Diebold-Mariano (1995) test with a HAC
(Newey-West) long-run variance. In-sample adequacy: AIC/BIC, Ljung-Box on
squared standardised residuals, Engle's ARCH-LM, and the Engle-Ng (1993)
sign-bias test for unmodelled asymmetry.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

from ._results import VolatilityFitResult

__all__ = [
    "qlike_loss",
    "mse_loss",
    "mincer_zarnowitz",
    "diebold_mariano",
    "DMResult",
    "MZResult",
    "ljung_box_squared",
    "arch_lm_test",
    "sign_bias_test",
    "forecast_race_table",
]


def _validate_pair(forecast: np.ndarray, proxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(forecast, dtype=float).ravel()
    p = np.asarray(proxy, dtype=float).ravel()
    if f.size != p.size:
        raise ValueError(f"forecast ({f.size}) and proxy ({p.size}) lengths differ")
    if f.size == 0:
        raise ValueError("empty forecast/proxy arrays")
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(p))):
        raise ValueError("forecast/proxy contain NaN or inf; clean inputs first")
    return f, p


def qlike_loss(forecast: np.ndarray, proxy: np.ndarray) -> np.ndarray:
    """Element-wise QLIKE loss ln(f) + p/f (Patton-robust; see module docs).

    ``forecast`` must be strictly positive; the proxy may contain zeros
    (e.g. a zero return day), which this form handles gracefully — the
    alternative normalised form p/f - ln(p/f) - 1 does not.
    """
    f, p = _validate_pair(forecast, proxy)
    if np.any(f <= 0):
        raise ValueError("QLIKE requires strictly positive variance forecasts")
    if np.any(p < 0):
        raise ValueError("variance proxy must be non-negative")
    return np.log(f) + p / f


def mse_loss(forecast: np.ndarray, proxy: np.ndarray) -> np.ndarray:
    """Element-wise squared error (f - p)^2 on the variance scale."""
    f, p = _validate_pair(forecast, proxy)
    return (f - p) ** 2


class MZResult(NamedTuple):
    """Mincer-Zarnowitz regression proxy_t = a + b * forecast_t + e_t.

    Unbiased, efficient forecasts have (a, b) = (0, 1). ``joint_pvalue`` is
    the HAC-robust Wald test of that joint hypothesis.
    """

    intercept: float
    slope: float
    intercept_se: float
    slope_se: float
    r2: float
    joint_stat: float
    joint_pvalue: float


def mincer_zarnowitz(
    forecast: np.ndarray, proxy: np.ndarray, hac_lags: int | None = None
) -> MZResult:
    """Mincer-Zarnowitz forecast-efficiency regression with HAC errors.

    Regresses the realised proxy on the forecast; slope 1 and intercept 0
    mean the forecast is (conditionally) unbiased and fully exploits its own
    information. A slope < 1 is the classic symptom of an over-reactive
    forecast; note the noisy proxy attenuates R^2 but not the slope.
    """
    f, p = _validate_pair(forecast, proxy)
    if hac_lags is None:
        hac_lags = int(np.floor(1.5 * f.size ** (1.0 / 3.0)))
    X = sm.add_constant(f)
    res = sm.OLS(p, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    # H0: intercept = 0 and slope = 1
    wald = res.wald_test((np.eye(2), np.array([0.0, 1.0])), scalar=True)
    return MZResult(
        intercept=float(res.params[0]),
        slope=float(res.params[1]),
        intercept_se=float(res.bse[0]),
        slope_se=float(res.bse[1]),
        r2=float(res.rsquared),
        joint_stat=float(wald.statistic),
        joint_pvalue=float(wald.pvalue),
    )


class DMResult(NamedTuple):
    """Diebold-Mariano test of equal predictive accuracy.

    ``stat`` is asymptotically N(0,1) under H0: E[loss1 - loss2] = 0.
    Negative stat => model 1 has *lower* loss (better). ``pvalue`` is
    two-sided.
    """

    stat: float
    pvalue: float
    mean_loss_diff: float


def diebold_mariano(
    loss1: np.ndarray,
    loss2: np.ndarray,
    h: int = 1,
    hac_lags: int | None = None,
    harvey_correction: bool = True,
) -> DMResult:
    """Diebold-Mariano (1995) test with Newey-West (HAC) long-run variance.

    Parameters
    ----------
    loss1, loss2 : array-like
        Per-period losses of the two competing forecasts (same loss function,
        same evaluation sample).
    h : int
        Forecast horizon; the default HAC lag is h - 1 (h-step optimal
        forecast errors are MA(h-1) under correct specification).
    hac_lags : int, optional
        Override the Newey-West truncation lag (Bartlett kernel).
    harvey_correction : bool
        Apply the Harvey-Leybourne-Newbold (1997) small-sample scaling and
        use the t_{n-1} distribution instead of the normal.
    """
    l1 = np.asarray(loss1, dtype=float).ravel()
    l2 = np.asarray(loss2, dtype=float).ravel()
    if l1.size != l2.size:
        raise ValueError("loss series must have equal length")
    d = l1 - l2
    n = d.size
    if n < 10:
        raise ValueError(f"need at least 10 loss observations for DM, got {n}")
    if hac_lags is None:
        hac_lags = max(h - 1, 0)
    d_c = d - d.mean()
    lrv = float(d_c @ d_c) / n
    for lag in range(1, hac_lags + 1):
        w = 1.0 - lag / (hac_lags + 1.0)  # Bartlett kernel
        lrv += 2.0 * w * float(d_c[lag:] @ d_c[:-lag]) / n
    if lrv <= 0:
        raise ValueError(
            "non-positive HAC variance estimate (degenerate loss differential); "
            "DM statistic undefined"
        )
    dm = d.mean() / np.sqrt(lrv / n)
    if harvey_correction:
        adj = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        dm *= adj
        pvalue = 2.0 * stats.t.sf(abs(dm), df=n - 1)
    else:
        pvalue = 2.0 * stats.norm.sf(abs(dm))
    return DMResult(stat=float(dm), pvalue=float(pvalue), mean_loss_diff=float(d.mean()))


# ---------------------------------------------------------------------------
# in-sample diagnostics
# ---------------------------------------------------------------------------

def ljung_box_squared(
    std_residuals: np.ndarray, lags: int = 10, model_df: int = 0
) -> pd.DataFrame:
    """Ljung-Box test on **squared** standardised residuals.

    If the variance model is adequate, z_t^2 should be serially uncorrelated;
    significant autocorrelation means unmodelled volatility clustering.
    Returns the statsmodels table (lb_stat, lb_pvalue).
    """
    z = np.asarray(std_residuals, dtype=float).ravel()
    if z.size <= lags + 1:
        raise ValueError(f"need more than {lags + 1} residuals for Ljung-Box({lags})")
    return acorr_ljungbox(z**2, lags=[lags], model_df=model_df)


def arch_lm_test(residuals: np.ndarray, lags: int = 10) -> dict[str, float]:
    """Engle's ARCH-LM test for remaining conditional heteroskedasticity.

    Applied to *standardised* residuals of a fitted model: a small p-value
    means ARCH effects remain (model inadequate). Applied to raw returns it
    is the standard pre-test for whether a GARCH-type model is needed at all.
    """
    x = np.asarray(residuals, dtype=float).ravel()
    if x.size <= 2 * lags:
        raise ValueError(f"need more than {2 * lags} observations for ARCH-LM({lags})")
    lm, lm_pvalue, fstat, f_pvalue = het_arch(x, nlags=lags)
    return {"lm_stat": float(lm), "lm_pvalue": float(lm_pvalue), "f_stat": float(fstat), "f_pvalue": float(f_pvalue)}


def sign_bias_test(result: VolatilityFitResult) -> pd.DataFrame:
    """Engle-Ng (1993) sign-bias test for unmodelled asymmetry.

    Regress z_t^2 on [1, S_{t-1}, S_{t-1} r_{t-1}, (1-S_{t-1}) r_{t-1}] where
    S = 1[r < 0]. Significant coefficients mean the fitted model's variance
    misses sign/size effects of lagged shocks — e.g. a symmetric GARCH fitted
    to leveraged data fails the negative-size-bias row, while GJR/EGARCH fits
    pass. Returns a table with sign_bias, negative_size_bias,
    positive_size_bias and joint rows (t/F stats and p-values).
    """
    z2 = result.std_residuals[1:] ** 2
    r_lag = result.returns[:-1]
    s_neg = (r_lag < 0).astype(float)
    X = sm.add_constant(np.column_stack([s_neg, s_neg * r_lag, (1.0 - s_neg) * r_lag]))
    ols = sm.OLS(z2, X).fit()
    joint = ols.f_test(np.eye(4)[1:])  # all three slopes zero
    rows = [
        {"test": "sign_bias", "stat": float(ols.tvalues[1]), "pvalue": float(ols.pvalues[1])},
        {"test": "negative_size_bias", "stat": float(ols.tvalues[2]), "pvalue": float(ols.pvalues[2])},
        {"test": "positive_size_bias", "stat": float(ols.tvalues[3]), "pvalue": float(ols.pvalues[3])},
        {"test": "joint_F", "stat": float(joint.statistic), "pvalue": float(joint.pvalue)},
    ]
    return pd.DataFrame(rows).set_index("test")


def forecast_race_table(
    forecasts: dict[str, np.ndarray],
    proxy: np.ndarray,
    benchmark: str | None = None,
    h: int = 1,
) -> pd.DataFrame:
    """Compare competing variance forecasts on QLIKE and MSE with DM tests.

    Parameters
    ----------
    forecasts : dict
        Model name -> variance forecast array (aligned with ``proxy``).
    proxy : array-like
        Realised variance proxy for the same dates.
    benchmark : str, optional
        Model against which DM tests are run (default: first key). Negative
        DM stat = the row's model beats the benchmark on QLIKE.

    Notes
    -----
    On *model confidence*: with a handful of models a full Model Confidence
    Set (Hansen-Lunde-Nash) is overkill; pairwise DM against a benchmark plus
    the loss ranking conveys the same message. Interpret |DM| < 2 as "no
    significant difference" — on 500 observations only large accuracy gaps
    are detectable, which is itself a desk-relevant fact.
    """
    if not forecasts:
        raise ValueError("forecasts dict is empty")
    if benchmark is None:
        benchmark = next(iter(forecasts))
    if benchmark not in forecasts:
        raise ValueError(f"benchmark {benchmark!r} not in forecasts")
    p = np.asarray(proxy, dtype=float).ravel()
    bench_q = qlike_loss(forecasts[benchmark], p)
    rows = []
    for name, f in forecasts.items():
        q = qlike_loss(f, p)
        m = mse_loss(f, p)
        row = {"model": name, "qlike": float(q.mean()), "mse": float(m.mean())}
        if name != benchmark:
            dm = diebold_mariano(q, bench_q, h=h)
            row["dm_stat_vs_benchmark"] = dm.stat
            row["dm_pvalue"] = dm.pvalue
        else:
            row["dm_stat_vs_benchmark"] = np.nan
            row["dm_pvalue"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows).set_index("model").sort_values("qlike")
    return out
