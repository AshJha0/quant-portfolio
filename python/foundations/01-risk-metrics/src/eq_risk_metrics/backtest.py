"""VaR backtesting: exception counting and Kupiec's proportion-of-failures test.

A VaR number that has never been backtested is a number, not a risk control
(``docs/DESK_GUIDE.md`` §3). The two things a desk checks are:

1. **Unconditional coverage** -- does a 99% VaR actually get exceeded on
   about 1% of days? That is Kupiec's POF test, implemented here.
2. **Independence** -- are the exceptions spread out, or clustered in a
   handful of stressed weeks? Clustering means the model is missing a
   volatility regime shift even when the average rate looks right. That is
   Christoffersen's test, which is *not* implemented here (see
   ``docs/METHODOLOGY.md``); this module deliberately covers the
   unconditional half only, and says so.

Sign convention matches the rest of the package: ``var`` values are
**positive loss fractions**, ``returns`` are signed (a loss is negative), so
an exception is ``return < -var``.
"""
from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["KupiecResult", "count_var_exceptions", "kupiec_pof_test"]


class KupiecResult(TypedDict):
    """Return type of :func:`kupiec_pof_test`."""

    n_observations: int
    n_exceptions: int
    observed_rate: float
    expected_rate: float
    expected_exceptions: float
    lr_statistic: float
    p_value: float
    reject_at_5pct: bool


def count_var_exceptions(
    returns: pd.Series, var: float | pd.Series
) -> pd.Series:
    """Boolean series flagging days whose realised loss exceeded VaR.

    An exception (a "breach", a "hit") on day ``t`` is ``r_t < -var_t``:
    the realised return fell below the negated positive-loss VaR figure.
    The comparison is strict, so a return landing exactly on the VaR
    threshold is not counted -- an arbitrary but conventional choice, and
    with continuous returns a measure-zero one.

    Parameters
    ----------
    returns : pandas.Series
        Realised daily returns, unitless, signed (loss negative).
    var : float or pandas.Series
        VaR as a **positive loss fraction**. A float applies the same
        static VaR to every day (what you get from a single-shot
        calculation); a Series lets you backtest a *rolling* VaR forecast,
        in which case it must share the index of ``returns`` -- this is
        the realistic case, since a desk re-estimates VaR daily.

    Returns
    -------
    pandas.Series
        Boolean series, same index as ``returns``, ``True`` on exception
        days.

    Raises
    ------
    ValueError
        If ``returns`` is empty, if a Series ``var`` does not align with
        ``returns``, or if any VaR value is negative (a negative "loss
        fraction" means the sign convention has been mixed up).
    """
    returns = pd.Series(returns)
    if len(returns) == 0:
        raise ValueError("count_var_exceptions: returns is empty")
    if isinstance(var, pd.Series):
        if not returns.index.equals(var.index):
            raise ValueError(
                "count_var_exceptions: a Series `var` must share the index of "
                f"`returns` (got {len(var)} VaR values for {len(returns)} returns)"
            )
        var_values = var
    else:
        if not np.isfinite(var):
            raise ValueError(f"count_var_exceptions: var must be finite, got {var!r}")
        var_values = pd.Series(float(var), index=returns.index)
    if (var_values.dropna() < 0).any():
        raise ValueError(
            "count_var_exceptions: VaR must be a POSITIVE loss fraction "
            "(this package's convention); a negative value means the sign "
            "convention has been mixed up somewhere upstream"
        )
    return returns < -var_values


def kupiec_pof_test(
    returns: pd.Series, var: float | pd.Series, confidence: float = 0.99
) -> KupiecResult:
    """Kupiec proportion-of-failures (unconditional coverage) test.

    Under the null that the VaR model is correctly calibrated, exceptions
    are i.i.d. Bernoulli with probability ``p = 1 - confidence``. With
    ``x`` exceptions in ``n`` days the likelihood-ratio statistic is

    ``LR_POF = -2 ln[ p^x (1-p)^(n-x) / (x/n)^x (1-x/n)^(n-x) ]``

    which is asymptotically chi-squared with 1 degree of freedom. A large
    statistic (small p-value) rejects the model: **too many** exceptions
    means the VaR understates risk (the serious direction -- the limit
    system has been quietly under-capitalising the book); **too few** also
    rejects, and means the VaR is too conservative, which wastes risk
    budget rather than endangering it. The test is two-sided in that sense,
    so always read ``observed_rate`` against ``expected_rate`` to see which
    side of the null you are on -- the p-value alone does not say.

    Assumptions and limits (worth stating before quoting a p-value):

    - **Asymptotic** chi-squared reference distribution. With the ~250-day
      window a desk typically uses, at 99% you expect only ~2.5 exceptions,
      and the test has very low power -- it will fail to reject a badly
      miscalibrated model far more often than the nominal 5% suggests.
      Basel's traffic-light approach exists partly because of this.
    - **Unconditional only.** It counts exceptions and ignores *when* they
      happened. A model that produces all its exceptions in one stressed
      fortnight passes Kupiec while being obviously broken; catching that
      needs Christoffersen's independence test, which is not implemented
      here.
    - **Same-horizon.** ``returns`` must be realised returns over the same
      horizon the VaR was quoted for (1 day, as everywhere in this package).

    Parameters
    ----------
    returns : pandas.Series
        Realised daily returns, unitless, signed (loss negative).
    var : float or pandas.Series
        VaR as a positive loss fraction -- static (float) or a rolling
        forecast aligned to ``returns`` (Series). See
        :func:`count_var_exceptions`.
    confidence : float
        The confidence level the VaR was quoted at, in (0, 1). The nominal
        exception rate is ``1 - confidence``.

    Returns
    -------
    KupiecResult
        ``n_observations`` (int), ``n_exceptions`` (int), ``observed_rate``
        (float, ``x/n``), ``expected_rate`` (float, ``1 - confidence``),
        ``expected_exceptions`` (float, ``n(1-confidence)``),
        ``lr_statistic`` (float, >= 0), ``p_value`` (float in [0, 1], upper
        tail of chi-squared(1)), and ``reject_at_5pct`` (bool).

    Raises
    ------
    ValueError
        If ``returns`` is empty, ``confidence`` is outside (0, 1), or the
        VaR input is invalid (see :func:`count_var_exceptions`).

    Examples
    --------
    A correctly calibrated model on its own sample is not rejected:

    >>> import numpy as np, pandas as pd
    >>> from eq_risk_metrics import var_historical, kupiec_pof_test
    >>> r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 1000))
    >>> res = kupiec_pof_test(r, var_historical(r, 0.99), 0.99)
    >>> res["reject_at_5pct"]
    False
    """
    returns = pd.Series(returns)
    if len(returns) == 0:
        raise ValueError("kupiec_pof_test: returns is empty")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError(
            "kupiec_pof_test: confidence must be a finite number strictly "
            f"between 0 and 1 (e.g. 0.99), got {confidence!r}"
        )
    exceptions = count_var_exceptions(returns, var)
    n = int(len(returns))
    x = int(exceptions.sum())
    p = 1.0 - confidence
    observed = x / n

    # Log-likelihood under the null (rate p) and the unrestricted MLE
    # (rate x/n), with the 0*log(0) = 0 convention applied explicitly so
    # the x = 0 and x = n corners are finite rather than NaN.
    def _loglik(rate: float) -> float:
        total = 0.0
        if x > 0:
            total += x * np.log(rate) if rate > 0 else -np.inf
        if n - x > 0:
            total += (n - x) * np.log1p(-rate) if rate < 1 else -np.inf
        return total

    lr = -2.0 * (_loglik(p) - _loglik(observed))
    # Clamp away floating-point noise: the unrestricted likelihood is a
    # maximum, so LR >= 0 mathematically, but it can come out at -1e-16.
    lr = float(max(lr, 0.0))
    p_value = float(stats.chi2.sf(lr, df=1))
    return {
        "n_observations": n,
        "n_exceptions": x,
        "observed_rate": observed,
        "expected_rate": p,
        "expected_exceptions": n * p,
        "lr_statistic": lr,
        "p_value": p_value,
        "reject_at_5pct": bool(p_value < 0.05),
    }
