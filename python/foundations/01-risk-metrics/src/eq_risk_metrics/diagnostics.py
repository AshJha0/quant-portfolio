"""Return-distribution diagnostics: skewness, kurtosis, normality testing."""
from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd
from scipy import stats

from ._validate import finite_series

__all__ = ["NormalityReport", "normality_report"]


class NormalityReport(TypedDict):
    """Return type of :func:`normality_report`."""

    skewness: float
    excess_kurtosis: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    normality_rejected_at_5pct: bool


def normality_report(returns: pd.Series) -> NormalityReport:
    """Skewness, excess kurtosis, and the Jarque-Bera normality test.

    The Jarque-Bera statistic is
    ``JB = n/6 * (S^2 + K^2/4)`` where ``S`` is sample skewness and
    ``K`` is sample excess kurtosis, asymptotically chi-squared(2)
    under the null of normality.

    Assumption: the chi-squared(2) reference distribution is an
    asymptotic (large-``n``) approximation; in small samples the test
    has poor size (over-rejects) and low power, so a rejection with
    ``n`` in the tens or low hundreds should be treated cautiously.

    Interpretation: for daily equity returns this test almost always
    rejects normality, with positive excess kurtosis (fat tails). That
    rejection is the direct justification for not trusting
    :func:`eq_risk_metrics.var_es.var_parametric` at high confidence --
    see ``docs/METHODOLOGY.md``.

    Parameters
    ----------
    returns : pandas.Series or array_like
        Simple (or log) daily returns, unitless. At least 3
        observations, all finite and not all equal; otherwise
        ``ValueError`` (skewness and kurtosis of a constant sample are 0/0).

    Returns
    -------
    NormalityReport
        ``skewness`` (float), ``excess_kurtosis`` (float, Fisher
        definition: 0 for a normal distribution), ``jarque_bera_stat``
        (float, >= 0), ``jarque_bera_pvalue`` (float in [0, 1]), and
        ``normality_rejected_at_5pct`` (bool, ``p < 0.05``).
    """
    returns = finite_series(returns, "normality_report", min_len=3)
    x = returns.to_numpy(dtype=float)
    # A sample is treated as constant when its spread is at or below
    # floating-point resolution at the sample's own scale. That covers an
    # exactly constant sample and one whose variance underflows to zero
    # (e.g. [0, 0, 5e-324]); either way skewness and kurtosis are 0/0.
    centred = x - x.mean()
    m2 = float(np.mean(centred * centred))
    scale = float(np.max(np.abs(x)))
    if not m2 > (np.finfo(float).eps * scale) ** 2:
        raise ValueError(
            "normality_report: the sample is constant to floating-point "
            "resolution, so skewness and kurtosis are undefined"
        )
    skewness = float(stats.skew(x))
    excess_kurtosis = float(stats.kurtosis(x))
    jb_stat, jb_p = stats.jarque_bera(x)
    if not all(np.isfinite([skewness, excess_kurtosis, jb_stat, jb_p])):
        raise ValueError(
            "normality_report: moments are not finite for this sample "
            "(variance too small relative to the data's scale)"
        )
    return {
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_p),
        "normality_rejected_at_5pct": jb_p < 0.05,
    }
