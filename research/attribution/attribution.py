"""W23a factor attribution: OLS with Newey-West errors, rolling betas, information ratio.

Plain idea: before calling a strategy's return "alpha", regress it on the
known premia it might simply be harvesting. For an FX book that is the dollar
factor DOL and the carry factor HML_FX; for an equity book the Fama-French
factors. What is left (the intercept) is the part the known factors do not
explain, and its t-statistic must allow for the fact that monthly returns are
autocorrelated and heteroskedastic.

Formal statement. ``y_t = a + b' X_t + e_t`` by OLS. The covariance of
``(a, b)`` is the Newey-West (1987) HAC estimator with Bartlett weights,
``V = (X'X)^-1 S (X'X)^-1``, ``S = G_0 + sum_{k=1}^{L} w_k (G_k + G_k')``,
``G_k = sum_t e_t e_{t-k} x_t x_{t-k}'``, ``w_k = 1 - k/(L+1)``. The default
lag is ``floor(4 (T/100)^(2/9))`` (Newey and West 1994). ``small_sample``
multiplies ``V`` by ``T/(T-K)``. The information ratio is the annualised
alpha over the annualised residual standard deviation,
``IR = a P / (sd(e) sqrt(P))`` with ``P`` periods per year (Grinold and
Kahn 2000, ch. 5).

Named factor sets:

* ``fx``: ``DOL`` (the average of the sorted portfolios) and ``HML_FX``
  (last minus first portfolio), built from ``research/lib/panel.lrv_sort``
  (Lustig, Roussanov and Verdelhan 2011, section II);
* ``ff3``: ``Mkt-RF``, ``SMB``, ``HML``; ``ff5``: plus ``RMW``, ``CMA``;
  ``ff5mom``: plus ``Mom`` (Kenneth French library, ``qp_data.french``).

Worked example (``tests/test_attribution.py``): ``y = 0.002 + 0.5 x`` with
no noise recovers ``a = 0.002`` and ``b = 0.5`` exactly. On random data the
coefficients and HAC standard errors equal ``statsmodels`` OLS with
``cov_type="HAC"`` to 1e-10.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

FACTOR_SETS = {
    "fx": ("DOL", "HML_FX"),
    "ff3": ("Mkt-RF", "SMB", "HML"),
    "ff5": ("Mkt-RF", "SMB", "HML", "RMW", "CMA"),
    "ff5mom": ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"),
}
MIN_OBS = 24


def default_lag(t: int) -> int:
    return int(math.floor(4.0 * (t / 100.0) ** (2.0 / 9.0)))


def fx_factors(ports: pd.DataFrame) -> pd.DataFrame:
    """DOL and HML_FX from an ``lrv_sort`` frame (columns P1..Pn, HML)."""
    p = ports[[c for c in ports.columns if c.startswith("P")]]
    return pd.DataFrame({"DOL": p.mean(axis=1, skipna=False), "HML_FX": ports["HML"]})


def select(factors: pd.DataFrame, name: str) -> pd.DataFrame:
    cols = FACTOR_SETS[name]
    missing = [c for c in cols if c not in factors.columns]
    if missing:
        raise KeyError(f"factor set {name!r} needs {missing}")
    return factors[list(cols)]


def _hac(xm: np.ndarray, e: np.ndarray, lag: int) -> np.ndarray:
    u = xm * e[:, None]
    s = u.T @ u
    for k in range(1, lag + 1):
        g = u[k:].T @ u[:-k]
        s += (1.0 - k / (lag + 1.0)) * (g + g.T)
    inv = np.linalg.inv(xm.T @ xm)
    return inv @ s @ inv


def ols_hac(y: pd.Series, factors: pd.DataFrame, lag: int | None = None,
            small_sample: bool = False, periods_per_year: int = 12) -> dict:
    """Regress ``y`` on ``factors`` (inner-joined, NaN rows dropped)."""
    j = pd.concat([y.rename("_y"), factors], axis=1, join="inner").dropna()
    t = len(j)
    if t < MIN_OBS:
        raise ValueError(f"only {t} aligned observations (< {MIN_OBS})")
    yv = j["_y"].to_numpy(dtype=float)
    names = ["alpha", *factors.columns]
    xm = np.column_stack([np.ones(t), j[list(factors.columns)].to_numpy(dtype=float)])
    coef, *_ = np.linalg.lstsq(xm, yv, rcond=None)
    e = yv - xm @ coef
    lag = default_lag(t) if lag is None else lag
    v = _hac(xm, e, lag)
    if small_sample:
        v *= t / (t - xm.shape[1])
    se = np.sqrt(np.diag(v))
    tss = float(((yv - yv.mean()) ** 2).sum())
    sd_e = float(e.std(ddof=xm.shape[1]))
    a = float(coef[0])
    return {
        "coef": pd.Series(coef, index=names),
        "se": pd.Series(se, index=names),
        "t": pd.Series(coef / se, index=names),
        "alpha_annual": a * periods_per_year,
        "alpha_t": float(coef[0] / se[0]),
        "information_ratio": a * periods_per_year / (sd_e * math.sqrt(periods_per_year))
        if sd_e > 0 else None,
        "r2": 1.0 - float(e @ e) / tss if tss > 0 else None,
        "nobs": t, "lag": lag,
        "resid": pd.Series(e, index=j.index),
    }


def rolling_betas(y: pd.Series, factors: pd.DataFrame, window: int = 36) -> pd.DataFrame:
    """OLS coefficients (alpha first) over a trailing window, row = window end."""
    j = pd.concat([y.rename("_y"), factors], axis=1, join="inner").dropna()
    if window <= factors.shape[1] + 1:
        raise ValueError("window too short for the number of factors")
    rows = {}
    for end in range(window, len(j) + 1):
        w = j.iloc[end - window:end]
        xm = np.column_stack([np.ones(window), w[list(factors.columns)].to_numpy(dtype=float)])
        coef, *_ = np.linalg.lstsq(xm, w["_y"].to_numpy(dtype=float), rcond=None)
        rows[j.index[end - 1]] = coef
    return pd.DataFrame.from_dict(rows, orient="index", columns=["alpha", *factors.columns])


def attribute(y: pd.Series, factors: pd.DataFrame, set_name: str, **kw: object) -> dict:
    """``ols_hac`` on a named factor set, with the set's name recorded."""
    res = ols_hac(y, select(factors, set_name), **kw)  # type: ignore[arg-type]
    res["factor_set"] = set_name
    return res
