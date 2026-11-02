"""HAR-RV for FX: realised variance on the FX trading day, then Corsi's model.

Plain idea
----------
Realised variance (RV) adds up squared intraday returns, so it measures one
day's variance almost model-free. HAR-RV forecasts tomorrow's RV from the
average RV of the last day, week and month; the three horizons stand for
traders who rebalance at different speeds, and together they mimic long
memory with plain OLS.

The FX-specific part is what "a day" means. Spot FX trades round the clock
from the Sunday Wellington open to the Friday New York close, and the market
rolls its value date at 17:00 New York time. The interbank convention, and
the one used by the FX realised-vol literature (Andersen, Bollerslev,
Diebold and Labys 2001, "The distribution of realized exchange rate
volatility", JASA 96, s.2), is therefore a day running from 17:00 to 17:00
New York, with weekend returns dropped. :func:`fx_daily_realized_variance`
builds that series from time-stamped prices; the New York clock follows
daylight saving, which a fixed UTC cut would get wrong for part of the year.

Formal statement
----------------
Corsi, F. (2009), "A simple approximate long-memory model of realized
volatility", J. Financial Econometrics 7(2), 174-196, eq. (8):

    RV_{t+1} = b0 + b_d RV_t + b_w RV_t^(w) + b_m RV_t^(m) + e_{t+1},

with ``RV^(w)`` and ``RV^(m)`` the means of the last 5 and 22 daily RVs,
estimated by OLS with Newey-West (1987) HAC standard errors. Daily RV is
``sum_j r_{t,j}^2`` over the day's intraday log returns (Andersen et al.
2003, Econometrica 71(2), eq. 4).

Equity twin: ``eq_vol.har_rv``. The regression is the same code path; the
tests check both packages give identical coefficients on one series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WEEK, MONTH = 5, 22
FX_DAY_CUT = "17:00"
FX_DAY_TZ = "America/New_York"

__all__ = [
    "FX_DAY_CUT",
    "FX_DAY_TZ",
    "MONTH",
    "WEEK",
    "HARResult",
    "fit_har",
    "fx_daily_realized_variance",
    "har_design",
    "realized_variance",
    "simulate_har",
]


def realized_variance(intraday_log_returns: np.ndarray) -> np.ndarray:
    """Daily RV from a ``(days, intraday_obs)`` array of log returns."""
    r = np.asarray(intraday_log_returns, dtype=float)
    if r.ndim != 2:
        raise ValueError("intraday_log_returns must be 2-D (days, obs)")
    if not np.all(np.isfinite(r)):
        raise ValueError("intraday_log_returns must be finite")
    return np.sum(r * r, axis=1)


def fx_daily_realized_variance(
    prices: pd.Series,
    cut: str = FX_DAY_CUT,
    tz: str = FX_DAY_TZ,
    min_obs: int = 2,
) -> pd.Series:
    """RV per FX trading day from a tz-aware, time-stamped price series.

    Each intraday log return is assigned to the FX day in which it ends; the
    day labelled ``D`` runs from ``cut`` on ``D-1`` to ``cut`` on ``D`` in
    ``tz`` (default 17:00 New York). Returns that span a day boundary are
    dropped, so the overnight-to-open jump does not count as intraday
    variance. Days labelled Saturday or Sunday (ticks between the Friday
    and Sunday cuts) are dropped, as are days with fewer than ``min_obs``
    returns.
    """
    if not isinstance(prices, pd.Series) or not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must be a pandas Series with a DatetimeIndex")
    if prices.index.tz is None:
        raise ValueError("prices index must be timezone-aware")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices index must be increasing")
    p = prices.to_numpy(dtype=float)
    if p.size < 2 or not np.all(np.isfinite(p)) or np.any(p <= 0.0):
        raise ValueError("need at least two finite, positive prices")
    hh, mm = (int(x) for x in cut.split(":"))
    # work on the local wall clock so the cut stays at 17:00 through DST changes
    wall = prices.index.tz_convert(tz).tz_localize(None)
    shift = pd.Timedelta(hours=24 - hh, minutes=-mm)
    # a tick at exactly the cut closes that day, so it belongs to the day it ends
    label = (wall + shift - pd.Timedelta(microseconds=1)).normalize()
    r = np.diff(np.log(p))
    lab_end = label[1:]
    same_day = np.asarray(label[1:] == label[:-1])
    df = pd.DataFrame({"day": lab_end[same_day], "r2": r[same_day] ** 2})
    g = df.groupby("day")["r2"].agg(["sum", "count"])
    g = g[(g["count"] >= min_obs) & (pd.DatetimeIndex(g.index).dayofweek < 5)]
    out = g["sum"].astype(float)
    out.name = "rv"
    return out


def har_design(rv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Regressor matrix ``[1, RV_d, RV_w, RV_m]`` and target ``RV_{t+1}``."""
    rv = np.asarray(rv, dtype=float)
    if rv.ndim != 1 or rv.size < MONTH + 10:
        raise ValueError(f"need a 1-D series of at least {MONTH + 10} days")
    if not np.all(np.isfinite(rv)) or np.any(rv < 0.0):
        raise ValueError("rv must be finite and non-negative")
    c = np.concatenate(([0.0], np.cumsum(rv)))
    t = np.arange(MONTH - 1, rv.size - 1)
    d = rv[t]
    w = (c[t + 1] - c[t + 1 - WEEK]) / WEEK
    m = (c[t + 1] - c[t + 1 - MONTH]) / MONTH
    X = np.column_stack([np.ones_like(d), d, w, m])
    return X, rv[t + 1]


@dataclass(frozen=True)
class HARResult:
    params: np.ndarray  # b0, b_d, b_w, b_m
    se_hac: np.ndarray
    r2: float
    n_obs: int
    last_rv: np.ndarray  # last MONTH observations, oldest first

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Iterated 1..horizon-day-ahead RV forecasts."""
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        hist = list(self.last_rv)
        out = []
        b0, bd, bw, bm = self.params
        for _ in range(horizon):
            f = b0 + bd * hist[-1] + bw * np.mean(hist[-WEEK:]) + bm * np.mean(hist[-MONTH:])
            out.append(f)
            hist.append(f)
        return np.array(out)

    def forecast_vol(self, horizon: int = 1, days_per_year: int = 260) -> float:
        """Annualised vol implied by the summed ``horizon``-day RV forecast.

        FX has about 260 trading days a year (five per week, no exchange
        holidays), not the equity 252.
        """
        total = float(np.sum(self.forecast(horizon)))
        return float(np.sqrt(max(total, 0.0) * days_per_year / horizon))


def _newey_west(X: np.ndarray, e: np.ndarray, lags: int) -> np.ndarray:
    n = X.shape[0]
    xe = X * e[:, None]
    S = xe.T @ xe
    for L in range(1, lags + 1):
        g = xe[L:].T @ xe[:-L]
        S += (1.0 - L / (lags + 1.0)) * (g + g.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    return np.sqrt(np.diag(XtX_inv @ S @ XtX_inv) * n / (n - X.shape[1]))


def fit_har(rv: np.ndarray | pd.Series, hac_lags: int = MONTH) -> HARResult:
    """OLS fit of HAR-RV with Newey-West standard errors."""
    arr = np.asarray(rv, dtype=float)
    X, y = har_design(arr)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    r2 = 1.0 - float(e @ e) / float(np.sum((y - y.mean()) ** 2))
    return HARResult(beta, _newey_west(X, e, hac_lags), r2, y.size, arr[-MONTH:].copy())


def simulate_har(params: np.ndarray, n: int, noise_sd: float, seed: int = 0) -> np.ndarray:
    """Simulate a HAR-RV path (Gaussian innovations, floored at a tiny positive)."""
    b0, bd, bw, bm = params
    if bd + bw + bm >= 1.0:
        raise ValueError("HAR persistence b_d + b_w + b_m must be < 1")
    rng = np.random.default_rng(seed)
    mean = b0 / (1.0 - bd - bw - bm)
    rv = list(np.full(MONTH, mean))
    for _ in range(n):
        f = b0 + bd * rv[-1] + bw * np.mean(rv[-WEEK:]) + bm * np.mean(rv[-MONTH:])
        rv.append(max(f + noise_sd * rng.standard_normal(), 1e-12))
    return np.array(rv[MONTH:])
