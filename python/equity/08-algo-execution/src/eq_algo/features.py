"""Point-in-time cross-sectional alpha features.

All feature functions operate on wide pandas DataFrames (index = trading dates,
columns = tickers) and obey **strict point-in-time (PIT) discipline**: the
feature value at date ``t`` uses only data with timestamp ``<= t``.  This is
enforced by construction (pandas ``rolling``/``shift`` windows *end* at ``t``;
lookbacks use ``shift``) and by the leakage tests in
``tests/test_features.py``, which mutate rows strictly after ``t`` and assert
that the feature at ``t`` is bit-identical.

Conventions
-----------
- Prices are close prices in currency units; returns derived here are simple
  or log returns as documented per function.
- ``lookback``/``window`` arguments are in trading days.
- Realised volatility is annualised with 252 trading days (ACT/365F is not
  used intraday; equity desk convention for daily bars is 252).
- Cross-sectional utilities (rank / z-score / winsorisation) act **row by
  row** (one date at a time) and therefore cannot leak through time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "momentum",
    "short_term_reversal",
    "realized_vol",
    "ma_crossover",
    "rsi",
    "turnover_zscore",
    "cs_rank",
    "cs_zscore",
    "winsorize",
]


def _check_frame(df: pd.DataFrame, name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame (dates x tickers)")
    if df.shape[0] == 0:
        raise ValueError(f"{name} has no rows")


def momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Price momentum with a skip window, e.g. classic 12-1 momentum.

    ``momentum_t = P_{t-skip} / P_{t-lookback} - 1``

    Parameters
    ----------
    prices : DataFrame
        Close prices, dates x tickers.
    lookback : int
        Total lookback in trading days (252 = 12 months).
    skip : int
        Most recent days excluded (21 = 1 month) to avoid contamination by
        short-term reversal.  ``skip < lookback`` required.

    Returns
    -------
    DataFrame
        PIT feature; NaN during the warm-up period.
    """
    _check_frame(prices, "prices")
    if not 0 <= skip < lookback:
        raise ValueError(f"require 0 <= skip < lookback, got skip={skip}, lookback={lookback}")
    return prices.shift(skip) / prices.shift(lookback) - 1.0


def short_term_reversal(prices: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """Short-term (1-month) reversal: minus the trailing ``lookback``-day return.

    ``reversal_t = -(P_t / P_{t-lookback} - 1)`` — high values mean the stock
    *fell* over the last month and is expected to bounce.
    """
    _check_frame(prices, "prices")
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    return -(prices / prices.shift(lookback) - 1.0)


def realized_vol(prices: pd.DataFrame, window: int = 63, annualize: bool = True) -> pd.DataFrame:
    """Realised volatility of daily log returns over a trailing window.

    Sample standard deviation (ddof=1) of ``ln(P_t/P_{t-1})`` over the window
    ending at ``t`` (inclusive), annualised with ``sqrt(252)`` if requested.
    """
    _check_frame(prices, "prices")
    if window < 2:
        raise ValueError("window must be >= 2")
    logret = np.log(prices / prices.shift(1))
    vol = logret.rolling(window, min_periods=window).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(252.0)
    return vol


def ma_crossover(prices: pd.DataFrame, fast: int = 20, slow: int = 100) -> pd.DataFrame:
    """Moving-average crossover: ``MA_fast / MA_slow - 1`` (trend strength).

    Both moving averages are simple, use windows ending at ``t`` inclusive.
    """
    _check_frame(prices, "prices")
    if not 1 <= fast < slow:
        raise ValueError(f"require 1 <= fast < slow, got fast={fast}, slow={slow}")
    ma_f = prices.rolling(fast, min_periods=fast).mean()
    ma_s = prices.rolling(slow, min_periods=slow).mean()
    return ma_f / ma_s - 1.0


def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Relative Strength Index using simple (Cutler) averaging.

    ``RSI = 100 * avg_gain / (avg_gain + avg_loss)`` with simple means of
    up-moves and down-moves over the window ending at ``t``.  The simple-mean
    variant (Cutler's RSI) is used instead of Wilder's exponential smoothing
    so values are exactly hand-computable; flat windows (no moves at all)
    return the neutral value 50.
    """
    _check_frame(prices, "prices")
    if window < 1:
        raise ValueError("window must be >= 1")
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    denom = avg_gain + avg_loss
    out = 100.0 * avg_gain / denom.mask(denom == 0.0)
    return out.where(~(denom == 0.0), 50.0)


def turnover_zscore(volumes: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Abnormal-volume feature: z-score of log volume vs its trailing window.

    ``z_t = (ln V_t - mean_{t-window+1..t} ln V) / std_{t-window+1..t} ln V``

    High values flag unusually heavy trading (news / flow).  Windows end at
    ``t`` inclusive; zero volumes are treated as missing.
    """
    _check_frame(volumes, "volumes")
    if window < 2:
        raise ValueError("window must be >= 2")
    logv = np.log(volumes.mask(volumes <= 0.0))
    mu = logv.rolling(window, min_periods=window).mean()
    sd = logv.rolling(window, min_periods=window).std(ddof=1)
    return (logv - mu) / sd.mask(sd == 0.0)


# ---------------------------------------------------------------------------
# Cross-sectional utilities (row-wise; cannot leak through time)
# ---------------------------------------------------------------------------

def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank per date, in (0, 1]; ties averaged."""
    _check_frame(df, "df")
    return df.rank(axis=1, pct=True)


def cs_zscore(df: pd.DataFrame, ddof: int = 0) -> pd.DataFrame:
    """Cross-sectional z-score per date: ``(x - row mean) / row std``.

    ``ddof=0`` (population std) by default.  Rows with zero dispersion or a
    single valid name return NaN.
    """
    _check_frame(df, "df")
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=ddof)
    sd = sd.mask(sd == 0.0)
    return df.sub(mu, axis=0).div(sd, axis=0)


def winsorize(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Cross-sectional winsorisation: clip each row at its own quantiles.

    Parameters
    ----------
    lower, upper : float
        Quantile bounds in [0, 1] with ``lower < upper``.
    """
    _check_frame(df, "df")
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError(f"require 0 <= lower < upper <= 1, got {lower}, {upper}")
    lo = df.quantile(lower, axis=1)
    hi = df.quantile(upper, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)
