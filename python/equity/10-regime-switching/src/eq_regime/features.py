"""Regime-relevant feature engineering from a multi-asset price panel.

All features are computed point-in-time: the value at date ``t`` uses only
observations up to and including ``t``.  Standardisation uses an EXPANDING
window (:func:`expanding_zscore`) — never the full sample — so z-scores are
free of lookahead by construction.  This is test-enforced with a mutation
test: perturbing future observations must leave all past feature values
bit-identical.

Feature set (each a column of the feature table):

* ``vol_{w}d``        — annualised realized vol of the equal-weight index,
                        rolling window ``w`` for several windows.
* ``dispersion``      — cross-sectional std of daily asset returns (rolling
                        mean-smoothed).
* ``avg_corr``        — average pairwise correlation over a rolling window.
* ``drawdown``        — depth of the equal-weight index below its running max
                        (0 at the peak, positive in drawdown).
* ``trend``           — equal-weight index vs its 200d moving average, in
                        fractional terms (price / MA - 1).
* ``credit_proxy``    — synthetic credit-spread proxy: short-window vol of the
                        spread between the high-beta and low-beta halves of
                        the panel (widens in stress, like IG/HY spreads).
* ``term_proxy``      — synthetic term-structure proxy: long-window minus
                        short-window realized vol (vol backwardation in
                        crises, contango in calm markets).

Units: vols annualised (ACT/252, log-returns); drawdown and trend fractional.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252

__all__ = [
    "expanding_zscore",
    "realized_vol",
    "return_dispersion",
    "average_pairwise_correlation",
    "drawdown_depth",
    "trend_strength",
    "credit_proxy_spread",
    "term_proxy",
    "build_features",
]


def expanding_zscore(df: pd.DataFrame | pd.Series, min_periods: int = 60) -> pd.DataFrame | pd.Series:
    """Point-in-time z-score with an expanding window (no lookahead).

    The z-score at date ``t`` is ``(x_t - mean(x_{1..t})) / std(x_{1..t})``
    using ONLY data up to and including ``t``.  Values before ``min_periods``
    observations are NaN.  Zero-variance prefixes yield NaN (not inf).

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Raw feature values, time-indexed.
    min_periods : int
        Minimum observations before a z-score is emitted.

    Returns
    -------
    Same type as input, expanding z-scores.
    """
    if min_periods < 2:
        raise ValueError(f"min_periods must be >= 2, got {min_periods}")
    mean = df.expanding(min_periods=min_periods).mean()
    std = df.expanding(min_periods=min_periods).std(ddof=1)
    if isinstance(std, pd.DataFrame):
        std = std.where(std > 0.0)
    else:
        std = std.where(std > 0.0)
    return (df - mean) / std


def _index_returns(returns: pd.DataFrame) -> pd.Series:
    """Equal-weight index daily log-return."""
    return returns.mean(axis=1)


def realized_vol(returns: pd.DataFrame, windows: tuple[int, ...] = (10, 21, 63)) -> pd.DataFrame:
    """Annualised rolling realized vol of the equal-weight index.

    Parameters
    ----------
    returns : pd.DataFrame
        (T x N) daily log-returns.
    windows : tuple of int
        Rolling window lengths in days.

    Returns
    -------
    pd.DataFrame
        Columns ``vol_{w}d`` for each window, annualised with sqrt(252).
    """
    if len(windows) == 0:
        raise ValueError("windows must be non-empty")
    idx = _index_returns(returns)
    out = {}
    for w in windows:
        if w < 2:
            raise ValueError(f"window must be >= 2, got {w}")
        out[f"vol_{w}d"] = idx.rolling(w).std(ddof=1) * np.sqrt(TRADING_DAYS)
    return pd.DataFrame(out, index=returns.index)


def return_dispersion(returns: pd.DataFrame, smooth: int = 10) -> pd.Series:
    """Cross-sectional standard deviation of daily returns, rolling-smoothed.

    Dispersion at date ``t`` is ``std_i(r_{i,t})`` (ddof=1) averaged over the
    trailing ``smooth`` days.

    Parameters
    ----------
    returns : pd.DataFrame
        (T x N) daily log-returns, N >= 2.
    smooth : int
        Trailing mean window (1 = raw daily dispersion).

    Returns
    -------
    pd.Series named ``dispersion``.
    """
    if returns.shape[1] < 2:
        raise ValueError("return_dispersion needs at least 2 assets")
    if smooth < 1:
        raise ValueError(f"smooth must be >= 1, got {smooth}")
    raw = returns.std(axis=1, ddof=1)
    out = raw.rolling(smooth).mean() if smooth > 1 else raw
    return out.rename("dispersion")


def average_pairwise_correlation(returns: pd.DataFrame, window: int = 63) -> pd.Series:
    """Rolling average pairwise correlation across assets.

    For each date ``t`` compute the (N x N) correlation matrix of the trailing
    ``window`` daily returns and average the ``N (N-1) / 2`` off-diagonal
    entries.  Vectorised via rolling sufficient statistics.

    Parameters
    ----------
    returns : pd.DataFrame
        (T x N) daily log-returns, N >= 2.
    window : int
        Rolling window length in days (>= 3).

    Returns
    -------
    pd.Series named ``avg_corr`` (NaN for the first ``window - 1`` dates).
    """
    if returns.shape[1] < 2:
        raise ValueError("average_pairwise_correlation needs at least 2 assets")
    if window < 3:
        raise ValueError(f"window must be >= 3, got {window}")
    x = returns.to_numpy(dtype=float)
    t_len, n = x.shape
    out = np.full(t_len, np.nan)
    # Rolling means via cumulative sums; per-date covariance on the window.
    cs = np.vstack([np.zeros((1, n)), np.cumsum(x, axis=0)])
    for t in range(window - 1, t_len):
        seg = x[t - window + 1 : t + 1]
        s = cs[t + 1] - cs[t + 1 - window]
        m = s / window
        segc = seg - m
        cov = segc.T @ segc
        d = np.sqrt(np.diag(cov))
        denom = np.outer(d, d)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / denom
        iu = np.triu_indices(n, k=1)
        vals = corr[iu]
        out[t] = np.nanmean(vals)
    return pd.Series(out, index=returns.index, name="avg_corr")


def drawdown_depth(prices: pd.DataFrame) -> pd.Series:
    """Depth of the equal-weight index below its running maximum.

    The index is the equal-weight average of prices normalised to 1.0 at the
    first date.  Drawdown is ``1 - index / running_max`` — 0 at new highs,
    positive (e.g. 0.25 = 25% below peak) in drawdowns.

    Parameters
    ----------
    prices : pd.DataFrame
        (T x N) price levels.

    Returns
    -------
    pd.Series named ``drawdown``.
    """
    norm = prices / prices.iloc[0]
    idx = norm.mean(axis=1)
    running_max = idx.cummax()
    return (1.0 - idx / running_max).rename("drawdown")


def trend_strength(prices: pd.DataFrame, ma_window: int = 200) -> pd.Series:
    """Equal-weight index versus its moving average: ``price / MA - 1``.

    Positive when the index trades above its trailing ``ma_window``-day mean
    (uptrend), negative below (downtrend).

    Parameters
    ----------
    prices : pd.DataFrame
        (T x N) price levels.
    ma_window : int
        Moving-average window in days.

    Returns
    -------
    pd.Series named ``trend`` (NaN during warmup).
    """
    if ma_window < 2:
        raise ValueError(f"ma_window must be >= 2, got {ma_window}")
    norm = prices / prices.iloc[0]
    idx = norm.mean(axis=1)
    ma = idx.rolling(ma_window).mean()
    return (idx / ma - 1.0).rename("trend")


def _beta_split(returns: pd.DataFrame, lookback: int = 252) -> tuple[list[str], list[str]]:
    """Split assets into high-vol and low-vol halves by trailing full-history vol.

    Deterministic given the panel; used only to construct synthetic factor
    proxies.  Uses the FIRST ``lookback`` days so the split itself does not
    drift (documented simplification: the split is part of the factor
    definition, computed once on the warmup window — no future data enters
    values at t beyond that fixed membership).
    """
    lookback = min(lookback, len(returns))
    vols = returns.iloc[:lookback].std(ddof=1).sort_values()
    cols = list(vols.index)
    half = len(cols) // 2
    return cols[half:], cols[:half]  # high-vol, low-vol


def credit_proxy_spread(returns: pd.DataFrame, window: int = 21, lookback: int = 252) -> pd.Series:
    """Synthetic credit-spread proxy from the price panel.

    Credit spreads widen when risky assets underperform safe ones with high
    volatility.  Proxy: annualised rolling vol of the daily return spread
    between the high-vol half and the low-vol half of the panel (membership
    fixed on the first ``lookback`` days).  Rises sharply in stress regimes,
    mimicking IG/HY spread behaviour without external data.

    Returns
    -------
    pd.Series named ``credit_proxy`` (annualised vol units).
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    hi, lo = _beta_split(returns, lookback)
    spread = returns[hi].mean(axis=1) - returns[lo].mean(axis=1)
    return (spread.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)).rename("credit_proxy")


def term_proxy(returns: pd.DataFrame, short_window: int = 21, long_window: int = 126) -> pd.Series:
    """Synthetic term-structure proxy: long-horizon minus short-horizon vol.

    In calm markets short-dated realized vol sits below long-dated (contango,
    positive value); in crises short-dated vol spikes above (backwardation,
    negative value) — the same shape as the VIX term structure.

    Returns
    -------
    pd.Series named ``term_proxy`` (annualised vol difference).
    """
    if not 2 <= short_window < long_window:
        raise ValueError(
            f"need 2 <= short_window < long_window, got {short_window}, {long_window}"
        )
    idx = _index_returns(returns)
    v_s = idx.rolling(short_window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    v_l = idx.rolling(long_window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    return (v_l - v_s).rename("term_proxy")


def build_features(
    prices: pd.DataFrame,
    vol_windows: tuple[int, ...] = (10, 21, 63),
    corr_window: int = 63,
    ma_window: int = 200,
    standardize: bool = True,
    zscore_min_periods: int = 60,
    dropna: bool = True,
) -> pd.DataFrame:
    """Build the full regime feature table from a price panel.

    Point-in-time by construction; when ``standardize`` is True each raw
    feature is passed through :func:`expanding_zscore` (expanding window —
    no lookahead).  Rows with any NaN (warmup) are dropped when ``dropna``.

    Parameters
    ----------
    prices : pd.DataFrame
        (T x N) daily price levels, N >= 2, T >= max window + warmup.
    vol_windows, corr_window, ma_window : window lengths in days.
    standardize : bool
        Apply expanding z-scores.
    zscore_min_periods : int
        Warmup for the expanding z-score.
    dropna : bool
        Drop warmup rows containing NaN.

    Returns
    -------
    pd.DataFrame
        Feature table indexed by date; guaranteed NaN-free after warmup when
        ``dropna`` is True.
    """
    if prices.shape[1] < 2:
        raise ValueError("build_features needs at least 2 assets")
    min_len = max(max(vol_windows), corr_window, ma_window) + zscore_min_periods + 5
    if len(prices) < min_len:
        raise ValueError(
            f"price history too short: {len(prices)} rows < required {min_len} "
            f"(max window + z-score warmup)"
        )
    returns = np.log(prices / prices.shift(1)).iloc[1:]

    feats = pd.concat(
        [
            realized_vol(returns, vol_windows),
            return_dispersion(returns),
            average_pairwise_correlation(returns, corr_window),
            drawdown_depth(prices).reindex(returns.index),
            trend_strength(prices, ma_window).reindex(returns.index),
            credit_proxy_spread(returns),
            term_proxy(returns),
        ],
        axis=1,
    )
    if standardize:
        feats = expanding_zscore(feats, min_periods=zscore_min_periods)
    if dropna:
        feats = feats.dropna()
    return feats
