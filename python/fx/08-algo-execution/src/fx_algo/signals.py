"""Signal combination, vol-targeted sizing and session/carry filters.

Positions are expressed in units of base-currency notional (1.0 = one
unit of base per unit leverage).  All transformations here are causal:
a position indexed at bar ``t`` uses information up to bar ``t`` only and
is applied by the backtester to the ``t -> t+1`` return.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .sessions import session_of_hour

__all__ = [
    "rolling_zscore",
    "combine_signals",
    "vol_target_positions",
    "session_filter",
    "carry_gate",
]


def rolling_zscore(x: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Causal rolling z-score: (x - rolling mean) / rolling std.

    Parameters
    ----------
    x : pandas.Series
    window : int
        Rolling window length in bars.
    min_periods : int, optional
        Minimum observations (default ``window // 2``).

    Returns
    -------
    pandas.Series
        NaN where insufficient history; zero-variance windows give 0.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    mp = window // 2 if min_periods is None else min_periods
    mu = x.rolling(window, min_periods=mp).mean()
    sd = x.rolling(window, min_periods=mp).std(ddof=0)
    z = (x - mu) / sd.replace(0.0, np.nan)
    return z.fillna(0.0)


def combine_signals(
    features: pd.DataFrame,
    weights: Mapping[str, float],
    zscore_window: int = 96,
    clip: float = 3.0,
) -> pd.Series:
    """Combine features into a single signal in [-clip, clip].

    Each named feature is z-scored causally then linearly combined:
    ``s_t = sum_k w_k * z_k(t)``, clipped at ``+-clip``.

    Parameters
    ----------
    features : pandas.DataFrame
        Feature matrix (see ``features.feature_matrix``).
    weights : Mapping[str, float]
        Feature name -> weight.  Names must exist as columns.
    zscore_window : int
        Window (bars) for the causal z-score.
    clip : float
        Symmetric clip on the combined signal.

    Returns
    -------
    pandas.Series
    """
    missing = [k for k in weights if k not in features.columns]
    if missing:
        raise ValueError(f"weights refer to unknown features: {missing}")
    s = pd.Series(0.0, index=features.index)
    for name, w in weights.items():
        s = s + w * rolling_zscore(features[name].fillna(0.0), zscore_window)
    return s.clip(-clip, clip).rename("signal")


def vol_target_positions(
    signal: pd.Series,
    returns: pd.Series,
    target_ann_vol: float = 0.10,
    bars_per_year: float = 24.0 * 261.0,
    vol_window: int = 96,
    max_leverage: float = 5.0,
) -> pd.Series:
    """Scale a unit signal to a vol-targeted position.

    ``pos_t = clip( signal_t * target / (sigma_t * sqrt(bars_per_year)),
    +-max_leverage )`` with ``sigma_t`` the causal rolling std of bar
    returns.  Annualisation uses ``bars_per_year`` (24h day, ~261 FX
    trading days).

    Parameters
    ----------
    signal : pandas.Series
        Combined signal (roughly unit scale).
    returns : pandas.Series
        Bar close-to-close returns aligned with ``signal``.
    target_ann_vol : float
        Annualised volatility target of the P&L, e.g. 0.10 = 10%.
    bars_per_year : float
        Bars per year for annualisation.
    vol_window : int
        Rolling window (bars) for the vol estimate.
    max_leverage : float
        Absolute position cap in base-notional units.

    Returns
    -------
    pandas.Series
        Position per bar; 0 where the vol estimate is unavailable.
    """
    if target_ann_vol <= 0:
        raise ValueError(f"target_ann_vol must be > 0, got {target_ann_vol}")
    sigma = returns.rolling(vol_window, min_periods=vol_window // 2).std(ddof=0)
    ann = sigma * np.sqrt(bars_per_year)
    scale = target_ann_vol / ann.replace(0.0, np.nan)
    pos = (signal * scale).clip(-max_leverage, max_leverage)
    return pos.fillna(0.0).rename("position")


def session_filter(
    positions: pd.Series,
    hours: pd.Series | np.ndarray,
    allowed_sessions: Sequence[str] = ("london", "overlap", "ny"),
) -> pd.Series:
    """Zero positions outside the allowed sessions.

    New risk is only put on when liquidity supports it (spread cost in
    Asia/late-NY sessions can exceed intraday alpha; see DESK_GUIDE.md).

    Parameters
    ----------
    positions : pandas.Series
        Target positions per bar.
    hours : array_like
        Hour-of-day of each bar end (mod 24 applied internally).
    allowed_sessions : sequence of str
        Sessions in which positions may be non-zero.

    Returns
    -------
    pandas.Series
    """
    sess = session_of_hour(np.asarray(hours, dtype=float))
    keep = np.isin(sess, list(allowed_sessions))
    out = positions.copy()
    out[~keep] = 0.0
    return out


def carry_gate(
    positions: pd.Series,
    hours: pd.Series | np.ndarray,
    carry: pd.Series | np.ndarray,
    rollover_hour: float = 21.0,
    bar_hours: float = 1.0,
) -> pd.Series:
    """Only allow positions to be held over the rollover if carry agrees.

    On the bar whose holding interval crosses the daily rollover
    (5pm NY = 21:00 London here), a position whose sign disagrees with the
    sign of the carry (r_base - r_quote) is flattened, so negative-carry
    exposure is never rolled overnight.

    Parameters
    ----------
    positions : pandas.Series
        Target positions per bar (position at bar t is held over t->t+1).
    hours : array_like
        Hour-of-day of each bar end.
    carry : array_like
        Annualised carry per bar (r_base - r_quote).
    rollover_hour : float
        Hour-of-day of the overnight rollover.
    bar_hours : float
        Bar length in hours (to locate the holding interval).

    Returns
    -------
    pandas.Series
    """
    h = np.asarray(hours, dtype=float) % 24.0
    c = np.asarray(carry, dtype=float)
    p = positions.to_numpy(dtype=float).copy()
    # The position set at bar t is held over (h_t, h_t + bar_hours]; it
    # crosses the rollover iff h_t < rollover <= h_t + bar_hours.
    crosses = (h < rollover_hour) & (h + bar_hours >= rollover_hour)
    disagree = np.sign(p) * np.sign(c) < 0
    p[crosses & disagree] = 0.0
    return pd.Series(p, index=positions.index, name=positions.name)
