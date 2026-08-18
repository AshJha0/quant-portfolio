"""Bar construction and intraday FX features with point-in-time discipline.

Every feature value indexed at bar-end time ``t`` is a deterministic
function of ticks with ``time < t`` only.  This is *test-enforced*: the
suite mutates future ticks and asserts features at or before the cutoff
are bit-identical (see tests/test_features.py).

Features (all causal, all in return/pip-free units unless stated):

* ``momentum``      — k-bar past return of bar closes (1h/4h momentum).
* ``reversion``     — gap of close below the running intraday TWAP mid,
  the OTC-FX analog of "reversion to session VWAP" (there is no
  consolidated volume tape in FX, so a TWAP mid replaces VWAP).
* ``breakout``      — London-open breakout of the Asia session range.
* ``carry``         — sign/level of the base-quote rate differential,
  used as a *filter* for which direction may be held overnight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "build_bars",
    "momentum",
    "reversion_to_session_mean",
    "london_open_breakout",
    "carry_feature",
    "feature_matrix",
]


def build_bars(ticks: pd.DataFrame, bar_hours: float = 1.0) -> pd.DataFrame:
    """Aggregate ticks-lite rows into OHLC bars.

    Ticks are assigned to half-open bars ``[k*bar_hours, (k+1)*bar_hours)``
    by their ``time_hours``; a tick exactly on a boundary opens the next
    bar.  The bar is indexed by its **end** time, so a bar's row contains
    only information available at that timestamp (point-in-time).

    Parameters
    ----------
    ticks : pandas.DataFrame
        Columns ``time_hours``, ``mid`` (and optionally ``bid``/``ask``).
    bar_hours : float
        Bar length in hours (1.0 and 4.0 are the standard grids here).

    Returns
    -------
    pandas.DataFrame
        Indexed by bar end time (absolute hours), columns ``open``,
        ``high``, ``low``, ``close``, ``twap_mid`` (mean tick mid — the
        FX analog of a VWAP bar price), ``n_ticks``, ``hour`` (hour of
        day of bar end, mod 24) and ``day``.

    Raises
    ------
    ValueError
        If ``bar_hours <= 0`` or ``ticks`` is empty.
    """
    if bar_hours <= 0:
        raise ValueError(f"bar_hours must be > 0, got {bar_hours}")
    if len(ticks) == 0:
        raise ValueError("ticks is empty")
    t = ticks["time_hours"].to_numpy(dtype=float)
    idx = np.floor(t / bar_hours + 1e-12).astype(int)
    g = ticks.groupby(idx)["mid"]
    bars = pd.DataFrame(
        {
            "open": g.first(),
            "high": g.max(),
            "low": g.min(),
            "close": g.last(),
            "twap_mid": g.mean(),
            "n_ticks": g.count(),
        }
    )
    end_time = (bars.index.to_numpy(dtype=float) + 1.0) * bar_hours
    bars.index = pd.Index(end_time, name="bar_end_hours")
    bars["hour"] = np.mod(end_time, 24.0)
    bars["day"] = np.floor((end_time - 1e-9) / 24.0).astype(int)
    return bars


def momentum(bars: pd.DataFrame, lookback_bars: int) -> pd.Series:
    """k-bar momentum: total close-to-close return over the lookback.

    ``momentum_t = close_t / close_{t-k} - 1`` — uses only bars up to and
    including ``t`` (point-in-time by construction).

    Parameters
    ----------
    bars : pandas.DataFrame
        Output of :func:`build_bars`.
    lookback_bars : int
        Lookback ``k >= 1``.

    Returns
    -------
    pandas.Series
        NaN for the first ``k`` bars.
    """
    if lookback_bars < 1:
        raise ValueError(f"lookback_bars must be >= 1, got {lookback_bars}")
    return bars["close"].pct_change(lookback_bars).rename(f"mom_{lookback_bars}")


def reversion_to_session_mean(bars: pd.DataFrame) -> pd.Series:
    """Mean-reversion gap to the running intraday TWAP mid.

    FX has no consolidated tape, so the equity "reversion to VWAP"
    becomes reversion to the running *time*-weighted average mid of the
    current trading day: ``(day_twap_t - close_t) / close_t``.  Positive
    when price sits below its day average (expect pull back up).  The
    running TWAP is an expanding mean of ``twap_mid`` within the day up
    to bar ``t`` — strictly causal.

    Returns
    -------
    pandas.Series
    """
    day = bars["day"]
    run_twap = bars.groupby(day)["twap_mid"].expanding().mean().droplevel(0)
    return ((run_twap - bars["close"]) / bars["close"]).rename("reversion")


def london_open_breakout(
    bars: pd.DataFrame,
    asia_end_hour: float = 7.0,
    window_end_hour: float = 10.0,
) -> pd.Series:
    """London-open breakout of the overnight Asia range.

    For bars ending in ``(asia_end_hour, window_end_hour]`` the feature is
    ``+1`` if the bar close breaks above the Asia-session high of the same
    day, ``-1`` if below the Asia low, else 0.  Outside the window it is
    0.  Uses only same-day bars that ended at or before ``asia_end_hour``
    plus the current close — causal.

    Returns
    -------
    pandas.Series of {-1.0, 0.0, +1.0}
    """
    out = pd.Series(0.0, index=bars.index, name="breakout")
    for _, day_bars in bars.groupby(bars["day"]):
        asia = day_bars[day_bars["hour"] <= asia_end_hour]
        if len(asia) == 0:
            continue
        hi, lo = asia["high"].max(), asia["low"].min()
        window = day_bars[
            (day_bars["hour"] > asia_end_hour) & (day_bars["hour"] <= window_end_hour)
        ]
        out.loc[window.index[window["close"] > hi]] = 1.0
        out.loc[window.index[window["close"] < lo]] = -1.0
    return out


def carry_feature(bars: pd.DataFrame, daily_panel: pd.DataFrame) -> pd.Series:
    """Broadcast the daily carry (r_base - r_quote) onto the bar grid.

    Each bar of day ``d`` receives the carry known at the **start** of day
    ``d`` (i.e. the panel row of day ``d``, which is fixed overnight before
    the day starts) — no forward-looking rate information.

    Parameters
    ----------
    bars : pandas.DataFrame
        Output of :func:`build_bars`.
    daily_panel : pandas.DataFrame
        Output of ``generate_daily_panel`` (column ``carry`` indexed by day).

    Returns
    -------
    pandas.Series
        Annualised rate differential per bar.
    """
    carry = daily_panel["carry"].reindex(bars["day"].to_numpy())
    return pd.Series(carry.to_numpy(), index=bars.index, name="carry")


def feature_matrix(
    bars: pd.DataFrame,
    daily_panel: pd.DataFrame | None = None,
    momentum_lookbacks: tuple[int, ...] = (1, 4),
) -> pd.DataFrame:
    """Assemble the standard causal feature matrix on a bar grid.

    Parameters
    ----------
    bars : pandas.DataFrame
        Output of :func:`build_bars` (1h grid recommended; the 4-bar
        momentum on a 1h grid is the 4h momentum).
    daily_panel : pandas.DataFrame, optional
        Daily carry panel; if omitted the carry column is 0.
    momentum_lookbacks : tuple of int
        Lookbacks for the momentum features.

    Returns
    -------
    pandas.DataFrame
        Columns ``mom_<k>`` for each lookback, ``reversion``,
        ``breakout``, ``carry``; indexed like ``bars``.
    """
    cols = [momentum(bars, k) for k in momentum_lookbacks]
    cols.append(reversion_to_session_mean(bars))
    cols.append(london_open_breakout(bars))
    if daily_panel is not None:
        cols.append(carry_feature(bars, daily_panel))
    else:
        cols.append(pd.Series(0.0, index=bars.index, name="carry"))
    return pd.concat(cols, axis=1)
