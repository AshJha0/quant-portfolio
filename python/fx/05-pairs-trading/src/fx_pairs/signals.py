"""Z-score signal generation: state machine, vol-targeted sizing, carry filter.

Timing convention (no lookahead)
--------------------------------
``positions[t]`` is decided from information up to and including the close of
``t`` and is *held from close t to close t+1*: the backtest engine applies it
to the return from ``t`` to ``t+1``.  Rolling statistics use windows ending at
``t``; the vol-target scale uses realised vol up to ``t``.

State machine
-------------
flat -> short spread when ``z >= entry``; flat -> long when ``z <= -entry``;
exit to flat when the z-score reverts inside ``|z| <= exit``; hard stop when
the z-score moves further to ``|z| >= stop`` (regime-break protection — see
the SNB case study in docs/VALIDATION.md); time stop after ``max_holding``
bars.  After any exit the machine is flat for at least one bar (no same-bar
reversal), so positions can never jump -1 -> +1 directly.

Carry-aware entry filter
------------------------
A spread trade's expected gross gain is roughly ``(|z| - exit) * sigma_spread``
(the z-distance it is expected to revert, in log units).  Holding it costs or
earns carry at ``carry_per_day`` (log units/day, for a +1 long-spread unit).
Expected holding time is proportional to the OU half-life.  The filter vetoes
entries whose expected carry drag over ``holding_multiple x half-life`` exceeds
the expected reversion gain; carry-favourable entries are never vetoed.  This
is deliberately a coarse desk-style filter (documented in METHODOLOGY.md):
it changes *which* trades are taken, never how existing ones are managed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._validation import require_finite

__all__ = [
    "zscore",
    "Trade",
    "generate_positions",
    "carry_entry_veto",
    "vol_target_scale",
]


def zscore(
    spread: pd.Series,
    window: int | None = None,
    mu: float | None = None,
    sigma: float | None = None,
) -> pd.Series:
    """Z-score of a spread with rolling or frozen (formation) statistics.

    Exactly one of ``window`` or ``(mu, sigma)`` must be supplied.  Rolling
    stats use ``min_periods=window`` so the warmup is NaN (no partial-window
    lookahead artefacts); frozen stats come from a formation window in
    walk-forward use.

    Parameters
    ----------
    spread : pandas.Series
        Log-spread series.
    window : int, optional
        Rolling window length in bars.
    mu, sigma : float, optional
        Frozen mean and standard deviation.
    """
    if window is not None:
        if mu is not None or sigma is not None:
            raise ValueError("supply either window or (mu, sigma), not both")
        if window < 2:
            raise ValueError("window must be >= 2")
        m = spread.rolling(window, min_periods=window).mean()
        s = spread.rolling(window, min_periods=window).std(ddof=1)
        return (spread - m) / s
    if mu is None or sigma is None:
        raise ValueError("supply either window or both mu and sigma")
    require_finite(mu=mu, sigma=sigma)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return (spread - mu) / sigma


@dataclass
class Trade:
    """One round-trip spread trade (indices into the signal series)."""

    entry: int
    exit: int
    side: int  # +1 long spread, -1 short spread
    exit_reason: str  # "exit" | "stop" | "time" | "eod"


def generate_positions(
    z: pd.Series | np.ndarray,
    entry: float = 2.0,
    exit_: float = 0.5,
    stop: float | None = 4.0,
    max_holding: int | None = None,
    allow_long: np.ndarray | None = None,
    allow_short: np.ndarray | None = None,
) -> tuple[np.ndarray, list[Trade]]:
    """Run the z-score state machine; returns unit positions and trades.

    Parameters
    ----------
    z : array-like
        Z-score series (NaN during warmup => forced flat).
    entry, exit_ : float
        Entry and exit thresholds, ``entry > exit_ >= 0``.
    stop : float, optional
        Hard stop at ``|z| >= stop`` (must exceed ``entry``); ``None``
        disables.
    max_holding : int, optional
        Time stop in bars; ``None`` disables.
    allow_long, allow_short : bool arrays, optional
        Per-bar entry permissions (used by the carry filter).  They gate
        *entries only* — open positions are managed regardless.

    Returns
    -------
    positions : numpy.ndarray
        Values in {-1, 0, +1}; ``positions[t]`` is held from close t to
        close t+1.
    trades : list of Trade
        Round-trips; an open position at the last bar is closed as ``"eod"``.
    """
    zv = np.asarray(z, dtype=float)
    n = len(zv)
    # `stop` in particular: `if stop <= entry` is False for NaN, so a NaN stop
    # used to pass validation and then disable the hard stop entirely --
    # `state * zt <= -nan` is always False. Losing the regime-break stop
    # silently is the most expensive failure this module can have.
    require_finite(entry=entry, exit_=exit_, stop=stop)
    if not (entry > exit_ >= 0.0):
        raise ValueError(f"need entry > exit_ >= 0, got entry={entry}, exit_={exit_}")
    if stop is not None and stop <= entry:
        raise ValueError(f"stop ({stop}) must exceed entry ({entry})")
    if max_holding is not None and max_holding < 1:
        raise ValueError("max_holding must be >= 1")
    for name, arr in (("allow_long", allow_long), ("allow_short", allow_short)):
        if arr is not None and len(arr) != n:
            raise ValueError(f"{name} must match z length")

    pos = np.zeros(n)
    trades: list[Trade] = []
    state = 0
    entry_idx = -1
    for t in range(n):
        zt = zv[t]
        if state == 0:
            if np.isfinite(zt):
                if zt >= entry and (allow_short is None or allow_short[t]):
                    state, entry_idx = -1, t
                elif zt <= -entry and (allow_long is None or allow_long[t]):
                    state, entry_idx = 1, t
        else:
            reason = None
            if not np.isfinite(zt):
                reason = "exit"
            elif stop is not None and state * zt <= -stop:
                # long stopped when z <= -stop, short stopped when z >= +stop
                reason = "stop"
            elif abs(zt) <= exit_ or state * zt >= exit_:
                # reverted inside the exit band or through the mean
                reason = "exit"
            elif max_holding is not None and t - entry_idx >= max_holding:
                reason = "time"
            if reason is not None:
                trades.append(Trade(entry=entry_idx, exit=t, side=state, exit_reason=reason))
                state = 0
                entry_idx = -1
        pos[t] = state
    if state != 0:
        trades.append(Trade(entry=entry_idx, exit=n - 1, side=state, exit_reason="eod"))
    return pos, trades


def carry_entry_veto(
    z: pd.Series | np.ndarray,
    sigma_spread: float,
    carry_per_day: float | pd.Series | np.ndarray,
    half_life: float,
    entry: float = 2.0,
    exit_: float = 0.5,
    holding_multiple: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Entry permission masks that skip trades with adverse expected carry.

    For a candidate long-spread entry at bar ``t`` (``z_t <= -entry``):
    expected reversion gain ~ ``(|z_t| - exit_) * sigma_spread`` (log units);
    expected carry P&L ~ ``+carry_per_day[t] * holding_multiple * half_life``
    (positive helps a long).  The entry is vetoed when the expected carry
    *drag* exceeds the expected gain, i.e. when

    ``side * carry_per_day * H < -(|z| - exit_) * sigma_spread``,
    ``H = holding_multiple * half_life``.

    Shorts are symmetric with carry sign flipped.  Carry-favourable or
    carry-neutral entries are never vetoed.

    Parameters
    ----------
    sigma_spread : float
        Formation-window spread standard deviation (log units).
    carry_per_day : float or array-like
        Daily carry of a +1 long-spread unit in log units/day (the engine's
        ``accr1 - beta * accr2`` per day).
    half_life : float
        OU half-life in bars; non-finite disables the veto (nothing to
        compare against).

    Returns
    -------
    (allow_long, allow_short) : bool arrays for :func:`generate_positions`.
    """
    zv = np.asarray(z, dtype=float)
    n = len(zv)
    require_finite(sigma_spread=sigma_spread, entry=entry, exit_=exit_,
                   holding_multiple=holding_multiple)
    if sigma_spread <= 0:
        raise ValueError("sigma_spread must be positive")
    cpd = np.full(n, float(carry_per_day)) if np.isscalar(carry_per_day) \
        else np.asarray(carry_per_day, dtype=float)
    if len(cpd) != n:
        raise ValueError("carry_per_day must be scalar or match z length")
    if not np.isfinite(half_life):
        return np.ones(n, dtype=bool), np.ones(n, dtype=bool)
    H = holding_multiple * half_life
    gain = np.maximum(np.abs(zv) - exit_, 0.0) * sigma_spread
    with np.errstate(invalid="ignore"):
        long_carry = cpd * H          # carry P&L if long spread
        short_carry = -cpd * H        # carry P&L if short spread
        allow_long = ~((long_carry < 0) & (-long_carry > gain))
        allow_short = ~((short_carry < 0) & (-short_carry > gain))
    return allow_long, allow_short


def vol_target_scale(
    spread: pd.Series,
    target_vol: float = 0.10,
    lookback: int = 63,
    ann_factor: float = 252.0,
    max_leverage: float = 10.0,
) -> pd.Series:
    """Position scale so a 1-unit spread position runs at ``target_vol``.

    ``scale_t = target_vol / realised_ann_vol_t`` where the realised vol is the
    rolling standard deviation of daily spread changes up to and including
    ``t`` (annualised).  The scale applied at ``t`` uses only data through
    ``t`` — combined with the ``positions[t]`` timing convention this is
    lookahead-free.  Capped at ``max_leverage``: this cap is exactly what a
    vol-targeted book does NOT have when a pegged spread's realised vol
    collapses — see the SNB floor case study, where uncapped vol targeting
    maximises leverage right before the break.

    Returns
    -------
    pandas.Series
        Scale factor (NaN during warmup — treat as 0 exposure).
    """
    require_finite(target_vol=target_vol, ann_factor=ann_factor,
                   max_leverage=max_leverage)
    if target_vol <= 0:
        raise ValueError("target_vol must be positive")
    if ann_factor <= 0:
        raise ValueError("ann_factor must be positive")
    if max_leverage <= 0:
        raise ValueError("max_leverage must be positive")
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    dvol = spread.diff().rolling(lookback, min_periods=lookback).std(ddof=1)
    ann = dvol * np.sqrt(ann_factor)
    scale = target_vol / ann
    return scale.clip(upper=max_leverage)
