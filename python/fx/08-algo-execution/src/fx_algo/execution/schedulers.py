"""Parent-order schedulers for OTC FX execution.

FX has no consolidated tape, so the equity VWAP-schedule toolkit does
not transfer directly: there is no market volume print to track.  The
desk-standard FX benchmarks are **TWAP**, **arrival price** and the
**WM/R 4pm London fix**; "volume" participation is participation of the
*modeled* session depth (POV-analog).  Every scheduler returns a signed
child-quantity array on the simulator's bucket grid summing exactly to
the parent quantity, with zeros on non-tradeable (weekend) buckets.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..sessions import FIX_HOUR_LONDON, FIX_WINDOW_MINUTES, fix_window_mask

__all__ = [
    "twap_schedule",
    "liquidity_weighted_schedule",
    "pov_schedule",
    "fix_schedule",
]


def _validate(parent_qty: float, n: int, tradeable: Optional[np.ndarray]) -> np.ndarray:
    if parent_qty == 0:
        raise ValueError("parent_qty must be non-zero")
    if n < 1:
        raise ValueError(f"need at least one bucket, got {n}")
    if tradeable is None:
        tradeable = np.ones(n, dtype=bool)
    tradeable = np.asarray(tradeable, dtype=bool)
    if len(tradeable) != n:
        raise ValueError("tradeable mask length must match n_buckets")
    if not tradeable.any():
        raise ValueError("no tradeable buckets in the horizon")
    return tradeable


def twap_schedule(
    parent_qty: float, n_buckets: int, tradeable: Optional[np.ndarray] = None
) -> np.ndarray:
    """Equal child quantities across all tradeable buckets.

    Parameters
    ----------
    parent_qty : float
        Signed parent quantity (mm base).
    n_buckets : int
        Grid length.
    tradeable : numpy.ndarray of bool, optional
        Weekend/blackout mask; non-tradeable buckets get 0.

    Returns
    -------
    numpy.ndarray
        Schedule summing exactly to ``parent_qty``.
    """
    mask = _validate(parent_qty, n_buckets, tradeable)
    q = np.zeros(n_buckets)
    q[mask] = parent_qty / mask.sum()
    return q


def liquidity_weighted_schedule(
    parent_qty: float, depths: np.ndarray, tradeable: Optional[np.ndarray] = None
) -> np.ndarray:
    """Child quantities proportional to the session depth profile.

    The FX replacement for a VWAP schedule: with no tape, the expected
    *tradeable* volume curve is the modeled depth-by-session profile, and
    the schedule participates at a constant fraction of it, concentrating
    flow in London/NY-overlap hours where spread and impact are lowest.

    Parameters
    ----------
    parent_qty : float
        Signed parent quantity (mm base).
    depths : numpy.ndarray
        Bucket depths (mm base per bucket), strictly positive on
        tradeable buckets.
    tradeable : numpy.ndarray of bool, optional
        Weekend/blackout mask.

    Returns
    -------
    numpy.ndarray
        ``q_t = parent * depth_t / sum(depth)`` over tradeable buckets.
    """
    depths = np.asarray(depths, dtype=float)
    mask = _validate(parent_qty, len(depths), tradeable)
    if np.any(depths[mask] <= 0):
        raise ValueError("depths must be > 0 on tradeable buckets")
    w = np.where(mask, depths, 0.0)
    return parent_qty * w / w.sum()


def pov_schedule(
    parent_qty: float,
    volumes: np.ndarray,
    participation: float,
    tradeable: Optional[np.ndarray] = None,
) -> np.ndarray:
    """POV-analog: participate at a capped rate of modeled volume.

    Fills ``q_t = min(participation * v_t, remaining)`` bucket by bucket
    until the parent is done.  ``volumes`` is *modeled* tradeable volume
    (session depth), since actual FX volume is unobservable.

    Parameters
    ----------
    parent_qty : float
        Signed parent quantity.
    volumes : numpy.ndarray
        Modeled volume per bucket (mm base).
    participation : float
        Maximum participation rate in (0, 1].
    tradeable : numpy.ndarray of bool, optional
        Weekend/blackout mask.

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    ValueError
        If the parent cannot be completed within the horizon at the
        given participation rate.
    """
    if not (0 < participation <= 1):
        raise ValueError(f"participation must be in (0,1], got {participation}")
    volumes = np.asarray(volumes, dtype=float)
    mask = _validate(parent_qty, len(volumes), tradeable)
    side = 1.0 if parent_qty > 0 else -1.0
    remaining = abs(parent_qty)
    q = np.zeros(len(volumes))
    for t in range(len(volumes)):
        if not mask[t] or remaining <= 0:
            continue
        take = min(participation * volumes[t], remaining)
        q[t] = side * take
        remaining -= take
    if remaining > 1e-9:
        raise ValueError(
            f"POV schedule incomplete: {remaining:.2f}mm unfilled at "
            f"{participation:.0%} participation — extend the horizon or raise the rate"
        )
    return q


def fix_schedule(
    parent_qty: float,
    times_hours: np.ndarray,
    dt_minutes: float,
    fix_hour: float = FIX_HOUR_LONDON,
    window_minutes: float = FIX_WINDOW_MINUTES,
) -> np.ndarray:
    """Fix-targeting schedule: TWAP inside the WM/R calculation window.

    The WM/R benchmark is (post-2015) a 5-minute TWAP-style window
    centred on 16:00 London; executing flat across exactly that window
    minimises tracking error to the print.  All quantity is placed,
    equally, in the buckets whose midpoints fall inside the window.

    Parameters
    ----------
    parent_qty : float
        Signed parent quantity.
    times_hours : numpy.ndarray
        Bucket start times (absolute hours).
    dt_minutes : float
        Bucket length in minutes.
    fix_hour : float
        Fix hour-of-day (London), default 16.0.
    window_minutes : float
        Window width in minutes, default 5.0.

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    ValueError
        If no bucket of the grid lies in the fix window.
    """
    times_hours = np.asarray(times_hours, dtype=float)
    _validate(parent_qty, len(times_hours), None)
    mask = fix_window_mask(times_hours, dt_minutes, fix_hour, window_minutes)
    if not mask.any():
        raise ValueError(
            f"no bucket in the fix window {fix_hour:.2f}h +- {window_minutes / 2:.1f}min; "
            "use a finer grid or a horizon covering the fix"
        )
    q = np.zeros(len(times_hours))
    q[mask] = parent_qty / mask.sum()
    return q
