"""FX trading sessions, pair liquidity profiles and time-grid utilities.

Conventions
-----------
* The trading day is the 24h OTC FX day expressed on a **London clock**:
  hour-of-day ``h`` is in ``[0, 24)`` with ``h = 0`` at London midnight.
* Time grids are expressed in **absolute hours** from the grid start (may
  exceed 24 for multi-day horizons); hour-of-day is ``t mod 24``.
* Pairs are quoted BASE/QUOTE (EURUSD = USD per 1 EUR).  All spreads are
  **full** quoted spreads in pips of the quote currency; a pip is
  ``pip_size`` price units.  Depth is quoted in **millions of base
  currency per minute** that the market absorbs at the modeled impact
  cost.  Volatility is quoted in **pips per sqrt-minute**.

The five stylized sessions (start-inclusive, end-exclusive, London clock):

===========  ============  =====================================
session      hours         notes
===========  ============  =====================================
``asia``     [0, 7)        Tokyo/Singapore/Sydney
``london``   [7, 12)       London morning
``overlap``  [12, 17)      London-NY overlap - deepest liquidity
``ny``       [17, 21)      NY afternoon after London close
``late``     [21, 24)      NY late / rollover - thinnest
===========  ============  =====================================

The WM/R "London 4pm" fix (16:00) falls inside the overlap session.
Since the 2015 reform the fix is computed over a 5-minute window
centred on 16:00 (widened from 60 seconds after the 2013 fix-rigging
scandal); see :func:`fix_window_mask` and docs/METHODOLOGY.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

__all__ = [
    "SESSION_NAMES",
    "SESSION_BOUNDS",
    "session_of_hour",
    "PairProfile",
    "EURUSD",
    "GBPUSD",
    "USDMXN",
    "make_time_grid",
    "weekend_mask",
    "fix_window_mask",
    "FIX_HOUR_LONDON",
    "FIX_WINDOW_MINUTES",
]

#: WM/R benchmark time, London clock (4pm London).
FIX_HOUR_LONDON: float = 16.0
#: Post-2015 WM/R calculation window width in minutes.
FIX_WINDOW_MINUTES: float = 5.0

SESSION_NAMES: tuple[str, ...] = ("asia", "london", "overlap", "ny", "late")

#: session -> (start_hour, end_hour), start-inclusive / end-exclusive.
SESSION_BOUNDS: dict[str, tuple[float, float]] = {
    "asia": (0.0, 7.0),
    "london": (7.0, 12.0),
    "overlap": (12.0, 17.0),
    "ny": (17.0, 21.0),
    "late": (21.0, 24.0),
}


def session_of_hour(hour_of_day: np.ndarray | float) -> np.ndarray:
    """Map hour-of-day (London clock, in [0, 24)) to session names.

    Parameters
    ----------
    hour_of_day : array_like or float
        Hours-of-day; values are taken mod 24 so absolute grid hours are
        accepted directly.

    Returns
    -------
    numpy.ndarray of str
        Session name per element (``"asia"``, ``"london"``, ``"overlap"``,
        ``"ny"`` or ``"late"``).
    """
    h = np.atleast_1d(np.asarray(hour_of_day, dtype=float)) % 24.0
    out = np.empty(h.shape, dtype=object)
    for name, (lo, hi) in SESSION_BOUNDS.items():
        out[(h >= lo) & (h < hi)] = name
    return out.astype(str)


@dataclass(frozen=True)
class PairProfile:
    """Static liquidity profile of a currency pair by session.

    Attributes
    ----------
    name : str
        Pair name, BASE/QUOTE convention (e.g. ``"EURUSD"``).
    pip_size : float
        Price units per pip (1e-4 for most pairs, 1e-2 for JPY quotes).
    s0 : float
        Reference spot (quote ccy per 1 base ccy).
    spread_pips : Mapping[str, float]
        Full quoted spread in pips by session.
    depth_mm_per_min : Mapping[str, float]
        Absorbable flow in millions of base ccy per minute by session.
    vol_pips_per_sqrt_min : Mapping[str, float]
        Mid volatility in pips per sqrt-minute by session.
    """

    name: str
    pip_size: float
    s0: float
    spread_pips: Mapping[str, float]
    depth_mm_per_min: Mapping[str, float]
    vol_pips_per_sqrt_min: Mapping[str, float]

    def _by_session(self, mapping: Mapping[str, float], hours: np.ndarray) -> np.ndarray:
        sess = session_of_hour(hours)
        return np.array([mapping[s] for s in sess], dtype=float)

    def spread_pips_at(self, hours: np.ndarray | float) -> np.ndarray:
        """Full quoted spread (pips) at each hour (absolute or hour-of-day)."""
        return self._by_session(self.spread_pips, np.atleast_1d(np.asarray(hours, float)))

    def depth_at(self, hours: np.ndarray | float) -> np.ndarray:
        """Depth (mm base/minute) at each hour."""
        return self._by_session(self.depth_mm_per_min, np.atleast_1d(np.asarray(hours, float)))

    def vol_at(self, hours: np.ndarray | float) -> np.ndarray:
        """Mid vol (pips/sqrt-minute) at each hour."""
        return self._by_session(self.vol_pips_per_sqrt_min, np.atleast_1d(np.asarray(hours, float)))


#: EURUSD: the most liquid pair; overlap spread ~0.2 pip, late ~1 pip.
EURUSD = PairProfile(
    name="EURUSD",
    pip_size=1e-4,
    s0=1.1000,
    spread_pips={"asia": 0.6, "london": 0.35, "overlap": 0.2, "ny": 0.4, "late": 1.0},
    depth_mm_per_min={"asia": 20.0, "london": 50.0, "overlap": 70.0, "ny": 40.0, "late": 8.0},
    vol_pips_per_sqrt_min={"asia": 0.9, "london": 1.8, "overlap": 2.2, "ny": 1.6, "late": 0.7},
)

#: GBPUSD: liquid major, wider than EURUSD, flash-crash prone in Asia hours.
GBPUSD = PairProfile(
    name="GBPUSD",
    pip_size=1e-4,
    s0=1.2700,
    spread_pips={"asia": 1.2, "london": 0.6, "overlap": 0.4, "ny": 0.8, "late": 2.0},
    depth_mm_per_min={"asia": 8.0, "london": 30.0, "overlap": 40.0, "ny": 22.0, "late": 4.0},
    vol_pips_per_sqrt_min={"asia": 1.3, "london": 2.4, "overlap": 2.8, "ny": 2.0, "late": 1.0},
)

#: USDMXN: EM pair; NY hours are its home liquidity, Asia is a desert.
USDMXN = PairProfile(
    name="USDMXN",
    pip_size=1e-4,
    s0=17.00,
    spread_pips={"asia": 150.0, "london": 60.0, "overlap": 40.0, "ny": 30.0, "late": 250.0},
    depth_mm_per_min={"asia": 0.5, "london": 3.0, "overlap": 5.0, "ny": 6.0, "late": 0.3},
    vol_pips_per_sqrt_min={"asia": 20.0, "london": 35.0, "overlap": 42.0, "ny": 38.0, "late": 15.0},
)


def make_time_grid(
    start_hour: float, horizon_hours: float, dt_minutes: float
) -> np.ndarray:
    """Build a bucket-start grid in absolute hours.

    Parameters
    ----------
    start_hour : float
        Absolute start hour (0 = London midnight of day 0).
    horizon_hours : float
        Total horizon covered by the grid.
    dt_minutes : float
        Bucket length in minutes; must divide the horizon to within 1e-9.

    Returns
    -------
    numpy.ndarray
        Bucket start times (absolute hours), length ``horizon*60/dt``.

    Raises
    ------
    ValueError
        If ``dt_minutes`` or ``horizon_hours`` is not strictly positive.
    """
    if dt_minutes <= 0:
        raise ValueError(f"dt_minutes must be > 0, got {dt_minutes}")
    if horizon_hours <= 0:
        raise ValueError(f"horizon_hours must be > 0, got {horizon_hours}")
    n = int(round(horizon_hours * 60.0 / dt_minutes))
    return start_hour + np.arange(n) * dt_minutes / 60.0


def weekend_mask(
    times_hours: np.ndarray,
    weekend_start_hour: float,
    weekend_end_hour: float,
) -> np.ndarray:
    """Tradeability mask: False inside the weekend gap.

    FX trades continuously from Monday Wellington open to Friday NY close;
    the weekend is a hard no-trading gap (no buckets, no fills).

    Parameters
    ----------
    times_hours : numpy.ndarray
        Absolute bucket-start hours.
    weekend_start_hour, weekend_end_hour : float
        Absolute hours delimiting the closed interval-open interval
        ``[start, end)`` during which trading is impossible.

    Returns
    -------
    numpy.ndarray of bool
        True where the bucket is tradeable.
    """
    t = np.asarray(times_hours, dtype=float)
    return ~((t >= weekend_start_hour) & (t < weekend_end_hour))


def fix_window_mask(
    times_hours: np.ndarray,
    dt_minutes: float,
    fix_hour: float = FIX_HOUR_LONDON,
    window_minutes: float = FIX_WINDOW_MINUTES,
) -> np.ndarray:
    """Boolean mask of buckets whose midpoint falls in the WM/R fix window.

    The window is ``[fix - w/2, fix + w/2)`` with ``w = window_minutes``
    (5 minutes centred on 16:00 London after the 2015 reform).  Bucket
    membership is decided by the bucket midpoint, so the mask is exact for
    grids of 1-minute buckets aligned to the minute.

    Parameters
    ----------
    times_hours : numpy.ndarray
        Absolute bucket-start hours.
    dt_minutes : float
        Bucket length in minutes.
    fix_hour : float
        Fix time as hour-of-day (London clock), default 16.0.
    window_minutes : float
        Window width, default 5.0.

    Returns
    -------
    numpy.ndarray of bool
    """
    t = np.asarray(times_hours, dtype=float)
    mid_hod = (t + 0.5 * dt_minutes / 60.0) % 24.0
    half_w = 0.5 * window_minutes / 60.0
    eps = 1e-9  # deterministic bucket selection at exact boundaries
    return (mid_hod >= fix_hour - half_w - eps) & (mid_hod < fix_hour + half_w - eps)
