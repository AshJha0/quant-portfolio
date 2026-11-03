"""Day count conventions, accrual fractions and coupon schedule generation.

The arithmetic lives in :mod:`qp_core.daycount`; this module keeps the
fixed-income convention list and sign rule (negative fractions allowed).

Conventions implemented
-----------------------
``ACT/365F``     actual days / 365 (fixed).  Used throughout this package to
                 map calendar dates onto the curve's time axis.
``ACT/360``      actual days / 360.  Money-market convention (deposits, FRAs).
``30/360US``     US (NASD) 30/360 bond basis, without the end-of-February
                 special rule (documented simplification).
``ACT/ACT-ISDA`` actual days split by calendar year, leap years / 366,
                 non-leap years / 365. Swap and some euro-area
                 conventions. Not the government-bond convention.
``ACT/ACT-ICMA`` actual days in the accrual period over (frequency x actual
                 days in the reference coupon period). Regular coupons are
                 exactly ``coupon / frequency``. US Treasuries, gilts and
                 most government bonds use it. It needs the reference
                 period, so call ``year_fraction`` with ``ref_start``,
                 ``ref_end`` and ``frequency``.

Schedules
---------
``generate_schedule`` rolls coupon dates backwards from maturity in steps of
``12 / frequency`` months (annual, semiannual, quarterly), producing a short
front stub if the effective date does not align. With ``eom=True`` (the
default) a month-end maturity keeps every coupon date on a month end, so a
28-Feb-2025 maturity pays on 31 Aug. Payment dates can be rolled with any
calendar from :mod:`qp_core.holidays` via :mod:`fi_rates.calendars` (``adjust=True``); accrual uses
unadjusted dates. 30/360US omits the end-of-February rule.
"""

from __future__ import annotations

import datetime as dt

from qp_core import daycount as _core
from qp_core.daycount import (
    SUPPORTED_FREQUENCIES,
    add_months,
    adjust_modified_following,
    generate_schedule,
    is_month_end,
    roll_months,
)

__all__ = [
    "SUPPORTED_CONVENTIONS",
    "SUPPORTED_FREQUENCIES",
    "add_months",
    "adjust_modified_following",
    "generate_schedule",
    "is_month_end",
    "roll_months",
    "year_fraction",
]

SUPPORTED_CONVENTIONS: tuple[str, ...] = (
    "ACT/365F",
    "ACT/360",
    "30/360US",
    "ACT/ACT-ISDA",
    "ACT/ACT-ICMA",
)


def year_fraction(
    start: dt.date,
    end: dt.date,
    convention: str = "ACT/365F",
    *,
    ref_start: dt.date | None = None,
    ref_end: dt.date | None = None,
    frequency: int | None = None,
) -> float:
    """Day count fraction between two dates.

    Parameters
    ----------
    start, end : datetime.date
        Accrual period start / end.  ``end`` may precede ``start``; the
        fraction is then negative (useful for horizon arithmetic).
    convention : str
        One of :data:`SUPPORTED_CONVENTIONS`.
    ref_start, ref_end, frequency
        Reference coupon period and coupons per year. Required for
        ``ACT/ACT-ICMA`` and ignored otherwise.

    Returns
    -------
    float
        Year fraction under the requested convention.
    """
    if convention not in SUPPORTED_CONVENTIONS:
        raise ValueError(
            f"Unknown day count convention {convention!r}; "
            f"supported: {SUPPORTED_CONVENTIONS}"
        )
    return _core.year_fraction(
        start, end, convention, ref_start=ref_start, ref_end=ref_end, frequency=frequency
    )
