"""Settlement holiday calendars: a thin re-export of :mod:`qp_core.holidays`.

The rules-based generator moved to ``qp_core`` in round 4 so the equity
credit package can use the same calendars without importing an FX package.
Everything that was importable from here still is. See
``qp_core.holidays`` for the rules, their sources and how to add one-off
closures.
"""

from __future__ import annotations

from qp_core.holidays import (
    CURRENCIES,
    MAX_YEAR,
    MIN_YEAR,
    add_closure,
    clear_closures,
    easter_sunday,
    extra_closures,
    holidays,
    pair_holidays,
)

__all__ = [
    "CURRENCIES",
    "MAX_YEAR",
    "MIN_YEAR",
    "add_closure",
    "clear_closures",
    "easter_sunday",
    "extra_closures",
    "holidays",
    "pair_holidays",
]
