"""W27: smart order router over several lit books and dark venues.

Modules
-------
allocation
    Cont-Kukanov order placement as an LP (no-oversubscription variant),
    with closed-form special cases.
fees
    Venue fee and latency schedules (data in ``python/execution/venues/``).
market
    The multi-venue market: lit venues are W25 reference books.
dark
    Midpoint, conditional and periodic-auction dark venues; counterparties
    and the predatory trader.
router
    The smart router: sweep, allocate, post, cancel, learn.
tca
    Per-venue routing TCA and markout-learned venue scores.
gaming
    The predator's detector, its AUC and a bootstrap confidence interval.

Only ``qp_exec.lob``'s published names are used from W25.
"""

from .allocation import (
    Allocation,
    deterministic_optimum,
    expected_cost,
    single_venue_optimum,
    solve_allocation,
)
from .fees import DEFAULT_DARK, DEFAULT_LIT, DarkSchedule, LitSchedule, load_fee_table
from .market import ExecFill, Market, MarketParams
from .router import RouterReport, SmartRouter, SweepLeg, sweep_plan
from .tca import VenueScores, VenueTCA, venue_tca

__all__ = [
    "DEFAULT_DARK", "DEFAULT_LIT", "Allocation", "DarkSchedule", "ExecFill", "LitSchedule",
    "Market", "MarketParams", "RouterReport", "SmartRouter", "SweepLeg", "VenueScores",
    "VenueTCA", "deterministic_optimum", "expected_cost", "load_fee_table",
    "single_venue_optimum", "solve_allocation", "sweep_plan", "venue_tca",
]
