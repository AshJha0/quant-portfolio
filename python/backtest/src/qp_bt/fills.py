"""Fill models: when, and at what price, a decision fills.

Plain idea
----------
A decision taken at the close of bar ``t`` cannot trade at that close for
free. It fills ``lag`` bars later, at a point ``fraction`` of the way along
the path from the previous close to that bar's close.

Formal statement
----------------
A decision at bar ``t`` fills in period ``s = t + lag`` (the period from
close ``s-1`` to close ``s``) at

    p_fill = p_{s-1} + fraction * (p_s - p_{s-1}).

Holdings before the fill earn ``a = fraction * r_s``; holdings after it earn
``b = (1 + r_s) / (1 + a) - 1``, where ``r_s = p_s / p_{s-1} - 1``.

* ``lag = 1, fraction = f`` is :class:`qp_core.execution.ExecutionConfig`
  mode ``"next"``: ``f = 1`` is the next close, ``f = 0.5`` stands in for an
  early-session fill when only closes exist.
* ``lag = 1, fraction = 0`` fills at ``p_t`` and books the trade in period
  ``t + 1`` (close auction, booked the next day).
* ``lag = 0, fraction = 1`` fills at ``p_t`` and books the trade in period
  ``t`` itself (close auction, booked on the decision day). ``lag = 0`` with
  any other fraction would trade before the decision was taken, and raises.

Both auction forms fill at the same price; they differ only in which ledger
row carries the trade. ``extra_cost_bp`` is the auction premium charged on
every traded unit of notional.

Worked example
--------------
``p_t = 100``, ``p_{t+1} = 102``, ``fraction = 0.5``, ``lag = 1``: the fill is
at 101, ``a = 0.01``, ``b = 102/101 - 1 = 0.0099010``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qp_core.execution import ExecutionConfig

__all__ = ["NextBarFill"]


@dataclass(frozen=True)
class NextBarFill:
    """Next-bar fill model with a signal-to-fill lag and optional partial fills.

    Attributes
    ----------
    fraction : float
        Point in [0, 1] along the fill period's close-to-close path.
    lag : int
        Bars from decision to fill period; 0 only with ``fraction = 1``.
    extra_cost_bp : float
        Premium in bp of traded notional (auction cost), >= 0.
    max_participation : float or None
        Event mode only: the most a unit order may fill in one bar, as a
        share of that bar's volume. The rest stays working and fills in
        later bars (partial fills). ``None`` fills in full.
    """

    fraction: float = 0.5
    lag: int = 1
    extra_cost_bp: float = 0.0
    max_participation: float | None = None

    def __post_init__(self) -> None:
        f = float(self.fraction)
        if not math.isfinite(f) or not 0.0 <= f <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {self.fraction!r}")
        if int(self.lag) != self.lag or self.lag < 0:
            raise ValueError(f"lag must be a non-negative integer, got {self.lag!r}")
        if self.lag == 0 and f != 1.0:
            raise ValueError("lag 0 fills at the decision close, so fraction must be 1")
        e = float(self.extra_cost_bp)
        if not math.isfinite(e) or e < 0:
            raise ValueError(f"extra_cost_bp must be finite and >= 0, got {self.extra_cost_bp!r}")
        if self.max_participation is not None:
            m = float(self.max_participation)
            if not math.isfinite(m) or not 0.0 < m <= 1.0:
                raise ValueError(f"max_participation must be in (0, 1], got {m!r}")

    @classmethod
    def from_execution(cls, execution: ExecutionConfig | None = None, *,
                       auction_booked: str = "next") -> NextBarFill:
        """The fill model equivalent to an :class:`ExecutionConfig`.

        ``auction_booked`` places a close-auction fill in the decision day's
        ledger row (``"decision"``, lag 0) or the next row (``"next"``,
        lag 1, fraction 0). The price is ``p_t`` either way.
        """
        ex = execution if execution is not None else ExecutionConfig()
        if ex.mode == "next":
            return cls(fraction=float(ex.fill_fraction), lag=1)
        if auction_booked == "decision":
            return cls(fraction=1.0, lag=0, extra_cost_bp=float(ex.auction_cost_bp))
        if auction_booked == "next":
            return cls(fraction=0.0, lag=1, extra_cost_bp=float(ex.auction_cost_bp))
        raise ValueError(f"auction_booked must be 'decision' or 'next', got {auction_booked!r}")

    def fill_price(self, p_prev: float, p_now: float) -> float:
        """``p_prev + fraction * (p_now - p_prev)``."""
        return p_prev + self.fraction * (p_now - p_prev)
