"""The fill model shared by the strategy projects.

A decision taken at the close of period ``t`` cannot trade at that close for
free. Two modes are allowed:

``"next"``
    Fill at ``p_t + f * (p_{t+1} - p_t)`` with ``f = fill_fraction`` in
    (0, 1]. ``f = 1`` is the next close. ``f = 0.5`` stands in for a
    next-open or early-session fill when only closes exist.
``"close_auction"``
    Fill at ``p_t`` and pay ``auction_cost_bp`` basis points of traded
    notional on top of the normal costs.

A same-close fill with no extra cost is not representable, on purpose: that
is the look-ahead the rule exists to stop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

__all__ = ["EXEC_MODES", "ExecutionConfig", "fill_price"]

EXEC_MODES: tuple[str, ...] = ("next", "close_auction")

_P = TypeVar("_P", float, NDArray[np.float64])


@dataclass(frozen=True)
class ExecutionConfig:
    """When and at what price a decision taken at the close of ``t`` fills.

    Attributes
    ----------
    mode : {"next", "close_auction"}
        See the module docstring.
    fill_fraction : float
        ``f`` in (0, 1]. Used by ``"next"`` only.
    auction_cost_bp : float
        Extra cost of a close-auction fill, basis points of traded notional,
        finite and >= 0. Projects that trade a different market set their own
        default (``fx_algo`` uses 0.5 bp).
    """

    mode: str = "next"
    fill_fraction: float = 0.5
    auction_cost_bp: float = 10.0

    def __post_init__(self) -> None:
        if self.mode not in EXEC_MODES:
            raise ValueError(f"mode must be 'next' or 'close_auction', got {self.mode!r}")
        f = float(self.fill_fraction)
        if not math.isfinite(f) or not (0.0 < f <= 1.0):
            raise ValueError(f"fill_fraction must be finite and in (0, 1], got {self.fill_fraction!r}")
        a = float(self.auction_cost_bp)
        if not math.isfinite(a) or a < 0:
            raise ValueError(f"auction_cost_bp must be finite and >= 0, got {self.auction_cost_bp!r}")

    @property
    def effective_fraction(self) -> float:
        """Share of the next move paid by the fill: 0 in close-auction mode."""
        return 0.0 if self.mode == "close_auction" else float(self.fill_fraction)


def fill_price(p_t: _P, p_next: _P, config: ExecutionConfig) -> _P:
    """Fill price of an order decided at the close ``p_t``.

    ``p_t + f * (p_next - p_t)`` with ``f = config.effective_fraction``, so a
    close-auction fill returns ``p_t``. Works elementwise on arrays.
    """
    return p_t + config.effective_fraction * (p_next - p_t)
