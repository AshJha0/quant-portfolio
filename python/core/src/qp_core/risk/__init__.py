"""Risk building blocks shared by the rates, FX and equity packages (W14, W14b).

Modules
-------
jacobian        Quote-to-node Jacobian, node-to-quote risk mapping, cross-gamma.
vega_grid       Bucketed vega on expiry x strike and expiry x tenor x strike grids.
smile_dynamics  Sticky strike / moneyness / delta deltas; premium-adjusted FX delta.
pnl_explain     Risk-based against full-revaluation P&L (HPL, RTPL, unexplained).

The modules are pure functions of caller-supplied pricers, so they do not
depend on any package's trade or curve types. The rates-specific wiring
(bootstrap Jacobian, curve carry and roll-down) is in ``fi_rates.risk``.
"""

from __future__ import annotations

from .jacobian import cross_gamma, fd_gradient, fd_jacobian, node_to_quote_risk, quote_to_node_risk
from .pnl_explain import PnLExplain, explain, taylor_attribution
from .smile_dynamics import (
    Smile,
    SmileDynamics,
    premium_adjusted_delta,
    rank_smile_dynamics,
    smile_delta,
    smile_gamma,
    smile_price,
)
from .vega_grid import VolGrid, vega_grid

__all__ = [
    "PnLExplain",
    "Smile",
    "SmileDynamics",
    "VolGrid",
    "cross_gamma",
    "explain",
    "fd_gradient",
    "fd_jacobian",
    "node_to_quote_risk",
    "premium_adjusted_delta",
    "quote_to_node_risk",
    "rank_smile_dynamics",
    "smile_delta",
    "smile_gamma",
    "smile_price",
    "taylor_attribution",
    "vega_grid",
]
