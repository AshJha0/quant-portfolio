"""Gamma and vega neutralisation with two listed options, then delta with the underlying.

Plain idea
----------
Shares have no gamma or vega, so to flatten those a book must trade
options. Two options give two unknowns for the two equations "gamma = 0"
and "vega = 0"; the shares then mop up whatever delta is left.

Formal statement
----------------
For book Greeks ``(D, G, V)`` and hedge options with per-unit Greeks
``(d_i, g_i, v_i)``, solve

    [g_1 g_2] [n_1]     [G]
    [v_1 v_2] [n_2] = - [V],     n_s = -(D + n_1 d_1 + n_2 d_2).

The system is singular when the two options have proportional gamma and
vega (for Black-Scholes, the same expiry), and the function raises then.

Worked example
--------------
>>> from qp_core.hedging.greeks import OptionGreeks, gamma_vega_hedge
>>> h = gamma_vega_hedge(OptionGreeks(0.5, 0.02, 30.0),
...                      OptionGreeks(0.6, 0.03, 10.0), OptionGreeks(0.4, 0.01, 20.0))
>>> [round(x, 6) for x in (h.n1, h.n2, h.n_underlying)]
[-0.2, -1.4, 0.18]
"""

from __future__ import annotations

from dataclasses import dataclass

from qp_core.errors import InputError

__all__ = ["GreekHedge", "OptionGreeks", "gamma_vega_hedge", "hedge_cost"]


@dataclass(frozen=True)
class OptionGreeks:
    """Delta, gamma and vega of one unit (or of a whole book)."""

    delta: float
    gamma: float
    vega: float


@dataclass(frozen=True)
class GreekHedge:
    """Units of hedge option 1, option 2 and the underlying to trade."""

    n1: float
    n2: float
    n_underlying: float


def gamma_vega_hedge(
    book: OptionGreeks, opt1: OptionGreeks, opt2: OptionGreeks, *, rcond: float = 1e-12
) -> GreekHedge:
    """Quantities that leave the book with zero delta, gamma and vega."""
    det = opt1.gamma * opt2.vega - opt2.gamma * opt1.vega
    scale = max(abs(opt1.gamma * opt2.vega), abs(opt2.gamma * opt1.vega), 1e-300)
    if abs(det) <= rcond * scale:
        raise InputError("hedge options have proportional gamma and vega; pick a different expiry")
    n1 = (-book.gamma * opt2.vega + opt2.gamma * book.vega) / det
    n2 = (-opt1.gamma * book.vega + book.gamma * opt1.vega) / det
    n_s = -(book.delta + n1 * opt1.delta + n2 * opt2.delta)
    return GreekHedge(n1, n2, n_s)


def hedge_cost(hedge: GreekHedge, half_spread1: float, half_spread2: float, half_spread_s: float) -> float:
    """Crossing cost of the hedge: sum of ``|units| * half-spread`` in price units."""
    if min(half_spread1, half_spread2, half_spread_s) < 0.0:
        raise InputError("half-spreads must be >= 0")
    return (
        abs(hedge.n1) * half_spread1 + abs(hedge.n2) * half_spread2 + abs(hedge.n_underlying) * half_spread_s
    )
