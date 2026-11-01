"""Drawdown, downside deviation and annualisation.

Conventions (one per quantity, used by every project)
-----------------------------------------------------
Downside deviation
    Full-sample root mean square of ``min(r - mar, 0)``: the squares are
    averaged over all ``n`` observations, not only the ones below ``mar``.
    This is the Sortino and Price (1994) definition and the one every
    project already used. Zero downside deviation makes the Sortino ratio
    undefined; what to return then (NaN, inf or an exception) stays a
    project decision and is documented in each project's ``sortino_ratio``.
Drawdown of a return series
    A positive fraction of the running peak of wealth, ``1 - W_t / max W``.
    Wealth starts at 1 and the running peak includes that start
    (``include_start=True``): a first-period loss is a drawdown. Two projects
    historically measured from the first post-return wealth instead
    (``eq_algo.evaluation.max_drawdown`` and ``fx_port.metrics.max_drawdown``);
    they keep that behaviour by passing ``include_start=False``, so their
    reported numbers do not move.
Drawdown of a P&L series
    Equity is ``cumsum(pnl)`` starting from 0, and the running peak includes
    that 0.
Annualisation
    A per-period mean scales by ``P`` and a per-period standard deviation or
    ratio by ``sqrt(P)``, with ``P`` periods per year (252 for daily).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "annualisation_factor",
    "check_periods_per_year",
    "downside_deviation",
    "drawdown_series",
    "max_drawdown_from_equity",
    "max_drawdown_from_pnl",
    "max_drawdown_from_returns",
    "wealth_curve",
]

Compounding = Literal["simple", "log"]


def check_periods_per_year(periods_per_year: float) -> float:
    """Return ``float(periods_per_year)``; raise unless finite and > 0."""
    p = float(periods_per_year)
    if not math.isfinite(p) or p <= 0:
        raise ValueError(f"periods_per_year must be finite and positive, got {periods_per_year}")
    return p


def annualisation_factor(periods_per_year: float) -> float:
    """``sqrt(periods_per_year)``: the scale for a volatility or a ratio."""
    return math.sqrt(check_periods_per_year(periods_per_year))


def downside_deviation(returns: ArrayLike, mar: float = 0.0) -> float:
    """``sqrt(mean(min(r - mar, 0)^2))`` over all observations.

    No finiteness or length check is made here: callers validate first, as
    their error messages differ. An empty input gives NaN.
    """
    r = np.asarray(returns, dtype=float)
    return float(np.sqrt(np.mean(np.minimum(r - mar, 0.0) ** 2)))


def wealth_curve(
    returns: ArrayLike,
    compounding: Compounding = "simple",
    *,
    include_start: bool = True,
) -> NDArray[np.float64]:
    """Wealth path from per-period returns.

    ``compounding="simple"`` gives ``cumprod(1 + r)``; ``"log"`` gives
    ``exp(cumsum(r))``. With ``include_start`` the starting wealth 1.0 is
    prepended, so the path has ``n + 1`` points.
    """
    r = np.asarray(returns, dtype=float)
    if compounding == "simple":
        w = np.cumprod(1.0 + r)
    elif compounding == "log":
        w = np.exp(np.cumsum(r))
    else:
        raise ValueError(f"compounding must be 'simple' or 'log', got {compounding!r}")
    if include_start:
        return np.concatenate([[1.0], w])
    return w


def drawdown_series(equity: ArrayLike) -> NDArray[np.float64]:
    """Drawdown from the running peak, in the units of ``equity`` (<= 0)."""
    e = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(e)
    return e - peak


def max_drawdown_from_equity(equity: ArrayLike) -> float:
    """Largest drop from a running peak of an equity curve, as a number >= 0.

    In the units of ``equity``. A rising curve gives ``0.0``, not ``-0.0``.
    """
    return float(-drawdown_series(equity).min()) + 0.0


def max_drawdown_from_returns(
    returns: ArrayLike,
    compounding: Compounding = "simple",
    *,
    include_start: bool = True,
) -> float:
    """Maximum drawdown of compounded wealth, as a positive fraction of peak.

    0.25 means a 25% fall from the running peak. See the module docstring for
    ``include_start``.
    """
    wealth = wealth_curve(returns, compounding, include_start=include_start)
    peak = np.maximum.accumulate(wealth)
    return float((1.0 - wealth / peak).max())


def max_drawdown_from_pnl(pnl: ArrayLike) -> float:
    """Maximum drawdown of ``cumsum(pnl)`` with the running peak floored at 0.

    In P&L units, >= 0. An empty series gives 0.0.
    """
    r = np.asarray(pnl, dtype=float)
    if r.size == 0:
        return 0.0
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(np.maximum(eq, 0.0))
    return float(np.max(peak - eq, initial=0.0))
