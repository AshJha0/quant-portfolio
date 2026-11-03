"""RFQ quoting: hit-ratio model, optimal markup, inventory skew.

Plain idea
----------
A client asks several dealers for a price. Quote too wide and you lose the
trade; too tight and you win but earn nothing. A fitted hit-ratio curve
``P(win | markup)`` turns that into a one-line optimisation: maximise
``markup * P(win)``. If the trade would also reduce the dealer's inventory
risk, that reduction has value, so the dealer quotes tighter on that side
(skew).

Formal statement
----------------
With the logistic hit ratio ``P(m) = 1 / (1 + exp(-(a + b m)))``, ``b < 0``,
the expected profit ``m P(m)`` is maximised at

    m* = (1 + W(exp(a - 1))) / (-b),

where ``W`` is the principal Lambert W function (set the derivative
``P + m P' = 0``, use ``P' = b P (1 - P)``, substitute ``z = -b m``).
Fermanian, Gueant and Rachez (2016) use this logistic form for RFQ hit
ratios. If winning also moves inventory by ``side * size`` and the dealer
values risk reduction at ``v = gamma * sigma^2 * q * size * side_sign``
(the Avellaneda-Stoikov 2008 reservation-price shift, one trade horizon),
the dealer maximises ``(m + v) P(m)``, whose optimum is
``m* = (1 + W(exp(a - b v - 1))) / (-b) - v``.

Worked example
--------------
>>> from qp_exec.rfq.quote import HitRatioModel, optimal_markup
>>> hm = HitRatioModel(intercept=1.0, slope=-200.0)
>>> round(optimal_markup(hm), 6)
0.007836
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "HitRatioModel",
    "expected_profit",
    "fit_hit_ratio",
    "lambert_w0",
    "optimal_markup",
    "skewed_markup",
]


def lambert_w0(x: float) -> float:
    """Principal branch ``W(x)`` for ``x >= 0`` (Halley iteration, 1e-15 relative)."""
    if not (x >= 0.0 and math.isfinite(x)):
        raise ValueError("lambert_w0 is implemented for finite x >= 0")
    if x == 0.0:
        return 0.0
    w = math.log1p(x) if x < 3.0 else math.log(x) - math.log(math.log(x))
    for _ in range(50):
        ew = math.exp(w)
        f = w * ew - x
        wp1 = w + 1.0
        step = f / (ew * wp1 - (w + 2.0) * f / (2.0 * wp1))
        w -= step
        if abs(step) <= 1e-15 * (1.0 + abs(w)):
            break
    return w


@dataclass(frozen=True)
class HitRatioModel:
    """Logistic hit ratio in markup (price units per unit notional, or bp if you fit in bp)."""

    intercept: float
    slope: float

    def __post_init__(self) -> None:
        if not (self.slope < 0.0 and math.isfinite(self.intercept)):
            raise ValueError("slope must be < 0 (wider quote wins less) and intercept finite")

    def p_win(self, markup: float) -> float:
        z = self.intercept + self.slope * markup
        return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def expected_profit(model: HitRatioModel, markup: float, inventory_value: float = 0.0) -> float:
    """``(m + v) P(m)``."""
    return (markup + inventory_value) * model.p_win(markup)


def optimal_markup(model: HitRatioModel) -> float:
    """``argmax m P(m)`` in closed form via Lambert W."""
    return (1.0 + lambert_w0(math.exp(model.intercept - 1.0))) / (-model.slope)


def skewed_markup(
    model: HitRatioModel,
    *,
    inventory: float,
    size: float,
    firm_side: int,
    risk_aversion: float,
    sigma: float,
    horizon: float = 1.0,
) -> float:
    """Optimal markup when winning moves inventory by ``firm_side * size``.

    ``firm_side`` is +1 if the firm buys on a win, -1 if it sells. The
    risk-reduction value is ``v = gamma sigma^2 T q size (-firm_side)``:
    positive when the trade shrinks ``|q|``. A long dealer (q > 0) asked to
    sell quotes tighter; asked to buy, wider.
    """
    if firm_side not in (1, -1) or size <= 0.0 or risk_aversion < 0.0 or sigma < 0.0:
        raise ValueError("firm_side in {+1,-1}, size > 0, risk_aversion and sigma >= 0")
    v = risk_aversion * sigma * sigma * horizon * inventory * size * (-firm_side)
    a, b = model.intercept, model.slope
    # Variable m' = m + v has hit ratio logistic(a - b v + b m').
    arg = a - b * v - 1.0
    if arg > 700.0:  # W(e^x) ~ x - log x for huge x
        w = arg - math.log(arg)
    else:
        w = lambert_w0(math.exp(arg))
    return (1.0 + w) / (-b) - v


def fit_hit_ratio(
    markups: npt.ArrayLike, won: npt.ArrayLike, *, max_iter: int = 100, tol: float = 1e-12
) -> tuple[HitRatioModel, npt.NDArray[np.float64]]:
    """Logistic MLE by Newton-Raphson; returns the model and coefficient SEs.

    SEs are the square roots of the inverse observed information diagonal.
    """
    m = np.asarray(markups, dtype=np.float64)
    y = np.asarray(won, dtype=np.float64)
    if m.ndim != 1 or m.shape != y.shape or m.size < 10:
        raise ValueError("markups and won must be 1-D, aligned, at least 10")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("won must be 0/1")
    x = np.column_stack([np.ones_like(m), m])
    beta = np.zeros(2)
    info = np.eye(2)
    for _ in range(max_iter):
        z = x @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        w = p * (1.0 - p)
        grad = x.T @ (y - p)
        info = (x * w[:, None]).T @ x
        step = np.linalg.solve(info, grad)
        beta = beta + step
        if float(np.max(np.abs(step))) < tol:
            break
    cov = np.linalg.inv(info)
    se: npt.NDArray[np.float64] = np.sqrt(np.diag(cov))
    return HitRatioModel(float(beta[0]), float(beta[1])), se
