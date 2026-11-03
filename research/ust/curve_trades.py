"""DV01-neutral UST slope and fly: carry, roll-down and realised returns (W23 D).

Plain idea: an upward-sloping curve pays you to hold longer bonds even if
nothing moves (carry), and as a bond ages it rolls down to a lower yield
(roll-down). A 2s10s steepener or flattener, sized so a parallel move does
not change its value (DV01-neutral), isolates the slope bet. Who pays: the
term premium, compensation for duration risk that varies with the slope
(Fama and Bliss 1987; Campbell and Shiller 1991; Cochrane and Piazzesi
2005). A known premium, not new alpha.

Formal statement. All legs are zero-coupon bonds on the GSW curve (no coupon
or financing basis; a labelled simplification). Log price
``p_t(n) = -n y_t(n)``. Over a holding period ``h`` years the log return of
an ``n``-year zero, financed at the ``h``-year zero rate, is

    rx_{t,t+h}(n) = p_{t+h}(n - h) - p_t(n) - h y_t(h).

Expected carry + roll-down assumes an unchanged curve:
``cr_t(n) = n y_t(n) - (n - h) y_t(n - h) - h y_t(h)``.

The DV01 of one unit of log-price exposure to ``n`` is ``n`` (dp/dy = -n),
so a DV01-neutral slope ``long a units of n_long, short b units of n_short``
uses ``a n_long = b n_short``. We set the long-end DV01 to 1 (a = 1/n_long,
b = 1/n_short): the trade returns ``rx(n_long)/n_long - rx(n_short)/n_short``,
a return per unit of DV01 measured in years, i.e. in yield units. A fly
long the belly, short both wings, DV01-split 50/50:
``rx(belly)/belly - 0.5 rx(w1)/w1 - 0.5 rx(w2)/w2``.

Position rule (time-series, registered separately): hold the side whose
expected carry + roll-down is positive, sign ``s_t = sign(cr_t)``.

Worked example (``tests/test_ust.py``): a flat 4 % curve has zero carry and
roll-down at every maturity, and a parallel 50 bp shift moves the
DV01-neutral 2s10s by ``-0.005 (10 - h)/10 + 0.005 (2 - h)/2`` per unit
DV01, ``= 0.005 h (1/10 - 1/2) = -0.002 h``, which with h = 1/12 is
``-0.0049583 + 0.0047917 = -0.0001667``: only the ageing of the legs over
the month, a thirtieth of the 50 bp move.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from gsw import zero_yield


def _y(params: pd.Series, n: float) -> float:
    return float(zero_yield(params, n)[0])


def carry_rolldown(params: pd.Series, n: float, h: float) -> float:
    """``n y(n) - (n - h) y(n - h) - h y(h)``, unchanged-curve excess log return."""
    if not n > h > 0:
        raise ValueError("need n > h > 0")
    return n * _y(params, n) - (n - h) * _y(params, n - h) - h * _y(params, h)


def excess_return(p0: pd.Series, p1: pd.Series, n: float, h: float) -> float:
    """Realised ``p_{t+h}(n-h) - p_t(n) - h y_t(h)``."""
    if not n > h > 0:
        raise ValueError("need n > h > 0")
    return -(n - h) * _y(p1, n - h) + n * _y(p0, n) - h * _y(p0, h)


def legs(kind: str, tenors: Sequence[float]) -> list[tuple[float, float]]:
    """(maturity, weight per unit log-price exposure), DV01-neutral."""
    if kind == "slope":
        short, long_ = tenors
        return [(long_, 1.0 / long_), (short, -1.0 / short)]
    if kind == "fly":
        w1, belly, w2 = tenors
        return [(belly, 1.0 / belly), (w1, -0.5 / w1), (w2, -0.5 / w2)]
    raise ValueError(f"unknown trade {kind!r}")


def trade_carry(params: pd.Series, kind: str, tenors: Sequence[float], h: float) -> float:
    return sum(w * carry_rolldown(params, n, h) for n, w in legs(kind, tenors))


def trade_return(p0: pd.Series, p1: pd.Series, kind: str, tenors: Sequence[float],
                 h: float) -> float:
    return sum(w * excess_return(p0, p1, n, h) for n, w in legs(kind, tenors))


def monthly_study(curve: pd.DataFrame, kind: str, tenors: Sequence[float],
                  cost_per_flip: float = 0.0) -> pd.DataFrame:
    """Month-end panel: expected carry, sign position, realised and net return.

    ``curve`` is GSW parameters by business day. Month ends are the last
    available date of each calendar month; ``h`` is the actual year fraction
    between consecutive month ends (days / 365.25). The position set at month
    end m earns the trade return to m+1. ``cost_per_flip`` (yield units per
    unit DV01, e.g. 0.0001 = 1 bp of DV01) is charged when the sign changes,
    twice for a full reversal, once for the first build.
    """
    me = curve.groupby(pd.DatetimeIndex(curve.index).to_period("M")).tail(1)
    dates = pd.DatetimeIndex(me.index)
    out, prev = [], 0.0
    for i in range(len(me) - 1):
        d0, d1 = dates[i], dates[i + 1]
        p0, p1 = me.iloc[i], me.iloc[i + 1]
        h = (d1 - d0).days / 365.25
        cr = trade_carry(p0, kind, tenors, h)
        pos = float(np.sign(cr))
        rx = trade_return(p0, p1, kind, tenors, h)
        cost = abs(pos - prev) * cost_per_flip
        out.append({"date": d0, "carry": cr, "position": pos, "rx": rx,
                    "net": pos * rx - cost})
        prev = pos
    return pd.DataFrame(out).set_index("date")
