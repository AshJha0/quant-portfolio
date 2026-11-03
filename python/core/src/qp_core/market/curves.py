"""Discount curves and volatility surfaces held by a market snapshot.

Plain idea
----------
A discount curve answers "what is one unit of this currency paid at time t
worth today?". It is stored as pillar times and discount factors; between
pillars the log of the discount factor is linear in time, which is the same
as a flat instantaneous forward rate on each segment. Past the last pillar
the last segment's forward rate carries on.

Formal statement
----------------
For pillars ``0 = t_0 < t_1 < ... < t_n`` with ``l_i = ln P(t_i)`` and
``l_0 = 0``,

    ln P(t) = l_i + (t - t_i) (l_{i+1} - l_i) / (t_{i+1} - t_i),  t_i <= t <= t_{i+1},

and for ``t > t_n`` the slope of the last segment is extended. This is the
rule of ``fx_rates.curve.DiscountCurve`` (Hagan and West 2006, "raw
interpolation"); the arithmetic is the same ``numpy.interp`` call, so a
curve copied from ``fx_rates`` gives bit-identical discount factors.

A :class:`VolSurface` is a grid of Black implied volatilities by expiry and
strike. Between expiries total variance ``sigma^2 t`` is linear in ``t``
(Gatheral 2006, ch. 3); between strikes volatility is linear and is flat
outside the grid. A one-expiry, one-strike surface is a flat volatility.

Worked example
--------------
>>> c = DiscountCurve((1.0, 2.0), (0.96, 0.92))
>>> round(c.df(1.5), 12)
0.939787209958
>>> round(c.shifted(0.0001).df(1.0) / c.df(1.0), 12)   # +1bp zero shift
0.999900005
>>> VolSurface.flat(0.2).vol(0.7, 123.0)
0.2
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import InputError

__all__ = ["DiscountCurve", "VolSurface"]


def _floats(xs: Sequence[float] | np.ndarray, what: str) -> tuple[float, ...]:
    out = tuple(float(x) for x in xs)
    if not out:
        raise InputError(f"{what} must not be empty")
    if not all(math.isfinite(x) for x in out):
        raise InputError(f"{what} must be finite")
    return out


@dataclass(frozen=True)
class DiscountCurve:
    """Discount curve on pillar times (years from the snapshot date).

    ``times`` strictly increasing and > 0; ``dfs`` strictly positive.
    Immutable: shocks return new curves.
    """

    times: tuple[float, ...]
    dfs: tuple[float, ...]

    def __init__(self, times: Sequence[float] | np.ndarray, dfs: Sequence[float] | np.ndarray) -> None:
        t = _floats(times, "times")
        d = _floats(dfs, "dfs")
        if len(t) != len(d):
            raise InputError("times and dfs must have equal length")
        if t[0] <= 0.0 or any(b <= a for a, b in zip(t, t[1:], strict=False)):
            raise InputError("pillar times must be > 0 and strictly increasing")
        if any(x <= 0.0 for x in d):
            raise InputError("discount factors must be > 0")
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "dfs", d)

    @classmethod
    def from_zero_rates(cls, times: Sequence[float], zeros: Sequence[float]) -> DiscountCurve:
        """Curve from continuously compounded zero rates: ``P(t) = exp(-z t)``."""
        t = _floats(times, "times")
        z = _floats(zeros, "zeros")
        if len(t) != len(z):
            raise InputError("times and zeros must have equal length")
        return cls(t, [math.exp(-zi * ti) for ti, zi in zip(t, z, strict=True)])

    @classmethod
    def flat(cls, rate: float, horizon: float = 50.0) -> DiscountCurve:
        """Flat continuously compounded curve (one pillar, extrapolated)."""
        return cls.from_zero_rates([horizon], [rate])

    def _logdf(self, t: float) -> float:
        x = np.concatenate(([0.0], np.asarray(self.times)))
        y = np.concatenate(([0.0], np.log(np.asarray(self.dfs))))
        if t > x[-1]:
            slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
            return float(y[-1] + slope * (t - x[-1]))
        return float(np.interp(np.atleast_1d(t), x, y)[0])

    def df(self, t: float) -> float:
        """Discount factor at ``t >= 0`` years."""
        if not math.isfinite(t) or t < 0.0:
            raise InputError(f"discount factor requested at invalid time {t!r}")
        return float(np.exp(np.atleast_1d(self._logdf(float(t))))[0])

    def zero_rate(self, t: float) -> float:
        """Continuously compounded zero rate at ``t > 0``."""
        if not t > 0.0:
            raise InputError("zero rate requires t > 0")
        return -self._logdf(float(t)) / float(t)

    def shifted(self, dz: float) -> DiscountCurve:
        """Parallel shift of every pillar zero rate by ``dz`` (decimal)."""
        return DiscountCurve(self.times, [d * math.exp(-dz * t) for t, d in zip(self.times, self.dfs, strict=True)])

    def bumped_pillar(self, i: int, dz: float) -> DiscountCurve:
        """Shift the zero rate of pillar ``i`` only (key-rate bump)."""
        d = list(self.dfs)
        d[i] = d[i] * math.exp(-dz * self.times[i])
        return DiscountCurve(self.times, d)

    def to_dict(self) -> dict[str, Any]:
        return {"times": list(self.times), "dfs": list(self.dfs)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DiscountCurve:
        return cls(d["times"], d["dfs"])


@dataclass(frozen=True)
class VolSurface:
    """Black implied volatility grid ``vols[i][j]`` at ``expiries[i]``, ``strikes[j]``."""

    expiries: tuple[float, ...]
    strikes: tuple[float, ...]
    vols: tuple[tuple[float, ...], ...]

    def __init__(
        self,
        expiries: Sequence[float],
        strikes: Sequence[float],
        vols: Sequence[Sequence[float]],
    ) -> None:
        e = _floats(expiries, "expiries")
        k = _floats(strikes, "strikes")
        v = tuple(_floats(row, "vols") for row in vols)
        if e[0] <= 0.0 or any(b <= a for a, b in zip(e, e[1:], strict=False)):
            raise InputError("expiries must be > 0 and strictly increasing")
        if any(b <= a for a, b in zip(k, k[1:], strict=False)):
            raise InputError("strikes must be strictly increasing")
        if len(v) != len(e) or any(len(r) != len(k) for r in v):
            raise InputError("vols must be len(expiries) rows of len(strikes)")
        if any(x <= 0.0 for r in v for x in r):
            raise InputError("vols must be > 0")
        object.__setattr__(self, "expiries", e)
        object.__setattr__(self, "strikes", k)
        object.__setattr__(self, "vols", v)

    @classmethod
    def flat(cls, vol: float) -> VolSurface:
        return cls([1.0], [1.0], [[vol]])

    def _smile(self, i: int, strike: float) -> float:
        return float(np.interp(strike, self.strikes, self.vols[i]))

    def vol(self, t: float, strike: float) -> float:
        """Volatility at expiry ``t > 0`` and ``strike``."""
        if not t > 0.0:
            raise InputError("vol requires t > 0")
        e = self.expiries
        if len(e) == 1 or t <= e[0]:
            return self._smile(0, strike)
        if t >= e[-1]:
            return self._smile(len(e) - 1, strike)
        j = int(np.searchsorted(e, t))
        w0 = self._smile(j - 1, strike) ** 2 * e[j - 1]
        w1 = self._smile(j, strike) ** 2 * e[j]
        w = w0 + (w1 - w0) * (t - e[j - 1]) / (e[j] - e[j - 1])
        return math.sqrt(w / t)

    def shifted(self, dvol: float) -> VolSurface:
        """Parallel shift of every grid volatility by ``dvol`` (decimal)."""
        return VolSurface(self.expiries, self.strikes, [[x + dvol for x in r] for r in self.vols])

    def to_dict(self) -> dict[str, Any]:
        return {"expiries": list(self.expiries), "strikes": list(self.strikes), "vols": [list(r) for r in self.vols]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VolSurface:
        return cls(d["expiries"], d["strikes"], d["vols"])
