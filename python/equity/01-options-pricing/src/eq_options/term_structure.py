"""Deterministic rate, dividend and volatility term structures for European options.

Plain idea: with rates, dividend yields and volatility that change over time
but are known today, the terminal log-spot is still normal. Its mean uses
the integrated rates and its variance the integrated variance, so Black-
Scholes holds with the averages over ``[0, T]``: the average rate, the
average yield and the root-mean-square volatility. Term structures only have
to supply those integrals.

Formal statement (Merton 1973, "Theory of rational option pricing", Bell J.
Econ. 4(1), 141-183, section 8). With instantaneous forward rate ``f(t)``,
yield ``g(t)`` and volatility ``s(t)``:
``r_bar = (1/T) int_0^T f``, ``q_bar = (1/T) int_0^T g``,
``sigma_bar^2 = (1/T) int_0^T s^2``, and the price is
``bs(S, K, T, r_bar, sigma_bar, q_bar)``.

Representation used here:

* :class:`ZeroCurve`: pillar times and continuously compounded zero rates;
  ``r(t) t`` is interpolated linearly, so the instantaneous forward rate is
  piecewise constant (the "raw" method of Hagan and West 2006,
  "Interpolation methods for curve construction", Appl. Math. Finance
  13(2), 89-129). The last forward rate is extended flat.
* :class:`VolTermStructure`: pillar times and implied (term) vols; total
  variance ``w(t) = sigma(t)^2 t`` is interpolated linearly, so the forward
  variance is piecewise constant. A decreasing ``w`` is calendar arbitrage
  and raises ``ValueError``.

Worked example: zero rates 2 % to 1y and 4 % to 2y give a 1y-to-2y forward
of 6 %; vols 20 % to 1y and 30 % to 2y give a forward vol of
``sqrt(0.09 * 2 - 0.04) = 37.4 %``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .black_scholes import OptionType, bs_price

__all__ = ["VolTermStructure", "ZeroCurve", "bs_price_ts", "flat_curve"]


def _pillars(times: Sequence[float], values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    if t.ndim != 1 or t.size == 0 or t.size != v.size:
        raise ValueError("times and values must be non-empty 1-D sequences of equal length")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(v))):
        raise ValueError("times and values must be finite")
    if t[0] <= 0.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError("times must be strictly positive and strictly increasing")
    return t, v


def _linear_from_origin(t: np.ndarray, w: np.ndarray, T: float) -> float:
    """Piecewise-linear function through (0, 0) and the pillars, last slope extended."""
    if T < 0.0 or not math.isfinite(T):
        raise ValueError(f"T must be finite and >= 0, got {T!r}")
    xs = np.concatenate(([0.0], t))
    ys = np.concatenate(([0.0], w))
    if T <= xs[-1]:
        return float(np.interp(T, xs, ys))
    slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return float(ys[-1] + slope * (T - xs[-1]))


@dataclass(frozen=True)
class ZeroCurve:
    """Continuously compounded zero curve with piecewise-constant forwards."""

    times: Sequence[float]
    zero_rates: Sequence[float]

    def __post_init__(self) -> None:
        _pillars(self.times, self.zero_rates)

    def integral(self, T: float) -> float:
        """``int_0^T f(u) du = r(T) T``."""
        t, z = _pillars(self.times, self.zero_rates)
        return _linear_from_origin(t, z * t, T)

    def zero_rate(self, T: float) -> float:
        """Average rate over ``[0, T]`` (the first pillar's rate at ``T = 0``)."""
        return float(self.zero_rates[0]) if T == 0.0 else self.integral(T) / T

    def discount(self, T: float) -> float:
        return math.exp(-self.integral(T))

    def forward_rate(self, t1: float, t2: float) -> float:
        if not t2 > t1:
            raise ValueError("need t2 > t1")
        return (self.integral(t2) - self.integral(t1)) / (t2 - t1)


def flat_curve(rate: float) -> ZeroCurve:
    """A flat curve at ``rate``."""
    return ZeroCurve((1.0,), (rate,))


@dataclass(frozen=True)
class VolTermStructure:
    """Implied-vol term structure, linear in total variance."""

    times: Sequence[float]
    vols: Sequence[float]

    def __post_init__(self) -> None:
        t, v = _pillars(self.times, self.vols)
        if np.any(v < 0.0):
            raise ValueError("vols must be >= 0")
        if np.any(np.diff(np.concatenate(([0.0], v * v * t))) < 0.0):
            raise ValueError("total variance decreases between pillars: calendar arbitrage")

    def total_variance(self, T: float) -> float:
        t, v = _pillars(self.times, self.vols)
        return _linear_from_origin(t, v * v * t, T)

    def vol(self, T: float) -> float:
        """Term (implied) vol to ``T``: ``sqrt(w(T) / T)``."""
        return float(self.vols[0]) if T == 0.0 else math.sqrt(self.total_variance(T) / T)

    def forward_vol(self, t1: float, t2: float) -> float:
        if not t2 > t1:
            raise ValueError("need t2 > t1")
        return math.sqrt((self.total_variance(t2) - self.total_variance(t1)) / (t2 - t1))


def bs_price_ts(
    S: float, K: float, T: float, rates: ZeroCurve, vols: VolTermStructure,
    dividends: ZeroCurve | None = None, option_type: OptionType = "call",
) -> float:
    """European price under deterministic term structures (Merton 1973)."""
    q_curve = dividends if dividends is not None else flat_curve(0.0)
    return bs_price(S, K, T, rates.zero_rate(T), vols.vol(T), q_curve.zero_rate(T), option_type)
