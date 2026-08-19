"""Discount curve with log-linear interpolation on discount factors.

Conventions
-----------
- Times are year fractions from the valuation date (t = 0), computed with the
  day-count module.
- ``DiscountCurve`` interpolates **linearly in log discount factor** between
  pillars, which is equivalent to piecewise-constant instantaneous forward
  rates.  Beyond the last pillar the last forward rate is extrapolated flat;
  before the first pillar the forward from 0 to the first pillar is used.
- Zero rates are **continuously compounded, annualised**:
  ``z(t) = -ln(DF(t)) / t``.
- Discount factors must be strictly positive but may exceed 1.0 — negative
  rates (EUR 2015-2022, JPY, CHF) are first-class citizens.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DiscountCurve"]

_ArrayLike = "float | np.ndarray"


class DiscountCurve:
    """Discount curve defined by pillar times and discount factors.

    Parameters
    ----------
    times : array_like
        Strictly increasing pillar times (years), all > 0.
    dfs : array_like
        Discount factors at the pillar times, all > 0 (DF > 1 allowed for
        negative rates).
    name : str
        Label used in reports (e.g. ``"USD"``, ``"EUR+basis"``).

    Raises
    ------
    ValueError
        On empty, non-increasing or non-positive inputs.
    """

    def __init__(self, times, dfs, name: str = "") -> None:
        t = np.atleast_1d(np.asarray(times, dtype=float))
        d = np.atleast_1d(np.asarray(dfs, dtype=float))
        if t.ndim != 1 or d.ndim != 1 or t.size != d.size or t.size == 0:
            raise ValueError("times and dfs must be non-empty 1-D arrays of equal length")
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(d)):
            raise ValueError("times and dfs must be finite")
        if t[0] <= 0.0:
            raise ValueError(f"first pillar time must be > 0, got {t[0]}")
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("pillar times must be strictly increasing")
        if np.any(d <= 0.0):
            raise ValueError("discount factors must be strictly positive")
        self.name = name
        # prepend the trivial node DF(0) = 1
        self._t = np.concatenate(([0.0], t))
        self._logdf = np.concatenate(([0.0], np.log(d)))

    # ------------------------------------------------------------------ #
    # constructors / accessors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_zero_rates(cls, times, zeros, name: str = "") -> "DiscountCurve":
        """Build a curve from continuously compounded annualised zero rates."""
        t = np.atleast_1d(np.asarray(times, dtype=float))
        z = np.atleast_1d(np.asarray(zeros, dtype=float))
        if t.size != z.size:
            raise ValueError("times and zeros must have equal length")
        return cls(t, np.exp(-z * t), name=name)

    @property
    def times(self) -> np.ndarray:
        """Pillar times (years), excluding the implicit t = 0 node."""
        return self._t[1:].copy()

    @property
    def dfs(self) -> np.ndarray:
        """Discount factors at the pillars."""
        return np.exp(self._logdf[1:])

    @property
    def zero_rates(self) -> np.ndarray:
        """Continuously compounded zero rates at the pillars."""
        return -self._logdf[1:] / self._t[1:]

    @property
    def n_pillars(self) -> int:
        return self._t.size - 1

    # ------------------------------------------------------------------ #
    # interpolation
    # ------------------------------------------------------------------ #
    def _interp_logdf(self, t: np.ndarray) -> np.ndarray:
        x, y = self._t, self._logdf
        out = np.interp(t, x, y)
        # np.interp clamps beyond the last pillar; extrapolate the last
        # segment's slope instead (flat instantaneous forward).
        if x.size >= 2:
            beyond = t > x[-1]
            if np.any(beyond):
                slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
                out = np.where(beyond, y[-1] + slope * (t - x[-1]), out)
        return out

    def df(self, t):
        """Discount factor(s) at time(s) ``t`` (years, >= 0).

        Scalar in, scalar (float) out; array in, array out.
        """
        arr = np.asarray(t, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"discount factor requested at non-finite time {t!r}")
        if np.any(arr < 0.0):
            raise ValueError("discount factor requested for negative time")
        out = np.exp(self._interp_logdf(np.atleast_1d(arr)))
        return float(out[0]) if arr.ndim == 0 else out

    def zero_rate(self, t):
        """Continuously compounded annualised zero rate(s) at ``t`` (> 0)."""
        arr = np.asarray(t, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"zero rate requested at non-finite time {t!r}")
        if np.any(arr <= 0.0):
            raise ValueError("zero rate requires t > 0")
        out = -self._interp_logdf(np.atleast_1d(arr)) / np.atleast_1d(arr)
        return float(out[0]) if arr.ndim == 0 else out

    def forward_rate(self, t1: float, t2: float) -> float:
        """Continuously compounded forward rate between ``t1`` and ``t2``.

        ``f(t1, t2) = (ln DF(t1) - ln DF(t2)) / (t2 - t1)``, 0 <= t1 < t2.
        """
        if not (0.0 <= t1 < t2):
            raise ValueError(f"require 0 <= t1 < t2, got t1={t1}, t2={t2}")
        l1 = float(self._interp_logdf(np.atleast_1d(float(t1)))[0])
        l2 = float(self._interp_logdf(np.atleast_1d(float(t2)))[0])
        return (l1 - l2) / (t2 - t1)

    # ------------------------------------------------------------------ #
    # bumping (risk)
    # ------------------------------------------------------------------ #
    def parallel_shift(self, bp: float) -> "DiscountCurve":
        """New curve with all pillar zero rates shifted by ``bp`` basis points."""
        if not np.isfinite(bp):
            raise ValueError(f"parallel shift bp must be finite, got {bp!r}")
        z = self.zero_rates + bp * 1e-4
        return DiscountCurve.from_zero_rates(self.times, z, name=self.name)

    def pillar_shift(self, index: int, bp: float) -> "DiscountCurve":
        """New curve with the zero rate of pillar ``index`` shifted by ``bp`` bp.

        With log-linear DF interpolation this key-rate bump only affects
        discount factors strictly between the neighbouring pillars
        (locality — tested in ``tests/test_risk.py``).
        """
        if not (0 <= index < self.n_pillars):
            raise ValueError(f"pillar index {index} out of range [0, {self.n_pillars})")
        if not np.isfinite(bp):
            raise ValueError(f"bump bp must be finite, got {bp!r}")
        z = self.zero_rates
        z[index] += bp * 1e-4
        return DiscountCurve.from_zero_rates(self.times, z, name=self.name)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"DiscountCurve(name={self.name!r}, pillars={self.n_pillars}, "
            f"t=[{self._t[1]:.2f}..{self._t[-1]:.2f}])"
        )
