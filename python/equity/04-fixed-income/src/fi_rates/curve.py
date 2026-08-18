"""Discount curve with pluggable interpolation and derived rate views.

Conventions
-----------
* Time is measured in **years** (ACT/365F from the valuation date when built
  from dates); ``t = 0`` is the valuation date with ``P(0) = 1``.
* Zero rates are **continuously compounded, annualised**:
  ``P(t) = exp(-z(t) * t)``.
* Forward rates between ``t1 < t2`` are continuously compounded:
  ``f(t1, t2) = (ln P(t1) - ln P(t2)) / (t2 - t1)``.
* Par rates are the fixed rate of a swap/bond with unit notional paying
  ``frequency`` times per year with equal accrual factors ``1/frequency``:
  ``par(T) = (1 - P(T)) / sum_i alpha_i P(t_i)``.

Interpolation schemes
---------------------
``loglinear_df`` (default)
    Linear in ``ln P(t)`` — equivalent to piecewise-constant instantaneous
    forwards.  The desk workhorse: local, always-positive discount factors,
    exact pillar repricing.
``linear_zero``
    Linear in zero rate ``z(t)``.  Simple but produces the well-known
    sawtooth in the forward curve (demonstrated in ``docs/VALIDATION.md``).
``pchip_zero``
    Monotone cubic (PCHIP) on zero rates — a "monotone-convex-lite" scheme:
    C1 forwards without the overshoot of a natural cubic spline.

Extrapolation policy
--------------------
* **Long end** (``t`` beyond the last pillar): flat zero-rate extrapolation,
  and every such query emits an :class:`ExtrapolationWarning` — extrapolation
  risk is real and documented in ``docs/VALIDATION.md``.
* **Short end** (``0 < t`` before the first pillar): silent, because
  discounting short stub cashflows is routine.  ``loglinear_df`` interpolates
  between the implicit anchor ``P(0) = 1`` and the first pillar (equivalent
  to a constant zero rate = first pillar zero); ``linear_zero`` and
  ``pchip_zero`` hold the first pillar zero flat.  All three therefore agree
  on the short end.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator

__all__ = ["DiscountCurve", "ExtrapolationWarning", "INTERPOLATIONS"]

INTERPOLATIONS: tuple[str, ...] = ("loglinear_df", "linear_zero", "pchip_zero")

_TMIN = 1e-10  # below this, treat t as "now" (df = 1, zero rate = short-end zero)


class ExtrapolationWarning(UserWarning):
    """Raised (as a warning) when the curve is queried outside its pillars."""


class DiscountCurve:
    """Discount curve over pillar times.

    Parameters
    ----------
    times : sequence of float
        Strictly increasing pillar times in years, all > 0.
    discount_factors : sequence of float
        Discount factors at the pillars, all > 0.  (May exceed 1 for
        negative-rate curves.)
    interpolation : str
        One of :data:`INTERPOLATIONS`.
    """

    def __init__(
        self,
        times: Sequence[float],
        discount_factors: Sequence[float],
        interpolation: str = "loglinear_df",
    ) -> None:
        t = np.asarray(times, dtype=float)
        p = np.asarray(discount_factors, dtype=float)
        if t.ndim != 1 or t.size == 0:
            raise ValueError("times must be a non-empty 1-D sequence")
        if t.shape != p.shape:
            raise ValueError("times and discount_factors must have equal length")
        if np.any(t <= 0):
            raise ValueError("all pillar times must be > 0")
        if np.any(np.diff(t) <= 0):
            raise ValueError("pillar times must be strictly increasing")
        if np.any(p <= 0):
            raise ValueError("discount factors must be > 0")
        if interpolation not in INTERPOLATIONS:
            raise ValueError(
                f"unknown interpolation {interpolation!r}; choose from {INTERPOLATIONS}"
            )
        self.times = t
        self.dfs = p
        self.interpolation = interpolation
        self._zeros = -np.log(p) / t
        self._pchip: PchipInterpolator | None = None
        if interpolation == "pchip_zero" and t.size >= 2:
            self._pchip = PchipInterpolator(t, self._zeros, extrapolate=False)

    # ------------------------------------------------------------------ ctors
    @classmethod
    def from_zero_rates(
        cls,
        times: Sequence[float],
        zero_rates: Sequence[float],
        interpolation: str = "loglinear_df",
    ) -> "DiscountCurve":
        """Build from continuously compounded zero rates ``z(t_i)``."""
        t = np.asarray(times, dtype=float)
        z = np.asarray(zero_rates, dtype=float)
        return cls(t, np.exp(-z * t), interpolation)

    # ------------------------------------------------------------- core query
    def _warn_extrapolation(self, t: np.ndarray) -> None:
        if np.any(t > self.times[-1] + 1e-12):
            warnings.warn(
                f"curve queried beyond last pillar t={self.times[-1]:.4f}y; "
                "flat zero-rate extrapolation applied",
                ExtrapolationWarning,
                stacklevel=3,
            )

    def zero_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        """Continuously compounded annualised zero rate ``z(t)``.

        For ``t <= 0`` (within tolerance) returns the first pillar's zero rate
        (flat short-end extrapolation).
        """
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        self._warn_extrapolation(t_arr)
        tc = np.clip(t_arr, self.times[0], self.times[-1])
        if self.times.size == 1:
            z = np.full_like(tc, self._zeros[0])
        elif self.interpolation == "linear_zero":
            z = np.interp(tc, self.times, self._zeros)
        elif self.interpolation == "pchip_zero":
            z = np.asarray(self._pchip(tc), dtype=float)  # type: ignore[misc]
        else:  # loglinear_df: linear in ln P = -z*t, with node (0, ln P = 0)
            knots_t = np.concatenate(([0.0], self.times))
            knots_lnp = np.concatenate(([0.0], np.log(self.dfs)))
            lnp = np.interp(tc, knots_t, knots_lnp)
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(tc > _TMIN, -lnp / np.maximum(tc, _TMIN), self._zeros[0])
        # Queries at (or numerically at) t = 0 return the first pillar's zero
        # rate — the flat short-end extrapolation limit.
        out = np.where(t_arr <= _TMIN, self._zeros[0], z)
        return float(out[0]) if np.isscalar(t) or np.ndim(t) == 0 else out

    def df(self, t: float | np.ndarray) -> float | np.ndarray:
        """Discount factor ``P(t)``; ``P(t) = 1`` for ``t <= 0``."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        self._warn_extrapolation(t_arr)
        if self.interpolation == "loglinear_df" and self.times.size >= 1:
            knots_t = np.concatenate(([0.0], self.times))
            knots_lnp = np.concatenate(([0.0], np.log(self.dfs)))
            tc = np.clip(t_arr, 0.0, self.times[-1])
            lnp = np.interp(tc, knots_t, knots_lnp)
            # flat-zero extrapolation beyond the last pillar
            beyond = t_arr > self.times[-1]
            if np.any(beyond):
                z_last = self._zeros[-1]
                lnp = np.where(beyond, -z_last * t_arr, lnp)
            p = np.exp(lnp)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ExtrapolationWarning)
                z = np.atleast_1d(np.asarray(self.zero_rate(t_arr), dtype=float))
            p = np.exp(-z * np.maximum(t_arr, 0.0))
        p = np.where(t_arr <= _TMIN, 1.0, p)
        return float(p[0]) if np.isscalar(t) or np.ndim(t) == 0 else p

    def forward_rate(self, t1: float, t2: float) -> float:
        """Continuously compounded forward rate between ``t1`` and ``t2``."""
        if t2 <= t1:
            raise ValueError(f"require t2 > t1, got t1={t1}, t2={t2}")
        p1 = float(np.asarray(self.df(t1)))
        p2 = float(np.asarray(self.df(t2)))
        return (np.log(p1) - np.log(p2)) / (t2 - t1)

    def simple_forward_rate(self, t1: float, t2: float) -> float:
        """Simply compounded forward: ``(P(t1)/P(t2) - 1) / (t2 - t1)``."""
        if t2 <= t1:
            raise ValueError(f"require t2 > t1, got t1={t1}, t2={t2}")
        p1 = float(np.asarray(self.df(t1)))
        p2 = float(np.asarray(self.df(t2)))
        return (p1 / p2 - 1.0) / (t2 - t1)

    def par_rate(self, maturity: float, frequency: int = 1) -> float:
        """Par swap/bond rate for maturity in years, fixed leg paying
        ``frequency`` times per year with accruals ``1/frequency``."""
        if maturity <= 0:
            raise ValueError(f"maturity must be > 0, got {maturity}")
        if frequency not in (1, 2, 4):
            raise ValueError(f"frequency must be 1, 2 or 4, got {frequency}")
        alpha = 1.0 / frequency
        n = int(round(maturity * frequency))
        if n < 1 or abs(n * alpha - maturity) > 1e-9:
            raise ValueError(
                f"maturity {maturity} is not a whole number of periods at "
                f"frequency {frequency}"
            )
        pay_times = alpha * np.arange(1, n + 1)
        dfs = np.asarray(self.df(pay_times))
        annuity = float(alpha * dfs.sum())
        return (1.0 - float(dfs[-1])) / annuity

    # ----------------------------------------------------------------- bumps
    def bumped_parallel(self, shift: float) -> "DiscountCurve":
        """New curve with all pillar zero rates shifted by ``shift`` (absolute,
        e.g. ``1e-4`` = 1bp)."""
        return DiscountCurve.from_zero_rates(
            self.times, self._zeros + shift, self.interpolation
        )

    def bumped_pillars(self, shifts: Sequence[float]) -> "DiscountCurve":
        """New curve with per-pillar zero-rate shifts (absolute)."""
        s = np.asarray(shifts, dtype=float)
        if s.shape != self.times.shape:
            raise ValueError(
                f"expected {self.times.size} shifts, got {s.size}"
            )
        return DiscountCurve.from_zero_rates(
            self.times, self._zeros + s, self.interpolation
        )

    @property
    def zero_rates(self) -> np.ndarray:
        """Pillar zero rates (continuously compounded, annualised)."""
        return self._zeros.copy()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"DiscountCurve(n={self.times.size}, "
            f"span=[{self.times[0]:.3f}, {self.times[-1]:.3f}]y, "
            f"interp={self.interpolation!r})"
        )
