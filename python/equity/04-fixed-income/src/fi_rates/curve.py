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
    Linear in ``ln P(t)``: equivalent to piecewise-constant instantaneous
    forwards.  The desk workhorse: local, always-positive discount factors,
    exact pillar repricing.
``linear_zero``
    Linear in zero rate ``z(t)``.  Simple but produces the well-known
    sawtooth in the forward curve (demonstrated in ``docs/VALIDATION.md``).
``pchip_zero``
    Monotone cubic (PCHIP) on zero rates. The zero curve is C1 and has no
    spline overshoot. The instantaneous forward ``f = z + t z'`` is only C0:
    it is continuous but kinks at the pillars. PCHIP is non-local, so a
    pillar bump moves the two neighbouring segments as well. It is not
    Hagan-West monotone-convex and does not guarantee positive forwards.
``monotone_convex``
    Hagan and West (2006), "Interpolation methods for curve construction",
    Applied Mathematical Finance 13(2), 89-129, Section 5 (the "monotone
    convex" method, without the amelioration step). The instantaneous
    forward is interpolated, not the zero rate. On each segment it is the
    discrete forward ``f^d_i = (r_i t_i - r_{i-1} t_{i-1}) / (t_i - t_{i-1})``
    plus a correction ``g(x)`` that integrates to zero, so every pillar
    reprices exactly. The node forwards are the time-weighted averages of
    the neighbouring discrete forwards (their eq. 21-23), and ``g`` is one of
    the four piecewise quadratics of their Section 5.2 chosen so the forward
    stays between its node values (no spurious oscillation). The forward is
    continuous, it is monotone wherever the discrete forwards are, and it is
    local: a pillar bump moves at most the two neighbouring segments on each
    side. The first segment ``(0, t_1]`` is part of the scheme, so the short
    end differs slightly from the other three schemes (the pillar still
    reprices).

    Positivity collar (``positive_forwards=True``, monotone_convex only).
    Plain idea: if every discrete forward is positive you usually want the
    interpolated forward positive too, and the plain scheme does not promise
    that (a steep drop between two segments can push a node forward below
    zero). Formal statement: Hagan and West (2006), Section 5 ("ensuring
    positivity"), clamp the node forwards to
    ``f_0 in [0, 2 f^d_1]``, ``f_i in [0, 2 min(f^d_i, f^d_{i+1})]`` for
    ``0 < i < n`` and ``f_n in [0, 2 f^d_n]``. With node values in that band
    every one of the four ``g`` shapes stays at or above zero, so the
    instantaneous forward is non-negative wherever the discrete forwards are
    positive; pillars still reprice exactly because ``g`` still integrates to
    zero. When some discrete forward is negative the clamp is meaningless and
    the constructor raises. Worked example: ``tests/test_round4_collar.py``.

Extrapolation policy
--------------------
* **Long end** (``t`` beyond the last pillar): flat zero-rate extrapolation,
  and every such query emits an :class:`ExtrapolationWarning`: extrapolation
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

__all__ = ["INTERPOLATIONS", "DiscountCurve", "ExtrapolationWarning"]

INTERPOLATIONS: tuple[str, ...] = ("loglinear_df", "linear_zero", "pchip_zero", "monotone_convex")

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
        times: Sequence[float] | np.ndarray,
        discount_factors: Sequence[float] | np.ndarray,
        interpolation: str = "loglinear_df",
        *,
        positive_forwards: bool = False,
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
        if positive_forwards and interpolation != "monotone_convex":
            raise ValueError("positive_forwards applies to monotone_convex only")
        self.interpolation = interpolation
        self.positive_forwards = bool(positive_forwards)
        self._zeros = -np.log(p) / t
        self._pchip: PchipInterpolator | None = None
        if interpolation == "pchip_zero" and t.size >= 2:
            self._pchip = PchipInterpolator(t, self._zeros, extrapolate=False)
        self._mc: _MonotoneConvex | None = None
        if interpolation == "monotone_convex":
            self._mc = _MonotoneConvex(t, -np.log(p), collar=self.positive_forwards)

    # ------------------------------------------------------------------ ctors
    @classmethod
    def from_zero_rates(
        cls,
        times: Sequence[float] | np.ndarray,
        zero_rates: Sequence[float] | np.ndarray,
        interpolation: str = "loglinear_df",
        *,
        positive_forwards: bool = False,
    ) -> DiscountCurve:
        """Build from continuously compounded zero rates ``z(t_i)``."""
        t = np.asarray(times, dtype=float)
        z = np.asarray(zero_rates, dtype=float)
        return cls(t, np.exp(-z * t), interpolation, positive_forwards=positive_forwards)

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
        elif self.interpolation == "monotone_convex":
            assert self._mc is not None
            tc = np.clip(t_arr, _TMIN, self.times[-1])
            z = self._mc.integral(tc) / tc
        else:  # loglinear_df: linear in ln P = -z*t, with node (0, ln P = 0)
            knots_t = np.concatenate(([0.0], self.times))
            knots_lnp = np.concatenate(([0.0], np.log(self.dfs)))
            lnp = np.interp(tc, knots_t, knots_lnp)
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(tc > _TMIN, -lnp / np.maximum(tc, _TMIN), self._zeros[0])
        # Queries at (or numerically at) t = 0 return the first pillar's zero
        # rate, the flat short-end extrapolation limit.
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

    def instantaneous_forward(self, t: float | np.ndarray) -> float | np.ndarray:
        """Instantaneous forward ``f(t) = -d ln P / dt``.

        Exact for ``monotone_convex`` (the interpolated object itself); for
        the other schemes a central difference of ``ln P`` with step 1e-6.
        Beyond the last pillar the flat-zero extrapolation gives ``z_n``.
        """
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        if self._mc is not None:
            inside = np.clip(t_arr, 0.0, self.times[-1])
            f = np.where(t_arr > self.times[-1], self._zeros[-1], self._mc.forward(inside))
        else:
            h = 1e-6
            lo = np.maximum(t_arr - h, 0.0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ExtrapolationWarning)
                f = -(np.log(np.asarray(self.df(t_arr + h), dtype=float))
                      - np.log(np.asarray(self.df(lo), dtype=float))) / (t_arr + h - lo)
        return float(f[0]) if np.isscalar(t) or np.ndim(t) == 0 else f

    # ----------------------------------------------------------------- bumps
    def bumped_parallel(self, shift: float) -> DiscountCurve:
        """New curve with all pillar zero rates shifted by ``shift`` (absolute,
        e.g. ``1e-4`` = 1bp)."""
        return DiscountCurve.from_zero_rates(
            self.times, self._zeros + shift, self.interpolation,
            positive_forwards=self.positive_forwards,
        )

    def bumped_pillars(self, shifts: Sequence[float] | np.ndarray) -> DiscountCurve:
        """New curve with per-pillar zero-rate shifts (absolute)."""
        s = np.asarray(shifts, dtype=float)
        if s.shape != self.times.shape:
            raise ValueError(
                f"expected {self.times.size} shifts, got {s.size}"
            )
        return DiscountCurve.from_zero_rates(
            self.times, self._zeros + s, self.interpolation,
            positive_forwards=self.positive_forwards,
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


class _MonotoneConvex:
    """Hagan-West (2006) monotone-convex forward on pillars ``t`` with
    ``y = -ln P(t) = r t``. ``integral(t)`` returns ``int_0^t f``.

    ``collar=True`` applies the Section 5 positivity bounds to the node
    forwards (see the module docstring)."""

    def __init__(self, t: np.ndarray, y: np.ndarray, collar: bool = False) -> None:
        tau = np.concatenate(([0.0], t))
        yy = np.concatenate(([0.0], y))
        dt = np.diff(tau)
        fd = np.diff(yy) / dt  # discrete forwards f^d_1..f^d_n
        n = fd.size
        f = np.empty(n + 1)  # node forwards f_0..f_n (Hagan-West eq. 21-23)
        if n == 1:
            f[:] = fd[0]
        else:
            w = dt[:-1] / (dt[:-1] + dt[1:])
            f[1:n] = w * fd[1:] + (1.0 - w) * fd[:-1]
            f[0] = fd[0] - 0.5 * (f[1] - fd[0])
            f[n] = fd[-1] - 0.5 * (f[n - 1] - fd[-1])
        if collar:
            if np.any(fd <= 0.0):
                raise ValueError("positive_forwards needs every discrete forward > 0")
            f[0] = np.clip(f[0], 0.0, 2.0 * fd[0])
            if n > 1:
                f[1:n] = np.clip(f[1:n], 0.0, 2.0 * np.minimum(fd[:-1], fd[1:]))
            f[n] = np.clip(f[n], 0.0, 2.0 * fd[-1])
        self.tau, self.yy, self.dt, self.fd, self.f = tau, yy, dt, fd, f

    @staticmethod
    def _g(g0: float, g1: float, x: float) -> tuple[float, float]:
        """(G(x), int_0^x G) for one segment; Hagan-West Section 5.2."""
        if g0 == 0.0 and g1 == 0.0:
            return 0.0, 0.0
        tiny = 1e-9 * max(abs(g0), abs(g1))
        if abs(g0) <= tiny or abs(g1) <= tiny:
            # Boundary between regions: one end sits at the discrete forward
            # (two equal discrete forwards side by side; "equal" up to
            # round-off, since ln P differences rarely cancel exactly). Every
            # region's limit here is a jump at the pillar (eta -> 0 or 1), so
            # we take the plain quadratic: continuous, integrates to zero,
            # and it dips (or peaks) by |g|/3 inside the segment.
            G = g0 * (1 - 4 * x + 3 * x * x) + g1 * (-2 * x + 3 * x * x)
            I = g0 * (x - 2 * x * x + x**3) + g1 * (-x * x + x**3)
            return G, I
        if (g0 < 0 and -0.5 * g0 <= g1 <= -2.0 * g0) or (g0 > 0 and -0.5 * g0 >= g1 >= -2.0 * g0):
            # region (i): plain quadratic
            G = g0 * (1 - 4 * x + 3 * x * x) + g1 * (-2 * x + 3 * x * x)
            I = g0 * (x - 2 * x * x + x**3) + g1 * (-x * x + x**3)
            return G, I
        if (g0 < 0 and g1 > -2.0 * g0) or (g0 > 0 and g1 < -2.0 * g0):
            # region (ii): flat then quadratic
            eta = (g1 + 2 * g0) / (g1 - g0)
            if x <= eta:
                return g0, g0 * x
            u = (x - eta) / (1 - eta)
            return g0 + (g1 - g0) * u * u, g0 * x + (g1 - g0) * (x - eta) * u * u / 3.0
        if (g0 > 0 and 0 > g1 > -0.5 * g0) or (g0 < 0 and 0 < g1 < -0.5 * g0):
            # region (iii): quadratic then flat
            eta = 3 * g1 / (g1 - g0)
            if x < eta:
                v = (eta - x) / eta
                return g1 + (g0 - g1) * v * v, g1 * x + (g0 - g1) * eta / 3.0 * (1 - v**3)
            return g1, g1 * x + (g0 - g1) * eta / 3.0
        # region (iv): g0, g1 of the same sign; two quadratics meeting at A
        eta = g1 / (g1 + g0)
        A = -g0 * g1 / (g0 + g1)
        if x < eta:
            v = (eta - x) / eta
            return A + (g0 - A) * v * v, A * x + (g0 - A) * eta / 3.0 * (1 - v**3)
        u = (x - eta) / (1 - eta)
        return (A + (g1 - A) * u * u,
                A * x + (g0 - A) * eta / 3.0 + (g1 - A) * (x - eta) * u * u / 3.0)

    def _eval(self, t: np.ndarray, want_integral: bool) -> np.ndarray:
        out = np.empty_like(t, dtype=float)
        for k, tk in enumerate(t.ravel()):
            i = int(np.searchsorted(self.tau, tk, side="left"))
            i = min(max(i, 1), self.fd.size)
            x = (tk - self.tau[i - 1]) / self.dt[i - 1]
            fd = self.fd[i - 1]
            G, I = self._g(self.f[i - 1] - fd, self.f[i] - fd, x)
            if want_integral:
                out.flat[k] = self.yy[i - 1] + self.dt[i - 1] * (fd * x + I)
            else:
                out.flat[k] = fd + G
        return out

    def integral(self, t: np.ndarray) -> np.ndarray:
        return self._eval(np.asarray(t, dtype=float), True)

    def forward(self, t: np.ndarray) -> np.ndarray:
        return self._eval(np.asarray(t, dtype=float), False)
