"""Total-variance volatility surface across expiries.

Construction
------------
Each expiry is an SVI slice in forward log-moneyness ``k = ln(K / F(T))``.
At build time every slice is sampled on a common dense ``k`` grid, giving a
matrix of *total variance* ``w(k, T_i) = sigma_imp(k, T_i)^2 * T_i``.

Why interpolate total variance and not vol
------------------------------------------
Absence of calendar arbitrage is equivalent (at fixed forward moneyness) to
total variance being non-decreasing in ``T``.  Linear interpolation of a
monotone quantity in ``T`` preserves monotonicity between pillars, so a
calendar-arbitrage-free set of pillars yields a calendar-arbitrage-free
interpolated surface.  Linear interpolation of *vol* in ``T`` offers no such
guarantee: two arbitrage-free pillar vols can interpolate to a total variance
that dips between them (short-dated high vol next to long-dated low vol),
creating negative forward variance.  Interpolating ``w`` also makes the
implied *forward variance* between pillars piecewise-constant and manifestly
non-negative whenever the pillars are calendar-free.

Extrapolation policy (documented and tested)
--------------------------------------------
* ``T < T_min``:  ``w(k, T) = w(k, T_min) * T / T_min`` -- total variance
  scales linearly to zero at ``T = 0``, i.e. implied vol is held flat at the
  first pillar's smile.  Equivalent to constant instantaneous variance before
  the first pillar.
* ``T > T_max``: linear continuation of the last inter-pillar slope of
  ``w`` in ``T``, floored at zero slope (so extrapolated total variance never
  decreases -- no synthetic calendar arbitrage).  With a single pillar the
  slope used is ``w(k, T_1)/T_1`` (flat vol).
* ``k`` outside the sampled grid: the SVI slices themselves extrapolate
  (linear wings in total variance), because slices are evaluated analytically
  in ``k``; there is no k-grid clipping.

Calendar enforcement
--------------------
``enforce_calendar=True`` replaces the pillar matrix with its running maximum
along ``T`` on the sampled ``k`` grid (monotone adjustment).  Queries then
interpolate the adjusted grid bilinearly (linear in ``k``, linear in ``w``
across ``T``).  Without enforcement, queries evaluate the SVI slices exactly
in ``k`` (grid only used for diagnostics), so pillar values are exact.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .smile import SVIParams, svi_total_variance

__all__ = ["VolSurface", "CalendarCheck", "check_calendar"]


@dataclass
class CalendarCheck:
    """Result of a calendar-spread arbitrage check on the pillar grid."""

    is_free: bool
    worst_violation: float  # most negative w(T_{i+1}) - w(T_i); 0 when free
    violations: list  # list of (T_lo, T_hi, k, w_lo, w_hi)


def check_calendar(
    expiries: np.ndarray,
    w_grid: np.ndarray,
    k_grid: np.ndarray,
    tol: float = 1e-12,
) -> CalendarCheck:
    """Check total variance is non-decreasing in T at every fixed k.

    Parameters
    ----------
    expiries : array
        Increasing pillar expiries, shape (nT,).
    w_grid : array
        Total variance, shape (nT, nK).
    k_grid : array
        Log-moneyness grid, shape (nK,).
    tol : float
        Decreases smaller than ``tol`` are ignored (numerical noise).
    """
    expiries = np.asarray(expiries, dtype=float)
    w_grid = np.asarray(w_grid, dtype=float)
    diffs = np.diff(w_grid, axis=0)
    violations = []
    worst = 0.0
    bad = np.argwhere(diffs < -tol)
    for i, j in bad:
        violations.append(
            (float(expiries[i]), float(expiries[i + 1]), float(k_grid[j]),
             float(w_grid[i, j]), float(w_grid[i + 1, j]))
        )
        worst = min(worst, float(diffs[i, j]))
    return CalendarCheck(is_free=len(violations) == 0, worst_violation=worst, violations=violations)


class VolSurface:
    """Implied-volatility surface built from per-expiry SVI slices.

    Parameters
    ----------
    expiries : array_like
        Pillar expiries in years, strictly increasing, all > 0.
    slices : sequence of SVIParams
        One raw-SVI slice per pillar, in forward log-moneyness.
    spot : float
        Spot price (>0), used for forward computation in ``vol(K, T)``.
    rate, div_yield : float
        Continuously compounded annualised risk-free rate and dividend yield
        (ACT/365F), defining the forward ``F(T) = S exp((r - q) T)``.
    k_grid : array_like, optional
        Log-moneyness grid used for calendar diagnostics/enforcement
        (default: 201 points on [-1.5, 1.5]).
    enforce_calendar : bool
        If True, apply the monotone (running-max in T) adjustment on the
        sampled grid and answer queries from the adjusted grid.

    Notes
    -----
    A single-expiry surface is valid; T-interpolation then degenerates to the
    single-pillar extrapolation policy documented in the module docstring.
    """

    def __init__(
        self,
        expiries: np.ndarray,
        slices: list[SVIParams],
        spot: float,
        rate: float,
        div_yield: float,
        k_grid: np.ndarray | None = None,
        enforce_calendar: bool = False,
    ) -> None:
        expiries = np.asarray(expiries, dtype=float)
        if expiries.ndim != 1 or expiries.size == 0:
            raise ValueError("expiries must be a non-empty 1-D array")
        if np.any(expiries <= 0.0):
            raise ValueError("all pillar expiries must be positive")
        if np.any(np.diff(expiries) <= 0.0):
            raise ValueError("pillar expiries must be strictly increasing")
        if len(slices) != expiries.size:
            raise ValueError(
                f"got {len(slices)} SVI slices for {expiries.size} expiries"
            )
        if spot <= 0.0:
            raise ValueError(f"spot must be positive, got {spot}")

        self.expiries = expiries
        self.slices = list(slices)
        self.spot = float(spot)
        self.rate = float(rate)
        self.div_yield = float(div_yield)
        self.k_grid = (
            np.linspace(-1.5, 1.5, 201) if k_grid is None else np.asarray(k_grid, dtype=float)
        )
        self.enforced = bool(enforce_calendar)

        # Sample slices on the k grid: total-variance pillar matrix (nT, nK).
        self.w_grid = np.vstack(
            [np.asarray(svi_total_variance(self.k_grid, p)) for p in self.slices]
        )
        self.calendar = check_calendar(self.expiries, self.w_grid, self.k_grid)
        if not self.calendar.is_free:
            msg = (
                f"surface has calendar arbitrage at {len(self.calendar.violations)} grid "
                f"points (worst total-variance decrease {self.calendar.worst_violation:.3e})"
            )
            if enforce_calendar:
                warnings.warn(msg + "; applying monotone (running-max) adjustment", UserWarning)
            else:
                warnings.warn(msg + "; pass enforce_calendar=True to adjust", UserWarning)
        if enforce_calendar:
            self.w_grid = np.maximum.accumulate(self.w_grid, axis=0)

    # ------------------------------------------------------------------ #

    def forward(self, T: float) -> float:
        """Forward price ``F(T) = S exp((r - q) T)``."""
        return self.spot * float(np.exp((self.rate - self.div_yield) * T))

    def _w_at_pillars(self, k: np.ndarray) -> np.ndarray:
        """Total variance at every pillar for the given k values, shape (nT, nk)."""
        k = np.atleast_1d(np.asarray(k, dtype=float))
        if self.enforced:
            # Interpolate the (adjusted) grid linearly in k; linear wings beyond.
            out = np.empty((self.expiries.size, k.size))
            for i in range(self.expiries.size):
                out[i] = np.interp(k, self.k_grid, self.w_grid[i])
                # extrapolate linearly using edge slopes
                lo = k < self.k_grid[0]
                hi = k > self.k_grid[-1]
                if lo.any():
                    slope = (self.w_grid[i, 1] - self.w_grid[i, 0]) / (
                        self.k_grid[1] - self.k_grid[0]
                    )
                    out[i, lo] = self.w_grid[i, 0] + slope * (k[lo] - self.k_grid[0])
                if hi.any():
                    slope = (self.w_grid[i, -1] - self.w_grid[i, -2]) / (
                        self.k_grid[-1] - self.k_grid[-2]
                    )
                    out[i, hi] = self.w_grid[i, -1] + slope * (k[hi] - self.k_grid[-1])
            return out
        return np.vstack([np.asarray(svi_total_variance(k, p)) for p in self.slices])

    def total_variance(self, k: np.ndarray | float, T: float) -> np.ndarray | float:
        """Total implied variance ``w(k, T)`` with linear-in-T interpolation.

        Parameters
        ----------
        k : array_like
            Forward log-moneyness ``ln(K / F(T))``.
        T : float
            Expiry in years (> 0).  Outside the pillar range the documented
            extrapolation policy applies (see module docstring).
        """
        if T <= 0.0:
            raise ValueError(f"T must be positive, got {T}")
        k_arr = np.atleast_1d(np.asarray(k, dtype=float))
        wp = self._w_at_pillars(k_arr)  # (nT, nk)
        Ts = self.expiries

        if T <= Ts[0]:
            w = wp[0] * (T / Ts[0])
        elif T >= Ts[-1]:
            if Ts.size == 1:
                slope = wp[-1] / Ts[-1]
            else:
                slope = (wp[-1] - wp[-2]) / (Ts[-1] - Ts[-2])
            slope = np.maximum(slope, 0.0)  # never extrapolate decreasing w
            w = wp[-1] + slope * (T - Ts[-1])
        else:
            i = int(np.searchsorted(Ts, T) - 1)
            t0, t1 = Ts[i], Ts[i + 1]
            lam = (T - t0) / (t1 - t0)
            w = (1.0 - lam) * wp[i] + lam * wp[i + 1]

        w = np.maximum(w, 1e-12)
        return w if np.ndim(k) else float(w[0])

    def vol_k(self, k: np.ndarray | float, T: float) -> np.ndarray | float:
        """Implied vol at forward log-moneyness ``k`` and expiry ``T``."""
        w = self.total_variance(k, T)
        return np.sqrt(np.asarray(w) / T) if np.ndim(k) else float(np.sqrt(w / T))

    def vol(self, K: np.ndarray | float, T: float) -> np.ndarray | float:
        """Implied vol at absolute strike ``K`` and expiry ``T``.

        Handles forward moneyness internally: ``k = ln(K / F(T))`` with
        ``F(T) = S exp((r - q) T)``.
        """
        K_arr = np.atleast_1d(np.asarray(K, dtype=float))
        if np.any(K_arr <= 0.0):
            raise ValueError("strikes must be positive")
        k = np.log(K_arr / self.forward(T))
        out = np.asarray(self.vol_k(k, T))
        return out if np.ndim(K) else float(out[0])
