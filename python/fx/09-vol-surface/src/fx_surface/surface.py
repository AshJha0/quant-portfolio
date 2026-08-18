"""FX volatility surface: delta-space pillars, total-variance interpolation.

Delta-space vs moneyness-space interpolation (vs equity)
--------------------------------------------------------
Equity surfaces interpolate total variance at fixed (log-)moneyness: the
smile is anchored to strikes.  The OTC FX market is *sticky-delta*: the
quoted objects (ATM, 25d, 10d) float with spot and vol, so the natural
interpolation coordinate between expiries is *delta*, not moneyness.  A
25-delta call at 1M and a 25-delta call at 3M sit at very different
moneyness, and interpolating at fixed moneyness across FX expiries skews
short-dated wings badly (the same k is many more deltas OTM at 1W than
at 1Y).  This surface therefore interpolates *total variance at fixed
delta* between pillar expiries, and answers ``vol(K, T)`` by a
fixed-point iteration (strike -> delta -> vol -> delta ...), which
converges in a handful of steps.

Calendar arbitrage is checked at fixed delta: total variance must be
non-decreasing in T along each delta coordinate.  (At fixed *strike*
this is the classical condition; at fixed delta it is the market-
consistent analogue for a sticky-delta surface and is what desks
monitor on ATM/RR/BF pillars.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import brentq

from .garman_kohlhagen import gk_delta, gk_forward
from .smile import SVISmile, VannaVolgaSmile
from .smile_from_quotes import (
    PILLAR_LABELS,
    SmileQuotes,
    atm_dns_strike,
    solve_pillar_strikes,
    vols_from_quotes,
)

__all__ = ["SmileSlice", "FXVolSurface", "build_slice", "build_surface"]


@dataclass
class SmileSlice:
    """One fitted expiry of the surface.

    Attributes
    ----------
    label : str
        Tenor label, e.g. ``"3m"``.
    T : float
        Year fraction.
    S, r_d, r_f : float
        Spot and the zero rates to this expiry.
    convention : str
        Native delta convention of the *quotes* (used for strike
        placement): ``spot | forward | spot_pa | forward_pa``.
    vols, strikes : dict
        Pillar vols and strikes keyed by ``('10p','25p','atm','25c','10c')``.
    smile : object
        Fitted smile with a ``vol(K)`` method (SVI or vanna-volga).
    """

    label: str
    T: float
    S: float
    r_d: float
    r_f: float
    convention: str
    vols: dict[str, float]
    strikes: dict[str, float]
    smile: object

    @property
    def F(self) -> float:
        return gk_forward(self.S, self.T, self.r_d, self.r_f)


def build_slice(
    label: str,
    T: float,
    S: float,
    r_d: float,
    r_f: float,
    quotes: SmileQuotes,
    convention: str = "spot",
    smile_model: str = "svi",
) -> SmileSlice:
    """Quotes -> five vols -> pillar strikes -> fitted smile, one expiry.

    ``smile_model`` is ``"svi"`` (fit all five pillars) or ``"vv"``
    (vanna-volga from the 25P/ATM/25C pillars only - the market
    construction; the 10d pillars are then out-of-sample checks).
    """
    vols = vols_from_quotes(quotes)
    strikes = solve_pillar_strikes(vols, S, T, r_d, r_f, convention)
    F = gk_forward(S, T, r_d, r_f)
    if smile_model == "svi":
        ks = np.array([strikes[p] for p in PILLAR_LABELS])
        vs = np.array([vols[p] for p in PILLAR_LABELS])
        smile = SVISmile.fit(ks, vs, F, T)
    elif smile_model == "vv":
        smile = VannaVolgaSmile(
            S, T, r_d, r_f,
            np.array([strikes["25p"], strikes["atm"], strikes["25c"]]),
            np.array([vols["25p"], vols["atm"], vols["25c"]]),
        )
    else:
        raise ValueError(f"unknown smile_model {smile_model!r}; use 'svi' or 'vv'")
    return SmileSlice(label, T, S, r_d, r_f, convention, vols, strikes, smile)


class FXVolSurface:
    """Term structure of fitted smiles with delta-space interpolation.

    Parameters
    ----------
    slices : sequence of SmileSlice
        At least one expiry; sorted by T internally.
    delta_convention : str
        The *common* delta coordinate used for inter-expiry
        interpolation and for ``vol(K, T)`` queries.  Broker pillar
        strikes keep their native per-expiry conventions; a single
        coordinate is required for a well-defined interpolation axis
        (default ``"forward"``, the usual choice for mixed-tenor books).
    """

    def __init__(
        self, slices: Sequence[SmileSlice], delta_convention: str = "forward"
    ) -> None:
        if len(slices) == 0:
            raise ValueError("surface needs at least one slice")
        self.slices = sorted(slices, key=lambda s: s.T)
        Ts = [s.T for s in self.slices]
        if len(set(Ts)) != len(Ts):
            raise ValueError("duplicate expiries in surface")
        self.delta_convention = delta_convention
        self._Ts = np.array(Ts)
        self._rds = np.array([s.r_d for s in self.slices])
        self._rfs = np.array([s.r_f for s in self.slices])
        self.S = self.slices[0].S

    # -- rates / forward for arbitrary T --------------------------------
    def rates(self, T: float) -> tuple[float, float]:
        """(r_d, r_f) zero rates at T: linear interp of pillar zeros, flat ends."""
        return (
            float(np.interp(T, self._Ts, self._rds)),
            float(np.interp(T, self._Ts, self._rfs)),
        )

    def forward(self, T: float) -> float:
        r_d, r_f = self.rates(T)
        return gk_forward(self.S, T, r_d, r_f)

    # -- per-slice delta coordinate lookups -----------------------------
    def _slice_vol_at_delta(
        self, sl: SmileSlice, delta: float | None, cp: int = 1
    ) -> float:
        """Vol on one slice at an (unsigned) delta in the surface's common
        convention; ``delta=None`` means the DNS ATM coordinate."""
        pa = self.delta_convention.endswith("_pa")
        if delta is None:
            # DNS strike is a fixed point K = F exp(+-sigma(K)^2 T / 2).
            K = sl.F
            for _ in range(100):
                K_new = atm_dns_strike(sl.F, float(sl.smile.vol(K)), sl.T, pa)
                if abs(K_new - K) < 1e-14 * sl.F:
                    K = K_new
                    break
                K = K_new
            return float(sl.smile.vol(K))
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta magnitude must be in (0,1), got {delta}")

        target = cp * delta

        def g(K: float) -> float:
            sig = float(sl.smile.vol(K))
            return gk_delta(sl.S, K, sl.T, sl.r_d, sl.r_f, sig, cp, self.delta_convention) - target

        v = sl.vols["atm"] * math.sqrt(sl.T)
        lo = sl.F * math.exp(-12.0 * v - 1e-6)
        hi = sl.F * math.exp(12.0 * v + 1e-6)
        K = brentq(g, lo, hi, xtol=1e-13 * sl.F)
        return float(sl.smile.vol(K))

    # -- public queries -------------------------------------------------
    def vol_delta(self, delta: float | None, T: float, cp: int = 1) -> float:
        """Vol at fixed (unsigned) delta and arbitrary expiry T.

        Total variance ``sigma^2 T`` is interpolated linearly in T at
        fixed delta between the bracketing pillar expiries; outside the
        pillar range the *vol* is extrapolated flat (documented choice -
        keeps short/long extrapolation arbitrage-safe at fixed delta).
        ``delta=None`` queries the ATM (DNS) coordinate.
        """
        if T <= 0.0:
            raise ValueError(f"T must be positive, got {T}")
        sls = self.slices
        if T <= sls[0].T:
            return self._slice_vol_at_delta(sls[0], delta, cp)
        if T >= sls[-1].T:
            return self._slice_vol_at_delta(sls[-1], delta, cp)
        i = int(np.searchsorted(self._Ts, T, side="right")) - 1
        s0, s1 = sls[i], sls[i + 1]
        v0 = self._slice_vol_at_delta(s0, delta, cp)
        v1 = self._slice_vol_at_delta(s1, delta, cp)
        w0, w1 = v0 * v0 * s0.T, v1 * v1 * s1.T
        w = w0 + (w1 - w0) * (T - s0.T) / (s1.T - s0.T)
        return math.sqrt(w / T)

    def vol_atm(self, T: float) -> float:
        """DNS ATM vol at arbitrary T."""
        return self.vol_delta(None, T)

    def vol(self, K: float, T: float, tol: float = 1e-12, max_iter: int = 100) -> float:
        """Vol at (strike, expiry) via the sticky-delta fixed point.

        Iterates sigma -> delta(K; sigma) -> vol_delta(delta, T) until
        the vol stabilises.  Consistency: ``vol(K, T) ==
        vol_delta(delta(K, vol(K,T)), T)`` by construction at the fixed
        point (tested).
        """
        if K <= 0.0:
            raise ValueError(f"strike must be positive, got {K}")
        r_d, r_f = self.rates(T)
        F = self.forward(T)
        cp = 1 if K >= F else -1
        sigma = self.vol_atm(T)
        for _ in range(max_iter):
            d = gk_delta(self.S, K, T, r_d, r_f, sigma, cp, self.delta_convention)
            sigma_new = self.vol_delta(abs(d), T, cp)
            if abs(sigma_new - sigma) < tol:
                return sigma_new
            sigma = 0.5 * (sigma + sigma_new)  # damped for wing stability
        return sigma

    # -- diagnostics ----------------------------------------------------
    def calendar_arbitrage_report(
        self, deltas: Sequence[float] = (0.10, 0.25), tol: float = 1e-10
    ) -> list[dict]:
        """Fixed-delta calendar check across consecutive pillar expiries.

        Coordinates checked: each put/call delta in ``deltas`` plus ATM.
        A violation is total variance *decreasing* in T by more than
        ``tol``.  Returns a list of violation records (empty = clean).
        """
        coords: list[tuple[str, float | None, int]] = [("atm", None, 1)]
        for d in deltas:
            coords.append((f"{int(round(d * 100))}p", d, -1))
            coords.append((f"{int(round(d * 100))}c", d, +1))
        violations = []
        for name, d, cp in coords:
            w_prev = None
            for sl in self.slices:
                v = self._slice_vol_at_delta(sl, d, cp)
                w = v * v * sl.T
                if w_prev is not None and w < w_prev[1] - tol:
                    violations.append(
                        {
                            "coordinate": name,
                            "T_from": w_prev[0],
                            "T_to": sl.T,
                            "w_from": w_prev[1],
                            "w_to": w,
                        }
                    )
                w_prev = (sl.T, w)
        return violations

    def is_calendar_arbitrage_free(self, **kwargs) -> bool:
        return len(self.calendar_arbitrage_report(**kwargs)) == 0


def build_surface(market, smile_model: str = "svi",
                  delta_convention: str = "forward") -> FXVolSurface:
    """Build a full surface from a market-quote object.

    ``market`` must expose ``S`` and an iterable ``slices`` of records
    with attributes ``label, T, r_d, r_f, quotes, convention`` (see
    :class:`fx_surface.data.synthetic.FXMarketData`).
    """
    slices = [
        build_slice(ms.label, ms.T, market.S, ms.r_d, ms.r_f, ms.quotes,
                    ms.convention, smile_model)
        for ms in market.slices
    ]
    return FXVolSurface(slices, delta_convention=delta_convention)
