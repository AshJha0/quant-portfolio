"""FX volatility surface: delta-space pillars, total-variance interpolation.

Delta-space interpolation (vs equity)
-------------------------------------
Equity surfaces interpolate total variance at fixed (log-)moneyness: the
smile is anchored to strikes.  OTC FX quotes ATM, 25d and 10d pillars in
delta space, so between pillar expiries this surface interpolates total
variance at fixed *delta*, and answers ``vol(K, T)`` by a damped
fixed-point iteration (strike -> delta -> vol -> delta ...).

Delta coordinate
----------------
By default (``delta_convention="native"``) every slice is read in the
delta convention its quotes use (spot, forward, spot_pa or forward_pa),
including the matching DNS ATM definition.  A surface built from quotes
therefore returns those quotes at the pillar expiries.  Between two
pillars, "the 25-delta call" means the 25-delta call of each pillar in
its own convention, which is how a broker term structure is read.  A
single common convention can still be forced, but then the pillar quotes
of slices quoted in another convention are not reproduced.

Premium-adjusted call deltas are not monotone in strike.  Strike solving
on a pa slice brackets on the falling branch above the smile-consistent
delta maximum (the market-standard OTM strike).

Calendar arbitrage
------------------
The static no-arbitrage condition is that total variance is
non-decreasing in T at fixed forward moneyness ``k = ln(K/F(T))``.  That
is what :meth:`FXVolSurface.calendar_arbitrage_report` checks.  The
fixed-delta report (:meth:`FXVolSurface.delta_calendar_report`) is a
monitoring view of the quoted pillars.  It neither implies nor is implied
by absence of calendar arbitrage.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .garman_kohlhagen import DELTA_CONVENTIONS, gk_delta, gk_forward
from .smile import SVISmile, VannaVolgaSmile
from .smile_from_quotes import (
    PILLAR_LABELS,
    SmileQuotes,
    atm_dns_strike,
    solve_pillar_strikes,
    strike_from_delta,
    vols_from_quotes,
)

__all__ = [
    "SmileSlice",
    "FXVolSurface",
    "SurfaceConvergenceError",
    "build_slice",
    "build_surface",
]


class SurfaceConvergenceError(RuntimeError):
    """The strike -> delta -> vol fixed point did not converge."""


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
    #: Largest |smile vol - quoted vol| at the five pillar strikes.
    pillar_fit_error: float = 0.0

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
    if smile_model == "svi":
        labels = PILLAR_LABELS
    else:
        labels = ("25p", "atm", "25c")
    err = max(abs(float(smile.vol(strikes[p])) - vols[p]) for p in labels)
    if smile_model == "svi" and err > 1e-8:
        warnings.warn(
            f"SVI cannot reproduce the {label} pillar quotes: max vol miss {err:.2e} "
            f"(active bounds {getattr(smile, 'active_bounds', ())}); the surface "
            "returns the fitted smile, not the quotes, at this expiry",
            UserWarning, stacklevel=2,
        )
    return SmileSlice(label, T, S, r_d, r_f, convention, vols, strikes, smile, err)


class FXVolSurface:
    """Term structure of fitted smiles with delta-space interpolation.

    Parameters
    ----------
    slices : sequence of SmileSlice
        At least one expiry; sorted by T internally.
    delta_convention : str
        ``"native"`` (default): each slice is read in its own quote
        convention, so pillar quotes are reproduced.  Or one of
        ``spot | forward | spot_pa | forward_pa`` to force a single
        common coordinate on every slice (pillar quotes of slices quoted
        in a different convention are then not reproduced).
    """

    def __init__(
        self, slices: Sequence[SmileSlice], delta_convention: str = "native"
    ) -> None:
        if len(slices) == 0:
            raise ValueError("surface needs at least one slice")
        if delta_convention != "native" and delta_convention not in DELTA_CONVENTIONS:
            raise ValueError(
                f"unknown delta convention {delta_convention!r}; expected 'native' "
                f"or one of {DELTA_CONVENTIONS}"
            )
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

    def slice_convention(self, sl: SmileSlice) -> str:
        """Delta convention used to read slice ``sl``."""
        return sl.convention if self.delta_convention == "native" else self.delta_convention

    # -- per-slice delta coordinate lookups -----------------------------
    def _slice_atm_strike(self, sl: SmileSlice) -> float:
        """DNS ATM strike on one slice: fixed point K = F exp(+-sigma(K)^2 T/2)."""
        pa = self.slice_convention(sl).endswith("_pa")
        K = sl.F
        for _ in range(200):
            K_new = atm_dns_strike(sl.F, float(sl.smile.vol(K)), sl.T, pa)
            if abs(K_new - K) < 1e-15 * sl.F:
                return K_new
            K = K_new
        raise SurfaceConvergenceError(f"DNS ATM fixed point did not converge on slice {sl.label}")

    def _slice_strike_at_delta(self, sl: SmileSlice, delta: float, cp: int) -> float:
        """Strike with smile-consistent delta ``cp*delta`` on one slice."""
        conv = self.slice_convention(sl)
        target = cp * delta

        def g(K: float) -> float:
            sig = float(sl.smile.vol(K))
            return gk_delta(sl.S, K, sl.T, sl.r_d, sl.r_f, sig, cp, conv) - target

        v = sl.vols["atm"] * math.sqrt(sl.T)
        start = sl.F
        floor = 0.0
        if conv.endswith("_pa") and cp == 1:
            # The pa call delta rises then falls in K.  Locate the
            # smile-consistent maximum and bracket on the falling branch.
            res = minimize_scalar(
                lambda x: -gk_delta(sl.S, sl.F * math.exp(x), sl.T, sl.r_d, sl.r_f,
                                    float(sl.smile.vol(sl.F * math.exp(x))), +1, conv),
                bounds=(-6.0 * v, 2.0 * v), method="bounded",
                options={"xatol": 1e-10},
            )
            start = floor = sl.F * math.exp(float(res.x))
            if g(start) <= 0.0:
                raise ValueError(
                    f"pa call delta {delta} unattainable on slice {sl.label}: "
                    f"maximum is {-float(res.fun):.6f}"
                )
        # Start from the flat-vol strike for this delta (closed form or the
        # pa falling branch), so the nearest root is the market one.
        try:
            guess = strike_from_delta(delta, cp, float(sl.smile.vol(sl.F)), sl.S, sl.T,
                                      sl.r_d, sl.r_f, conv)
            if guess > floor:
                start = guess
        except ValueError:
            pass
        # Every delta used here (unadjusted, pa put, pa call on its falling
        # branch) decreases in K: step up while g > 0, down while g < 0.
        a = start
        ga = g(a)
        if ga == 0.0:
            return a
        up = ga > 0.0
        fac = math.exp(0.25 * max(v, 1e-4))
        for _ in range(400):
            b = a * fac if up else max(a / fac, floor)
            gb = g(b)
            if not math.isfinite(gb):
                break
            if (gb > 0.0) != up or gb == 0.0:
                lo, hi = (a, b) if a < b else (b, a)
                return brentq(g, lo, hi, xtol=1e-15 * sl.F,
                              rtol=4.0 * np.finfo(float).eps, maxiter=500)
            if b == a:
                break
            a, ga = b, gb
        raise ValueError(
            f"delta {target} not bracketed on slice {sl.label} "
            f"(convention {conv}); outside the range where the smile is defined"
        )

    def _slice_vol_at_delta(
        self, sl: SmileSlice, delta: float | None, cp: int = 1
    ) -> float:
        """Vol on one slice at an (unsigned) delta in that slice's
        convention; ``delta=None`` means the DNS ATM coordinate."""
        if delta is None:
            return float(sl.smile.vol(self._slice_atm_strike(sl)))
        if not (math.isfinite(delta) and 0.0 < delta < 1.0):
            raise ValueError(f"delta magnitude must be in (0,1), got {delta}")
        if cp not in (+1, -1):
            raise ValueError(f"cp must be +1 or -1, got {cp}")
        return float(sl.smile.vol(self._slice_strike_at_delta(sl, delta, cp)))

    def _bracket(self, T: float) -> tuple[list[SmileSlice], list[float]]:
        """Pillar slices and linear-in-T weights used at expiry T."""
        sls = self.slices
        if T <= sls[0].T:
            return [sls[0]], [1.0]
        if T >= sls[-1].T:
            return [sls[-1]], [1.0]
        i = int(np.searchsorted(self._Ts, T, side="right")) - 1
        s0, s1 = sls[i], sls[i + 1]
        lam = (T - s0.T) / (s1.T - s0.T)
        return [s0, s1], [1.0 - lam, lam]

    # -- public queries -------------------------------------------------
    def vol_delta(self, delta: float | None, T: float, cp: int = 1) -> float:
        """Vol at fixed (unsigned) delta and arbitrary expiry T.

        Total variance ``sigma^2 T`` is interpolated linearly in T at
        fixed delta between the bracketing pillar expiries.  Outside the
        pillar range the vol is held flat.  ``delta=None`` queries the
        ATM (DNS) coordinate.
        """
        if not (math.isfinite(T) and T > 0.0):
            raise ValueError(f"T must be positive and finite, got {T}")
        sls, wts = self._bracket(T)
        if len(sls) == 1:
            return self._slice_vol_at_delta(sls[0], delta, cp)
        w = sum(lam * self._slice_vol_at_delta(sl, delta, cp) ** 2 * sl.T
                for sl, lam in zip(sls, wts))
        return math.sqrt(w / T)

    def vol_atm(self, T: float) -> float:
        """DNS ATM vol at arbitrary T (pa DNS on premium-adjusted slices)."""
        return self.vol_delta(None, T)

    def vol(self, K: float, T: float, tol: float = 1e-12, max_iter: int = 200) -> float:
        """Vol at (strike, expiry) via the fixed-delta fixed point.

        At a trial vol sigma, the delta of K at expiry T is computed in
        each bracketing slice's convention, each slice is read at that
        delta, and total variance is interpolated in T.  The iteration is
        damped.  Raises :class:`SurfaceConvergenceError` if it does not
        settle within ``max_iter`` steps.
        """
        if not (math.isfinite(K) and K > 0.0):
            raise ValueError(f"strike must be positive and finite, got {K}")
        if not (math.isfinite(T) and T > 0.0):
            raise ValueError(f"T must be positive and finite, got {T}")
        r_d, r_f = self.rates(T)
        F = self.forward(T)
        cp = 1 if K >= F else -1
        sls, wts = self._bracket(T)

        def step(sigma: float) -> float:
            if len(sls) == 1:
                sl = sls[0]
                d = gk_delta(self.S, K, T, r_d, r_f, sigma, cp, self.slice_convention(sl))
                return self._slice_vol_at_delta(sl, abs(d), cp)
            w = 0.0
            for sl, lam in zip(sls, wts):
                d = gk_delta(self.S, K, T, r_d, r_f, sigma, cp, self.slice_convention(sl))
                w += lam * self._slice_vol_at_delta(sl, abs(d), cp) ** 2 * sl.T
            return math.sqrt(w / T)

        sigma = self.vol_atm(T)
        for _ in range(max_iter):
            sigma_new = step(sigma)
            if abs(sigma_new - sigma) < tol:
                return sigma_new
            sigma = 0.5 * (sigma + sigma_new)  # damped for wing stability
        raise SurfaceConvergenceError(
            f"vol(K={K}, T={T}) fixed point did not converge in {max_iter} steps "
            f"(last iterate {sigma})"
        )

    # -- diagnostics ----------------------------------------------------
    def calendar_arbitrage_report(
        self,
        k_grid: Sequence[float] | None = None,
        tol: float = 1e-10,
        deltas: Sequence[float] | None = None,
    ) -> list[dict]:
        """Calendar check at fixed forward moneyness across pillar expiries.

        For each consecutive pair of pillars and each ``k = ln(K/F(T))``
        on ``k_grid`` (default 121 points on [-1, 1] scaled by the longer
        pillar's ATM standard deviation, at least +-0.3), a violation is
        total variance ``w(k, T)`` decreasing in T by more than ``tol``.
        This is the static no-arbitrage condition.  Returns violation
        records (empty means clean).

        ``deltas`` is accepted for backward compatibility and forwards to
        :meth:`delta_calendar_report` with a ``DeprecationWarning``.
        """
        if deltas is not None:
            import warnings
            warnings.warn(
                "calendar_arbitrage_report(deltas=...) is deprecated; the fixed-delta "
                "view is delta_calendar_report()",
                DeprecationWarning, stacklevel=2,
            )
            return self.delta_calendar_report(deltas=deltas, tol=tol)
        violations = []
        for s0, s1 in zip(self.slices, self.slices[1:]):
            if k_grid is None:
                half = max(4.0 * s1.vols["atm"] * math.sqrt(s1.T), 0.3)
                ks = np.linspace(-half, half, 121)
            else:
                ks = np.asarray(k_grid, dtype=float)
            w0 = np.array([float(s0.smile.vol(s0.F * math.exp(k))) ** 2 * s0.T for k in ks])
            w1 = np.array([float(s1.smile.vol(s1.F * math.exp(k))) ** 2 * s1.T for k in ks])
            for k, a, b in zip(ks, w0, w1):
                if b < a - tol:
                    violations.append(
                        {"k": float(k), "T_from": s0.T, "T_to": s1.T,
                         "w_from": float(a), "w_to": float(b)}
                    )
        return violations

    def delta_calendar_report(
        self, deltas: Sequence[float] = (0.10, 0.25), tol: float = 1e-10
    ) -> list[dict]:
        """Fixed-delta total-variance monitor across pillar expiries.

        Coordinates: ATM plus each put/call delta in ``deltas``.  A record
        is total variance decreasing in T by more than ``tol``.  This is a
        quote-monitoring view, not a no-arbitrage test.
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
                        {"coordinate": name, "T_from": w_prev[0], "T_to": sl.T,
                         "w_from": w_prev[1], "w_to": w}
                    )
                w_prev = (sl.T, w)
        return violations

    def is_calendar_arbitrage_free(self, **kwargs) -> bool:
        """True when :meth:`calendar_arbitrage_report` finds no violation."""
        return len(self.calendar_arbitrage_report(**kwargs)) == 0


def build_surface(market, smile_model: str = "svi",
                  delta_convention: str = "native") -> FXVolSurface:
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
