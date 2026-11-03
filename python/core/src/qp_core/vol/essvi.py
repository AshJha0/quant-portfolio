"""eSSVI fitted slice by slice, with static-arbitrage constraints.

Plain idea
----------
Power-law SSVI uses one ``rho`` and one ``(eta, gamma)`` for the whole
surface. That is too stiff for five-point FX smiles: on the USDJPY preset it
misses the worst pillar by about one vol point (fx_surface VALIDATION F10).
eSSVI keeps the SSVI shape of each slice but gives every expiry its own
three numbers ``(theta, rho, psi)``. Each slice is fitted to its pillars in
expiry order, and the fit may only move inside a region that rules out
butterfly arbitrage in the slice and wing crossings against the previous
slice.

Result on the F10 presets (``tests/test_vol_essvi.py``): eSSVI by slice
fits EURUSD within 0.03 vol, but USDJPY's one-year slice still misses by
0.115 vol, and an unconstrained fit does no better, so three numbers per
slice are the limit there. Raw SVI per slice (:func:`fit_svi_slices`, five
numbers, the plan's alternative) with the same arbitrage checks enforced
fits both presets within 0.05 vol. Use SVI per slice for pricing FX
smiles; eSSVI remains the smoother, cheaper parametrisation when its
error is acceptable.

Formal statement
----------------
One slice, ``k = ln(K / F_T)``, ``psi = theta * phi`` (Hendriks and Martini
2019, "The extended SSVI volatility surface", J. Computational Finance
22(5); the slice is Gatheral and Jacquier 2014 eq. 4.1 with
``psi = theta phi``):

    w(k) = theta/2 * (1 + rho psi k / theta
                      + sqrt((psi k / theta + rho)^2 + 1 - rho^2))

Butterfly arbitrage is ruled out when (Gatheral and Jacquier 2014,
Theorem 4.2, rewritten in ``psi``)

    psi (1 + |rho|) < 4        and        psi^2 (1 + |rho|) <= 4 theta.

Between consecutive slices ``1 -> 2`` the fit imposes

    theta_2 >= theta_1,   psi_2 (1 + rho_2) >= psi_1 (1 + rho_1),
    psi_2 (1 - rho_2) >= psi_1 (1 - rho_1).

The first is ``w`` increasing at the money; the other two say the wing
slopes ``psi (1 +- rho) / 2`` do not fall with maturity, so the slices
cannot cross far out in either wing. These are necessary for calendar
no-arbitrage but not sufficient on their own (Hendriks and Martini 2019,
s.3, add a condition on ``theta_2 / theta_1``). So the fit does not claim
more: :func:`calendar_arbitrage_free` checks ``w_2(k) >= w_1(k)`` on a dense
grid and :class:`ESSVIFit` reports the result.

Worked example
--------------
>>> s = ESSVISlice(theta=0.01, rho=-0.3, psi=0.1)
>>> round(float(s.total_variance(0.0)), 12)
0.01
>>> s.butterfly_free()
True
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "ESSVIFit",
    "ESSVISlice",
    "SVIFit",
    "SVISlice",
    "calendar_arbitrage_free",
    "essvi_total_variance",
    "fit_essvi_slices",
    "fit_svi_slices",
]


def essvi_total_variance(
    k: np.ndarray | float, theta: float, rho: float, psi: float
) -> np.ndarray | float:
    """Total variance of one eSSVI slice at log-moneyness ``k``."""
    k = np.asarray(k, dtype=float)
    pk = psi * k / theta
    return 0.5 * theta * (1.0 + rho * pk + np.sqrt((pk + rho) ** 2 + 1.0 - rho * rho))


@dataclass(frozen=True)
class ESSVISlice:
    """One eSSVI slice: ATM-forward total variance, correlation, ``psi = theta phi``."""

    theta: float
    rho: float
    psi: float

    def __post_init__(self) -> None:
        if not self.theta > 0.0:
            raise ValueError(f"theta must be > 0, got {self.theta}")
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")
        if not self.psi > 0.0:
            raise ValueError(f"psi must be > 0, got {self.psi}")

    def total_variance(self, k: np.ndarray | float) -> np.ndarray | float:
        return essvi_total_variance(k, self.theta, self.rho, self.psi)

    def implied_vol(self, k: np.ndarray | float, T: float) -> np.ndarray | float:
        return np.sqrt(np.asarray(self.total_variance(k)) / T)

    def butterfly_free(self) -> bool:
        """Gatheral-Jacquier Theorem 4.2 in ``psi`` form (a sufficient condition)."""
        a = 1.0 + abs(self.rho)
        return bool(self.psi * a < 4.0 and self.psi**2 * a <= 4.0 * self.theta * (1.0 + 1e-9))


def calendar_arbitrage_free(
    slices: Sequence[ESSVISlice], k_grid: np.ndarray | None = None, tol: float = 1e-12
) -> bool:
    """True when ``w_{i+1}(k) >= w_i(k) - tol`` for each consecutive pair on ``k_grid``."""
    grid = np.linspace(-3.0, 3.0, 1201) if k_grid is None else np.asarray(k_grid, dtype=float)
    for a, b in zip(slices[:-1], slices[1:], strict=True):
        if np.any(np.asarray(b.total_variance(grid)) < np.asarray(a.total_variance(grid)) - tol):
            return False
    return True


@dataclass(frozen=True)
class ESSVIFit:
    """Fitted slices, the worst absolute vol error per slice, and the arbitrage checks."""

    slices: tuple[ESSVISlice, ...]
    max_vol_error: tuple[float, ...]
    butterfly_free: bool
    calendar_free: bool


def _theta_at_forward(k: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(k)
    k, w = k[order], w[order]
    if not k[0] < 0.0 < k[-1]:
        raise ValueError("pillar strikes must bracket the forward (k = 0)")
    return float(np.interp(0.0, k, w))


# scipy-stubs types constraints as a TypedDict; plain dicts are what SLSQP takes.
Constraint = Any


def _constraints(
    scale: float, prev: ESSVISlice | None
) -> tuple[Callable[[np.ndarray], tuple[float, float, float]], list[Constraint]]:
    def u(x: np.ndarray) -> tuple[float, float, float]:
        return float(x[0]) * scale, float(x[1]), float(x[2])

    lim = 4.0 * (1.0 - 1e-9)
    cons: list[Constraint] = [
        {"type": "ineq", "fun": lambda x: lim - u(x)[2] * (1.0 + u(x)[1])},
        {"type": "ineq", "fun": lambda x: lim - u(x)[2] * (1.0 - u(x)[1])},
        {"type": "ineq", "fun": lambda x: (4.0 * u(x)[0] - u(x)[2] ** 2 * (1.0 + u(x)[1])) / scale},
        {"type": "ineq", "fun": lambda x: (4.0 * u(x)[0] - u(x)[2] ** 2 * (1.0 - u(x)[1])) / scale},
    ]
    if prev is not None:
        p = prev
        cons += [
            {"type": "ineq", "fun": lambda x: (u(x)[0] - p.theta) / p.theta},
            {"type": "ineq", "fun": lambda x: u(x)[2] * (1.0 + u(x)[1]) - p.psi * (1.0 + p.rho)},
            {"type": "ineq", "fun": lambda x: u(x)[2] * (1.0 - u(x)[1]) - p.psi * (1.0 - p.rho)},
        ]
    return u, cons


def _essvi_objective(
    k: np.ndarray, v_arr: np.ndarray, T: float, u: Callable[[np.ndarray], tuple[float, float, float]]
) -> Callable[[np.ndarray], float]:
    def obj(x: np.ndarray) -> float:
        th, rho, psi = u(x)
        w = np.asarray(essvi_total_variance(k, th, rho, psi))
        return float(np.sum((np.sqrt(np.maximum(w, 1e-300) / T) - v_arr) ** 2)) * 1e4

    return obj


def fit_essvi_slices(
    strikes: Sequence[Sequence[float]],
    vols: Sequence[Sequence[float]],
    forwards: Sequence[float],
    expiries: Sequence[float],
) -> ESSVIFit:
    """Fit one eSSVI slice per expiry, in expiry order, under the constraints above.

    ``strikes[i]`` and ``vols[i]`` are the pillar strikes and Black vols of
    expiry ``expiries[i]`` (at least three points, bracketing the forward).
    The objective is the sum of squared vol errors. SLSQP is started from a
    few correlations and the best feasible answer is kept.
    """
    n = len(expiries)
    if n == 0 or not (len(strikes) == len(vols) == len(forwards) == n):
        raise ValueError("strikes, vols, forwards and expiries must have the same non-zero length")
    if any(t2 <= t1 for t1, t2 in zip(expiries[:-1], expiries[1:], strict=True)):
        raise ValueError("expiries must be strictly increasing")
    fitted: list[ESSVISlice] = []
    errors: list[float] = []
    prev: ESSVISlice | None = None
    for K, v, F, T in zip(strikes, vols, forwards, expiries, strict=True):
        K_arr = np.asarray(K, dtype=float)
        v_arr = np.asarray(v, dtype=float)
        if K_arr.size < 3 or K_arr.size != v_arr.size:
            raise ValueError("each slice needs at least three (strike, vol) pairs")
        if T <= 0.0 or F <= 0.0 or np.any(K_arr <= 0.0) or np.any(v_arr <= 0.0):
            raise ValueError("expiries, forwards, strikes and vols must be positive")
        k = np.log(K_arr / F)
        th0 = _theta_at_forward(k, v_arr**2 * T)
        if prev is not None:
            th0 = max(th0, prev.theta)
        u, cons = _constraints(th0, prev)

        obj = _essvi_objective(k, v_arr, T, u)

        best = None
        rho_starts = ([prev.rho] if prev is not None else []) + [-0.5, -0.2, 0.0, 0.2]
        psi0 = prev.psi * 1.05 if prev is not None else 0.5 * float(np.sqrt(th0))
        for r0 in rho_starts:
            res = minimize(
                obj, np.array([1.0, r0, psi0]), method="SLSQP", constraints=cons,
                bounds=[(1e-3, 50.0), (-0.999, 0.999), (1e-8, 4.0)],
                options={"maxiter": 1000, "ftol": 1e-18},
            )
            feasible = all(c["fun"](res.x) >= -1e-10 for c in cons)
            if feasible and (best is None or res.fun < best.fun):
                best = res
        if best is None:
            raise ValueError(f"no feasible eSSVI slice found at T={T}")
        sl = ESSVISlice(*u(best.x))
        fitted.append(sl)
        errors.append(float(np.max(np.abs(np.asarray(sl.implied_vol(k, T)) - v_arr))))
        prev = sl
    return ESSVIFit(
        slices=tuple(fitted),
        max_vol_error=tuple(errors),
        butterfly_free=all(s.butterfly_free() for s in fitted),
        calendar_free=calendar_arbitrage_free(fitted),
    )


# ---------------------------------------------------------------------------
# Raw SVI per slice with explicit arbitrage checks (the plan's alternative).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SVISlice:
    """Raw SVI (Gatheral 2004): ``w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + s^2))``."""

    a: float
    b: float
    rho: float
    m: float
    s: float

    def total_variance(self, k: np.ndarray | float) -> np.ndarray:
        x = np.asarray(k, dtype=float) - self.m
        return self.a + self.b * (self.rho * x + np.sqrt(x * x + self.s * self.s))

    def implied_vol(self, k: np.ndarray | float, T: float) -> np.ndarray:
        return np.sqrt(self.total_variance(k) / T)

    def density_factor(self, k: np.ndarray | float) -> np.ndarray:
        """Gatheral and Jacquier (2014) eq. 2.1, ``g(k)``; no butterfly arbitrage iff ``g >= 0``."""
        x = np.asarray(k, dtype=float) - self.m
        r = np.sqrt(x * x + self.s * self.s)
        w = self.total_variance(k)
        w1 = self.b * (self.rho + x / r)
        w2 = self.b * self.s * self.s / r**3
        kk = np.asarray(k, dtype=float)
        return (1.0 - kk * w1 / (2.0 * w)) ** 2 - 0.25 * w1 * w1 * (1.0 / w + 0.25) + 0.5 * w2


def _svi_grid(theta: float) -> np.ndarray:
    return np.asarray(np.linspace(-10.0, 10.0, 401) * np.sqrt(theta), dtype=float)


@dataclass(frozen=True)
class SVIFit:
    """Per-slice raw SVI fit with the arbitrage checks that were enforced."""

    slices: tuple[SVISlice, ...]
    max_vol_error: tuple[float, ...]
    butterfly_free: bool
    calendar_free: bool


def _svi_objective(
    k: np.ndarray, v_arr: np.ndarray, T: float, make: Callable[[np.ndarray], SVISlice]
) -> Callable[[np.ndarray], float]:
    def obj(x: np.ndarray) -> float:
        w = make(x).total_variance(k)
        return float(np.sum((np.sqrt(np.maximum(w, 1e-300) / T) - v_arr) ** 2)) * 1e4

    return obj


def fit_svi_slices(
    strikes: Sequence[Sequence[float]],
    vols: Sequence[Sequence[float]],
    forwards: Sequence[float],
    expiries: Sequence[float],
) -> SVIFit:
    """Fit raw SVI per expiry with no-butterfly and no-calendar constraints.

    Constraints, all imposed in the fit and re-checked afterwards:
    ``g(k) >= 0`` on a grid of +-10 ATM standard deviations (Gatheral and
    Jacquier 2014, Lemma 2.2); Lee's (2004) moment bound on the wings,
    ``b (1 + |rho|) <= 2``; and ``w_i(k) >= w_{i-1}(k)`` on the union of the
    two slices' grids. Five parameters for five pillars, so the fit is
    close to exact when the pillars are themselves arbitrage-free.
    """
    n = len(expiries)
    if n == 0 or not (len(strikes) == len(vols) == len(forwards) == n):
        raise ValueError("strikes, vols, forwards and expiries must have the same non-zero length")
    if any(t2 <= t1 for t1, t2 in zip(expiries[:-1], expiries[1:], strict=True)):
        raise ValueError("expiries must be strictly increasing")
    fitted: list[SVISlice] = []
    errors: list[float] = []
    butterfly_ok = True
    calendar_ok = True
    prev: SVISlice | None = None
    prev_grid = np.zeros(0)
    for K, v, F, T in zip(strikes, vols, forwards, expiries, strict=True):
        K_arr = np.asarray(K, dtype=float)
        v_arr = np.asarray(v, dtype=float)
        if K_arr.size < 5 or K_arr.size != v_arr.size:
            raise ValueError("raw SVI needs at least five (strike, vol) pairs per slice")
        if T <= 0.0 or F <= 0.0 or np.any(K_arr <= 0.0) or np.any(v_arr <= 0.0):
            raise ValueError("expiries, forwards, strikes and vols must be positive")
        k = np.log(K_arr / F)
        th = _theta_at_forward(k, v_arr**2 * T)
        grid = _svi_grid(th)
        cal_grid = np.union1d(grid, prev_grid)
        sc = th  # scale a and s by theta / sqrt(theta) so SLSQP sees O(1) numbers

        def make(x: np.ndarray, sc: float = sc) -> SVISlice:
            return SVISlice(a=x[0] * sc, b=x[1] * np.sqrt(sc), rho=x[2],
                            m=x[3] * np.sqrt(sc), s=max(x[4], 1e-8) * np.sqrt(sc))

        obj = _svi_objective(k, v_arr, T, make)

        cons: list[Constraint] = [
            {"type": "ineq", "fun": lambda x, g=grid, make=make: make(x).density_factor(g)},
            {"type": "ineq", "fun": lambda x, make=make: 2.0 - make(x).b * (1.0 + abs(make(x).rho))},
            {"type": "ineq", "fun": lambda x, g=grid, make=make, sc=sc: make(x).total_variance(g) / sc},
        ]
        if prev is not None:
            p = prev
            cons.append({"type": "ineq",
                         "fun": lambda x, g=cal_grid, make=make, p=p, sc=sc:
                         (make(x).total_variance(g) - p.total_variance(g)) / sc})
        best = None
        starts = [np.array([0.7, 0.3, r0, 0.0, 0.5]) for r0 in (-0.5, -0.2, 0.0)]
        if prev is not None:
            ps = np.sqrt(sc)
            starts.insert(0, np.array([prev.a / sc, prev.b / ps, prev.rho, prev.m / ps, prev.s / ps]))
        for x0 in starts:
            res = minimize(obj, x0, method="SLSQP", constraints=cons,
                           bounds=[(-5.0, 5.0), (1e-6, 50.0), (-0.999, 0.999), (-5.0, 5.0), (1e-4, 20.0)],
                           options={"maxiter": 2000, "ftol": 1e-18})
            feasible = all(np.all(np.asarray(c["fun"](res.x)) >= -1e-10) for c in cons)
            if feasible and (best is None or res.fun < best.fun):
                best = res
        if best is None:
            raise ValueError(f"no arbitrage-free SVI slice found at T={T}")
        sl = make(best.x)
        butterfly_ok &= bool(np.all(sl.density_factor(grid) >= -1e-10))
        if prev is not None:
            calendar_ok &= bool(np.all(sl.total_variance(cal_grid) >= prev.total_variance(cal_grid) - 1e-12))
        fitted.append(sl)
        errors.append(float(np.max(np.abs(sl.implied_vol(k, T) - v_arr))))
        prev, prev_grid = sl, grid
    return SVIFit(tuple(fitted), tuple(errors), butterfly_ok, calendar_ok)
