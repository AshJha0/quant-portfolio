"""Arbitrage-free SABR: the density by a conservative finite-difference PDE.

Plain idea: Hagan's SABR formula is an expansion, and at low strikes and
long expiries it implies a negative probability density, which is an
arbitrage (:func:`qp_core.vol.sabr.sabr_density_check` finds it). Hagan and
co-authors fixed this by going back one step: instead of expanding the
option price, they write down a one-dimensional forward (Fokker-Planck)
equation whose local volatility is the SABR effective volatility, and solve
it numerically. The scheme is built so that probability is conserved
exactly, the forward stays a martingale exactly, and the density cannot go
negative. Option prices are integrals against a non-negative density, so
call prices are convex and decreasing in strike by construction.

Formal statement (Hagan, Kumar, Lesniewski and Woodward 2014,
"Arbitrage-free SABR", Wilmott 2014(69), 60-75, section 2). With
``C(F) = F^beta`` on the shifted forward,

* ``y(F) = (F^{1-beta} - f^{1-beta}) / (1 - beta)`` (``ln(F/f)`` at beta = 1),
* ``D(F) = sqrt(alpha^2 + 2 rho alpha nu y + nu^2 y^2)``,
* ``Gamma(F) = (C(F) - C(f)) / (F - f)``, ``E(T, F) = exp(rho nu alpha Gamma(F) T)``,
* ``M(F) = D(F)^2 C(F)^2 / 2``,

the density ``Q(T, F)`` solves ``Q_T = (M E Q)_FF`` on ``(F_min, F_max)``
with absorbing ends: the mass ``P_L`` that reaches ``F_min`` and ``P_R``
that reaches ``F_max`` obey ``dP_L/dT = (M E Q)_F(F_min)`` and
``dP_R/dT = -(M E Q)_F(F_max)``, and ``Q(0, F) = delta(F - f)``. A Taylor
expansion of the implied vol of this local-vol model reproduces Hagan's
formula to first order in ``T``: ``D`` gives the
``(2 - 3 rho^2) nu^2 / 24`` term and the time average of ``E^{1/2}`` the
``rho beta nu alpha / (4 f^{1-beta})`` term. At long expiries the two
differ at second order, and the PDE is the one without arbitrage.

Grid (Le Floc'h and Kennedy 2014, "Finite difference techniques for
arbitrage-free SABR", J. Comput. Finance 20(3), 51-79, sections 3-4).
Cells are uniform in Hagan's normalised variable ``z``, with
``y(z) = (alpha / nu) (sinh(nu z) + rho (cosh(nu z) - 1))``, so they are
fine near the forward and coarse in the tails; a uniform grid in ``F``
needs thousands of cells when ``nu^2 T`` is large. ``z`` spans
``+/- n_sd sqrt(T)``, cut at the ``z`` of ``F = 0`` (the absorbing zero
boundary for ``beta < 1``), and ``z = 0`` is a cell centre so that the
initial delta sits on one cell. The finite-volume scheme on that
non-uniform grid holds cell masses ``q_j``, with face fluxes
``phi_{j+1/2} = (G_{j+1} - G_j) / (F_{j+1} - F_j)``,
``G_j = M_j E_j q_j / Delta_j`` (``Delta_j`` the cell width), mirror ghost
cells at both ends (the Hagan et al. 2014 condition ``G_0 = -G_1``), so
``phi_{1/2} = G_1 / (F_1 - F_min)`` and
``phi_{J+1/2} = -G_J / (F_max - F_J)``. Summation by parts shows that
``sum q_j + P_L + P_R`` and ``sum F_j q_j + F_min P_L + F_max P_R`` are
conserved by the semi-discrete system for any linear time stepper, so
total probability is 1 and the mean is ``f`` to rounding.

Time steps:

* ``scheme="implicit"``: backward Euler. ``I - dt A`` is a column
  diagonally dominant M-matrix, so its inverse is entrywise non-negative
  and the masses stay non-negative for any step (first order in time).
* ``scheme="tr_bdf2"`` (default): TR-BDF2, which Le Floc'h and Kennedy
  recommend for this PDE (second order, L-stable). Its trapezoidal stage
  is not provably positive, so :meth:`SabrPdeResult.check` is there to
  assert it; the tests run it on the arbitrage case.

Prices. Each cell's mass is spread uniformly over the widest interval
centred on ``F_j`` that fits in the cell; this keeps the mean of every
cell at ``F_j`` (so put-call parity is exact) and makes the call price a
smooth convex function of strike.

Worked example (the expansion's arbitrage case in the tests): ``f = 0.03``,
``T = 10``, ``alpha = 0.02``, ``beta = 0.5``, ``rho = -0.3``, ``nu = 0.6``.
The expansion's density is negative between strikes of about 0.1 % and
1.5 %; the PDE density (default 200 cells) is non-negative everywhere,
9.2 % of the mass is absorbed at zero, and the at-the-money Black vol is
13.68 % against 14.26 % from the expansion (the gap is the
second-order-in-``T`` difference described above; at ``T = 0.5`` the two
agree to 2e-6 at the money).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq

from .sabr import black76_price

__all__ = ["SabrPdeResult", "sabr_pde"]


@dataclass(frozen=True)
class SabrPdeResult:
    """Cell masses on the PDE grid plus the boundary masses, in shifted units."""

    forward: float  # shifted forward f
    expiry: float
    shift: float
    centres: np.ndarray  # shifted cell centres F_1 .. F_J
    faces: np.ndarray  # shifted faces F_min = F_{1/2} .. F_{J+1/2} = F_max
    mass: np.ndarray  # q_j, probability in cell j
    p_low: float  # mass absorbed at F_min
    p_high: float  # mass absorbed at F_max

    @property
    def f_min(self) -> float:
        return float(self.faces[0])

    @property
    def f_max(self) -> float:
        return float(self.faces[-1])

    @property
    def density(self) -> np.ndarray:
        """Cell-average density ``q_j / Delta_j``."""
        return self.mass / np.diff(self.faces)

    @property
    def total_mass(self) -> float:
        return float(self.mass.sum() + self.p_low + self.p_high)

    @property
    def mean(self) -> float:
        return float(self.centres @ self.mass + self.f_min * self.p_low + self.f_max * self.p_high)

    def check(self, tol: float = 1e-14) -> None:
        """Raise ``ValueError`` if any mass is below ``-tol``."""
        low = min(float(self.mass.min()), self.p_low, self.p_high)
        if low < -tol:
            raise ValueError(f"negative probability mass {low:.3e}")

    def call(self, strike: float) -> float:
        """Undiscounted call on the (unshifted) strike."""
        k = strike + self.shift
        if not math.isfinite(k):
            raise ValueError("strike must be finite")
        c = self.centres
        a = np.minimum(c - self.faces[:-1], self.faces[1:] - c)  # half-width
        lo, hi = c - a, c + a
        payoff = np.where(lo >= k, c - k, 0.0)
        part = (lo < k) & (hi > k)
        payoff[part] = (hi[part] - k) ** 2 / (4.0 * a[part])
        value = float(self.mass @ payoff)
        value += self.p_high * max(self.f_max - k, 0.0) + self.p_low * max(self.f_min - k, 0.0)
        return value

    def put(self, strike: float) -> float:
        """Undiscounted put by parity, exact because mass and mean are conserved."""
        return self.call(strike) - (self.forward - (strike + self.shift))

    def implied_vol(self, strike: float) -> float:
        """Black-76 vol (on the shifted forward and strike) of the PDE price."""
        k = strike + self.shift
        f = self.forward
        kind: Literal["call", "put"] = "call" if k >= f else "put"
        price = self.call(strike) if kind == "call" else self.put(strike)
        intrinsic = max(f - k, 0.0) if kind == "call" else max(k - f, 0.0)
        if not price > intrinsic:
            raise ValueError("price at or below intrinsic: no implied vol")

        def g(s: float) -> float:
            return black76_price(f, k, self.expiry, 0.0, s, kind) - price

        return float(brentq(g, 1e-8, 20.0, xtol=1e-14, rtol=1e-14))


def sabr_pde(
    F: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    shift: float = 0.0,
    n_cells: int = 200,
    n_steps: int | None = None,
    n_sd: float = 4.0,
    scheme: str = "tr_bdf2",
) -> SabrPdeResult:
    """Solve the arbitrage-free SABR density PDE (see the module docstring).

    Parameters
    ----------
    F, T : float
        Forward and expiry in years (``T > 0``).
    alpha, beta, rho, nu : float
        SABR parameters, ``alpha > 0``, ``0 <= beta <= 1``, ``-1 < rho < 1``,
        ``nu >= 0``.
    shift : float
        Shifted SABR: the model runs on ``F + shift``.
    n_cells, n_steps : int
        Approximate number of cells, and time steps (default ``n_cells // 2``,
        at least 20).
    n_sd : float
        Grid half-width in ``z``, in units of ``sqrt(T)``.
    scheme : {"tr_bdf2", "implicit"}
    """
    for name, v in (
        ("F", F),
        ("T", T),
        ("alpha", alpha),
        ("beta", beta),
        ("rho", rho),
        ("nu", nu),
        ("shift", shift),
        ("n_sd", n_sd),
    ):
        if not math.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v!r}")
    if (
        F + shift <= 0.0
        or T <= 0.0
        or alpha <= 0.0
        or nu < 0.0
        or not 0.0 <= beta <= 1.0
        or not -1.0 < rho < 1.0
        or n_sd <= 0.0
    ):
        raise ValueError(
            "need F + shift > 0, T > 0, alpha > 0, nu >= 0, 0 <= beta <= 1, -1 < rho < 1, n_sd > 0"
        )
    if n_cells < 10:
        raise ValueError("n_cells must be >= 10")
    if scheme not in ("tr_bdf2", "implicit"):
        raise ValueError(f"scheme must be 'tr_bdf2' or 'implicit', got {scheme!r}")
    n_steps = max(20, n_cells // 2) if n_steps is None else int(n_steps)
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    f = F + shift
    omb = 1.0 - beta

    def y_of_z(z_in: float | np.ndarray) -> np.ndarray:
        z = np.asarray(z_in, dtype=float)
        if nu == 0.0:
            return alpha * z
        return alpha / nu * (np.sinh(nu * z) + rho * (np.cosh(nu * z) - 1.0))

    def f_of_y(y_in: np.ndarray) -> np.ndarray:
        y = np.asarray(y_in, dtype=float)
        if omb == 0.0:
            return f * np.exp(y)
        out: np.ndarray = np.power(np.maximum(f**omb + omb * y, 0.0), 1.0 / omb)
        return out

    z_hi = n_sd * math.sqrt(T)
    z_lo = -z_hi
    if omb > 0.0:
        y_zero = -(f**omb) / omb  # y at F = 0
        if float(y_of_z(z_lo)) <= y_zero:
            z_lo = brentq(lambda z: float(y_of_z(z)) - y_zero, z_lo, 0.0, xtol=1e-15)
    dz0 = (z_hi - z_lo) / n_cells
    j0 = max(2, int(math.ceil(-z_lo / dz0)))
    dz = -z_lo / (j0 - 0.5)
    n = max(j0 + 2, int(math.ceil((z_hi - z_lo) / dz)))
    zf = z_lo + dz * np.arange(n + 1)
    zc = z_lo + dz * (np.arange(1, n + 1) - 0.5)
    yc = y_of_z(zc)
    faces = f_of_y(y_of_z(zf))
    centres = f_of_y(yc)
    centres[j0 - 1] = f
    width = np.diff(faces)
    if np.any(width <= 0.0) or np.any(centres <= faces[:-1]) or np.any(centres >= faces[1:]):
        raise ValueError("grid degenerated; reduce n_sd or n_cells")

    cc = np.power(centres, beta)
    d2 = alpha * alpha + 2.0 * rho * alpha * nu * yc + nu * nu * yc * yc
    m = 0.5 * d2 * cc * cc / width  # M_j / Delta_j: G_j = c_j(t) q_j
    dfc = centres - f
    with np.errstate(invalid="ignore", divide="ignore"):
        gam = np.where(
            np.abs(dfc) > 1e-14 * f, (cc - f**beta) / dfc, beta * f ** (beta - 1.0) if beta != 0.0 else 0.0
        )
    gap = np.diff(centres)  # F_{j+1} - F_j
    gl = centres[0] - faces[0]  # mirror distance, left
    gr = faces[-1] - centres[-1]  # mirror distance, right

    def coeff(t: float) -> np.ndarray:
        out: np.ndarray = m * np.exp(rho * nu * alpha * gam * t)
        return out

    def apply_a(q: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, float, float]:
        g = c * q
        phi = np.diff(g) / gap  # interior faces
        out = np.empty_like(g)
        fl, fr = g[0] / gl, g[-1] / gr  # boundary outflows
        out[1:-1] = phi[1:] - phi[:-1]
        out[0] = phi[0] - fl
        out[-1] = -fr - phi[-1]
        return out, float(fl), float(fr)

    def implicit_solve(rhs: np.ndarray, c: np.ndarray, theta: float) -> tuple[np.ndarray, float, float]:
        # (I - theta A) x = rhs; A is tridiagonal in the cell masses.
        diag = np.empty(n)
        diag[1:-1] = 1.0 + theta * c[1:-1] * (1.0 / gap[1:] + 1.0 / gap[:-1])
        diag[0] = 1.0 + theta * c[0] * (1.0 / gap[0] + 1.0 / gl)
        diag[-1] = 1.0 + theta * c[-1] * (1.0 / gap[-1] + 1.0 / gr)
        ab = np.zeros((3, n))
        ab[0, 1:] = -theta * c[1:] / gap  # row j, column j+1
        ab[1] = diag
        ab[2, :-1] = -theta * c[:-1] / gap  # row j+1, column j
        x = np.asarray(solve_banded((1, 1), ab, rhs), dtype=float)
        return x, float(c[0] * x[0] / gl), float(c[-1] * x[-1] / gr)

    q = np.zeros(n)
    q[j0 - 1] = 1.0
    pl = pr = 0.0
    dt = T / n_steps
    gma = 2.0 - math.sqrt(2.0)
    a1 = 1.0 / (gma * (2.0 - gma))
    a0 = (1.0 - gma) ** 2 / (gma * (2.0 - gma))
    th = (1.0 - gma) / (2.0 - gma) * dt
    half = 0.5 * gma * dt
    for k in range(n_steps):
        t0, t1 = k * dt, (k + 1) * dt
        if scheme == "implicit":
            q, fl, fr = implicit_solve(q, coeff(t1), dt)
            pl += dt * fl
            pr += dt * fr
            continue
        aq, fl0, fr0 = apply_a(q, coeff(t0))
        qs, fls, frs = implicit_solve(q + half * aq, coeff(t0 + gma * dt), half)
        pls = pl + half * (fl0 + fls)
        prs = pr + half * (fr0 + frs)
        qn, fl1, fr1 = implicit_solve(a1 * qs - a0 * q, coeff(t1), th)
        pl = a1 * pls - a0 * pl + th * fl1
        pr = a1 * prs - a0 * pr + th * fr1
        q = qn
    return SabrPdeResult(f, T, shift, centres, faces, q, float(pl), float(pr))
