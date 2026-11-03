"""Independent checks for the fixed-income maths pack.

The key in ``keys/answers.md`` uses annuity formulas, duration and
convexity identities and the Vasicek and Black closed forms. The checks here
avoid them: bond prices are summed cash flow by cash flow in exact rational
arithmetic, durations and convexities come from those sums or from
Richardson-extrapolated finite differences of the price, yields and rates
are root-found or solved as linear systems, the Vasicek bond is priced by
integrating its Riccati ODEs numerically, and the caplet by quadrature.
``career-prep/tests/test_quant_packs.py`` compares them.

Run ``python career-prep/fixed-income/verify.py`` to print the table.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

Number = Fraction | float


def _flows(
    coupon: Fraction, years: int, freq: int = 1, face: Fraction = Fraction(100)
) -> list[tuple[Fraction, Fraction]]:
    """(time in years, cash flow) for a bullet bond."""
    n = years * freq
    c = face * coupon / freq
    return [(Fraction(i, freq), c + (face if i == n else 0)) for i in range(1, n + 1)]


def _pv(flows: list[tuple[Fraction, Fraction]], y: Fraction, freq: int = 1) -> Fraction:
    """Exact present value at a yield compounded ``freq`` times a year."""
    return sum((cf / (1 + y / freq) ** int(t * freq) for t, cf in flows), Fraction(0))


def _pv_float(flows: list[tuple[Fraction, Fraction]], y: float) -> float:
    return float(sum(float(cf) * (1 + y) ** (-float(t)) for t, cf in flows))


BOND = _flows(Fraction(6, 100), 5)
Y5 = Fraction(5, 100)


def _macaulay(flows: list[tuple[Fraction, Fraction]], y: Fraction) -> Fraction:
    return sum((t * cf / (1 + y) ** int(t) for t, cf in flows), Fraction(0)) / _pv(flows, y)


def _deriv(f: Callable[[float], float], x: float, h: float, order: int) -> float:
    def d(hh: float) -> float:
        if order == 1:
            return (f(x + hh) - f(x - hh)) / (2 * hh)
        return (f(x + hh) - 2 * f(x) + f(x - hh)) / (hh * hh)

    return (4 * d(h / 2) - d(h)) / 3


def _bootstrap() -> np.ndarray:
    """Discount factors from a 1y zero at 96 and a 2y 5% annual bond at 99, as a linear system."""
    a = np.array([[100.0, 0.0], [5.0, 105.0]])
    return np.linalg.solve(a, np.array([96.0, 99.0]))


def _par_yield_2y() -> float:
    d1, d2 = _bootstrap()
    return float(brentq(lambda c: 100 * c * (d1 + d2) + 100 * d2 - 100, 0.0, 0.2, xtol=1e-15))


DFS = (Fraction(97, 100), Fraction(94, 100), Fraction(91, 100))


def _float_leg_pv() -> Fraction:
    """Float leg valued by projecting each forward from the curve, not by the 1 - DF_n shortcut."""
    prev = Fraction(1)
    tot = Fraction(0)
    for df in DFS:
        fwd = prev / df - 1  # one-year forward rate
        tot += fwd * df
        prev = df
    return tot


def _swap_rate() -> Fraction:
    annuity = sum(DFS, Fraction(0))
    # solve rate * annuity = float leg PV (a 1x1 linear system)
    return _float_leg_pv() / annuity


def _vasicek_ode(a: float = 0.5, b: float = 0.05, s: float = 0.01, r0: float = 0.03, t: float = 2.0) -> float:
    """P = exp(A(tau) - B(tau) r0); B' = 1 - a B, A' = a b ... integrated in time-to-maturity."""

    def rhs(_tau: float, y: np.ndarray) -> list[float]:
        bb, _aa = y
        return [1 - a * bb, -a * b * bb + 0.5 * s * s * bb * bb]

    sol = solve_ivp(rhs, (0.0, t), [0.0, 0.0], rtol=1e-13, atol=1e-15)
    bb, aa = sol.y[:, -1]
    return float(math.exp(aa - bb * r0))


def _caplet_quad() -> float:
    """Notional x tau x DF x E[(F_T - K)^+] with F lognormal, integrated in the normal variable."""
    f0, k, vol, expiry, tau, df, notional = 0.05, 0.05, 0.20, 1.0, 0.25, 0.94, 1_000_000.0
    sd = vol * math.sqrt(expiry)
    c = 1 / math.sqrt(2 * math.pi)
    ev = quad(
        lambda z: max(f0 * math.exp(-0.5 * sd * sd + sd * z) - k, 0.0) * c * math.exp(-z * z / 2),
        -14,
        14,
        points=[0.5 * sd],
        epsabs=1e-15,
        limit=200,
    )[0]
    return notional * tau * df * float(ev)


def _ytm_polyroots() -> float:
    """YTM of a 3y 5% annual bond at 98 from the roots of its price polynomial in v = 1/(1+y)."""
    # 5 v + 5 v^2 + 105 v^3 - 98 = 0
    roots = np.roots([105.0, 5.0, 5.0, -98.0])
    v = next(float(r.real) for r in roots if abs(r.imag) < 1e-12 and 0 < r.real < 1)
    return 1 / v - 1


ZERO10 = [(Fraction(10), Fraction(100))]
Y4 = Fraction(4, 100)

EXACT: dict[str, Callable[[], Number]] = {
    "F01": lambda: _pv(BOND, Y5),
    "F02": lambda: _macaulay(BOND, Y5),
    "F03": lambda: -_deriv(lambda y: _pv_float(BOND, y), 0.05, 1e-3, 1) / float(_pv(BOND, Y5)),
    "F04": lambda: _deriv(lambda y: _pv_float(BOND, y), 0.05, 1e-3, 2) / float(_pv(BOND, Y5)),
    "F05": lambda: (_pv_float(BOND, 0.0499) - _pv_float(BOND, 0.0501)) / 2,
    "F06": lambda: float(brentq(lambda z: (1 + z) ** -2 - 0.9, 0.0, 0.5, xtol=1e-15)),
    "F07": lambda: Fraction(105, 100) ** 2 / Fraction(104, 100) - 1,
    "F08": lambda: float(_bootstrap()[1]),
    "F09": _par_yield_2y,
    "F10": lambda: float(brentq(lambda c: math.exp(c) - 1.05, 0.0, 0.2, xtol=1e-15)),
    "F11": lambda: _pv(_flows(Fraction(4, 100), 2, freq=2), Y5, freq=2),
    "F12": _ytm_polyroots,
    "F13": lambda: _pv(ZERO10, Y4),
    "F14": lambda: _macaulay(ZERO10, Y4) / (1 + Y4),
    "F15": lambda: (
        (Fraction(4) * _macaulay(ZERO10, Y4) / (1 + Y4) + Fraction(6) * _macaulay(BOND, Y5) / (1 + Y5)) / 10
    ),
    "F16": lambda: (
        Fraction(1_000_000)
        * (Fraction(6, 100) - Fraction(5, 100))
        * Fraction(1, 4)
        / (1 + Fraction(6, 100) * Fraction(1, 4))
    ),
    "F17": _swap_rate,
    "F18": lambda: 100 * (Fraction(4, 100) * sum(DFS, Fraction(0)) - _float_leg_pv()),
    "F19": _vasicek_ode,
    "F20": _caplet_quad,
    "F21": lambda: (
        sum((t * (t + 1) * cf / (1 + Y4) ** (int(t) + 2) for t, cf in ZERO10), Fraction(0)) / _pv(ZERO10, Y4)
    ),
    "F22": lambda: _pv(BOND, Y5) - _pv(BOND, Fraction(6, 100)),
}

TOL: dict[str, float] = {"F03": 1e-9, "F04": 1e-6, "F05": 1e-6, "F19": 1e-10, "F20": 1e-10}


# ---------------------------------------------------------------- Monte Carlo
def _mc_vasicek() -> float:
    """E[exp(-int r)] by exact OU stepping with a trapezoid integral."""
    a, b, s, r0, t = 0.5, 0.05, 0.01, 0.03, 2.0
    steps, paths = 200, 100_000
    dt = t / steps
    e = math.exp(-a * dt)
    sd = s * math.sqrt((1 - e * e) / (2 * a))
    rng = np.random.default_rng(319)
    r = np.full(paths, r0)
    integral = np.zeros(paths)
    for _ in range(steps):
        nxt = b + (r - b) * e + sd * rng.standard_normal(paths)
        integral += 0.5 * (r + nxt) * dt
        r = nxt
    return float(np.exp(-integral).mean())


MONTE_CARLO: dict[str, tuple[Callable[[], float], float]] = {
    "F19": (_mc_vasicek, 2e-4),
}


if __name__ == "__main__":  # pragma: no cover
    for pid, fn in EXACT.items():
        val = fn()
        mc = MONTE_CARLO.get(pid)
        extra = f"  MC {mc[0]():.6f}" if mc else ""
        print(f"{pid}  {val!s:>40}  {float(val):.12g}{extra}")
