"""Independent checks for the linear algebra and numerical methods pack.

Every answer in ``keys/answers.md`` is recomputed here by running the method
the question describes (a real Newton or bisection loop, a real quadrature
rule, a real finite-difference quotient) or by a ``numpy``/``scipy`` routine
that does not share the hand derivation in the key. The amplification
factors are checked by building the full scheme matrix and taking its
spectrum. ``career-prep/tests/test_numerics_pack.py`` compares the key with
this file.

Run ``python career-prep/numerics/verify.py`` to print the table.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np
from scipy import integrate, linalg

Number = Fraction | float


def _eig(a: list[list[float]]) -> np.ndarray:
    return np.linalg.eigvalsh(np.array(a, dtype=float))


def _equicorr(n: int, rho: float) -> np.ndarray:
    return np.full((n, n), rho) + (1 - rho) * np.eye(n)


def _psd_boundary() -> float:
    # smallest rho for which the 3x3 equicorrelation matrix is PSD, by bisection
    lo, hi = -1.0, 0.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if np.linalg.eigvalsh(_equicorr(3, mid)).min() >= -1e-15 else (mid, hi)
    return hi


def _clip_diag() -> float:
    w, v = np.linalg.eigh(np.array([[1.0, 2.0], [2.0, 1.0]]))
    repaired = v @ np.diag(np.clip(w, 0, None)) @ v.T
    return float(repaired[0, 0])


def _det_lu() -> float:
    p, _lower, upper = linalg.lu(np.array([[2.0, 1, 1], [4, 3, 3], [8, 7, 9]]))
    return float(np.linalg.det(p) * np.prod(np.diag(upper)))


def _det_exact(a: list[list[Fraction]]) -> Fraction:
    # cofactor expansion, exact
    if len(a) == 1:
        return a[0][0]
    return sum(
        ((-1) ** j * a[0][j] * _det_exact([r[:j] + r[j + 1:] for r in a[1:]]) for j in range(len(a))),
        Fraction(0),
    )


def _newton(f: Callable[[Fraction], Fraction], df: Callable[[Fraction], Fraction],
            x: Fraction, steps: int) -> Fraction:
    for _ in range(steps):
        x = x - f(x) / df(x)
    return x


def _bisect_count() -> Fraction:
    a, b, n = 0.0, 2.0, 0
    while b - a >= 1e-6:
        m = (a + b) / 2
        a, b = (m, b) if m * m < 2 else (a, m)
        n += 1
    return Fraction(n)


def _secant() -> Fraction:
    x0, x1 = Fraction(1), Fraction(2)

    def f(x: Fraction) -> Fraction:
        return x * x - 2

    return x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))


def _trap_error_ratio() -> float:
    exact = math.e - 1
    errs = []
    for n in (64, 128):
        x = np.linspace(0, 1, n + 1)
        errs.append(abs(integrate.trapezoid(np.exp(x), x) - exact))
    return float(errs[0] / errs[1])


def _scheme_lowest_eig(theta: float, r: float, m: int = 200) -> float:
    """Spectral radius of the heat-equation step matrix on m interior points.

    theta = 0 explicit, 1/2 Crank-Nicolson, 1 implicit; r = D dt / dx^2,
    Dirichlet boundaries. Returns the smallest eigenvalue, which belongs to
    the highest-frequency (sawtooth) mode.
    """
    lap = (np.diag(-2 * np.ones(m)) + np.diag(np.ones(m - 1), 1) + np.diag(np.ones(m - 1), -1))
    eye = np.eye(m)
    step = np.linalg.solve(eye - theta * r * lap, eye + (1 - theta) * r * lap)
    # the two factors commute and are symmetric, so the step matrix is symmetric
    w = np.linalg.eigvalsh((step + step.T) / 2)
    return float(w.min())


def _explicit_dt_max() -> float:
    # largest dt with spectral radius <= 1 for dx = 0.01, D = 1, by bisection
    dx = 0.01
    lo, hi = 0.0, 1e-3
    for _ in range(40):
        mid = (lo + hi) / 2
        ok = _scheme_lowest_eig(0.0, mid / dx**2, 100) >= -1.0
        lo, hi = (mid, hi) if ok else (lo, mid)
    return lo


def _eps() -> Fraction:
    e = 1.0
    while 1.0 + e / 2 != 1.0:
        e /= 2
    return Fraction(e)


def _power_ratio() -> float:
    a = np.array([[2.0, 1.0], [1.0, 2.0]])  # eigenvalues 3 and 1
    v = np.array([1.0, 0.0])
    top = np.array([1.0, 1.0]) / math.sqrt(2)
    errs = []
    for _ in range(12):
        v = a @ v
        v /= np.linalg.norm(v)
        errs.append(np.linalg.norm(v - top))
    return float(errs[-1] / errs[-2])


EXACT: dict[str, Callable[[], Number]] = {
    "N01": lambda: float(_eig([[2, 1], [1, 2]]).max()),
    "N02": lambda: float(_eig(_equicorr(3, 0.5).tolist()).max() / 3),
    "N03": _psd_boundary,
    "N04": lambda: _det_exact([[Fraction(1), Fraction(9, 10), Fraction(7, 10)],
                               [Fraction(9, 10), Fraction(1), Fraction(3, 10)],
                               [Fraction(7, 10), Fraction(3, 10), Fraction(1)]]),
    "N05": lambda: float(np.linalg.cholesky(np.array([[4.0, 2.0], [2.0, 5.0]]))[1, 1]),
    "N06": _clip_diag,
    "N07": lambda: float(np.linalg.cond(np.array([[1.0, 1.0], [1.0, 1.0001]]))),
    "N08": _det_lu,
    "N09": lambda: float(abs(np.linalg.qr(np.array([[3.0, 1.0], [4.0, 2.0]]))[1][1, 1])),
    "N10": lambda: _newton(lambda x: x * x - 2, lambda x: 2 * x, Fraction(1), 2),
    "N11": lambda: _newton(lambda x: x * x, lambda x: 2 * x, Fraction(1), 10),
    "N12": _bisect_count,
    "N13": _secant,
    "N14": lambda: float(integrate.trapezoid([0.0, 0.25, 1.0], [0.0, 0.5, 1.0])),
    "N15": lambda: float(integrate.simpson([0.0, 0.25, 1.0], x=[0.0, 0.5, 1.0])),
    "N16": lambda: float(sum(w * x**4 for x, w in zip(*np.polynomial.legendre.leggauss(2), strict=True))),
    "N17": _trap_error_ratio,
    "N18": lambda: (math.exp(0.1) - math.exp(0.0)) / 0.1,
    "N19": lambda: (math.exp(0.1) - math.exp(-0.1)) / 0.2,
    "N20": lambda: math.sqrt(float(_eps())),
    "N21": _explicit_dt_max,
    "N22": lambda: _scheme_lowest_eig(0.5, 10.0, 400),
    "N23": lambda: _scheme_lowest_eig(1.0, 10.0, 400),
    "N24": _eps,
    "N25": _power_ratio,
}

# Tolerances for checks that are limits (a grid or a finite iteration count
# approaches the key) rather than exact evaluations.
REL_TOL: dict[str, float] = {"N17": 1e-3, "N21": 1e-3, "N22": 1e-3, "N23": 1e-3, "N25": 1e-6}


if __name__ == "__main__":  # pragma: no cover
    for pid, fn in EXACT.items():
        val = fn()
        print(f"{pid}  {val!s:>24}  {float(val):.12g}")
