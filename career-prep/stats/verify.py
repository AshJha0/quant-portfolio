"""Independent checks for the statistics and econometrics pack.

Every answer in ``keys/answers.md`` is recomputed here from the raw data in
the question, by a route that does not copy the written solution: direct
summation over the data, a ``numpy``/``scipy`` routine, or a closed form
evaluated numerically. A seeded Monte Carlo estimate is added where a
simulation is cheap; ``career-prep/tests/test_stats_pack.py`` checks both.

Run ``python career-prep/stats/verify.py`` to print the table.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np
from scipy import stats

Number = Fraction | float


def _s01() -> Fraction:
    x = [Fraction(v) for v in (2, 4, 4, 6)]
    m = sum(x, Fraction(0)) / len(x)
    return sum(((v - m) ** 2 for v in x), Fraction(0)) / (len(x) - 1)


def _s05() -> Fraction:
    z = stats.norm.ppf(0.975)
    return Fraction(math.ceil(z * z * 0.25 / 0.01**2))


def _ols(x: list[int], y: list[int]) -> tuple[Fraction, Fraction]:
    """Slope and R^2 by solving the normal equations exactly."""
    n = len(x)
    sx, sy = sum(x), sum(y)
    sxx = sum(v * v for v in x)
    sxy = sum(a * b for a, b in zip(x, y, strict=True))
    slope = Fraction(n * sxy - sx * sy, n * sxx - sx * sx)
    icpt = Fraction(sy, n) - slope * Fraction(sx, n)
    ybar = Fraction(sy, n)
    sse = sum(((b - icpt - slope * a) ** 2 for a, b in zip(x, y, strict=True)), Fraction(0))
    sst = sum(((b - ybar) ** 2 for b in y), Fraction(0))
    return slope, 1 - sse / sst


def _ar1_var(phi: float) -> float:
    """Unconditional variance of an AR(1) from its MA(infinity) weights."""
    return float(sum(phi ** (2 * k) for k in range(400)))


def _half_life(phi: float) -> float:
    lo, hi = 0.0, 100.0
    for _ in range(200):  # bisection on phi**h = 1/2
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if phi**mid > 0.5 else (lo, mid)
    return lo


def _exp_mle() -> float:
    _loc, scale = stats.expon.fit([1, 2, 3, 6], floc=0)  # scipy's own MLE
    return 1 / scale


def _var_mean_equicorr(n: int, rho: Fraction) -> Fraction:
    cov = [[Fraction(1) if i == j else rho for j in range(n)] for i in range(n)]
    return sum((sum(r, Fraction(0)) for r in cov), Fraction(0)) / (n * n)


def _ess_ar1(n: int, phi: float) -> float:
    """n divided by the long-run variance factor sum_k rho_k (large-n limit)."""
    lrv = sum(phi ** abs(k) for k in range(-500, 501))
    return n / lrv


def _chi2() -> Fraction:
    obs = [5, 8, 9, 8, 10, 20]
    e = Fraction(sum(obs), 6)
    return sum(((o - e) ** 2 / e for o in obs), Fraction(0))


def _beta_post_mean() -> Fraction:
    # posterior is proportional to p^7 (1-p)^3: mean = B(9,4)/B(8,4) by direct
    # integration of polynomials with exact coefficients
    def integral(a: int, b: int) -> Fraction:
        # integral of p^a (1-p)^b on [0,1] via binomial expansion
        return sum(
            (Fraction(math.comb(b, k) * (-1) ** k, a + k + 1) for k in range(b + 1)),
            Fraction(0),
        )

    return integral(8, 3) / integral(7, 3)


def _normal_post_mean() -> float:
    grid = np.linspace(-10, 10, 200_001)
    w = stats.norm.pdf(grid) * stats.norm.pdf(2.0, loc=grid)
    return float((grid * w).sum() / w.sum())


def _fisher_lower() -> float:
    z = math.atanh(0.5) - stats.norm.ppf(0.975) / math.sqrt(103 - 3)
    return math.tanh(z)


def _compound() -> Fraction:
    return Fraction(11, 10) * Fraction(9, 10) - 1


EXACT: dict[str, Callable[[], Number]] = {
    "S01": _s01,
    "S02": lambda: 20 / math.sqrt(100),
    "S03": lambda: Fraction(60 - 50) / Fraction(5),  # sd = sqrt(100 * 1/4) = 5
    "S04": lambda: float(2 * stats.norm.sf(2.0)),
    "S05": _s05,
    "S06": lambda: _ols([1, 2, 3, 4], [2, 3, 5, 6])[0],
    "S07": lambda: _ols([1, 2, 3, 4], [2, 3, 5, 6])[1],
    "S08": lambda: math.sqrt(sum(0.01**2 for _ in range(252))),  # variances add
    "S09": lambda: math.sqrt((1 + 1.0**2 / 2) / 4),
    "S10": lambda: _ar1_var(0.5),
    "S11": lambda: _half_life(0.8),
    "S12": _exp_mle,
    "S13": lambda: Fraction(5, 100) / 20,
    "S14": lambda: float(1 - stats.binom.pmf(0, 20, 0.05)),
    "S15": lambda: float(stats.norm.sf(stats.norm.ppf(0.95), loc=2.5)),
    "S16": lambda: float(np.array([1, 1]) @ np.array([[4, 3], [3, 9]]) @ np.array([1, 1])),
    "S17": lambda: _var_mean_equicorr(10, Fraction(1, 5)),
    "S18": lambda: _ess_ar1(100, 0.5),
    "S19": _chi2,
    "S20": lambda: Fraction(100, 6) * (Fraction(1, 4) + Fraction(1, 4)),
    "S21": _beta_post_mean,
    "S22": _normal_post_mean,
    "S23": lambda: float(stats.binom.pmf(0, 10, 0.1)),
    "S24": _fisher_lower,
    "S25": _compound,
}


# --------------------------------------------------------------- simulation
def _mc(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


N_MC = 400_000


def _mc_p_value() -> float:
    return float((np.abs(_mc(4).standard_normal(N_MC)) > 2).mean())


def _mc_any_false_positive() -> float:
    p = _mc(14).random((N_MC // 4, 20))
    return float((p < 0.05).any(axis=1).mean())


def _mc_power() -> float:
    z = _mc(15).standard_normal(N_MC) + 2.5
    return float((z > stats.norm.ppf(0.95)).mean())


def _mc_equicorr_mean() -> float:
    rng = _mc(17)
    common = rng.standard_normal((N_MC // 4, 1))
    x = math.sqrt(0.2) * common + math.sqrt(0.8) * rng.standard_normal((N_MC // 4, 10))
    return float(x.mean(axis=1).var())


def _mc_bootstrap_absent() -> float:
    draws = _mc(23).integers(0, 10, (N_MC // 4, 10))
    return float((draws != 0).all(axis=1).mean())


def _mc_ar1_var() -> float:
    rng = _mc(10)
    e = rng.standard_normal((40_000, 200))
    x = np.zeros(40_000)
    for t in range(200):
        x = 0.5 * x + e[:, t]
    return float(x.var())


MONTE_CARLO: dict[str, tuple[Callable[[], float], float]] = {
    "S04": (_mc_p_value, 0.003),
    "S10": (_mc_ar1_var, 0.04),
    "S14": (_mc_any_false_positive, 0.006),
    "S15": (_mc_power, 0.004),
    "S17": (_mc_equicorr_mean, 0.006),
    "S23": (_mc_bootstrap_absent, 0.006),
}


if __name__ == "__main__":  # pragma: no cover
    for pid, fn in EXACT.items():
        val = fn()
        mc = MONTE_CARLO.get(pid)
        extra = f"  MC {mc[0]():.4f}" if mc else ""
        print(f"{pid}  {val!s:>24}  {float(val):.10f}{extra}")
