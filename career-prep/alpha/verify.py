"""Independent checks for the alpha research pack.

The key quotes closed forms; this file reaches each value by simulation,
numerical integration, root-finding or a library routine instead.
``career-prep/tests/test_numeric_packs.py`` compares the two.

Run ``python career-prep/alpha/verify.py`` to print the table.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np
from scipy import integrate, optimize
from scipy.stats import norm, spearmanr

Number = Fraction | float


def _expected_max_normal(n: int) -> float:
    # E[max] = integral of x * n phi(x) Phi(x)^(n-1)
    val, _ = integrate.quad(lambda x: x * n * norm.pdf(x) * norm.cdf(x) ** (n - 1), -12, 12,
                            limit=200)
    return float(val)


def _half_life() -> float:
    return float(optimize.brentq(lambda t: 0.9**t - 0.5, 0.1, 100, xtol=1e-14))


SIG_X = [1.2, -0.4, 0.3, 2.0, -1.1, 0.8]
RET_Y = [0.5, 0.1, -0.2, 0.9, -0.3, 0.2]


def _psr_min_trl() -> float:
    # smallest n with PSR(n) = 0.95, PSR = Phi(SR sqrt(n - 1) / sqrt(1 + SR^2 / 2))
    sr = 0.1
    return float(optimize.brentq(
        lambda n: norm.cdf(sr * math.sqrt(n - 1) / math.sqrt(1 + 0.5 * sr * sr)) - 0.95,
        2, 10_000, xtol=1e-12))


def _purged_train_size() -> int:
    """Enumerate: fold 3 of 5 on 1000 days, 10-day labels, 1% embargo."""
    n, horizon, embargo = 1000, 10, 10
    test = range(400, 600)
    keep = 0
    for t in range(n):
        if t in test:
            continue
        overlaps = any(u in test for u in range(t, t + horizon))
        embargoed = test.stop <= t < test.stop + embargo
        keep += not (overlaps or embargoed)
    return keep


EXACT: dict[str, Callable[[], Number]] = {
    "A01": lambda: 0.05 * math.sqrt(400),
    "A02": lambda: Fraction(15, 10) / 10,
    "A03": lambda: 1.0 * math.sqrt(4),
    "A04": lambda: math.sqrt((1 + 0.5 * 0.3**2) / 60),
    "A05": lambda: float(norm.isf(0.05 / 20)),
    "A06": lambda: _expected_max_normal(100),
    "A07": _half_life,
    "A08": lambda: Fraction(8, 100) / 20,
    "A09": lambda: 0.03 / 0.06 * math.sqrt(60),
    "A10": lambda: float(spearmanr(SIG_X, RET_Y).statistic),
    "A11": lambda: float(norm.sf(1 / math.sqrt(1 / 3))),
    "A12": _psr_min_trl,
    "A13": lambda: 1 - 0.95**20,
    "A14": lambda: Fraction(_purged_train_size()),
}

TOLERANCE: dict[str, float] = {}

N_MC = 200_000


def _mc(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _mc_sr_se() -> float:
    r = _mc(4).normal(0.3, 1.0, (20_000, 60))
    return float((r.mean(axis=1) / r.std(axis=1, ddof=1)).std())


def _mc_null_sharpe() -> float:
    years = 3
    r = _mc(11).standard_normal((N_MC, years * 12)) * (0.1 / math.sqrt(12))
    sr = r.mean(axis=1) * 12 / (0.1)  # known vol, annualised
    return float((sr > 1).mean())


MONTE_CARLO: dict[str, tuple[Callable[[], float], float]] = {
    "A04": (_mc_sr_se, 0.004),
    "A06": (lambda: float(_mc(6).standard_normal((20_000, 100)).max(axis=1).mean()), 0.01),
    "A11": (_mc_null_sharpe, 0.002),
    "A13": (lambda: float((_mc(13).random((N_MC, 20)) < 0.05).any(axis=1).mean()), 0.004),
}


if __name__ == "__main__":  # pragma: no cover
    for pid, fn in EXACT.items():
        mc = MONTE_CARLO.get(pid)
        extra = f"  MC {mc[0]():.6g}" if mc else ""
        print(f"{pid}  {float(fn()):.10g}{extra}")
