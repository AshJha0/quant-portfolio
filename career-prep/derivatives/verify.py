"""Independent checks for the derivatives and products pack.

The key in ``keys/answers.md`` uses the Black-Scholes closed forms and
backward induction. Nothing here calls a Black-Scholes formula: prices are
risk-neutral expectations integrated by quadrature against the lognormal
density, Greeks are pathwise derivatives under the integral (or a
Richardson-extrapolated finite difference of the integrated price), tree
answers are enumerated path by path (the American one over every possible
exercise rule), and a seeded simulation gives a second check where it is
cheap. ``career-prep/tests/test_quant_packs.py`` compares them.

Run ``python career-prep/derivatives/verify.py`` to print the table.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

Number = Fraction | float

# Base case used throughout unless a problem says otherwise.
S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0


def _expect(
    f: Callable[[float, float], float],
    s0: float = S0,
    r: float = R,
    sig: float = SIG,
    t: float = T,
    q: float = 0.0,
    kink: float = K,
) -> float:
    """E^Q[f(S_T, Z)] with S_T = s0 exp((r - q - sig^2/2) t + sig sqrt(t) Z), Z ~ N(0,1).

    ``kink`` is the level of S_T where the payoff has a corner; the quadrature
    is split there so the corner does not cost accuracy.
    """
    drift = (r - q - 0.5 * sig * sig) * t
    vol = sig * math.sqrt(t)
    c = 1 / math.sqrt(2 * math.pi)
    zk = (math.log(kink / s0) - drift) / vol
    pts = sorted({0.0, zk}) if -14 < zk < 14 else [0.0]
    return float(
        quad(
            lambda z: f(s0 * math.exp(drift + vol * z), z) * c * math.exp(-z * z / 2),
            -14,
            14,
            epsabs=1e-13,
            epsrel=1e-13,
            limit=400,
            points=pts,
        )[0]
    )


def _disc(payoff: Callable[[float], float], **kw: float) -> float:
    r = kw.get("r", R)
    t = kw.get("t", T)
    return math.exp(-r * t) * _expect(lambda s, _z: payoff(s), **kw)


def call(k: float = K, **kw: float) -> float:
    return _disc(lambda s: max(s - k, 0.0), kink=k, **kw)


def put(k: float = K, **kw: float) -> float:
    return _disc(lambda s: max(k - s, 0.0), kink=k, **kw)


def _richardson(f: Callable[[float], float], x: float, h: float, order: int) -> float:
    """First (order=1) or second (order=2) central difference with one Richardson step."""

    def d(hh: float) -> float:
        if order == 1:
            return (f(x + hh) - f(x - hh)) / (2 * hh)
        return (f(x + hh) - 2 * f(x) + f(x - hh)) / (hh * hh)

    return (4 * d(h / 2) - d(h)) / 3


def _delta_pathwise(s0: float = S0) -> float:
    """dC/dS0 = e^{-rT} E[1{S_T > K} S_T / S0]."""
    return math.exp(-R * T) * _expect(lambda s, _z: (s > K) * s / s0, s0=s0)


def _vega_pathwise() -> float:
    """dC/dsigma = e^{-rT} E[1{S_T > K} S_T (sqrt(T) Z - sigma T)]."""
    return math.exp(-R * T) * _expect(lambda s, z: (s > K) * s * (math.sqrt(T) * z - SIG * T))


def _rho_pathwise() -> float:
    """dC/dr = e^{-rT} E[1{S_T > K} K T] (the S_T terms cancel against discounting)."""
    return math.exp(-R * T) * _expect(lambda s, _z: (s > K) * K * T)


def _implied_vol(price: float) -> float:
    return float(brentq(lambda v: call(sig=v) - price, 0.01, 1.0, xtol=1e-13))


def _tree_paths(steps: int) -> list[tuple[int, ...]]:
    return list(itertools.product((1, 0), repeat=steps))


def _euro_tree(
    payoff: Callable[[Fraction], Fraction],
    s0: Fraction,
    u: Fraction,
    d: Fraction,
    growth: Fraction,
    steps: int,
) -> Fraction:
    """Sum over every path: risk-neutral probability times discounted payoff."""
    p = (growth - d) / (u - d)
    tot = Fraction(0)
    for path in _tree_paths(steps):
        ups = sum(path)
        tot += p**ups * (1 - p) ** (steps - ups) * payoff(s0 * u**ups * d ** (steps - ups))
    return tot / growth**steps


def _american_put_all_rules() -> Fraction:
    """Two-step American put: maximise over every exercise rule, not by backward induction.

    A rule says, for each node before expiry (root, up, down), whether to exercise.
    The value of a rule is its expected discounted payoff; the price is the best rule.
    """
    s0, u, d, g, k = Fraction(100), Fraction(11, 10), Fraction(9, 10), Fraction(21, 20), Fraction(105)
    p = (g - d) / (u - d)
    best = Fraction(-1)
    for ex_root, ex_up, ex_dn in itertools.product((False, True), repeat=3):
        if ex_root:
            val = max(k - s0, Fraction(0))
        else:
            val = Fraction(0)
            for first, pr1 in ((u, p), (d, 1 - p)):
                ex = ex_up if first == u else ex_dn
                s1 = s0 * first
                if ex:
                    val += pr1 * max(k - s1, Fraction(0)) / g
                    continue
                for second, pr2 in ((u, p), (d, 1 - p)):
                    val += pr1 * pr2 * max(k - s1 * second, Fraction(0)) / g**2
        best = max(best, val)
    return best


def _margrabe_quad() -> float:
    """E[(S1 - S2)^+] for two correlated lognormals, by a 2-D quadrature in (Z1, Z2)."""
    s1, s2, v1, v2, rho = 100.0, 100.0, 0.2, 0.3, 0.5
    c = 1 / (2 * math.pi)

    def inner(z1: float) -> float:
        x1 = s1 * math.exp(-0.5 * v1 * v1 + v1 * z1)

        def g(z2: float) -> float:
            x2 = s2 * math.exp(-0.5 * v2 * v2 + v2 * (rho * z1 + math.sqrt(1 - rho**2) * z2))
            return max(x1 - x2, 0.0) * c * math.exp(-(z1 * z1 + z2 * z2) / 2)

        # kink where x1 = x2, solved for z2
        kink = ((math.log(s1 / s2) - 0.5 * v1 * v1 + v1 * z1 + 0.5 * v2 * v2) / v2 - rho * z1) / math.sqrt(
            1 - rho**2
        )
        pts = [kink] if -12 < kink < 12 else None
        return float(quad(g, -12, 12, points=pts, epsabs=1e-13, limit=200)[0])

    return float(quad(inner, -12, 12, epsabs=1e-12, limit=200)[0])


def _chooser_quad() -> float:
    """Chooser at t=0.5: value at 0.5 is max(C, P) with T-0.5 to go; integrate over S_0.5."""
    tc = 0.5

    def at_choice(s: float, _z: float) -> float:
        return max(call(s0=s, t=T - tc), put(s0=s, t=T - tc))

    # max(C, P) has its corner where C = P, i.e. at S = K exp(-r (T - tc)) by parity
    return math.exp(-R * tc) * _expect(at_choice, t=tc, kink=K * math.exp(-R * (T - tc)))


EXACT: dict[str, Callable[[], Number]] = {
    "D01": call,
    "D02": put,
    "D03": _delta_pathwise,
    "D04": lambda: _richardson(_delta_pathwise, S0, 0.5, 1),
    "D05": _vega_pathwise,
    "D06": lambda: _expect(lambda s, _z: s, q=0.02),
    "D07": lambda: _disc(lambda s: float(s > K)),
    "D08": lambda: _disc(lambda s: s * (s > K)),
    "D09": lambda: _implied_vol(12.0),
    "D10": lambda: _euro_tree(
        lambda s: max(s - 100, Fraction(0)), Fraction(100), Fraction(11, 10), Fraction(9, 10), Fraction(1), 2
    ),
    "D11": _american_put_all_rules,
    "D12": lambda: _disc(lambda s: abs(s - K)),
    "D13": lambda: (
        _disc(lambda s: max(s - 95.0, 0.0), kink=95.0) - _disc(lambda s: max(s - 105.0, 0.0), kink=105.0)
    ),
    "D14": lambda: _richardson(lambda x: _disc(lambda s: abs(s - K), s0=x), S0, 0.5, 1),
    "D15": _margrabe_quad,
    "D16": _chooser_quad,
    "D17": lambda: _expect(lambda s, _z: float(s > K)),
    "D18": lambda: _expect(lambda s, _z: float(s > K), r=0.10),
    "D19": lambda: -_richardson(lambda tt: call(t=tt), T, 0.02, 1),
    "D20": _rho_pathwise,
    "D21": lambda: _euro_tree(
        lambda s: max(s - 105, Fraction(0)),
        Fraction(100),
        Fraction(11, 10),
        Fraction(9, 10),
        Fraction(21, 20),
        2,
    ),
    "D22": lambda: _richardson(lambda x: call(s0=x), S0, 0.5, 2),
}

# Relative tolerance between key and check; finite differences get a looser one.
TOL: dict[str, float] = {"D04": 1e-6, "D14": 1e-7, "D19": 1e-6, "D22": 1e-5, "D15": 1e-8, "D16": 1e-8}


# ---------------------------------------------------------------- Monte Carlo
N_MC = 400_000


def _st(seed: int, mu: float = R, s0: float = S0) -> np.ndarray:
    z = np.random.default_rng(seed).standard_normal(N_MC)
    return np.asarray(s0 * np.exp((mu - 0.5 * SIG**2) * T + SIG * math.sqrt(T) * z))


def _mc_margrabe() -> float:
    rng = np.random.default_rng(215)
    z1, z2 = rng.standard_normal((2, N_MC))
    x1 = 100 * np.exp(-0.02 + 0.2 * z1)
    x2 = 100 * np.exp(-0.045 + 0.3 * (0.5 * z1 + math.sqrt(0.75) * z2))
    return float(np.maximum(x1 - x2, 0).mean())


DF = math.exp(-R * T)
MONTE_CARLO: dict[str, tuple[Callable[[], float], float]] = {
    # id: (estimator, absolute tolerance)
    "D01": (lambda: DF * float(np.maximum(_st(201) - K, 0).mean()), 0.08),
    "D02": (lambda: DF * float(np.maximum(K - _st(202), 0).mean()), 0.05),
    "D07": (lambda: DF * float((_st(207) > K).mean()), 0.005),
    "D15": (_mc_margrabe, 0.1),
    "D17": (lambda: float((_st(217) > K).mean()), 0.005),
    "D18": (lambda: float((_st(218, mu=0.10) > K).mean()), 0.005),
}


if __name__ == "__main__":  # pragma: no cover
    for pid, fn in EXACT.items():
        val = fn()
        mc = MONTE_CARLO.get(pid)
        extra = f"  MC {mc[0]():.4f}" if mc else ""
        print(f"{pid}  {val!s:>24}  {float(val):.12g}{extra}")
