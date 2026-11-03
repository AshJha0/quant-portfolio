"""Second derivation route for every stochastic, derivatives and fixed-income key.

W32 asks that each pricing-flavoured answer be derived twice, by routes that
share no code. Route B already exists: each pack's ``verify.py`` recomputes
the answer numerically (quadrature, ODE or boundary-value solves, finite
differences, path enumeration) and never calls a closed form. Route A lives
here: the closed-form derivation the key's prose describes, written out in
code from scratch, with no import from ``verify.py`` and no numerical
integration. ``ROUTES`` records, per answer, the formula route A uses, so a
reader can see which identity was checked.

Each answer must then agree three ways: route A with the written key, route
A with route B, and (by ``test_quant_packs.py``) route B with the key. A key
that is wrong in the same way as one route is caught by the other.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKS = {"stochastic": "S", "derivatives": "D", "fixed-income": "F"}

Number = Fraction | float


def _n(x: float) -> float:
    """Standard normal CDF via erfc (no scipy, so no shared code with route B)."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------- Black-Scholes
S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0


def _d12(s: float = S0, k: float = K, r: float = R, sig: float = SIG, t: float = T) -> tuple[float, float]:
    d1 = (math.log(s / k) + (r + 0.5 * sig * sig) * t) / (sig * math.sqrt(t))
    return d1, d1 - sig * math.sqrt(t)


def _bs_call(s: float = S0, k: float = K, r: float = R, sig: float = SIG, t: float = T) -> float:
    d1, d2 = _d12(s, k, r, sig, t)
    return s * _n(d1) - k * math.exp(-r * t) * _n(d2)


def _bs_put(s: float = S0, k: float = K, r: float = R, sig: float = SIG, t: float = T) -> float:
    d1, d2 = _d12(s, k, r, sig, t)
    return k * math.exp(-r * t) * _n(-d2) - s * _n(-d1)


def _implied_vol_newton(target: float) -> float:
    sig = 0.2
    for _ in range(50):
        d1, _d2 = _d12(sig=sig)
        step = (_bs_call(sig=sig) - target) / (S0 * _phi(d1) * math.sqrt(T))
        sig -= step
        if abs(step) < 1e-15:
            break
    return sig


def _tree_backward(
    up: Fraction, down: Fraction, gross: Fraction, k: Fraction, american_put: bool
) -> Fraction:
    """Recombining two-step tree, priced by backward induction (route B enumerates paths)."""
    p = (gross - down) / (up - down)
    s = Fraction(100)
    vals = [
        (
            max(k - s * up**j * down ** (2 - j), Fraction(0))
            if american_put
            else max(s * up**j * down ** (2 - j) - k, Fraction(0))
        )
        for j in range(3)
    ]
    for step in (1, 0):
        nxt = []
        for j in range(step + 1):
            cont = (p * vals[j + 1] + (1 - p) * vals[j]) / gross
            if american_put:
                cont = max(cont, k - s * up**j * down ** (step - j))
            nxt.append(cont)
        vals = nxt
    return vals[0]


def _margrabe() -> float:
    sig = math.sqrt(0.2**2 + 0.3**2 - 2 * 0.5 * 0.2 * 0.3)
    return 100 * (_n(sig / 2) - _n(-sig / 2))


def _chooser() -> float:
    # Rubinstein: call(K, T) + put(K e^{-r(T - t)}, t).
    return _bs_call() + _bs_put(k=K * math.exp(-R * 0.5), t=0.5)


# ---------------------------------------------------------------- fixed income
Y, C, MAT = Fraction(5, 100), Fraction(6), 5


def _annuity_price(c: Fraction, y: Fraction, n: int, face: Fraction = Fraction(100)) -> Fraction:
    v = 1 / (1 + y)
    return c * (1 - v**n) / y + face * v**n


def _macaulay_closed(c: Fraction, y: Fraction, n: int) -> Fraction:
    """Closed-form Macaulay duration of a level-coupon bond (Tuckman and Serrat, ch. 4)."""
    cr = c / 100
    return (1 + y) / y - (1 + y + n * (cr - y)) / (cr * ((1 + y) ** n - 1) + y)


def _convexity_sum(c: Fraction, y: Fraction, n: int) -> Fraction:
    flows = [(t, c + (100 if t == n else 0)) for t in range(1, n + 1)]
    return sum(
        (Fraction(t * (t + 1)) * cf / (1 + y) ** (t + 2) for t, cf in flows), Fraction(0)
    ) / _annuity_price(c, y, n)


def _ytm_secant() -> float:
    def f(y: float) -> float:
        return 5 * (1 - (1 + y) ** -3) / y + 100 * (1 + y) ** -3 - 98

    a, b = 0.05, 0.06
    for _ in range(100):
        fa, fb = f(a), f(b)
        if fb == fa:
            break
        a, b = b, b - fb * (b - a) / (fb - fa)
    return b


def _vasicek_closed(
    a: float = 0.5, b: float = 0.05, s: float = 0.01, r0: float = 0.03, t: float = 2.0
) -> float:
    bb = (1 - math.exp(-a * t)) / a
    aa = (b - s * s / (2 * a * a)) * (bb - t) - s * s * bb * bb / (4 * a)
    return math.exp(aa - bb * r0)


def _black_caplet() -> float:
    f = k = 0.05
    vol, tfix, tau, df, notional = 0.2, 1.0, 0.25, 0.94, 1_000_000
    d1 = (math.log(f / k) + 0.5 * vol * vol * tfix) / (vol * math.sqrt(tfix))
    d2 = d1 - vol * math.sqrt(tfix)
    return notional * tau * df * (f * _n(d1) - k * _n(d2))


_DF = [Fraction(97, 100), Fraction(94, 100), Fraction(91, 100)]
_P13 = 100 / Fraction(104, 100) ** 10
_P_REF = _annuity_price(C, Y, MAT)
_MOD_REF = _macaulay_closed(C, Y, MAT) / (1 + Y)


def _f08_df2() -> Fraction:
    df1 = Fraction(96, 100)
    return (99 - 5 * df1) / 105


# ---------------------------------------------------------------- the register
ROUTES: dict[str, tuple[Callable[[], Number], str]] = {
    # stochastic
    "S01": (lambda: Fraction(3), "W_3 ~ N(0, 3), second moment = variance"),
    "S02": (lambda: Fraction(2), "Cov(W_s, W_t) = min(s, t)"),
    "S03": (lambda: Fraction(1, 3), "double integral of min(s, u) over the unit square"),
    "S04": (lambda: Fraction(3), "fourth moment of N(0, 1) = 3 sigma^4"),
    "S18": (lambda: Fraction(3 + 1), "W_2 = W_1 + Z: E[W_1^4] + E[W_1^2] E[Z^2]"),
    "S22": (lambda: math.exp(2) - math.e, "lognormal: E[e^{2W}] - E[e^W]^2"),
    "S05": (lambda: Fraction(2), "Ito isometry: int_0^2 s ds"),
    "S14": (lambda: Fraction(1, 2), "Fubini: int_0^1 s ds"),
    "S19": (lambda: Fraction(1, 2), "Stratonovich = W_1^2 / 2, so expectation 1/2"),
    "S06": (lambda: math.e, "normal MGF e^{t/2} at t = 2"),
    "S07": (lambda: 2 * (1 - _n(1.0)), "reflection: 2 P(W_1 >= 1)"),
    "S15": (lambda: Fraction(4), "Wald: a / mu = 2 / 0.5"),
    "S16": (lambda: math.exp(-1.0), "exponential martingale: e^{-2 |mu| a}"),
    "S08": (lambda: 100 * math.exp(0.05), "S_0 e^{mu t}"),
    "S09": (lambda: 100 * math.exp(0.05 - 0.02), "exp of the log mean, S_0 e^{(mu - sigma^2/2) t}"),
    "S10": (lambda: 1e4 * math.exp(0.1) * (math.exp(0.04) - 1), "S_0^2 e^{2 mu t} (e^{sigma^2 t} - 1)"),
    "S11": (lambda: 1 - math.exp(-2.0), "theta + (x_0 - theta) e^{-kappa t}"),
    "S12": (lambda: 0.0625 * (1 - math.exp(-4.0)), "sigma^2 / (2 kappa) (1 - e^{-2 kappa t})"),
    "S13": (lambda: Fraction(1, 16), "sigma^2 / (2 kappa)"),
    "S17": (lambda: Fraction(1), "Girsanov: under Q, W_1 ~ N(1, 1)"),
    "S20": (lambda: 1e4 * math.exp(0.09), "e^{-r} S_0^2 e^{2 r + sigma^2}"),
    "S21": (lambda: Fraction(3), "Feynman-Kac: x^2 + (T - t) at x = 1, T - t = 2"),
    # derivatives
    "D01": (_bs_call, "Black-Scholes call S N(d1) - K e^{-rT} N(d2)"),
    "D02": (_bs_put, "Black-Scholes put K e^{-rT} N(-d2) - S N(-d1)"),
    "D06": (lambda: 100 * math.exp(0.05 - 0.02), "cost of carry S e^{(r - q) T}"),
    "D07": (lambda: math.exp(-R) * _n(_d12()[1]), "e^{-rT} N(d2)"),
    "D08": (lambda: S0 * _n(_d12()[0]), "S N(d1)"),
    "D09": (lambda: _implied_vol_newton(12.0), "Newton on the closed-form price with closed-form vega"),
    "D12": (lambda: _bs_call() + _bs_put(), "call + put"),
    "D13": (lambda: _bs_call(k=95) - _bs_call(k=105), "C(95) - C(105)"),
    "D03": (lambda: _n(_d12()[0]), "N(d1)"),
    "D04": (lambda: _phi(_d12()[0]) / (S0 * SIG), "phi(d1) / (S sigma sqrt T)"),
    "D22": (lambda: _phi(_d12()[0]) / (S0 * SIG), "parity: put gamma = call gamma"),
    "D05": (lambda: S0 * _phi(_d12()[0]), "S phi(d1) sqrt T"),
    "D14": (lambda: 2 * _n(_d12()[0]) - 1, "2 N(d1) - 1"),
    "D19": (
        lambda: -S0 * _phi(_d12()[0]) * SIG / 2 - R * K * math.exp(-R) * _n(_d12()[1]),
        "-S phi(d1) sigma / (2 sqrt T) - r K e^{-rT} N(d2)",
    ),
    "D20": (lambda: K * math.exp(-R) * _n(_d12()[1]), "K T e^{-rT} N(d2)"),
    "D10": (
        lambda: _tree_backward(Fraction(11, 10), Fraction(9, 10), Fraction(1), Fraction(100), False),
        "backward induction",
    ),
    "D21": (
        lambda: _tree_backward(Fraction(11, 10), Fraction(9, 10), Fraction(21, 20), Fraction(105), False),
        "backward induction",
    ),
    "D11": (
        lambda: _tree_backward(Fraction(11, 10), Fraction(9, 10), Fraction(21, 20), Fraction(105), True),
        "backward induction with early-exercise max",
    ),
    "D15": (_margrabe, "Margrabe closed form, sigma^2 = s1^2 + s2^2 - 2 rho s1 s2"),
    "D16": (_chooser, "Rubinstein chooser: C(K, T) + P(K e^{-r(T - t)}, t)"),
    "D17": (lambda: _n(_d12()[1]), "N(d2)"),
    "D18": (lambda: _n(_d12(r=0.10)[1]), "N(d2) with r -> mu = 10%"),
    # fixed income
    "F01": (lambda: _annuity_price(C, Y, MAT), "annuity factor c (1 - v^n) / y + 100 v^n"),
    "F02": (
        lambda: _macaulay_closed(C, Y, MAT),
        "closed-form Macaulay (1+y)/y - (1+y+n(c-y)) / (c((1+y)^n - 1) + y)",
    ),
    "F03": (lambda: _MOD_REF, "Macaulay / (1 + y)"),
    "F04": (lambda: _convexity_sum(C, Y, MAT), "sum t(t+1) CF v^{t+2} / P"),
    "F05": (lambda: float(_MOD_REF * _P_REF) * 1e-4, "modified duration x price x 1 bp"),
    "F22": (lambda: _P_REF - 100, "at 6% the 6% bond prices at par"),
    "F11": (
        lambda: _annuity_price(Fraction(2), Fraction(25, 1000), 4),
        "semiannual annuity, 4 periods at 2.5%",
    ),
    "F12": (_ytm_secant, "secant iteration on the annuity-form price"),
    "F13": (lambda: _P13, "100 / 1.04^10"),
    "F14": (lambda: Fraction(10) / Fraction(104, 100), "zero: Macaulay = T, modified = T / (1 + y)"),
    "F21": (lambda: Fraction(10 * 11) / Fraction(104, 100) ** 2, "zero: T (T + 1) / (1 + y)^2"),
    "F15": (
        lambda: Fraction(4, 10) * Fraction(10) / Fraction(104, 100) + Fraction(6, 10) * _MOD_REF,
        "value-weighted modified durations",
    ),
    "F06": (lambda: 0.9 ** (-0.5) - 1, "DF^{-1/T} - 1"),
    "F10": (lambda: math.log(1.05), "ln(1 + y)"),
    "F07": (lambda: Fraction(105, 100) ** 2 / Fraction(104, 100) - 1, "(1 + z2)^2 / (1 + z1) - 1"),
    "F08": (_f08_df2, "99 = 5 DF1 + 105 DF2 with DF1 = 0.96"),
    "F09": (lambda: (1 - _f08_df2()) / (Fraction(96, 100) + _f08_df2()), "par yield (1 - DF2) / (DF1 + DF2)"),
    "F17": (lambda: (1 - _DF[2]) / sum(_DF, Fraction(0)), "par swap rate (1 - DF3) / sum DF"),
    "F18": (
        lambda: 4 * sum(_DF, Fraction(0)) - 100 * (1 - _DF[2]),
        "fixed annuity minus floating leg 100 (1 - DF3)",
    ),
    "F16": (
        lambda: Fraction(1_000_000) * Fraction(1, 100) * Fraction(1, 4) / (1 + Fraction(6, 100) / 4),
        "N (L - K) tau / (1 + L tau)",
    ),
    "F19": (_vasicek_closed, "Vasicek affine bond exp(A - B r_0)"),
    "F20": (_black_caplet, "Black caplet N tau DF (F N(d1) - K N(d2))"),
}


# ---------------------------------------------------------------- tests
def _load(pack: str) -> ModuleType:
    name = f"cp_verify2_{pack.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(name, ROOT / pack / "verify.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


VERIFY = {p: _load(p) for p in PACKS}


def _key(pack: str) -> dict[str, str]:
    pat = re.compile(rf"^- \*\*({PACKS[pack]}\d\d)\*\* `([^`]+)`", re.M)
    return dict(pat.findall((ROOT / pack / "keys" / "answers.md").read_text("utf-8")))


KEYS = {pid: val for p in PACKS for pid, val in _key(p).items()}
PACK_OF = {pid: p for p in PACKS for pid in _key(p)}


def test_every_answer_has_a_recorded_second_route() -> None:
    assert set(ROUTES) == set(KEYS)
    assert all(len(note) >= 3 for _f, note in ROUTES.values())


def test_route_a_shares_no_code_with_route_b() -> None:
    src = Path(__file__).read_text("utf-8")
    body = src.split("# ---------------------------------------------------------------- tests")[0]
    assert "verify" not in body.split('"""', 2)[2].replace("verify.py", "")
    assert "scipy" not in body.split('"""', 2)[2].replace("no scipy", "")


def _tol(pid: str) -> float:
    return float(getattr(VERIFY[PACK_OF[pid]], "TOL", {}).get(pid, 1e-9))


@pytest.mark.parametrize("pid", sorted(ROUTES))
def test_route_a_matches_key(pid: str) -> None:
    got = ROUTES[pid][0]()
    written = Fraction(KEYS[pid])
    if isinstance(got, Fraction):
        assert got == written, (pid, got, written)
    else:
        assert got == pytest.approx(float(written), rel=1e-10, abs=1e-12), pid


@pytest.mark.parametrize("pid", sorted(ROUTES))
def test_route_a_matches_route_b(pid: str) -> None:
    a = ROUTES[pid][0]()
    b = VERIFY[PACK_OF[pid]].EXACT[pid]()
    if isinstance(a, Fraction) and isinstance(b, Fraction):
        assert a == b, pid
    else:
        assert float(a) == pytest.approx(float(b), rel=_tol(pid), abs=1e-12), pid


def test_a_wrong_route_a_is_caught() -> None:
    """Mutation check: a closed form off by one part in 10^8 fails the key comparison."""
    assert _bs_call() * (1 + 1e-8) != pytest.approx(float(Fraction(KEYS["D01"])), rel=1e-10)
