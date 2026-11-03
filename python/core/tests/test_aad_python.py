"""W19: adjoint Greeks against analytic Greeks and bump-and-reprice.

Oracles. Black-Scholes: ``qp_core.black_scholes.carry_greeks`` (closed form)
and central bumps. Monte Carlo: central bumps of the same estimator on the
same normals (common random numbers), and Black-Scholes within three
standard errors for the European payoff. Swap: central bumps of each pillar
``ln P`` and, when ``fi_rates`` is importable, central bumps of each quote
with the curve rebuilt by ``fi_rates.bootstrap.bootstrap_ois_dated``.
The plan's bar (W19) is agreement to 1e-6; it is applied as a relative
tolerance with a small absolute floor for Greeks near zero.
"""

from __future__ import annotations

import math
import pathlib
import time
from types import SimpleNamespace

import numpy as np
import pytest
from qp_core.aad import (
    Swap,
    Tape,
    bs_greeks_aad,
    bs_price,
    exp,
    log,
    mc_greeks_aad,
    mc_price,
    ncdf,
    normals,
    pillar_greeks,
    pv_float,
    quote_greeks,
    sqrt,
    swap_pv,
)
from qp_core.black_scholes import carry_greeks

FIXTURE = (pathlib.Path(__file__).resolve().parents[2]
           / "equity/04-fixed-income/tests/data/dated_bootstrap_usd.txt")
BASE = (100.0, 100.0, 1.0, 0.05, 0.02, 0.20)  # S, K, T, r, q, sigma


def close(a: float, b: float, rel: float = 1e-6, floor: float = 1e-9) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b)) + floor


def central(f, x: list[float], i: int, h: float) -> float:
    up, dn = list(x), list(x)
    up[i] += h
    dn[i] -= h
    return (f(up) - f(dn)) / (2.0 * h)


# ---------------------------------------------------------------- tape ---


def test_tape_elementary_partials():
    t = Tape()
    a, b = t.var(1.3), t.var(0.7)
    y = (a * b - a / b + 2.0 / a - 3.0 + b) * exp(a) + log(b) + sqrt(a) + ncdf(a - b) - (-a)
    g = t.gradient(y, [a, b])

    def f(v):
        x0, x1 = v
        return ((x0 * x1 - x0 / x1 + 2.0 / x0 - 3.0 + x1) * math.exp(x0) + math.log(x1)
                + math.sqrt(x0) + 0.5 * math.erfc(-(x0 - x1) / math.sqrt(2.0)) + x0)

    assert close(y.value, f([1.3, 0.7]), 1e-15)
    for i in range(2):
        assert close(g[i], central(f, [1.3, 0.7], i, 1e-6), 1e-8)


def test_tape_rejects_foreign_output():
    t1, t2 = Tape(), Tape()
    x = t2.var(1.0)
    with pytest.raises(ValueError):
        t1.backward(x * 2.0)


def test_rewind_keeps_input_adjoints():
    t = Tape()
    x = t.var(2.0)
    m = t.mark()
    t.zero_adjoints()
    for _ in range(3):
        t.backward(x * x, stop=m)
        t.rewind(m)
    assert t.adjoint(x) == pytest.approx(12.0) and len(t) == m


# --------------------------------------------------------- Black-Scholes ---


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize(("S", "K", "T", "sigma"), [(100, 100, 1.0, 0.2), (80, 100, 0.25, 0.35),
                                                    (130, 100, 3.0, 0.15)])
def test_bs_matches_analytic_greeks(is_call, S, K, T, sigma):
    r, q = 0.05, 0.02
    a = bs_greeks_aad(S, K, T, r, q, sigma, is_call)
    c = carry_greeks(S, K, T, r, q, sigma, is_call)
    assert close(a.price, c.price, 1e-12)
    assert close(a.delta, c.delta, 1e-12)
    assert close(a.vega, c.vega, 1e-12)
    assert close(a.rho, c.rho, 1e-12)
    assert close(a.rho_q, c.rho_q, 1e-12)
    assert close(-a.dT, c.theta, 1e-12)


def test_bs_matches_bumps():
    a = bs_greeks_aad(*BASE)
    x = list(BASE)

    def f(v):
        return bs_price(*v)

    for i, got in [(0, a.delta), (1, a.dK), (2, a.dT), (3, a.rho), (4, a.rho_q), (5, a.vega)]:
        assert close(got, central(f, x, i, 1e-5 * max(1.0, abs(x[i])))), i


def test_bs_worked_example():
    a = bs_greeks_aad(*BASE)
    assert round(a.price, 4) == 9.2270 and round(a.delta, 4) == 0.5869
    assert round(a.vega, 2) == 37.90


# ------------------------------------------------------------ Monte Carlo ---


@pytest.mark.parametrize("payoff", ["european", "asian"])
def test_mc_pathwise_matches_bumps_same_normals(payoff):
    z = normals(2000, 12)
    a = mc_greeks_aad(*BASE, normals=z, payoff=payoff)
    x = list(BASE)

    def f(v):
        return mc_price(*v, normals=z, payoff=payoff)

    assert close(a.price, f(x), 1e-12)
    for i, got in [(0, a.delta), (1, a.dK), (2, a.dT), (3, a.rho), (4, a.rho_q), (5, a.vega)]:
        assert close(got, central(f, x, i, 1e-6 * max(1.0, abs(x[i])))), (payoff, i, got)


def test_mc_european_converges_to_black_scholes():
    z = normals(20000, 1, seed=7)
    a = mc_greeks_aad(*BASE, normals=z)
    b = bs_greeks_aad(*BASE)
    assert abs(a.price - b.price) < 3 * a.std_error
    assert abs(a.delta - b.delta) < 0.01 and abs(a.vega - b.vega) / b.vega < 0.03


def test_mc_rejects_unknown_payoff():
    with pytest.raises(ValueError):
        mc_greeks_aad(*BASE, normals=normals(2, 2), payoff="lookback")
    with pytest.raises(ValueError):
        mc_price(*BASE, normals=normals(2, 2), payoff="lookback")


# ------------------------------------------------------------------- swap ---


def _parse_usd_ois():
    lines = [ln.split("#", 1)[0].split() for ln in FIXTURE.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    quotes, expect, i, on = [], [], 0, False
    while i < len(lines):
        tok = lines[i]
        i += 1
        if tok[0] == "build":
            on = tok[1] == "usd_ois"
        elif not on:
            continue
        elif tok[0] == "end":
            break
        elif tok[0] == "expect":
            expect.append((int(tok[1]), float(tok[2])))
        elif tok[0] == "fwd":
            quotes.append(SimpleNamespace(start=int(tok[1]), end=int(tok[2]),
                                          accrual=float(tok[3]), rate=float(tok[4])))
        elif tok[0] == "swap":
            nf, nl = int(tok[2]), int(tok[3])
            fx = tuple(SimpleNamespace(pay=int(a[1]), accrual=float(a[2])) for a in lines[i:i + nf])
            fl = tuple(SimpleNamespace(start=int(a[1]), end=int(a[2]), pay=int(a[3]),
                                       accrual=float(a[4])) for a in lines[i + nf:i + nf + nl])
            i += nf + nl
            quotes.append(SimpleNamespace(rate=float(tok[1]), fixed=fx, floating=fl))
    return quotes, expect


QUOTES, EXPECT = _parse_usd_ois()
DAYS = [d for d, _ in EXPECT]
DFS = [v for _, v in EXPECT]
# The test swap: the periods of the SOFR OIS 7Y quote, payer, 4 % fixed, 1m notional.
Q7 = next(q for q in QUOTES if hasattr(q, "fixed") and len(q.fixed) == 7)
SWAP = Swap(Q7.fixed, Q7.floating, 0.04, 1e6, True)


def test_fixture_parsed():
    assert len(QUOTES) == len(DAYS) == 18


def test_swap_pv_float_and_tape_agree():
    t = Tape()
    xs = [t.var(math.log(v)) for v in DFS]
    assert close(swap_pv(DAYS, xs, SWAP).value, pv_float(DAYS, DFS, SWAP), 1e-15)
    rec = Swap(SWAP.fixed, SWAP.floating, SWAP.rate, SWAP.notional, payer=False)
    assert pv_float(DAYS, DFS, rec) == -pv_float(DAYS, DFS, SWAP)


def test_swap_pillar_greeks_match_bumps():
    pg = pillar_greeks(DAYS, DFS, SWAP)
    x = [math.log(v) for v in DFS]

    def f(v):
        return pv_float(DAYS, [math.exp(u) for u in v], SWAP)

    for j in range(len(x)):
        assert close(pg.d_log_df[j], central(f, x, j, 1e-6), 1e-6, 1e-6), j
    assert all(g == 0.0 for d, g in zip(DAYS, pg.d_log_df, strict=True) if d > 2700)


def _to_fi(fb, quotes):
    out = []
    for q in quotes:
        if hasattr(q, "fixed"):
            out.append(fb.SwapQuote(q.rate, tuple(fb.FixedPeriod(f.pay, f.accrual) for f in q.fixed),
                                    tuple(fb.FloatPeriod(p.start, p.end, p.pay, p.accrual)
                                          for p in q.floating)))
        else:
            out.append(fb.ForwardQuote(q.start, q.end, q.accrual, q.rate))
    return out


def test_swap_quote_greeks_match_rebootstrap():
    fb = pytest.importorskip("fi_rates.bootstrap")
    base = fb.bootstrap_ois_dated(_to_fi(fb, QUOTES)).curve
    assert list(base.days) == DAYS
    qg = quote_greeks(QUOTES, DAYS, list(base.dfs), SWAP)
    h = 1e-6
    for k in range(len(QUOTES)):
        pv = []
        for s in (h, -h):
            qs = list(QUOTES)
            qs[k] = SimpleNamespace(**{**vars(QUOTES[k]), "rate": QUOTES[k].rate + s})
            c = fb.bootstrap_ois_dated(_to_fi(fb, qs)).curve
            pv.append(pv_float(list(c.days), list(c.dfs), SWAP))
        bump = (pv[0] - pv[1]) / (2 * h)
        assert close(qg.d_quote[k], bump, 1e-6, 1e-4), (k, qg.d_quote[k], bump)


def test_quote_greeks_need_one_quote_per_pillar():
    with pytest.raises(ValueError):
        quote_greeks(QUOTES[:-1], DAYS, DFS, SWAP)


# ------------------------------------------------------------- cost ratio ---


def _best(fn, reps: int = 5) -> float:
    best = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def test_cost_ratio_reported():
    """Print AAD time / price time; asserted only loosely (shared runners are noisy).

    Pure-Python tape: every operation allocates a ``Var`` and appends four
    list entries, so the ratio is well above the C++/Rust figures. The plan's
    < 5x bar (W19) is checked on the compiled ports; the handback has numbers.
    """
    z = normals(500, 12)
    ratios = {
        "bs": _best(lambda: bs_greeks_aad(*BASE)) / _best(lambda: bs_price(*BASE)),
        "mc": _best(lambda: mc_greeks_aad(*BASE, normals=z), 3)
        / _best(lambda: mc_price(*BASE, normals=z), 3),
        "swap": _best(lambda: pillar_greeks(DAYS, DFS, SWAP)) / _best(lambda: pv_float(DAYS, DFS, SWAP)),
    }
    print("python aad cost ratios", {k: round(v, 2) for k, v in ratios.items()})
    assert all(np.isfinite(v) and v < 200 for v in ratios.values())
