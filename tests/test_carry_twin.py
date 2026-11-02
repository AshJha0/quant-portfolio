"""Equity with dividend yield q and FX with foreign rate r_f are one model.

Black-Scholes-Merton with a continuous yield and Garman-Kohlhagen are the
same formula in cost-of-carry form, b = r - q = r_d - r_f. These tests pin
that the two packages agree, and say how closely:

* bit for bit: price, d1/d2, gamma, vega, vanna, volga (scalar and array),
  and every array Greek except theta;
* within a few ulps of max(1, S, K, |value|): the scalar Greeks price, delta, rho and
  theta (the FX package evaluates N(x) through scipy.stats.norm.cdf, the
  equity package through 0.5 erfc(-x / sqrt 2)), and the array theta (the two
  packages add the theta terms in a different order);
* implied vol to max(1e-12 sigma, 64 eps price / vega) where both solvers
  return a positive vol
  (the packages form the forward differently: S e^{(r-q)T} against
  S e^{-r_f T} / e^{-r_d T}).

The tolerances are the ones measured before the shared kernel existed; the
shared kernel must not loosen any of them.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

eq = pytest.importorskip("eq_options")
fx = pytest.importorskip("fx_options")

from eq_options.arrays import bs_greeks_array, bs_price_array  # noqa: E402
from eq_options.black_scholes import bs_price, d1_d2  # noqa: E402
from eq_options.black_scholes import implied_vol as eq_iv  # noqa: E402
from eq_options.greeks import bs_greeks  # noqa: E402
from fx_options.arrays import gk_greeks_array, gk_price_array  # noqa: E402
from fx_options.garman_kohlhagen import d1, d2, gk_price  # noqa: E402
from fx_options.garman_kohlhagen import implied_vol as fx_iv  # noqa: E402
from fx_options.greeks import analytic_greeks  # noqa: E402

EPS = 2.220446049250313e-16

GRID = list(
    itertools.product(
        [50.0, 100.0, 137.3],
        [80.0, 100.0, 125.0],
        [1 / 365, 0.25, 2.0, 10.0],
        [-0.01, 0.0, 0.05],
        [0.0, 0.03, -0.005],
        [0.05, 0.2, 0.8],
        ["call", "put"],
    )
)
PAIRS = [
    ("price", "price"),
    ("delta", "delta_spot"),
    ("gamma", "gamma"),
    ("vega", "vega"),
    ("theta", "theta"),
    ("rho", "rho_domestic"),
    ("vanna", "vanna"),
    ("volga", "volga"),
]
BIT_EXACT_SCALAR = {"gamma", "vega", "vanna", "volga"}


def _cols():
    cols = [np.array([g[j] for g in GRID]) for j in range(6)]
    return (*cols, np.array([g[6] for g in GRID]))


@pytest.mark.parametrize(("S", "K", "T", "r", "q", "v", "ot"), GRID[::7])
def test_scalar_price_and_d1_are_bit_identical(S, K, T, r, q, v, ot):
    assert bs_price(S, K, T, r, v, q, ot) == gk_price(S, K, T, r, q, v, ot)
    assert d1_d2(S, K, T, r, v, q) == (d1(S, K, T, r, q, v), d2(S, K, T, r, q, v))


def test_scalar_price_and_d1_bit_identical_on_full_grid():
    for S, K, T, r, q, v, ot in GRID:
        assert bs_price(S, K, T, r, v, q, ot) == gk_price(S, K, T, r, q, v, ot)
        assert d1_d2(S, K, T, r, v, q)[0] == d1(S, K, T, r, q, v)


def test_scalar_boundary_branches_agree():
    # T == 0 and sigma == 0 take explicit branches in both packages.
    for ot in ("call", "put"):
        assert bs_price(100.0, 90.0, 0.0, 0.05, 0.2, 0.02, ot) == gk_price(
            100.0, 90.0, 0.0, 0.05, 0.02, 0.2, ot
        )
        assert bs_price(100.0, 90.0, 1.0, 0.05, 0.0, 0.02, ot) == gk_price(
            100.0, 90.0, 1.0, 0.05, 0.02, 0.0, ot
        )


def test_scalar_greeks_agree():
    for S, K, T, r, q, v, ot in GRID:
        e = bs_greeks(S, K, T, r, v, q, ot)
        f = analytic_greeks(S, K, T, r, q, v, ot)
        for a, b in PAIRS:
            x, y = getattr(e, a), getattr(f, b)
            if a in BIT_EXACT_SCALAR:
                assert x == y, (a, S, K, T, r, q, v, ot)
            else:
                assert abs(x - y) <= 8 * EPS * max(1.0, S, K, abs(x)), (a, x, y)
        # FX's foreign rho is the equity dividend rho: -phi S T e^{-qT} N(phi d1).
        assert f.rho_foreign == pytest.approx(-T * S * e.delta, rel=1e-14, abs=1e-14)


def test_array_prices_bit_identical():
    S, K, T, r, q, v, ot = _cols()
    a = bs_price_array(S, K, T=T, r=r, sigma=v, q=q, option_type=ot)
    b = gk_price_array(S, K, T=T, r_d=r, r_f=q, sigma=v, option_type=ot)
    assert np.array_equal(a, b)


def test_array_greeks_bit_identical_except_theta():
    S, K, T, r, q, v, ot = _cols()
    e = bs_greeks_array(S, K, T=T, r=r, sigma=v, q=q, option_type=ot)
    f = gk_greeks_array(S, K, T=T, r_d=r, r_f=q, sigma=v, option_type=ot)
    for a, b in PAIRS:
        if a == "theta":
            scale = np.maximum(1.0, np.abs(e[a]))
            assert np.all(np.abs(e[a] - f[b]) <= 4 * EPS * scale)
        else:
            assert np.array_equal(e[a], f[b]), a


def test_implied_vol_agrees():
    checked = 0
    for S, K, T, r, q, v, ot in GRID:
        p = bs_price(S, K, T, r, v, q, ot)
        try:
            a = eq_iv(p, S, K, T, r, q, ot)
            b = fx_iv(p, S, K, T, r, q, ot)
        except ValueError:
            continue  # quote too close to a bound for one of the two rules
        if a > 0.0 and b > 0.0:
            # The information limit of the quote: one ulp of the price moves
            # the vol by about eps * price / vega.
            vega = bs_greeks(S, K, T, r, v, q, ot).vega
            tol = max(1e-12 * v, 64 * EPS * max(p, 1e-300) / vega)
            assert abs(a - b) <= tol, (S, K, T, r, q, v, ot, a, b)
            checked += 1
    assert checked > 1000
