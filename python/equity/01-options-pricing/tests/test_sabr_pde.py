"""Arbitrage-free SABR PDE (Hagan et al. 2014 on a Le Floc'h-Kennedy z grid).

Oracles: exact conservation laws of the scheme (mass and forward), the
lognormal closed form (beta = 1, nu = 0), Hagan's expansion where it is
accurate (short expiry), grid convergence, and no-arbitrage checks on the
prices (non-negative Breeden-Litzenberger density, monotone and convex
calls, exact put-call parity) in a case where the expansion itself fails.
"""

import numpy as np
import pytest
from eq_options.sabr import sabr_density_check, sabr_vol
from eq_options.sabr_pde import sabr_pde

ARB = (0.03, 10.0, 0.02, 0.5, -0.3, 0.6)  # expansion density negative here
SHORT = (0.03, 0.5, 0.02, 0.5, -0.3, 0.3)


@pytest.mark.parametrize("scheme", ["tr_bdf2", "implicit"])
@pytest.mark.parametrize(
    "args", [ARB, SHORT, (1.0, 1.0, 0.2, 1.0, -0.5, 0.5), (0.01, 2.0, 0.005, 0.0, 0.2, 0.3)]
)
def test_mass_and_forward_conserved(scheme, args):
    r = sabr_pde(*args, scheme=scheme)
    assert r.total_mass == pytest.approx(1.0, abs=1e-12)
    assert r.mean == pytest.approx(args[0], abs=1e-12 * max(args[0], 1.0))


@pytest.mark.parametrize("scheme", ["tr_bdf2", "implicit"])
def test_density_non_negative_where_the_expansion_has_arbitrage(scheme):
    _, d_exp = sabr_density_check(*ARB[:2], *ARB[2:], np.linspace(0.0005, 0.1, 200))
    assert d_exp.min() < 0.0  # the expansion fails here
    r = sabr_pde(*ARB, scheme=scheme)
    r.check()
    assert r.mass.min() >= 0.0 and r.p_low >= 0.0 and r.p_high >= 0.0
    assert r.p_low > 0.05  # real mass absorbed at zero
    ks = np.linspace(0.0005, 0.1, 200)
    calls = np.array([r.call(k) for k in ks])
    h = ks[1] - ks[0]
    bl = (calls[:-2] - 2.0 * calls[1:-1] + calls[2:]) / h**2
    assert bl.min() >= -1e-9 * np.abs(bl).max()  # convex in strike
    assert np.all(np.diff(calls) <= 1e-15)  # decreasing in strike


def test_put_call_parity_is_exact():
    r = sabr_pde(*ARB)
    for k in (0.001, 0.02, 0.03, 0.05, 0.2):
        assert r.call(k) - r.put(k) == pytest.approx(0.03 - k, abs=1e-15)


def test_lognormal_limit_is_flat_black_vol():
    r = sabr_pde(1.0, 2.0, 0.25, 1.0, 0.0, 0.0, n_cells=400)
    for k in (0.6, 1.0, 1.5):
        assert r.implied_vol(k) == pytest.approx(0.25, abs=1e-4)


def test_short_expiry_matches_hagan_expansion():
    r = sabr_pde(*SHORT)
    f = SHORT[0]
    assert r.implied_vol(f) == pytest.approx(sabr_vol(f, f, *SHORT[1:], variant="hagan"), abs=2e-5)
    for k in (0.7 * f, 1.3 * f):
        assert r.implied_vol(k) == pytest.approx(sabr_vol(f, k, *SHORT[1:], variant="hagan"), abs=5e-4)


def test_grid_convergence():
    v = [sabr_pde(*ARB, n_cells=n).implied_vol(0.03) for n in (100, 200, 400, 800)]
    d = np.abs(np.diff(v))
    assert d[-1] < 2e-5 and d[0] > d[-1]


def test_tr_bdf2_and_implicit_agree_on_a_fine_grid():
    a = sabr_pde(*ARB, n_cells=400, scheme="tr_bdf2").implied_vol(0.03)
    b = sabr_pde(*ARB, n_cells=400, n_steps=4000, scheme="implicit").implied_vol(0.03)
    assert a == pytest.approx(b, abs=2e-5)


def test_shifted_negative_forward():
    r = sabr_pde(-0.002, 1.0, 0.01, 0.5, 0.0, 0.2, shift=0.03)
    assert r.mean == pytest.approx(0.028, abs=1e-14)
    assert r.implied_vol(0.001) == pytest.approx(
        sabr_vol(-0.002, 0.001, 1.0, 0.01, 0.5, 0.0, 0.2, shift=0.03, variant="hagan"), abs=2e-3
    )


def test_input_validation():
    with pytest.raises(ValueError):
        sabr_pde(0.03, 0.0, 0.02, 0.5, 0.0, 0.3)
    with pytest.raises(ValueError):
        sabr_pde(0.03, 1.0, 0.02, 1.5, 0.0, 0.3)
    with pytest.raises(ValueError):
        sabr_pde(0.03, 1.0, 0.02, 0.5, 0.0, 0.3, scheme="explicit")
    with pytest.raises(ValueError):
        sabr_pde(0.03, 1.0, 0.02, 0.5, 0.0, 0.3, n_cells=5)
    with pytest.raises(ValueError):
        sabr_pde(*SHORT).implied_vol(0.0)  # deep ITM put side: price is intrinsic-bounded
