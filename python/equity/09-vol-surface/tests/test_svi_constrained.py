"""Constrained SVI: Lee wing bound and Durrleman no-butterfly at the optimum.

The old fitter was box-bounded least squares that only warned after the
fact; these tests fail on it (it returns the arbitrageable slice).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from eq_surface.smile import (
    LEE_SLOPE_MAX,
    SVIParams,
    durrleman_g,
    fit_svi,
    svi_total_variance,
)


def _dense_min_g(p: SVIParams, k_lo: float, k_hi: float) -> float:
    k = np.concatenate([np.linspace(k_lo, k_hi, 4001), p.m + p.sigma * np.linspace(-6, 6, 601)])
    return float(np.min(np.asarray(durrleman_g(k, p))))


def test_v_shaped_quotes_fit_arbitrage_free():
    k = np.linspace(-0.4, 0.4, 11)
    w_bad = 0.04 + 0.5 * np.abs(k)
    res = fit_svi(k, w_bad, T=1.0)
    assert res.arb_free and res.lee_ok
    assert "butterfly" in res.active_constraints
    assert _dense_min_g(res.params, -1.4, 1.4) >= -1e-10
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loose = fit_svi(k, w_bad, T=1.0, arbitrage_free=False)
    assert loose.min_g < -1e-3  # the old path keeps the arbitrage
    assert res.rmse_vol > loose.rmse_vol  # the constraint costs fit quality


def test_steep_wings_respect_lee_bound():
    """Quotes from an SVI with wing slope b(1+|rho|) = 3 > 2 (Lee) on a wide
    strike range: the constrained fit lands on the bound, the old fit does not."""
    bad = SVIParams(a=0.02, b=2.0, rho=-0.5, m=0.0, sigma=0.2)
    k = np.linspace(-1.5, 1.0, 21)
    w = np.asarray(svi_total_variance(k, bad))
    res = fit_svi(k, w, T=2.0)
    p = res.params
    assert p.b * (1.0 + abs(p.rho)) <= LEE_SLOPE_MAX + 1e-9
    assert res.active_constraints  # a constraint binds: the data want more slope
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loose = fit_svi(k, w, T=2.0, arbitrage_free=False)
    assert loose.params.b * (1.0 + abs(loose.params.rho)) > 2.5


def test_default_weights_are_vol_space_and_custom_weights_validated():
    k = np.linspace(-0.5, 0.5, 9)
    w = 0.04 + 0.05 * k**2
    with pytest.raises(ValueError, match="weights"):
        fit_svi(k, w, T=1.0, weights=np.full(9, -1.0))
    res = fit_svi(k, w, T=1.0, weights=np.ones(9))
    assert res.arb_free


@settings(max_examples=12, deadline=None)
@given(
    a=st.floats(0.005, 0.05),
    b=st.floats(0.02, 0.4),
    rho=st.floats(-0.8, 0.3),
    m=st.floats(-0.1, 0.1),
    sig=st.floats(0.05, 0.4),
)
def test_fit_recovers_arbitrage_free_svi(a, b, rho, m, sig):
    """Property: quotes from an arbitrage-free SVI are fitted to < 1e-4 vol
    and the fit is itself arbitrage-free on a dense grid."""
    true = SVIParams(a, b, rho, m, sig)
    k = np.linspace(-0.6, 0.4, 15)
    if _dense_min_g(true, -1.6, 1.4) < 1e-6:
        return  # the sampled slice itself has butterfly arbitrage
    w = np.asarray(svi_total_variance(k, true))
    res = fit_svi(k, w, T=1.0)
    assert res.rmse_vol < 1e-4
    assert res.lee_ok
    assert _dense_min_g(res.params, -1.6, 1.4) >= -1e-10
