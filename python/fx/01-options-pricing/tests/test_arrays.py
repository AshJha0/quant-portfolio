"""gk_price_array against the scalar gk_price."""

import itertools

import numpy as np
import pytest
from fx_options import gk_price, gk_price_array

GRID = list(
    itertools.product(
        [0.6, 1.08, 1.1, 150.0],
        [0.5, 1.0, 1.1, 160.0],
        [0.0, 1 / 365, 0.5, 3.0],
        [0.0, 1e-13, 0.05, 0.3, 1.5],
        [-0.01, 0.04],
        ["call", "put"],
    )
)


def test_matches_scalar_on_boundary_grid():
    S, K, T, sig, rd, ot = (np.array(c) for c in zip(*GRID, strict=False))
    got = gk_price_array(S, K, T=T, r_d=rd, r_f=0.02, sigma=sig, option_type=ot)
    want = np.array([gk_price(s, k, t, d, 0.02, v, o) for s, k, t, v, d, o in GRID])
    assert np.all(np.abs(got - want) <= 1e-12 * np.maximum(1.0, np.maximum(S, K)))


def test_keyword_only_and_broadcast():
    with pytest.raises(TypeError):
        gk_price_array(1.1, 1.1, 1.0, 0.02, 0.01, 0.1, "call")  # type: ignore[misc]
    p = gk_price_array(
        1.1, np.array([1.0, 1.1, 1.2]), T=0.5, r_d=0.03, r_f=0.01, sigma=0.1, option_type="put"
    )
    assert p.shape == (3,)
    assert p[1] == pytest.approx(gk_price(1.1, 1.1, 0.5, 0.03, 0.01, 0.1, "put"), rel=1e-14)


def test_errors_name_field_and_index():
    with pytest.raises(ValueError, match=r"K is invalid.*index 1"):
        gk_price_array(1.1, [1.0, 0.0], T=1.0, r_d=0.0, r_f=0.0, sigma=0.1, option_type="call")
    with pytest.raises(ValueError, match="option_type"):
        gk_price_array(1.1, 1.0, T=1.0, r_d=0.0, r_f=0.0, sigma=0.1, option_type="digital")
    with pytest.raises(ValueError, match="finite double at index 0"):
        gk_price_array([1.1], 1.0, T=1.0, r_d=-800.0, r_f=0.0, sigma=0.1, option_type="put")


def test_greeks_match_scalar():
    from fx_options import GreeksResult, analytic_greeks, gk_greeks_array

    rows = [g for g in GRID if g[3] * g[2] ** 0.5 > 1e-12]
    S, K, T, sig, rd, ot = (np.array(c) for c in zip(*rows, strict=True))
    got = gk_greeks_array(S, K, T=T, r_d=rd, r_f=0.02, sigma=sig, option_type=ot)
    assert set(got) == set(GreeksResult.__dataclass_fields__)
    for i, (s, k, t, v, d, o) in enumerate(rows):
        g = analytic_greeks(s, k, t, d, 0.02, v, o)
        for f, arr in got.items():
            assert arr[i] == pytest.approx(getattr(g, f), rel=1e-11, abs=1e-12 * max(1, s, k)), f
    with pytest.raises(ValueError, match=r"sigma must be > 0.*index 0"):
        gk_greeks_array(1.1, 1.0, T=1.0, r_d=0.0, r_f=0.0, sigma=0.0, option_type="call")
    with pytest.raises(ValueError, match="d1 undefined"):
        gk_greeks_array(1.1, 1.0, T=1.0, r_d=0.0, r_f=0.0, sigma=1e-13, option_type="put")
    with pytest.raises(TypeError):
        gk_greeks_array(1.1, 1.0, 1.0, 0.0, 0.0, 0.1, "call")  # type: ignore[misc]
