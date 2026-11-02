"""bs_price_array / bs_greeks_array against the scalar reference."""

import itertools
import time

import numpy as np
import pytest
from eq_options import BSGreeks, bs_greeks, bs_greeks_array, bs_price, bs_price_array

GRID = list(
    itertools.product(
        [0.0, 50.0, 99.5, 100.0, 180.0],
        [0.0, 80.0, 100.0, 140.0],
        [0.0, 1 / 365, 0.5, 3.0],
        [0.0, 0.05, 0.3, 1.5],
        [-0.01, 0.04],
        ["call", "put"],
    )
)


def _cols(rows):
    S, K, T, sig, r, ot = (np.array(c) for c in zip(*rows, strict=False))
    return S, K, T, sig, r, ot


def test_price_matches_scalar_on_boundary_grid():
    S, K, T, sig, r, ot = _cols(GRID)
    got = bs_price_array(S, K, T=T, r=r, sigma=sig, q=0.015, option_type=ot)
    want = np.array([bs_price(*row[:3], row[4], row[3], 0.015, row[5]) for row in GRID])
    tol = 1e-12 * np.maximum(1.0, np.maximum(S, K))
    assert np.all(np.abs(got - want) <= tol)


def test_greeks_match_scalar():
    rows = [g for g in GRID if min(g[0], g[1], g[2], g[3]) > 0]
    S, K, T, sig, r, ot = _cols(rows)
    got = bs_greeks_array(S, K, T=T, r=r, sigma=sig, q=0.015, option_type=ot)
    assert set(got) == {f for f in BSGreeks.__dataclass_fields__}
    for i, row in enumerate(rows):
        g = bs_greeks(row[0], row[1], row[2], row[4], row[3], 0.015, row[5])
        for f, v in got.items():
            assert v[i] == pytest.approx(getattr(g, f), rel=1e-11, abs=1e-12 * max(1, row[0], row[1]))


def test_broadcasts_scalars_and_shapes():
    K = np.linspace(80, 120, 5).reshape(5, 1)
    sig = np.array([0.1, 0.2, 0.3])
    p = bs_price_array(100.0, K, T=1.0, r=0.02, sigma=sig)
    assert p.shape == (5, 3)
    assert p[2, 1] == pytest.approx(bs_price(100.0, 100.0, 1.0, 0.02, 0.2), rel=1e-14)


def test_arguments_after_k_are_keyword_only():
    with pytest.raises(TypeError):
        bs_price_array(100.0, 100.0, 1.0, 0.02, 0.2)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "msg"),
    [("S", -1.0, "S must be"), ("sigma", np.nan, "sigma must be"), ("r", np.inf, "r must be finite")],
)
def test_reports_field_and_first_bad_index(field, value, msg):
    kw = {"S": np.full(4, 100.0), "K": 100.0, "T": 1.0, "r": np.zeros(4), "sigma": np.full(4, 0.2)}
    kw[field] = np.array(kw[field], dtype=float) * np.ones(4)
    kw[field][2] = value
    S, K = kw.pop("S"), kw.pop("K")
    with pytest.raises(ValueError, match=f"{msg}.*index 2"):
        bs_price_array(S, K, **kw)


def test_bad_option_type_and_overflow():
    with pytest.raises(ValueError, match="option_type"):
        bs_price_array(100.0, 100.0, T=1.0, r=0.0, sigma=0.2, option_type=["call", "straddle"])
    with pytest.raises(ValueError, match="finite double at index 1"):
        bs_price_array(100.0, 100.0, T=1.0, r=np.array([0.0, -800.0]), sigma=0.2, option_type="put")
    with pytest.raises(ValueError, match="> 0"):
        bs_greeks_array(100.0, 100.0, T=0.0, r=0.0, sigma=0.2)


def test_array_is_faster_than_a_scalar_loop():
    # Loose floor so the test is stable on a loaded CI runner; the measured
    # ratio is in docs/METHODOLOGY.md.
    rng = np.random.default_rng(1)
    n = 20000
    S, T, sig = 100 * np.exp(rng.uniform(-0.3, 0.3, n)), rng.uniform(0.02, 2, n), rng.uniform(0.05, 0.6, n)
    t0 = time.perf_counter()
    loop = [
        bs_price(s, 100.0, t, 0.03, v) for s, t, v in zip(S.tolist(), T.tolist(), sig.tolist(), strict=False)
    ]
    t1 = time.perf_counter()
    arr = bs_price_array(S, 100.0, T=T, r=0.03, sigma=sig)
    t2 = time.perf_counter()
    assert np.allclose(arr, loop, rtol=1e-13, atol=1e-12)
    assert (t1 - t0) > 5 * (t2 - t1)
