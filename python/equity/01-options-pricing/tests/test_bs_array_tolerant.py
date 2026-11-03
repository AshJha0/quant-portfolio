"""W22: the scalar ``bs_price`` accepts arrays and agrees with itself.

Scalars still give a float (the scalar API is kept); any array, list or
tuple argument gives an array from ``bs_price_array``. Per element the two
agree within 1e-15 * max(1, S, K), boundary rows included.
"""

from __future__ import annotations

import numpy as np
import pytest
from eq_options import bs_price


def _grid(n: int = 20_000, seed: int = 1):
    rng = np.random.default_rng(seed)
    S = rng.uniform(1, 500, n)
    K = rng.uniform(1, 500, n)
    T = rng.uniform(1e-3, 10, n)
    r = rng.uniform(-0.05, 0.15, n)
    v = rng.uniform(0.01, 2.0, n)
    q = rng.uniform(-0.05, 0.15, n)
    # Boundary rows: T, sigma, S and K zero.
    T[:4], v[4:8], S[8:12], K[12:16] = 0.0, 0.0, 0.0, 0.0
    ot = np.where(rng.random(n) < 0.5, "call", "put")
    return S, K, T, r, v, q, ot


def test_array_call_agrees_with_scalar_calls():
    S, K, T, r, v, q, ot = _grid()
    arr = bs_price(S, K, T, r, v, q, ot)
    assert isinstance(arr, np.ndarray) and arr.shape == S.shape
    ref = np.array([bs_price(float(S[i]), float(K[i]), float(T[i]), float(r[i]), float(v[i]),
                             float(q[i]), str(ot[i])) for i in range(S.size)])
    scale = np.maximum(1.0, np.maximum(S, K))
    assert np.all(np.abs(arr - ref) <= 1e-15 * scale)


def test_scalar_call_still_returns_float():
    p = bs_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.02, "call")
    assert type(p) is float
    assert p == pytest.approx(9.227005508154036, abs=1e-12)


def test_broadcasting_and_lists():
    strikes = [90.0, 100.0, 110.0]
    out = bs_price(100.0, strikes, 0.5, 0.03, 0.2)
    assert out.shape == (3,)
    for k, p in zip(strikes, out, strict=True):
        assert abs(p - bs_price(100.0, k, 0.5, 0.03, 0.2)) <= 1e-15 * max(1.0, 100.0, k)


def test_array_errors_name_the_index():
    with pytest.raises(ValueError, match=r"sigma .* at index 1"):
        bs_price(np.array([100.0, 100.0]), 100.0, 1.0, 0.0, np.array([0.2, -0.1]))
