"""W22: the scalar ``gk_price`` accepts arrays and agrees with itself.

Scalars still give a float; any array, list or tuple argument gives an
array from ``gk_price_array``. Per element the two agree within
1e-15 * max(1, S, K), boundary rows included.
"""

from __future__ import annotations

import numpy as np
from fx_options.garman_kohlhagen import gk_price


def test_array_call_agrees_with_scalar_calls():
    rng = np.random.default_rng(2)
    n = 20_000
    S = rng.uniform(0.5, 2.0, n)
    K = rng.uniform(0.5, 2.0, n)
    T = rng.uniform(1e-3, 5.0, n)
    rd = rng.uniform(-0.01, 0.1, n)
    rf = rng.uniform(-0.01, 0.1, n)
    v = rng.uniform(0.01, 1.0, n)
    T[:4], v[4:8] = 0.0, 0.0
    ot = np.where(rng.random(n) < 0.5, "call", "put")
    arr = gk_price(S, K, T, rd, rf, v, ot)
    assert isinstance(arr, np.ndarray) and arr.shape == S.shape
    ref = np.array([gk_price(float(S[i]), float(K[i]), float(T[i]), float(rd[i]), float(rf[i]),
                             float(v[i]), str(ot[i])) for i in range(n)])
    scale = np.maximum(1.0, np.maximum(S, K))
    assert np.all(np.abs(arr - ref) <= 1e-15 * scale)


def test_scalar_call_still_returns_float():
    assert type(gk_price(1.1, 1.1, 0.5, 0.03, 0.01, 0.1, "call")) is float


def test_list_of_strikes():
    out = gk_price(1.10, [1.05, 1.10, 1.15], 0.25, 0.04, 0.02, 0.08, "put")
    assert out.shape == (3,)
    assert np.all(np.diff(out) > 0)
