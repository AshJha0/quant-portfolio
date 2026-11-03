"""W21: ``engine.bs_price_batch`` runs the fast C++ kernel by default and
the bit-exact one with ``exact=True``; Python and Rust are always exact."""

from __future__ import annotations

import numpy as np
import pytest
from eq_options.engine import bs_price_batch
from qp_core.engines import engine_available


def _cols(n=3000, seed=5):
    rng = np.random.default_rng(seed)
    S, K = rng.uniform(1, 500, n), rng.uniform(1, 500, n)
    T, r = rng.uniform(1e-3, 10, n), rng.uniform(-0.05, 0.15, n)
    v, q = rng.uniform(0.01, 2.0, n), rng.uniform(-0.05, 0.15, n)
    return S, K, T, r, v, q, rng.random(n) < 0.5


@pytest.mark.skipif(not engine_available("cpp"), reason="qp_cpp not installed")
def test_cpp_default_fast_within_bound_and_exact_matches_python():
    S, K, T, r, v, q, c = _cols()
    ref = bs_price_batch(S, K, T, r, v, q, c)
    fast = bs_price_batch(S, K, T, r, v, q, c, engine="cpp")
    exact = bs_price_batch(S, K, T, r, v, q, c, engine="cpp", exact=True)
    scale = np.maximum(1.0, np.maximum(S, K))
    assert np.all(np.abs(fast - exact) <= 1e-13 * scale)
    # The exact C++ batch equals the C++ scalar; against the Python reference
    # the cross-language tolerance of engine.py applies (64 eps (1 + d1^2) ...),
    # bounded here by the looser 1e-12 * max(1, S, K).
    assert np.all(np.abs(exact - ref) <= 1e-12 * scale)


@pytest.mark.skipif(not engine_available("rust"), reason="qp_rust not installed")
def test_rust_ignores_exact_flag():
    S, K, T, r, v, q, c = _cols(200)
    a = bs_price_batch(S, K, T, r, v, q, c, engine="rust")
    b = bs_price_batch(S, K, T, r, v, q, c, engine="rust", exact=True)
    assert np.array_equal(a, b)
