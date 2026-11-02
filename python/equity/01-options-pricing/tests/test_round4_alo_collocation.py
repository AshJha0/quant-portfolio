"""Round 4: Chebyshev collocation for the ALO double boundary (q < r < 0 put).

The 400-step march stays as the reference; the extrapolated Leisen-Reimer
tree is the independent check.
"""

import time

import numpy as np
import pytest
from eq_options import alo as alo_mod
from eq_options.alo import alo_double_boundary, alo_price
from eq_options.lr_tree import lr_price


def _lr_richardson(*a, n=2001):
    return 2.0 * lr_price(*a, "american", 2 * n - 1) - lr_price(*a, "american", n)


GRID = [
    (100.0, 100.0, 1.0, -0.005, 0.2, -0.02),
    (100.0, 100.0, 1.0, -0.01, 0.1, -0.03),
    (95.0, 100.0, 3.0, -0.01, 0.2, -0.02),     # band closes before T
    (80.0, 100.0, 0.5, -0.01, 0.3, -0.03),
    (120.0, 100.0, 2.0, -0.002, 0.15, -0.01),
]


@pytest.mark.parametrize("a", GRID)
def test_collocation_matches_march_to_1e5(a):
    S, K, T, r, s, q = a
    col = alo_price(S, K, T, r, s, q, "put")
    march = alo_price(S, K, T, r, s, q, "put", double_boundary_method="march")
    assert col == pytest.approx(march, abs=1e-5 * K / 100.0 * 2)  # march itself is ~1e-5


@pytest.mark.parametrize("a", GRID)
def test_collocation_matches_extrapolated_tree(a):
    assert alo_price(*a, "put") == pytest.approx(_lr_richardson(*a, "put"), abs=1e-5)


def test_collocation_long_maturity_band_closed():
    # T = 30 >> tau* ~ 11: the 1600-step march Richardson value is 34.282374
    a = (100.0, 100.0, 30.0, -0.005, 0.2, -0.02)
    assert alo_price(*a, "put") == pytest.approx(34.282374, abs=2e-5)


def test_collocation_converges_in_nodes():
    a = (95.0, 100.0, 3.0, -0.01, 0.2, -0.02, "put")
    ref = alo_price(*a, n_cheb=14, n_quad_db=40)
    e = [abs(alo_price(*a, n_cheb=n) - ref) for n in (4, 8)]
    assert e[1] < e[0] and e[1] < 2e-6


def test_closing_time_agrees_with_march():
    K, T, r, s, q = 100.0, 3.0, -0.01, 0.2, -0.02
    u_m, lo_m, hi_m = alo_double_boundary(K, T, r, s, q, method="march")
    u_c, lo_c, hi_c = alo_double_boundary(K, T, r, s, q)
    # the march stops at the last grid time before tau*; collocation one step short of tau*
    assert u_c[-1] == pytest.approx(u_m[-1], rel=2e-2)
    assert lo_c[0] == pytest.approx(K * r / q) and hi_c[0] == K
    assert np.all(hi_c > lo_c)
    # interior boundary values agree with the march to well under 0.1 % of strike
    hi_i = np.interp(u_m[1:-5], u_c, hi_c)
    lo_i = np.interp(u_m[1:-5], u_c, lo_c)
    assert np.max(np.abs(hi_i - hi_m[1:-5])) < 0.1
    assert np.max(np.abs(lo_i - lo_m[1:-5])) < 0.1


def test_collocation_call_symmetry():
    a = (100.0, 100.0, 1.0, -0.03, 0.2, -0.01)
    assert alo_price(*a, "call") == pytest.approx(
        alo_price(100.0, 100.0, 1.0, -0.01, 0.2, -0.03, "put"), abs=1e-12)


def test_collocation_inside_band_is_intrinsic():
    K, T, r, s, q = 100.0, 0.25, -0.01, 0.2, -0.02
    u, lo, hi = alo_double_boundary(K, T, r, s, q)
    assert u[-1] == T
    s_in = 0.5 * (lo[-1] + hi[-1])
    assert alo_price(s_in, K, T, r, s, q, "put") == K - s_in


def test_falls_back_to_march_when_solve_fails(monkeypatch):
    a = (100.0, 100.0, 1.0, -0.005, 0.2, -0.02, "put")
    march = alo_price(*a, double_boundary_method="march")
    monkeypatch.setattr(alo_mod, "_collocate", lambda *args, **kw: None)
    assert alo_price(*a) == march


def test_bad_method_and_node_counts_rejected():
    a = (100.0, 100.0, 1.0, -0.005, 0.2, -0.02, "put")
    with pytest.raises(ValueError, match="double_boundary_method"):
        alo_price(*a, double_boundary_method="tree")
    with pytest.raises(ValueError, match="n_cheb"):
        alo_price(*a, n_cheb=2)
    with pytest.raises(ValueError, match="double_boundary_method"):
        alo_double_boundary(100.0, 1.0, -0.005, 0.2, -0.02, method="x")


def _best_time(fn, n):
    best = np.inf
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def test_collocation_is_fast():
    """Speed guard: about 15-20 ms a price on a quiet core, ~10x the march.

    A fixed wall-clock bound alone failed on a busy machine (parallel test
    runs pushed a 15 ms price to 70-150 ms), so the main check is relative:
    collocation must beat the 400-step march, timed in the same process
    and interleaved so both see the same load, by at least 3x. The loose
    absolute bound still catches a fall back to the march or a fine grid.
    """
    a = (100.0, 100.0, 1.0, -0.005, 0.2, -0.02, "put")
    alo_price(*a)  # warm-up: imports and first-call allocation
    col, march = np.inf, np.inf
    for _ in range(3):
        col = min(col, _best_time(lambda: alo_price(*a), 5))
        march = min(march, _best_time(
            lambda: alo_price(*a, double_boundary_method="march"), 1))
    assert col < march / 3.0
    assert col < 0.5


@pytest.mark.parametrize("n_march", [1600])
def test_fine_march_does_not_stop_at_first_step(n_march):
    # Regression: before the expm1 / parity form of the march residual, a
    # 1600-step grid stopped after one node here (the first nodes sit at
    # tau ~ 2e-7 and K - x - put cancelled to noise) and returned a wrong price.
    a = (80.0, 100.0, 0.5, -0.01, 0.3, -0.03)
    u, B, Y = alo_mod._double_boundary(100.0, 0.5, -0.01, -0.03, 0.3, n_march)
    assert u.size == n_march + 1 and u[-1] == pytest.approx(0.5)
    assert np.all(np.diff(B) <= 0.0) and np.all(np.diff(Y) >= 0.0)
    p = alo_price(*a, option_type="put", n_march=n_march, double_boundary_method="march")
    ref = alo_price(*a, option_type="put")
    assert p == pytest.approx(ref, abs=1e-5)
