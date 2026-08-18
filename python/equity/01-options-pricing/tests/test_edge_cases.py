"""Edge cases: T=0, sigma=0, K=0, S=0, huge vol, invalid inputs, no NaNs."""

import itertools
import math

import numpy as np
import pytest

from eq_options import (
    black76_price,
    bs_greeks,
    bs_price,
    crr_price,
    forward_price,
    mc_price,
)


# ---------- T = 0: intrinsic ----------

@pytest.mark.parametrize(("S", "K"), [(105.0, 100.0), (95.0, 100.0), (100.0, 100.0)])
def test_t_zero_returns_intrinsic_bs(S: float, K: float) -> None:
    assert bs_price(S, K, 0.0, 0.05, 0.2, 0.01, "call") == max(S - K, 0.0)
    assert bs_price(S, K, 0.0, 0.05, 0.2, 0.01, "put") == max(K - S, 0.0)


def test_t_zero_returns_intrinsic_tree_and_mc() -> None:
    assert crr_price(105, 100, 0.0, 0.05, 0.2, 0.0, "call", "american", 100) == 5.0
    assert mc_price(95, 100, 0.0, 0.05, 0.2, 0.0, "put", n_paths=100).value == 5.0


# ---------- sigma = 0: discounted forward intrinsic ----------

def test_sigma_zero_discounted_forward_intrinsic() -> None:
    S, K, T, r, q = 100.0, 95.0, 1.0, 0.05, 0.02
    F = forward_price(S, T, r, q)
    expected = math.exp(-r * T) * max(F - K, 0.0)
    assert bs_price(S, K, T, r, 0.0, q, "call") == pytest.approx(expected, abs=1e-14)
    # OTM-forward call is worthless at zero vol
    assert bs_price(100, 120, 1.0, 0.02, 0.0, 0.0, "call") == 0.0


def test_sigma_zero_tree_matches_bs() -> None:
    for otype in ("call", "put"):
        for K in (80.0, 100.0, 120.0):
            tree = crr_price(100, K, 1.0, 0.05, 0.0, 0.02, otype, "european", 50)
            bs = bs_price(100, K, 1.0, 0.05, 0.0, 0.02, otype)
            assert tree == pytest.approx(bs, abs=1e-12)


def test_sigma_zero_american_put_deterministic() -> None:
    """r>0, sigma=0, ITM put: exercising immediately beats waiting."""
    price = crr_price(80, 100, 1.0, 0.05, 0.0, 0.0, "put", "american", 200)
    assert price == pytest.approx(20.0, abs=1e-10)


# ---------- K = 0 and S = 0 ----------

def test_k_zero_call_is_discounted_forward_put_worthless() -> None:
    S, T, r, q = 100.0, 2.0, 0.03, 0.02
    assert bs_price(S, 0.0, T, r, 0.3, q, "call") == pytest.approx(
        S * math.exp(-q * T), abs=1e-12
    )
    assert bs_price(S, 0.0, T, r, 0.3, q, "put") == 0.0
    assert crr_price(S, 0.0, T, r, 0.3, q, "call", "european", 100) == pytest.approx(
        S * math.exp(-q * T), abs=1e-12
    )


def test_s_zero_put_is_discounted_strike_call_worthless() -> None:
    K, T, r = 100.0, 1.0, 0.04
    assert bs_price(0.0, K, T, r, 0.2, 0.0, "call") == 0.0
    assert bs_price(0.0, K, T, r, 0.2, 0.0, "put") == pytest.approx(
        K * math.exp(-r * T), abs=1e-12
    )
    # American put on a worthless stock: exercise now, get full K
    assert crr_price(0.0, K, T, r, 0.2, 0.0, "put", "american", 100) == K


# ---------- very large sigma ----------

def test_very_large_sigma_call_approaches_discounted_spot() -> None:
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.01
    c = bs_price(S, K, T, r, 20.0, q, "call")
    assert c == pytest.approx(S * math.exp(-q * T), rel=1e-6)
    assert c <= S * math.exp(-q * T)


def test_very_large_sigma_put_approaches_discounted_strike() -> None:
    S, K, T, r = 100.0, 100.0, 1.0, 0.03
    p = bs_price(S, K, T, r, 20.0, 0.0, "put")
    assert p == pytest.approx(K * math.exp(-r * T), rel=1e-6)


# ---------- invalid inputs raise ValueError ----------

@pytest.mark.parametrize(
    ("S", "K", "T", "sigma"),
    [(-1.0, 100, 1, 0.2), (100, -5.0, 1, 0.2), (100, 100, -0.1, 0.2),
     (100, 100, 1, -0.2), (float("nan"), 100, 1, 0.2)],
)
def test_negative_or_nan_inputs_raise_everywhere(
    S: float, K: float, T: float, sigma: float
) -> None:
    with pytest.raises(ValueError):
        bs_price(S, K, T, 0.05, sigma, 0.0, "call")
    with pytest.raises(ValueError):
        crr_price(S, K, T, 0.05, sigma, 0.0, "call", "european", 50)
    with pytest.raises(ValueError):
        black76_price(S, K, T, 0.05, sigma, "call")
    with pytest.raises(ValueError):
        mc_price(S, K, T, 0.05, sigma, 0.0, "call", n_paths=100)


def test_bad_option_type_raises() -> None:
    with pytest.raises(ValueError, match="option_type"):
        bs_price(100, 100, 1, 0.05, 0.2, 0.0, "straddle")  # type: ignore[arg-type]


# ---------- negative rates ----------

def test_negative_rates_all_models_consistent() -> None:
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, -0.02, 0.2, 0.0
    bs = bs_price(S, K, T, r, sigma, q, "call")
    tree = crr_price(S, K, T, r, sigma, q, "call", "european", 1000)
    b76 = black76_price(forward_price(S, T, r, q), K, T, r, sigma, "call")
    mc = mc_price(S, K, T, r, sigma, q, "call", n_paths=100_000, seed=2)
    assert tree == pytest.approx(bs, abs=5e-3)
    assert b76 == pytest.approx(bs, abs=1e-10)
    assert abs(mc.value - bs) <= 3 * mc.std_error


def test_negative_rate_american_call_no_dividend_premium() -> None:
    """With r < 0 and q = 0 early exercise of an American CALL can pay
    (discounted strike grows): premium must be >= 0 and finite."""
    amer = crr_price(120, 100, 1.0, -0.05, 0.2, 0.0, "call", "american", 500)
    euro = crr_price(120, 100, 1.0, -0.05, 0.2, 0.0, "call", "european", 500)
    assert amer >= euro - 1e-12
    assert amer >= 20.0 - 1e-9  # at least intrinsic


# ---------- no NaNs anywhere in the valid domain ----------

def test_no_nans_price_sweep() -> None:
    grid = itertools.product(
        [1e-3, 1.0, 100.0, 1e5],        # S
        [1e-3, 100.0, 1e5],             # K
        [1e-6, 0.05, 1.0, 10.0],        # T
        [-0.05, 0.0, 0.10],             # r
        [1e-6, 0.2, 2.0, 8.0],          # sigma
        [0.0, 0.06],                    # q
        ["call", "put"],
    )
    for S, K, T, r, sigma, q, otype in grid:
        v = bs_price(S, K, T, r, sigma, q, otype)
        assert np.isfinite(v), (S, K, T, r, sigma, q, otype)
        assert v >= -1e-12


def test_no_nans_greeks_sweep() -> None:
    for S in (10.0, 100.0, 1000.0):
        for K in (50.0, 100.0, 200.0):
            for T in (0.01, 0.5, 5.0):
                for sigma in (0.05, 0.3, 1.5):
                    for otype in ("call", "put"):
                        g = bs_greeks(S, K, T, 0.03, sigma, 0.01, otype)
                        for value in g.as_dict().values():
                            assert np.isfinite(value)


def test_no_nans_tree_and_mc_extremes() -> None:
    assert np.isfinite(crr_price(100, 100, 10.0, 0.05, 1.5, 0.0, "put", "american", 2000))
    assert np.isfinite(crr_price(1e-3, 1e5, 0.01, 0.0, 0.05, 0.0, "call", "european", 500))
    res = mc_price(100, 100, 10.0, 0.05, 1.5, 0.0, "call", n_paths=10_000, seed=4)
    assert np.isfinite(res.value) and np.isfinite(res.std_error)
