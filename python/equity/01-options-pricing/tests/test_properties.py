"""Property-based invariants: homogeneity, monotonicity, continuity of limits,
Inf rejection, and American-exercise dominance relations.

These complement test_black_scholes.py / test_edge_cases.py with checks that
constrain the *shape* of the pricing functions rather than individual values,
so they hold for any correct implementation (and would catch sign errors,
wrong discounting, or broken limit handling immediately).
"""

import math

import pytest

from eq_options import (
    black76_price,
    bs_greeks,
    bs_price,
    crr_price,
    forward_price,
    implied_vol,
    mc_price,
)

INF = float("inf")


# ---------- Inf rejection (CONVENTIONS item 6: NaN/Inf) ----------

@pytest.mark.parametrize(
    ("S", "K", "T", "sigma"),
    [(INF, 100, 1, 0.2), (100, INF, 1, 0.2), (100, 100, INF, 0.2),
     (100, 100, 1, INF), (-INF, 100, 1, 0.2)],
)
def test_infinite_inputs_raise_everywhere(
    S: float, K: float, T: float, sigma: float
) -> None:
    with pytest.raises(ValueError, match="finite"):
        bs_price(S, K, T, 0.05, sigma, 0.0, "call")
    with pytest.raises(ValueError, match="finite"):
        crr_price(S, K, T, 0.05, sigma, 0.0, "call", "european", 50)
    with pytest.raises(ValueError, match="finite"):
        black76_price(S, K, T, 0.05, sigma, "call")
    with pytest.raises(ValueError, match="finite"):
        mc_price(S, K, T, 0.05, sigma, 0.0, "put", n_paths=100)


def test_infinite_or_nan_price_rejected_by_implied_vol() -> None:
    for bad in (INF, -INF, float("nan")):
        with pytest.raises(ValueError):
            implied_vol(bad, 100, 100, 1.0, 0.05, 0.0, "call")


# ---------- homogeneity of degree 1 in (S, K) ----------

@pytest.mark.parametrize("lam", [0.01, 0.5, 3.0, 1000.0])
@pytest.mark.parametrize("otype", ["call", "put"])
def test_bs_price_homogeneous_degree_one(lam: float, otype: str) -> None:
    """V(lam*S, lam*K) = lam * V(S, K) exactly in Black-Scholes."""
    S, K, T, r, sigma, q = 100.0, 95.0, 0.75, 0.03, 0.25, 0.01
    base = bs_price(S, K, T, r, sigma, q, otype)
    scaled = bs_price(lam * S, lam * K, T, r, sigma, q, otype)
    assert scaled == pytest.approx(lam * base, rel=1e-12)


# ---------- monotonicity ----------

def test_put_monotone_increasing_in_vol() -> None:
    vols = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    prices = [bs_price(100, 100, 1, 0.02, v, 0.01, "put") for v in vols]
    assert all(b > a for a, b in zip(prices, prices[1:]))


def test_call_delta_monotone_in_spot() -> None:
    """Convexity in S: call delta is nondecreasing in S."""
    spots = [60.0, 80.0, 100.0, 120.0, 150.0]
    deltas = [bs_greeks(s, 100, 1, 0.03, 0.2, 0.0, "call").delta for s in spots]
    assert all(b > a for a, b in zip(deltas, deltas[1:]))


def test_call_price_monotone_in_expiry_q_zero() -> None:
    """With q=0 and r>=0 a European call is nondecreasing in T."""
    expiries = [0.05, 0.25, 0.5, 1.0, 2.0, 5.0]
    prices = [bs_price(100, 100, t, 0.03, 0.2, 0.0, "call") for t in expiries]
    assert all(b > a for a, b in zip(prices, prices[1:]))


def test_american_put_monotone_in_expiry() -> None:
    """More optionality can never hurt an American holder."""
    expiries = [0.1, 0.5, 1.0, 2.0]
    prices = [
        crr_price(100, 100, t, 0.05, 0.2, 0.01, "put", "american", 400)
        for t in expiries
    ]
    assert all(b >= a - 1e-10 for a, b in zip(prices, prices[1:]))


def test_implied_vol_monotone_in_price() -> None:
    """Higher premium must map to higher implied vol (vega > 0)."""
    S, K, T, r, q = 100.0, 105.0, 0.5, 0.02, 0.0
    prices = [bs_price(S, K, T, r, v, q, "call") for v in (0.15, 0.25, 0.40)]
    ivs = [implied_vol(p, S, K, T, r, q, "call") for p in prices]
    assert ivs[0] < ivs[1] < ivs[2]
    assert ivs == pytest.approx([0.15, 0.25, 0.40], abs=1e-8)


# ---------- continuity of the documented limits ----------

def test_t_to_zero_limit_is_continuous() -> None:
    """price(T->0+) ~ price(T=0) = intrinsic (no jump at the boundary).

    The ATM time value decays like 0.4 * sigma * sqrt(T) * S, which sets
    the attainable tolerance.
    """
    T = 1e-10
    gap = 0.4 * 0.2 * math.sqrt(T) * 105.0  # ATM time-value scale
    for S, K, otype in [(105.0, 100.0, "call"), (95.0, 100.0, "put"),
                        (100.0, 100.0, "call")]:
        tiny = bs_price(S, K, T, 0.05, 0.2, 0.01, otype)
        limit = bs_price(S, K, 0.0, 0.05, 0.2, 0.01, otype)
        assert tiny == pytest.approx(limit, abs=10.0 * gap)


def test_sigma_to_zero_limit_is_continuous() -> None:
    """price(sigma=1e-9) ~ price(sigma=0) = discounted forward intrinsic."""
    for K in (80.0, 100.0, 120.0):
        tiny = bs_price(100, K, 1.0, 0.05, 1e-9, 0.02, "call")
        limit = bs_price(100, K, 1.0, 0.05, 0.0, 0.02, "call")
        assert tiny == pytest.approx(limit, abs=1e-6)


# ---------- American dominance and convexity on the tree ----------

def test_american_at_least_intrinsic_and_european_grid() -> None:
    for S in (70.0, 100.0, 130.0):
        for otype in ("call", "put"):
            amer = crr_price(S, 100, 1.0, 0.05, 0.25, 0.03, otype, "american", 200)
            euro = crr_price(S, 100, 1.0, 0.05, 0.25, 0.03, otype, "european", 200)
            sign = 1.0 if otype == "call" else -1.0
            assert amer >= max(sign * (S - 100.0), 0.0) - 1e-10
            assert amer >= euro - 1e-10


def test_american_put_convex_in_strike() -> None:
    """Butterfly on the tree must be non-negative (arbitrage-free)."""
    h = 5.0
    for K in (90.0, 100.0, 110.0):
        fly = (
            crr_price(100, K - h, 1.0, 0.05, 0.2, 0.0, "put", "american", 300)
            - 2 * crr_price(100, K, 1.0, 0.05, 0.2, 0.0, "put", "american", 300)
            + crr_price(100, K + h, 1.0, 0.05, 0.2, 0.0, "put", "american", 300)
        )
        assert fly >= -1e-10


# ---------- Black-76 shape properties ----------

def test_black76_call_decreasing_convex_in_strike() -> None:
    F, T, r, sigma = 100.0, 1.0, 0.03, 0.25
    strikes = [70.0, 85.0, 100.0, 115.0, 130.0]
    calls = [black76_price(F, k, T, r, sigma, "call") for k in strikes]
    assert all(b < a for a, b in zip(calls, calls[1:]))
    # discrete convexity on the equally spaced grid
    for a, b, c in zip(calls, calls[1:], calls[2:]):
        assert a - 2 * b + c >= -1e-12


def test_black76_price_bounds() -> None:
    """e^{-rT} max(F-K, 0) <= C <= e^{-rT} F."""
    F, K, T, r, sigma = 100.0, 90.0, 2.0, 0.04, 0.5
    c = black76_price(F, K, T, r, sigma, "call")
    df = math.exp(-r * T)
    assert df * max(F - K, 0.0) - 1e-12 <= c <= df * F + 1e-12


# ---------- cross-engine parity on a stressed contract ----------

def test_parity_and_forward_consistency_stressed() -> None:
    """Deep OTM, short-dated, negative rate: all identities still hold."""
    S, K, T, r, sigma, q = 100.0, 160.0, 0.05, -0.01, 0.6, 0.04
    c = bs_price(S, K, T, r, sigma, q, "call")
    p = bs_price(S, K, T, r, sigma, q, "put")
    assert c - p == pytest.approx(
        S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-10
    )
    F = forward_price(S, T, r, q)
    assert black76_price(F, K, T, r, sigma, "call") == pytest.approx(c, abs=1e-10)
