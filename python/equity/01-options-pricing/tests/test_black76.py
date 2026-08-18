"""Black-76: equivalence with BSM on the forward, Greeks vs finite differences."""

import math

import pytest

from eq_options import (
    b76_d1_d2,
    black76_greeks,
    black76_price,
    bs_price,
    forward_price,
)

CASES = [
    # (S, K, T, r, sigma, q)
    (100.0, 100.0, 1.0, 0.05, 0.20, 0.00),
    (100.0, 120.0, 0.5, 0.03, 0.35, 0.02),
    (80.0, 60.0, 2.0, -0.01, 0.15, 0.04),
    (250.0, 240.0, 0.1, 0.02, 0.25, 0.01),
]


@pytest.mark.parametrize(("S", "K", "T", "r", "sigma", "q"), CASES)
@pytest.mark.parametrize("otype", ["call", "put"])
def test_black76_equals_bsm_on_model_forward(
    S: float, K: float, T: float, r: float, sigma: float, q: float, otype: str
) -> None:
    """Black-76 with F = S e^{(r-q)T} reproduces BSM to 1e-10."""
    F = forward_price(S, T, r, q)
    assert black76_price(F, K, T, r, sigma, otype) == pytest.approx(
        bs_price(S, K, T, r, sigma, q, otype), abs=1e-10
    )


def test_black76_equals_bs_q_eq_r_when_forward_is_spot() -> None:
    """With q = r the forward equals spot and Black-76(F=S) == BS."""
    S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.04, 0.3
    assert black76_price(S, K, T, r, sigma, "call") == pytest.approx(
        bs_price(S, K, T, r, sigma, r, "call"), abs=1e-12
    )


def test_black76_put_call_parity_on_forward() -> None:
    """C - P = e^{-rT} (F - K)."""
    F, K, T, r, sigma = 2000.0, 1950.0, 0.5, 0.03, 0.18
    c = black76_price(F, K, T, r, sigma, "call")
    p = black76_price(F, K, T, r, sigma, "put")
    assert c - p == pytest.approx(math.exp(-r * T) * (F - K), abs=1e-10)


def test_black76_greeks_vs_finite_differences() -> None:
    F, K, T, r, sigma = 100.0, 95.0, 0.75, 0.04, 0.28
    for otype in ("call", "put"):
        g = black76_greeks(F, K, T, r, sigma, otype)
        h = 1e-5 * F

        def price(f: float = F, t: float = T, rr: float = r, s: float = sigma) -> float:
            return black76_price(f, K, t, rr, s, otype)

        assert g.delta == pytest.approx(
            (price(f=F + h) - price(f=F - h)) / (2 * h), rel=1e-6
        )
        assert g.gamma == pytest.approx(
            (price(f=F + h) - 2 * price() + price(f=F - h)) / h**2, rel=1e-4
        )
        hv = 1e-5
        assert g.vega == pytest.approx(
            (price(s=sigma + hv) - price(s=sigma - hv)) / (2 * hv), rel=1e-6
        )
        ht = 1e-6
        assert g.theta == pytest.approx(
            -(price(t=T + ht) - price(t=T - ht)) / (2 * ht), rel=1e-4
        )
        hr = 1e-6
        assert g.rho == pytest.approx(
            (price(rr=r + hr) - price(rr=r - hr)) / (2 * hr), rel=1e-4
        )


def test_black76_rho_is_minus_T_times_price() -> None:
    g = black76_greeks(500.0, 520.0, 1.5, 0.05, 0.22, "put")
    assert g.rho == pytest.approx(-1.5 * g.price, rel=1e-12)


def test_b76_d1_d2_atm_forward() -> None:
    d1, d2 = b76_d1_d2(100.0, 100.0, 4.0, 0.3)
    assert d1 == pytest.approx(0.3, abs=1e-14)  # sigma*sqrt(T)/2 = 0.3*2/2
    assert d2 == pytest.approx(-0.3, abs=1e-14)


def test_black76_edge_cases() -> None:
    # expiry: intrinsic
    assert black76_price(105.0, 100.0, 0.0, 0.05, 0.2, "call") == 5.0
    # zero vol: discounted intrinsic
    assert black76_price(105.0, 100.0, 1.0, 0.05, 0.0, "call") == pytest.approx(
        5.0 * math.exp(-0.05), abs=1e-12
    )
    with pytest.raises(ValueError):
        black76_price(-1.0, 100.0, 1.0, 0.05, 0.2, "call")
