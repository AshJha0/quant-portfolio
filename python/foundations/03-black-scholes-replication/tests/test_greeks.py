"""Analytic Greeks vs central finite differences of the price function,
for both calls and puts, plus the put-call-parity Greek relations
documented in ``eq_bs_replication.black_scholes.put_greeks``.
"""
import math

from eq_bs_replication import call_greeks, call_price, put_greeks, put_price

S, K, r, sigma, T = 100.0, 105.0, 0.03, 0.25, 0.75


def test_call_greeks_match_finite_differences():
    g = call_greeks(S, K, r, sigma, T)
    h = 1e-4
    delta_fd = (call_price(S + h, K, r, sigma, T)
                - call_price(S - h, K, r, sigma, T)) / (2 * h)
    vega_fd = (call_price(S, K, r, sigma + h, T)
               - call_price(S, K, r, sigma - h, T)) / (2 * h)
    gamma_fd = (call_price(S + h, K, r, sigma, T)
                - 2 * call_price(S, K, r, sigma, T)
                + call_price(S - h, K, r, sigma, T)) / h**2
    rho_fd = (call_price(S, K, r + h, sigma, T)
              - call_price(S, K, r - h, sigma, T)) / (2 * h)
    # theta here is dV/dt (calendar time); T is time-to-expiry, so
    # dV/dt = -dV/dT.
    theta_fd = -(call_price(S, K, r, sigma, T + h)
                 - call_price(S, K, r, sigma, T - h)) / (2 * h)
    assert abs(g.delta - delta_fd) < 1e-6
    assert abs(g.vega - vega_fd) < 1e-4
    assert abs(g.gamma - gamma_fd) < 1e-4
    assert abs(g.rho - rho_fd) < 1e-4
    assert abs(g.theta - theta_fd) < 1e-4


def test_put_greeks_match_finite_differences():
    g = put_greeks(S, K, r, sigma, T)
    h = 1e-4
    delta_fd = (put_price(S + h, K, r, sigma, T)
                - put_price(S - h, K, r, sigma, T)) / (2 * h)
    vega_fd = (put_price(S, K, r, sigma + h, T)
               - put_price(S, K, r, sigma - h, T)) / (2 * h)
    gamma_fd = (put_price(S + h, K, r, sigma, T)
                - 2 * put_price(S, K, r, sigma, T)
                + put_price(S - h, K, r, sigma, T)) / h**2
    rho_fd = (put_price(S, K, r + h, sigma, T)
              - put_price(S, K, r - h, sigma, T)) / (2 * h)
    theta_fd = -(put_price(S, K, r, sigma, T + h)
                 - put_price(S, K, r, sigma, T - h)) / (2 * h)
    assert abs(g.delta - delta_fd) < 1e-6
    assert abs(g.vega - vega_fd) < 1e-4
    assert abs(g.gamma - gamma_fd) < 1e-4
    assert abs(g.rho - rho_fd) < 1e-4
    assert abs(g.theta - theta_fd) < 1e-4


def test_put_greeks_parity_relations():
    # Cross-check put_greeks (derived via put-call parity) directly
    # against the parity identities stated in its docstring.
    c = call_greeks(S, K, r, sigma, T)
    p = put_greeks(S, K, r, sigma, T)
    disc_k = K * math.exp(-r * T)
    assert abs(p.delta - (c.delta - 1.0)) < 1e-12
    assert abs(p.gamma - c.gamma) < 1e-12
    assert abs(p.vega - c.vega) < 1e-12
    assert abs(p.theta - (c.theta + r * disc_k)) < 1e-12
    assert abs(p.rho - (c.rho - T * disc_k)) < 1e-12


def test_delta_bounds():
    # Call delta in (0, 1), put delta in (-1, 0).
    for s in (60.0, 100.0, 150.0):
        cd = call_greeks(s, K, r, sigma, T).delta
        pd = put_greeks(s, K, r, sigma, T).delta
        assert 0.0 < cd < 1.0
        assert -1.0 < pd < 0.0


def test_gamma_positive_and_symmetric():
    # Gamma is identical for calls and puts (parity) and always
    # non-negative (both prices are convex in S).
    cg = call_greeks(S, K, r, sigma, T).gamma
    pg = put_greeks(S, K, r, sigma, T).gamma
    assert cg > 0
    assert abs(cg - pg) < 1e-12


def test_vega_positive_and_symmetric():
    cv = call_greeks(S, K, r, sigma, T).vega
    pv = put_greeks(S, K, r, sigma, T).vega
    assert cv > 0
    assert abs(cv - pv) < 1e-12
