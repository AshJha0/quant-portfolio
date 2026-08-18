"""Analytic Greeks vs central finite differences; sign/bound properties."""

import math

import pytest

from eq_options import bs_greeks, bs_price, compare_greeks, crr_price, fd_greeks

SCENARIOS = [
    # (S, K, T, r, sigma, q): ATM, OTM call, ITM call, short-dated, neg rate
    (100.0, 100.0, 1.0, 0.05, 0.20, 0.00),
    (100.0, 120.0, 0.5, 0.03, 0.30, 0.02),
    (100.0, 80.0, 2.0, 0.01, 0.15, 0.04),
    (100.0, 100.0, 0.05, 0.02, 0.25, 0.00),
    (100.0, 95.0, 1.0, -0.02, 0.35, 0.01),
]
GREEK_NAMES = ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga"]


@pytest.mark.parametrize(("S", "K", "T", "r", "sigma", "q"), SCENARIOS)
@pytest.mark.parametrize("otype", ["call", "put"])
def test_analytic_vs_finite_difference_all_greeks(
    S: float, K: float, T: float, r: float, sigma: float, q: float, otype: str
) -> None:
    """Every analytic Greek matches central FD to 1e-4 relative tolerance."""
    ana = bs_greeks(S, K, T, r, sigma, q, otype)
    num = fd_greeks(bs_price, S, K, T, r, sigma, q, otype)
    for name in GREEK_NAMES:
        a, n = getattr(ana, name), getattr(num, name)
        assert a == pytest.approx(n, rel=1e-4, abs=1e-6), name


@pytest.mark.parametrize(("S", "K", "T", "r", "sigma", "q"), SCENARIOS)
@pytest.mark.parametrize("otype", ["call", "put"])
def test_gamma_and_vega_nonnegative(
    S: float, K: float, T: float, r: float, sigma: float, q: float, otype: str
) -> None:
    g = bs_greeks(S, K, T, r, sigma, q, otype)
    assert g.gamma >= 0.0
    assert g.vega >= 0.0


@pytest.mark.parametrize(("S", "K", "T", "r", "sigma", "q"), SCENARIOS)
def test_delta_bounds(S: float, K: float, T: float, r: float, sigma: float, q: float) -> None:
    """0 <= call delta <= e^{-qT}; -e^{-qT} <= put delta <= 0."""
    dfq = math.exp(-q * T)
    dc = bs_greeks(S, K, T, r, sigma, q, "call").delta
    dp = bs_greeks(S, K, T, r, sigma, q, "put").delta
    assert 0.0 <= dc <= dfq + 1e-12
    assert -dfq - 1e-12 <= dp <= 0.0


@pytest.mark.parametrize(("S", "K", "T", "r", "sigma", "q"), SCENARIOS)
def test_greek_parity_relations(
    S: float, K: float, T: float, r: float, sigma: float, q: float
) -> None:
    """delta_c - delta_p = e^{-qT}; gamma and vega identical for call/put."""
    gc = bs_greeks(S, K, T, r, sigma, q, "call")
    gp = bs_greeks(S, K, T, r, sigma, q, "put")
    assert gc.delta - gp.delta == pytest.approx(math.exp(-q * T), abs=1e-12)
    assert gc.gamma == pytest.approx(gp.gamma, abs=1e-14)
    assert gc.vega == pytest.approx(gp.vega, abs=1e-12)
    assert gc.vanna == pytest.approx(gp.vanna, abs=1e-12)
    assert gc.volga == pytest.approx(gp.volga, abs=1e-10)


def test_theta_negative_for_typical_long_options() -> None:
    """ATM and OTM options with r >= 0, q = 0 decay: theta < 0."""
    assert bs_greeks(100, 100, 1, 0.05, 0.2, 0.0, "call").theta < 0
    assert bs_greeks(100, 120, 1, 0.05, 0.2, 0.0, "call").theta < 0
    assert bs_greeks(100, 100, 0.05, 0.02, 0.3, 0.0, "put").theta < 0
    assert bs_greeks(100, 80, 1, 0.0, 0.2, 0.0, "put").theta < 0  # OTM put, r=0


def test_theta_can_be_positive_deep_itm_put() -> None:
    """Deep ITM European put with high r: pull-to-discounted-strike gives theta > 0."""
    assert bs_greeks(50, 100, 1, 0.10, 0.15, 0.0, "put").theta > 0


def test_rho_signs() -> None:
    assert bs_greeks(100, 100, 1, 0.05, 0.2, 0.0, "call").rho > 0
    assert bs_greeks(100, 100, 1, 0.05, 0.2, 0.0, "put").rho < 0


def test_atm_delta_near_half() -> None:
    d = bs_greeks(100, 100, 0.25, 0.0, 0.2, 0.0, "call").delta
    assert 0.5 < d < 0.55  # slightly above 1/2 from the sigma^2/2 drift


def test_vega_peaks_near_atm_forward() -> None:
    S, T, r, q, sigma = 100.0, 1.0, 0.03, 0.01, 0.2
    f = S * math.exp((r - q) * T)
    v_atm = bs_greeks(S, f, T, r, sigma, q, "call").vega
    v_itm = bs_greeks(S, f * 0.7, T, r, sigma, q, "call").vega
    v_otm = bs_greeks(S, f * 1.4, T, r, sigma, q, "call").vega
    assert v_atm > v_itm and v_atm > v_otm


def test_gamma_integrates_delta() -> None:
    """delta(S+h) - delta(S-h) ~ 2h * gamma at small h."""
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0
    h = 0.01
    d_up = bs_greeks(S + h, K, T, r, sigma, q, "call").delta
    d_dn = bs_greeks(S - h, K, T, r, sigma, q, "call").delta
    g = bs_greeks(S, K, T, r, sigma, q, "call").gamma
    assert (d_up - d_dn) / (2 * h) == pytest.approx(g, rel=1e-5)


def test_fd_greeks_works_on_binomial_pricer() -> None:
    """fd_greeks is generic: works on the CRR tree and lands near BS delta."""

    def tree(S: float, K: float, T: float, r: float, sigma: float,
             q: float, option_type: str, **kw: object) -> float:
        return crr_price(S, K, T, r, sigma, q, option_type, "european", 500)

    fd = fd_greeks(tree, 100, 100, 1, 0.05, 0.2, 0.0, "call", rel_bump=1e-3)
    ana = bs_greeks(100, 100, 1, 0.05, 0.2, 0.0, "call")
    assert fd.delta == pytest.approx(ana.delta, abs=5e-3)


def test_compare_greeks_table_small_errors() -> None:
    table = compare_greeks(100, 105, 0.5, 0.03, 0.25, 0.01, "put")
    assert set(table) == {"price", *GREEK_NAMES}
    for name, row in table.items():
        scale = max(abs(row["analytic"]), 1.0)
        assert row["abs_err"] / scale < 1e-4, name


def test_theta_finite_difference_convention() -> None:
    """theta = dV/dt: one year of decay ~ theta * dt for small dt."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    dt = 1e-4
    v_now = bs_price(S, K, T, r, sigma, 0.0, "call")
    v_later = bs_price(S, K, T - dt, r, sigma, 0.0, "call")
    theta = bs_greeks(S, K, T, r, sigma, 0.0, "call").theta
    assert (v_later - v_now) / dt == pytest.approx(theta, rel=1e-3)
