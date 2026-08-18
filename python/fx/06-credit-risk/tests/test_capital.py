"""EL, Basel standardized RW, Vasicek identities, sovereign vs corporate tails."""

import numpy as np
import pytest
from scipy.stats import norm

from fx_credit.capital import (
    SOVEREIGN_RHO,
    basel_corporate_correlation,
    capital_table,
    expected_loss,
    standardized_rw,
    vasicek_capital,
    vasicek_conditional_pd,
)


def test_expected_loss_exact_tiny_book():
    el = expected_loss(
        np.array([0.01, 0.02]), np.array([0.40, 0.50]), np.array([100.0, 200.0])
    )
    assert el == pytest.approx([0.40, 2.00], abs=1e-14)


def test_expected_loss_scalar():
    assert expected_loss(0.02, 0.55, 1_000.0) == pytest.approx(11.0, abs=1e-12)


def test_expected_loss_validation():
    with pytest.raises(ValueError, match="PD and LGD"):
        expected_loss(1.5, 0.5, 100.0)
    with pytest.raises(ValueError, match="EAD"):
        expected_loss(0.01, 0.5, -1.0)


def test_vasicek_rho_zero_identity():
    """rho=0: conditional PD equals unconditional PD for any factor draw."""
    for pd1 in (0.001, 0.02, 0.3):
        assert vasicek_conditional_pd(pd1, 0.0, 2.5) == pytest.approx(pd1, abs=1e-14)


def test_vasicek_median_factor_identity():
    """x=0: conditional PD = Phi(Phi^-1(pd)/sqrt(1-rho))."""
    pd1, rho = 0.02, 0.2
    expected = norm.cdf(norm.ppf(pd1) / np.sqrt(1 - rho))
    assert vasicek_conditional_pd(pd1, rho, 0.0) == pytest.approx(expected, abs=1e-14)


def test_vasicek_conditional_monotone_in_factor():
    xs = np.linspace(-3, 3, 20)
    vals = [vasicek_conditional_pd(0.02, 0.3, x) for x in xs]
    assert np.all(np.diff(vals) > 0)


def test_vasicek_degenerate_pds():
    assert vasicek_conditional_pd(0.0, 0.3, 3.0) == 0.0
    assert vasicek_conditional_pd(1.0, 0.3, -3.0) == 1.0


def test_vasicek_invalid_rho_raises():
    with pytest.raises(ValueError, match="rho"):
        vasicek_conditional_pd(0.01, 1.0, 0.0)


def test_vasicek_matches_granular_portfolio_simulation():
    """Non-circular check: simulate a one-factor default model on 5000 names;
    the portfolio loss-rate quantile must match the ASRF conditional PD."""
    pd1, rho, alpha = 0.05, 0.30, 0.95
    rng = np.random.default_rng(10)
    n_draws, n_names = 4000, 5000
    x = rng.standard_normal(n_draws)
    cond = np.array([vasicek_conditional_pd(pd1, rho, xi) for xi in x])
    losses = rng.binomial(n_names, cond) / n_names
    sim_q = np.quantile(losses, alpha)
    analytic = vasicek_conditional_pd(pd1, rho, norm.ppf(alpha))
    assert sim_q == pytest.approx(analytic, abs=0.01)


def test_capital_zero_at_boundary_pds():
    assert vasicek_capital(0.0, 0.45, SOVEREIGN_RHO) == 0.0
    assert vasicek_capital(1.0, 0.45, SOVEREIGN_RHO) == 0.0  # sure default = pure EL


def test_capital_positive_typical():
    k = vasicek_capital(0.02, 0.45, SOVEREIGN_RHO)
    assert 0.0 < k < 0.45


def test_capital_monotone_in_alpha():
    ks = [vasicek_capital(0.02, 0.45, 0.3, a) for a in (0.9, 0.99, 0.999)]
    assert np.all(np.diff(ks) > 0)


def test_capital_invalid_alpha():
    with pytest.raises(ValueError, match="alpha"):
        vasicek_capital(0.02, 0.45, 0.3, alpha=1.0)


def test_sovereign_correlation_fattens_tail():
    """Higher sovereign asset correlation => more tail capital than the Basel
    corporate correlation at every PD in the working range (the ordering that
    justifies internal sovereign models vs the 0% regulatory floor)."""
    for pd1 in (0.001, 0.005, 0.02, 0.06, 0.15):
        rho_c = basel_corporate_correlation(pd1)
        assert SOVEREIGN_RHO > rho_c
        k_sov = vasicek_capital(pd1, 0.45, SOVEREIGN_RHO)
        k_corp = vasicek_capital(pd1, 0.45, rho_c)
        assert k_sov > k_corp


def test_corporate_correlation_bounds_and_monotone():
    pds = np.linspace(1e-4, 0.2, 50)
    rho = basel_corporate_correlation(pds)
    assert np.all((rho >= 0.12 - 1e-12) & (rho <= 0.24 + 1e-12))
    assert np.all(np.diff(rho) < 0)  # decreasing in PD per Basel formula


def test_standardized_rw_table():
    assert standardized_rw("AAA") == 0.0
    assert standardized_rw("AA") == 0.0  # Basel allows 0% RW for AAA/AA sovereigns
    assert standardized_rw("A") == 0.20
    assert standardized_rw("BBB") == 0.50
    assert standardized_rw("BB") == 1.00
    assert standardized_rw("CCC") == 1.50
    with pytest.raises(ValueError, match="unknown rating"):
        standardized_rw("D+")


def test_capital_table_regulatory_wedge():
    """AAA sovereign: standardized capital is exactly 0 while internal economic
    capital is strictly positive — the documented model-vs-regulation wedge."""
    tab = capital_table(["AAA", "BB"], [0.0001, 0.02], lgd=0.45, ead=100.0)
    aaa = tab[tab["rating"] == "AAA"].iloc[0]
    assert aaa["std_capital"] == 0.0
    assert aaa["k_sovereign"] > 0.0
    bb = tab[tab["rating"] == "BB"].iloc[0]
    assert bb["k_sovereign"] > bb["k_corp_rho"] > 0.0
    assert bb["el"] == pytest.approx(0.02 * 0.45 * 100.0, abs=1e-12)


def test_capital_monotone_in_pd_working_range():
    pds = np.linspace(1e-4, 0.2, 40)
    ks = vasicek_capital(pds, 0.45, SOVEREIGN_RHO)
    assert np.all(np.diff(ks) > 0)
