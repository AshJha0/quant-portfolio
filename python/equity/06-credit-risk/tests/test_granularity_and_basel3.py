"""Granularity adjustment vs exact quadrature, MC quantile CIs, Basel III parameters."""

from __future__ import annotations

import numpy as np
import pytest
from eq_credit.portfolio_risk import (
    asset_correlation,
    basel_k,
    finite_portfolio_quantile,
    granularity_adjustment,
    mc_quantile_ci,
    simulate_portfolio_losses,
    vasicek_quantile,
)

PD, RHO, Q = 0.02, 0.15, 0.999


@pytest.mark.parametrize("n", [200, 1_000, 5_000, 20_000])
def test_ga_matches_exact_quadrature_within_one_lattice_step(n):
    # The exact quantile of a homogeneous book lives on the lattice k/N.
    # Vasicek + GA approximates it with O(1/N^2) error, so the two must
    # agree to within one lattice step 1/N (LGD = 1).
    exact = finite_portfolio_quantile(Q, PD, RHO, n)
    approx = float(vasicek_quantile(Q, PD, RHO)) + granularity_adjustment(
        PD, 1.0, 1.0, RHO, q=Q, n_loans=n)
    assert abs(exact - approx) <= 1.0 / n + 1e-12


def test_ga_is_order_one_over_n_and_positive():
    g1 = granularity_adjustment(PD, 0.45, 1.0, RHO, n_loans=1_000)
    g2 = granularity_adjustment(PD, 0.45, 1.0, RHO, n_loans=10_000)
    assert g1 > 0
    assert g1 / g2 == pytest.approx(10.0, rel=1e-6)


def test_ga_heterogeneous_book_matches_homogeneous_limit():
    # equal PDs, LGDs and EADs as arrays give the same GA as n_loans
    a = granularity_adjustment(np.full(500, PD), np.full(500, 0.4), np.ones(500), RHO)
    b = granularity_adjustment(PD, 0.4, 1.0, RHO, n_loans=500)
    assert a == pytest.approx(b, rel=1e-12)


def test_ga_grows_with_lgd_variance():
    a = granularity_adjustment(PD, 0.4, 1.0, RHO, n_loans=500)
    b = granularity_adjustment(PD, 0.4, 1.0, RHO, n_loans=500, lgd_var_xi=0.25)
    assert b > a


def test_mc_ci_covers_exact_quantile():
    # 95% order-statistic CI must contain the exact quantile for the large
    # majority of seeds; with 20 seeds, >= 16 covered is a loose bound
    # (P(<16 | p = 0.95) < 0.02).
    exact = finite_portfolio_quantile(Q, PD, RHO, 200)
    covered = 0
    for seed in range(20):
        ls = simulate_portfolio_losses(PD, 1.0, 1.0, RHO, n_sims=10_000,
                                       seed=seed, n_loans=200)
        _, lo, hi = mc_quantile_ci(ls, Q)
        covered += lo <= exact <= hi
    assert covered >= 16


def test_finite_quantile_validates():
    with pytest.raises(ValueError):
        finite_portfolio_quantile(1.0, PD, RHO, 100)
    with pytest.raises(ValueError):
        finite_portfolio_quantile(Q, PD, RHO, 0)
    with pytest.raises(ValueError):
        granularity_adjustment(PD, 0.4, 1.0, 1.5, n_loans=10)


# ---------------------------------------------------------- Basel III final

def test_basel3_pd_floor_is_5bp():
    assert basel_k(0.0003, 0.45)[0] == basel_k(0.0005, 0.45)[0]
    assert basel_k(0.0006, 0.45)[0] > basel_k(0.0005, 0.45)[0]
    # Basel II floor still available
    assert basel_k(0.0003, 0.45, pd_floor=0.0003)[0] < basel_k(0.0005, 0.45)[0]


def test_maturity_floor_and_cap():
    assert basel_k(0.01, 0.45, 0.25)[0] == basel_k(0.01, 0.45, 1.0)[0]
    assert basel_k(0.01, 0.45, 7.0)[0] == basel_k(0.01, 0.45, 5.0)[0]


def test_financial_institution_multiplier():
    r = asset_correlation(0.01)
    assert asset_correlation(0.01, financial_multiplier=True) == pytest.approx(1.25 * r)
    assert basel_k(0.01, 0.45, financial_multiplier=True)[0] > basel_k(0.01, 0.45)[0]


def test_defaulted_exposure_capital():
    assert basel_k(1.0, 0.45)[0] == 0.0
    assert basel_k(1.0, 0.45, el_best=0.35)[0] == pytest.approx(0.10)
