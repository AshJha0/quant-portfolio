"""Tests for EL, Basel IRB formulas (independent re-derivation) and the
Vasicek one-factor loss distribution (analytic + Monte Carlo)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from eq_credit.portfolio_risk import (
    asset_correlation,
    basel_k,
    basel_report,
    economic_capital,
    el_by_bucket,
    expected_loss,
    maturity_adjustment_b,
    risk_weighted_assets,
    simulate_portfolio_losses,
    vasicek_cdf,
    vasicek_quantile,
)


# ------------------------------------------------------------------------- EL
def test_el_hand_computed_exact_tiny_book() -> None:
    pd_ = np.array([0.01, 0.05])
    lgd = np.array([0.4, 0.5])
    ead = np.array([100.0, 200.0])
    el, total = expected_loss(pd_, lgd, ead)
    np.testing.assert_allclose(el, [0.01 * 0.4 * 100, 0.05 * 0.5 * 200], atol=1e-15)
    assert total == pytest.approx(0.4 + 5.0, abs=1e-12)


def test_el_downturn_haircut() -> None:
    el, total = expected_loss(0.02, 0.5, 1000.0, downturn_lgd_haircut=0.2)
    assert total == pytest.approx(0.02 * 0.6 * 1000.0, abs=1e-12)
    # LGD capped at 1.
    _, t2 = expected_loss(0.02, 0.9, 1000.0, downturn_lgd_haircut=0.5)
    assert t2 == pytest.approx(0.02 * 1.0 * 1000.0, abs=1e-12)


def test_el_by_bucket_aggregation() -> None:
    df = pd.DataFrame(
        {
            "rating": ["A", "A", "B"],
            "pd": [0.01, 0.01, 0.10],
            "lgd": [0.5, 0.5, 0.5],
            "ead": [100.0, 100.0, 100.0],
        }
    )
    out = el_by_bucket(df, "rating").set_index("rating")
    assert out.loc["A", "el"] == pytest.approx(1.0)
    assert out.loc["B", "el"] == pytest.approx(5.0)
    assert out.loc["A", "n"] == 2


def test_el_validation_errors() -> None:
    with pytest.raises(ValueError, match="PD"):
        expected_loss(1.5, 0.5, 100.0)
    with pytest.raises(ValueError, match="LGD"):
        expected_loss(0.5, 1.5, 100.0)
    with pytest.raises(ValueError, match="EAD"):
        expected_loss(0.5, 0.5, -100.0)
    with pytest.raises(ValueError, match="haircut"):
        expected_loss(0.5, 0.5, 100.0, downturn_lgd_haircut=-0.1)


# ---------------------------------------------------------------------- Basel
def test_basel_k_reproduces_independent_hand_calculation() -> None:
    # Reference point: PD = 1%, LGD = 45%, M = 2.5 (the BCBS explanatory-note
    # parameterisation).  Re-derive K here writing the formula independently
    # with the exact regulatory constants.
    PD, LGD, M = 0.01, 0.45, 2.5
    w = (1 - np.exp(-50 * PD)) / (1 - np.exp(-50))
    R = 0.12 * w + 0.24 * (1 - w)
    b = (0.11852 - 0.05478 * np.log(PD)) ** 2
    cond = norm.cdf((norm.ppf(PD) + np.sqrt(R) * norm.ppf(0.999)) / np.sqrt(1 - R))
    K_ref = (LGD * cond - PD * LGD) * (1 + (M - 2.5) * b) / (1 - 1.5 * b)
    assert basel_k(PD, LGD, M)[0] == pytest.approx(K_ref, abs=1e-14)
    # And the resulting risk weight matches the published Basel II corporate
    # curve value of 92.32% at this point (BCBS explanatory note, July 2005).
    assert K_ref * 12.5 == pytest.approx(0.9232, abs=0.0005)


def test_asset_correlation_limits() -> None:
    # R interpolates from 0.24 (PD -> 0) down to 0.12 (PD -> 1).
    assert asset_correlation(0.0) == pytest.approx(0.24, abs=1e-12)
    assert asset_correlation(1.0) == pytest.approx(0.12, abs=1e-10)
    r = asset_correlation(np.array([0.001, 0.01, 0.1]))
    assert np.all(np.diff(r) < 0)  # decreasing in PD


def test_sme_size_adjustment() -> None:
    # Smallest SME (S <= 5m): R reduced by the full 0.04.
    assert asset_correlation(0.01, 5.0) == pytest.approx(
        asset_correlation(0.01) - 0.04, abs=1e-12
    )
    # Large corporate (S >= 50m): no adjustment.
    assert asset_correlation(0.01, 50.0) == pytest.approx(
        asset_correlation(0.01), abs=1e-12
    )
    # Adjustment lowers K.
    assert basel_k(0.01, 0.45, 2.5, sales_millions=10.0)[0] < basel_k(0.01, 0.45, 2.5)[0]


def test_maturity_adjustment_formula_and_direction() -> None:
    b = maturity_adjustment_b(0.01)
    assert b == pytest.approx((0.11852 - 0.05478 * np.log(0.01)) ** 2, abs=1e-15)
    # Longer maturity => more capital; M = 2.5 is the neutral point.
    k1 = basel_k(0.01, 0.45, 1.0)[0]
    k25 = basel_k(0.01, 0.45, 2.5)[0]
    k5 = basel_k(0.01, 0.45, 5.0)[0]
    assert k1 < k25 < k5
    # Maturity adjustment magnitude decreases with PD.
    assert maturity_adjustment_b(0.001) > maturity_adjustment_b(0.05)


def test_k_monotone_increasing_over_relevant_pd_range() -> None:
    # K rises with PD over the practical rating scale (3bp to 20%)...
    pds = np.linspace(0.0003, 0.20, 200)
    k = basel_k(pds, 0.45, 2.5)
    assert np.all(np.diff(k) > 0)
    # ...but turns over at very high PD (EL takes over the 99.9% quantile):
    assert basel_k(0.99, 0.45, 2.5)[0] < basel_k(0.30, 0.45, 2.5)[0]


def test_pd_floor_applied() -> None:
    assert basel_k(0.0, 0.45, 2.5)[0] == basel_k(0.0003, 0.45, 2.5)[0]
    assert basel_k(0.0001, 0.45, 2.5)[0] == basel_k(0.0003, 0.45, 2.5)[0]


def test_k_scales_linearly_in_lgd() -> None:
    assert basel_k(0.01, 0.90, 2.5)[0] == pytest.approx(
        2.0 * basel_k(0.01, 0.45, 2.5)[0], rel=1e-12
    )


def test_rwa_identity_and_report() -> None:
    k = basel_k(0.02, 0.45, 2.5)
    assert risk_weighted_assets(k, 1_000.0)[0] == pytest.approx(k[0] * 12.5 * 1_000.0)
    rep = basel_report(np.array([0.001, 0.01, 0.05]))
    assert list(rep["pd"]) == [0.001, 0.01, 0.05]
    assert (rep["K"] > 0).all() and rep["K"].is_monotonic_increasing


def test_basel_input_validation() -> None:
    with pytest.raises(ValueError, match="PD"):
        basel_k(1.5, 0.45)
    with pytest.raises(ValueError, match="LGD"):
        basel_k(0.01, 1.5)
    with pytest.raises(ValueError, match="maturity"):
        basel_k(0.01, 0.45, -1.0)


# -------------------------------------------------------------------- Vasicek
def test_vasicek_quantile_cdf_round_trip() -> None:
    for q in [0.5, 0.95, 0.999]:
        x = vasicek_quantile(q, pd_=0.02, rho=0.15)
        assert vasicek_cdf(x, 0.02, 0.15) == pytest.approx(q, abs=1e-10)


def test_vasicek_median_below_mean_right_skew() -> None:
    # The Vasicek default-rate distribution is right-skewed for low PD:
    # median < mean (= PD).
    assert vasicek_quantile(0.5, 0.02, 0.15) < 0.02


def test_vasicek_higher_rho_fatter_tail() -> None:
    q_lo = vasicek_quantile(0.999, 0.02, 0.05)
    q_hi = vasicek_quantile(0.999, 0.02, 0.30)
    assert q_hi > q_lo > 0.02


def test_vasicek_matches_basel_conditional_pd() -> None:
    # The Basel K conditional-PD term IS the Vasicek 99.9% quantile with
    # rho = R(PD).
    PD = 0.01
    R = float(asset_correlation(PD))
    cond = norm.cdf((norm.ppf(PD) + np.sqrt(R) * norm.ppf(0.999)) / np.sqrt(1 - R))
    assert vasicek_quantile(0.999, PD, R) == pytest.approx(cond, abs=1e-12)


def test_vasicek_input_validation() -> None:
    with pytest.raises(ValueError, match="PD"):
        vasicek_cdf(0.1, 0.0, 0.2)
    with pytest.raises(ValueError, match="rho"):
        vasicek_quantile(0.999, 0.02, 1.5)


def test_mc_converges_to_analytic_within_se() -> None:
    # Homogeneous portfolio, large N: the mean simulated loss rate must match
    # PD*LGD within 3 standard errors, and the simulated CDF at the analytic
    # 95% quantile must be ~0.95.
    PD, RHO, LGD = 0.02, 0.15, 1.0
    losses = simulate_portfolio_losses(
        PD, LGD, 1.0, RHO, n_sims=40_000, seed=2, n_loans=2_000
    )
    se = losses.std(ddof=1) / np.sqrt(len(losses))
    assert abs(losses.mean() - PD * LGD) < 3 * se
    x95 = vasicek_quantile(0.95, PD, RHO)
    frac = (losses <= x95).mean()
    assert abs(frac - 0.95) < 3 * np.sqrt(0.95 * 0.05 / len(losses)) + 0.01


def test_mc_quantile_approaches_analytic_as_n_grows() -> None:
    PD, RHO = 0.02, 0.15
    q_inf = vasicek_quantile(0.99, PD, RHO)
    q_small = np.quantile(
        simulate_portfolio_losses(PD, 1.0, 1.0, RHO, 20_000, seed=3, n_loans=50), 0.99
    )
    q_large = np.quantile(
        simulate_portfolio_losses(PD, 1.0, 1.0, RHO, 20_000, seed=3, n_loans=5_000),
        0.99,
    )
    assert abs(q_large - q_inf) < abs(q_small - q_inf)
    assert q_large == pytest.approx(q_inf, rel=0.10)


def test_finite_portfolio_tail_at_least_infinitely_granular() -> None:
    # Concentration/granularity: the finite-portfolio 99.9% loss quantile sits
    # at or above the analytic infinitely granular quantile.
    PD, RHO = 0.02, 0.15
    q_inf = vasicek_quantile(0.999, PD, RHO)
    for n_loans, seed in [(100, 4), (1_000, 5)]:
        losses = simulate_portfolio_losses(
            PD, 1.0, 1.0, RHO, n_sims=30_000, seed=seed, n_loans=n_loans
        )
        assert np.quantile(losses, 0.999) >= q_inf - 1e-9


def test_simulation_seeded_reproducible() -> None:
    a = simulate_portfolio_losses(0.02, 0.5, 1.0, 0.2, 1_000, seed=9, n_loans=100)
    b = simulate_portfolio_losses(0.02, 0.5, 1.0, 0.2, 1_000, seed=9, n_loans=100)
    np.testing.assert_array_equal(a, b)


def test_economic_capital_from_losses_and_quantile() -> None:
    losses = np.array([0.0, 0.01, 0.02, 0.10])
    el = 0.0325
    ec = economic_capital(losses, el, q=1.0)
    assert ec == pytest.approx(0.10 - 0.0325, abs=1e-12)
    assert economic_capital(0.08, 0.03) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="empty"):
        economic_capital(np.array([]), 0.0)
