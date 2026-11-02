"""Round 3 credit models: LGD/EAD, calibration with MoC, migration, CDS.

Every test is an oracle: a hand computation, a closed form or a round trip.
"""

import numpy as np
import pytest
from eq_credit.calibration import (
    PD_FLOOR_CRR,
    calibrate_to_central_tendency,
    calibrate_with_moc,
    long_run_average,
    moc_estimation_error,
)
from eq_credit.cds import HazardCurve, bootstrap_hazards, cds_legs, par_spread
from eq_credit.lgd_ead import beta_moments_fit, downturn_lgd, ead_from_ccf, realised_ccf, workout_lgd
from eq_credit.migration import cohort_matrix, cumulative_pd, generator, marginal_pd, pd_at
from scipy.linalg import expm
from scipy.special import expit, logit
from scipy.stats import t as student_t

# --- LGD / EAD -------------------------------------------------------------


def test_workout_lgd_hand_value():
    # 100 EAD; recover 50 at 1y and 30 at 2y, cost 5 at 1y; 10% discount.
    lgd = workout_lgd(100.0, np.array([50.0, 30.0]), np.array([5.0, 0.0]), np.array([1.0, 2.0]), 0.10)
    assert lgd == pytest.approx(1.0 - (45.0 / 1.1 + 30.0 / 1.21) / 100.0, abs=1e-14)


def test_workout_lgd_zero_rate_is_undiscounted():
    assert workout_lgd(80.0, np.array([20.0, 20.0]), np.zeros(2), np.array([0.5, 3.0]), 0.0) == pytest.approx(
        0.5
    )


def test_ead_and_realised_ccf_round_trip():
    drawn, limit, ccf = np.array([40.0, 10.0]), np.array([100.0, 50.0]), np.array([0.75, 0.2])
    ead = ead_from_ccf(drawn, limit, ccf)
    np.testing.assert_allclose(ead, [85.0, 18.0])
    np.testing.assert_allclose(realised_ccf(drawn, limit, ead), ccf)
    assert np.isnan(realised_ccf(np.array([100.0]), np.array([100.0]), np.array([100.0]))[0])


def test_beta_fit_recovers_parameters():
    x = np.random.default_rng(3).beta(2.0, 5.0, size=200_000)
    a, b = beta_moments_fit(x)
    assert a == pytest.approx(2.0, rel=0.03) and b == pytest.approx(5.0, rel=0.03)


def test_downturn_lgd_picks_worst_years():
    assert downturn_lgd(
        np.array([0.3, 0.5, 0.4, 0.6]), np.array([0.01, 0.05, 0.02, 0.04]), 2
    ) == pytest.approx(0.55)


def test_lgd_input_validation():
    with pytest.raises(ValueError):
        workout_lgd(0.0, np.ones(1), np.ones(1), np.ones(1), 0.1)
    with pytest.raises(ValueError):
        ead_from_ccf(1.0, 2.0, 1.5)


# --- calibration -----------------------------------------------------------


def test_central_tendency_hits_target_and_keeps_order():
    raw = np.random.default_rng(1).uniform(0.001, 0.2, 500)
    cal, c = calibrate_to_central_tendency(raw, 0.02)
    assert cal.mean() == pytest.approx(0.02, abs=1e-12)
    assert np.array_equal(np.argsort(raw), np.argsort(cal))
    np.testing.assert_allclose(logit(cal) - logit(raw), c, atol=1e-9)


def test_single_obligor_calibration_is_exact():
    cal, c = calibrate_to_central_tendency(np.array([0.1]), 0.03)
    assert cal[0] == pytest.approx(0.03, abs=1e-14)
    assert c == pytest.approx(logit(0.03) - logit(0.1), abs=1e-10)


def test_moc_c_is_t_half_width():
    dr = np.array([0.01, 0.02, 0.03, 0.015, 0.025])
    expected = student_t.ppf(0.9, 4) * np.std(dr, ddof=1) / np.sqrt(5)
    assert moc_estimation_error(dr, 0.9) == pytest.approx(expected, rel=1e-14)
    assert long_run_average(dr) == pytest.approx(0.02)


def test_calibrate_with_moc_adds_margins_and_floors():
    dr = np.array([0.01, 0.02, 0.03])
    raw = expit(np.linspace(-9.0, -2.0, 50))
    res = calibrate_with_moc(raw, dr, moc_a=0.001, moc_b=0.0005)
    expected = np.clip(res.pd_calibrated + 0.0015 + res.moc_c, PD_FLOOR_CRR, 1.0)
    np.testing.assert_allclose(res.pd_final, expected, rtol=1e-15, atol=1e-18)
    assert res.pd_final.min() >= PD_FLOOR_CRR
    assert res.central_tendency == pytest.approx(0.02)


# --- migration -------------------------------------------------------------


def test_two_state_lifetime_pd_closed_form():
    p = np.array([[0.97, 0.03], [0.0, 1.0]])
    cum = cumulative_pd(p, 10)[:, 0]
    np.testing.assert_allclose(cum, 1.0 - 0.97 ** np.arange(1, 11), rtol=1e-14)
    marg = marginal_pd(p, 10)[:, 0]
    np.testing.assert_allclose(marg, 0.03 * 0.97 ** np.arange(10), rtol=1e-12)


def test_cohort_matrix_counts():
    start = np.array([0, 0, 0, 0, 1, 1])
    end = np.array([0, 0, 1, 2, 1, 2])
    p = cohort_matrix(start, end, 3)
    np.testing.assert_allclose(p, [[0.5, 0.25, 0.25], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])


def test_generator_round_trip_on_embeddable_matrix():
    q = np.array([[-0.10, 0.08, 0.02], [0.05, -0.15, 0.10], [0.0, 0.0, 0.0]])
    p = expm(q)
    np.testing.assert_allclose(generator(p), q, atol=1e-10)
    np.testing.assert_allclose(pd_at(q, 1.0), p[:-1, -1], atol=1e-14)


def test_generator_fix_gives_valid_generator():
    # Many grade-skipping zeros make the raw log have negative off-diagonals.
    p = np.array(
        [[0.90, 0.10, 0.0, 0.0], [0.05, 0.85, 0.10, 0.0], [0.0, 0.10, 0.80, 0.10], [0.0, 0.0, 0.0, 1.0]]
    )
    q = generator(p)
    off = ~np.eye(4, dtype=bool)
    assert np.all(q[off] >= 0.0)
    np.testing.assert_allclose(q.sum(axis=1), 0.0, atol=1e-12)
    assert np.abs(expm(q) - p).max() < 0.02


# --- CDS -------------------------------------------------------------------


def test_credit_triangle_limit():
    curve = HazardCurve(np.array([5.0]), np.array([0.02]))
    # Discrete premium converges to lambda (1 - R) as the grid is refined.
    errs = [abs(par_spread(curve, 5.0, 0.03, 0.4, freq=f) - 0.012) for f in (20, 200, 2000)]
    assert errs[2] < 1e-5 * 0.012
    assert errs[0] > 5 * errs[1] > 25 * errs[2]


def test_protection_leg_flat_closed_form():
    lam, r, t, rec = 0.03, 0.02, 4.0, 0.35
    prot, _ = cds_legs(HazardCurve(np.array([t]), np.array([lam])), t, r, rec)
    assert prot == pytest.approx((1 - rec) * lam / (lam + r) * (1 - np.exp(-(lam + r) * t)), rel=1e-14)


def test_bootstrap_reprices_par_spreads():
    mats = np.array([1.0, 3.0, 5.0, 7.0, 10.0])
    spreads = np.array([0.0060, 0.0080, 0.0100, 0.0115, 0.0125])
    curve = bootstrap_hazards(mats, spreads, rate=0.03, recovery=0.4)
    for m, s in zip(mats, spreads, strict=True):
        assert par_spread(curve, m, 0.03, 0.4) == pytest.approx(s, abs=1e-12)
    assert np.all(curve.hazards > 0.0)


def test_bootstrap_flat_spreads_give_flat_hazard():
    curve = bootstrap_hazards(np.array([1.0, 2.0, 5.0]), np.full(3, 0.01), 0.02, 0.4)
    np.testing.assert_allclose(curve.hazards, curve.hazards[0], rtol=1e-9)


def test_bootstrap_rejects_arbitrage():
    with pytest.raises(ValueError):
        bootstrap_hazards(np.array([1.0, 5.0]), np.array([0.05, 0.001]), 0.02, 0.4)
