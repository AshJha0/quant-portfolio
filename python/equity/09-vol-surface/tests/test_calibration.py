"""Heston calibration: recovery, fit quality, identifiability diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

import eq_surface as es
from eq_surface.heston import FellerWarning

S0, R, Q = 100.0, 0.02, 0.01


def test_recovers_true_params_clean_data(clean_calibration):
    """Documented tolerances: rho and v0 tight; kappa/xi looser (the ridge)."""
    res, true, _ = clean_calibration
    got = res.params
    assert abs(got.v0 - true.v0) < 1e-3          # tight: sets short-dated ATM level
    assert abs(got.rho - true.rho) < 0.02        # tight: sets skew sign/size
    assert abs(got.theta - true.theta) < 0.1 * true.theta
    assert abs(got.kappa - true.kappa) < 0.35 * true.kappa   # loose: ridge with xi
    assert abs(got.xi - true.xi) < 0.35 * true.xi            # loose: ridge with kappa


def test_rmse_below_02_vol_points_clean(clean_calibration):
    res, _, _ = clean_calibration
    assert res.rmse_vol_points < 0.2
    for T, rmse in res.rmse_by_expiry.items():
        assert rmse < 0.2, T


def test_jacobian_condition_number_reported_and_large(clean_calibration):
    """Vanilla surfaces under-identify (kappa, xi): the Jacobian is
    ill-conditioned by construction, and we report it rather than hide it."""
    res, _, _ = clean_calibration
    assert np.isfinite(res.condition_number)
    assert res.condition_number > 1e2  # documented ridge -> large condition number
    assert res.jac_singular_values.shape == (5,)
    assert np.all(np.diff(res.jac_singular_values) <= 0)  # descending


def test_feller_violation_warned_not_raised(clean_calibration):
    res, true, caught = clean_calibration
    # The DEFAULT_TRUE_HESTON truth violates Feller (ratio 0.8); the fitted
    # parameters land close to it, so calibration must have warned.
    assert res.feller_ratio < 1.0
    assert any(issubclass(w.category, FellerWarning) for w in caught)


def test_noisy_data_still_converges(noisy_calibration):
    """0.3 vol points of seeded noise: fit degrades gracefully, params sane."""
    res, true = noisy_calibration
    assert res.rmse_vol_points < 0.6  # of the order of the injected noise
    got = res.params
    assert abs(got.v0 - true.v0) < 0.01
    assert abs(got.rho - true.rho) < 0.15
    assert 0.0 < got.theta < 0.2
    assert res.success or res.rmse_vol_points < 0.6


def test_report_contains_key_fields(clean_calibration):
    res, _, _ = clean_calibration
    text = res.report()
    for token in ("v0", "kappa", "theta", "rho", "xi", "RMSE", "condition number", "Feller"):
        assert token in text


def test_start_costs_and_metadata(clean_calibration):
    res, _, _ = clean_calibration
    assert res.n_starts == 2
    assert len(res.start_costs) == 2
    assert 0 <= res.best_start < res.n_starts
    assert res.n_quotes == 27


def test_model_ivs_helper_matches_direct_inversion(calib_market):
    expiries, strikes, ivs, true = calib_market
    model = es.heston_model_ivs(S0, R, Q, expiries, strikes, true)
    # generation and re-inversion are the same operation -> near-exact match
    for iv_in, iv_out in zip(ivs, model):
        assert np.allclose(iv_in, iv_out, atol=1e-10, equal_nan=True)


def test_validation_errors():
    T = np.array([0.5])
    K = [np.array([90.0, 100.0, 110.0])]
    IV = [np.array([0.21, 0.20, 0.19])]
    with pytest.raises(ValueError, match="spot"):
        es.calibrate_heston(-1.0, R, Q, T, K, IV)
    with pytest.raises(ValueError, match="n_starts"):
        es.calibrate_heston(S0, R, Q, T, K, IV, n_starts=0)
    with pytest.raises(ValueError, match="match expiries"):
        es.calibrate_heston(S0, R, Q, T, K + K, IV)
    with pytest.raises(ValueError, match="shapes differ"):
        es.calibrate_heston(S0, R, Q, T, K, [np.array([0.2, 0.2])])
    with pytest.raises(ValueError, match="at least 5 quotes"):
        es.calibrate_heston(S0, R, Q, T, K, IV)  # only 3 quotes
    with pytest.raises(ValueError, match="no valid quotes"):
        es.calibrate_heston(S0, R, Q, T, K, [np.array([np.nan, np.nan, np.nan])])
    with pytest.raises(ValueError, match="positive"):
        es.calibrate_heston(S0, R, Q, np.array([-0.5]), K, IV)
