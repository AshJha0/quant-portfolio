"""Calibration: ground-truth recovery and the FX rho pattern."""

import numpy as np
import pytest

from fx_surface import CalibrationSlice, calibrate_heston
from fx_surface.data import calibration_slices

from conftest import TRUE_HESTON


def test_ground_truth_recovery(ground_truth_calibration):
    """Quotes generated from known Heston params -> calibration recovers
    them.  rho/v0 tight; theta moderate; kappa/xi individually loose
    (the documented kappa-xi ridge), but their ratio xi^2/kappa tight."""
    p = ground_truth_calibration.params
    t = TRUE_HESTON
    assert p.rho == pytest.approx(t.rho, abs=0.02)
    assert p.v0 == pytest.approx(t.v0, rel=0.02)
    assert p.theta == pytest.approx(t.theta, rel=0.10)
    assert p.kappa == pytest.approx(t.kappa, rel=0.50)
    assert p.xi == pytest.approx(t.xi, rel=0.30)
    assert p.xi**2 / p.kappa == pytest.approx(t.xi**2 / t.kappa, rel=0.15)


def test_ground_truth_rmse_near_zero(ground_truth_calibration):
    assert ground_truth_calibration.rmse_vol_pts < 0.01  # vol points


def test_eurusd_calibrates_clean_small_rho(eurusd_calibration):
    res = eurusd_calibration
    assert res.rmse_vol_pts < 0.25  # clean preset: < 0.25 vol pts
    assert abs(res.params.rho) < 0.30  # mild, fairly symmetric smile


def test_usdjpy_calibrates_clean_large_negative_rho(usdjpy_calibration):
    res = usdjpy_calibration
    assert res.rmse_vol_pts < 0.25
    assert res.params.rho < -0.40  # persistent JPY-call skew


def test_rho_pattern_across_pairs(eurusd_calibration, usdjpy_calibration):
    """The economic pattern: USDJPY skew is much more negative than
    EURUSD's."""
    assert usdjpy_calibration.params.rho < eurusd_calibration.params.rho - 0.2


def test_vol_level_pattern(eurusd_calibration, usdjpy_calibration):
    """USDJPY preset trades ~2.5-3 vol pts over EURUSD -> higher v0/theta."""
    assert usdjpy_calibration.params.v0 > eurusd_calibration.params.v0
    assert usdjpy_calibration.params.theta > eurusd_calibration.params.theta


def test_vega_weights_normalised(eurusd_calibration):
    w = eurusd_calibration.weights
    assert np.max(w) == pytest.approx(1.0)
    assert np.min(w) >= 0.05
    assert len(w) == 30  # 5 pillars x 6 expiries


def test_model_vols_shape_and_finiteness(usdjpy_calibration):
    res = usdjpy_calibration
    assert res.model_vols.shape == res.market_vols.shape == (30,)
    assert np.all(np.isfinite(res.model_vols))
    assert res.max_err_vol_pts >= res.rmse_vol_pts


def test_single_expiry_calibration(ground_truth_market):
    """A single smile also calibrates (fewer identified params, but the
    optimiser must run and fit well)."""
    m = ground_truth_market
    sl = calibration_slices(m)[2]
    res = calibrate_heston(m.S, [sl], max_nfev=200)
    assert res.rmse_vol_pts < 0.05
    assert res.params.rho < 0.0  # skew direction still identified


def test_calibration_validation():
    with pytest.raises(ValueError, match="at least one"):
        calibrate_heston(1.10, [])
    with pytest.raises(ValueError, match="same shape"):
        CalibrationSlice(T=1.0, r_d=0.04, r_f=0.03,
                         strikes=np.array([1.0, 1.1]), vols=np.array([0.1]))
    with pytest.raises(ValueError, match="T must be positive"):
        CalibrationSlice(T=0.0, r_d=0.04, r_f=0.03,
                         strikes=np.array([1.0]), vols=np.array([0.1]))
