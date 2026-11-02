"""HAR-RV oracle tests (Corsi 2009)."""

import numpy as np
import pytest
from eq_vol.har_rv import fit_har, har_design, realized_variance, simulate_har

TRUE = np.array([1e-5, 0.35, 0.30, 0.20])


def test_realized_variance_is_sum_of_squares():
    r = np.array([[0.01, -0.02, 0.005], [0.0, 0.03, -0.01]])
    np.testing.assert_allclose(realized_variance(r), [0.000525, 0.001], rtol=1e-14)


def test_design_matches_naive_loop():
    rv = np.random.default_rng(1).random(60)
    X, y = har_design(rv)
    for i, t in enumerate(range(21, 59)):
        assert X[i, 1] == rv[t]
        assert X[i, 2] == pytest.approx(rv[t - 4 : t + 1].mean(), abs=1e-15)
        assert X[i, 3] == pytest.approx(rv[t - 21 : t + 1].mean(), abs=1e-15)
        assert y[i] == rv[t + 1]


def test_noiseless_series_recovers_exact_params():
    """With zero noise the OLS fit reproduces the generating recursion."""
    rng = np.random.default_rng(2)
    rv = list(rng.random(22) * 1e-4)
    for _ in range(300):
        rv.append(TRUE[0] + TRUE[1] * rv[-1] + TRUE[2] * np.mean(rv[-5:])
                  + TRUE[3] * np.mean(rv[-22:]))
    res = fit_har(np.array(rv))
    np.testing.assert_allclose(res.params, TRUE, rtol=1e-6, atol=1e-12)


def test_ols_consistency_on_simulated_har():
    rv = simulate_har(TRUE, 20_000, noise_sd=2e-5, seed=4)
    res = fit_har(rv)
    np.testing.assert_allclose(res.params[1:], TRUE[1:], atol=4 * res.se_hac[1:].max() + 0.02)
    assert np.all(res.se_hac > 0)


def test_forecast_converges_to_unconditional_mean():
    rv = simulate_har(TRUE, 5_000, noise_sd=1e-5, seed=5)
    res = fit_har(rv)
    f = res.forecast(2_000)
    b0, bd, bw, bm = res.params
    assert f[-1] == pytest.approx(b0 / (1 - bd - bw - bm), rel=1e-6)
    assert res.forecast(1)[0] == pytest.approx(
        b0 + bd * rv[-1] + bw * rv[-5:].mean() + bm * rv[-22:].mean(), rel=1e-12)
