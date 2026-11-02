"""Euler attribution: closed form, Euler sum, gradient, incremental, convergence."""

import numpy as np
import pytest
from eq_var.risk_attribution import (
    parametric_attribution,
    parametric_incremental,
    simulation_attribution,
    simulation_incremental,
)
from scipy.stats import norm

W = np.array([1.0e6, -4.0e5, 2.5e5])
VOL = np.array([0.02, 0.015, 0.03])
CORR = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.4], [0.2, 0.4, 1.0]])
COV = np.outer(VOL, VOL) * CORR


def test_two_asset_hand_example():
    # w = (1, 1), sigma = (1, 1), rho = 0: sigma_p = sqrt(2); each component is
    # half of z sqrt(2) by symmetry.
    a = parametric_attribution([1.0, 1.0], np.eye(2), 0.01)
    z = norm.ppf(0.99)
    assert a.total == pytest.approx(z * np.sqrt(2.0), rel=1e-14)
    np.testing.assert_allclose(a.component, [z / np.sqrt(2.0)] * 2, rtol=1e-14)
    np.testing.assert_allclose(a.percent, [0.5, 0.5], rtol=1e-14)


@pytest.mark.parametrize("measure", ["var", "es"])
def test_euler_sum_and_gradient(measure):
    mu = np.array([1e-4, -2e-4, 5e-5])
    a = parametric_attribution(W, COV, 0.025, measure=measure, mean=mu)
    assert a.component.sum() == pytest.approx(a.total, rel=1e-13)
    h = 1.0
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        up = parametric_attribution(W + e, COV, 0.025, measure=measure, mean=mu).total
        dn = parametric_attribution(W - e, COV, 0.025, measure=measure, mean=mu).total
        assert a.marginal[i] == pytest.approx((up - dn) / (2 * h), rel=1e-6)


def test_es_total_matches_closed_form():
    a = parametric_attribution(W, COV, 0.025, measure="es")
    sig = np.sqrt(W @ COV @ W)
    assert a.total == pytest.approx(sig * norm.pdf(norm.ppf(0.975)) / 0.025, rel=1e-13)


def test_incremental_close_to_marginal_for_small_trade():
    dw = np.array([1e3, 0.0, -5e2])
    inc = parametric_incremental(W, dw, COV, 0.01)
    lin = parametric_attribution(W, COV, 0.01).marginal @ dw
    assert inc == pytest.approx(lin, rel=1e-3)


def test_simulation_es_converges_to_parametric():
    rng = np.random.default_rng(7)
    r = rng.multivariate_normal(np.zeros(3), COV, size=400_000)
    sim = simulation_attribution(r * W, 0.025, measure="es")
    ref = parametric_attribution(W, COV, 0.025, measure="es")
    np.testing.assert_allclose(sim.component, ref.component, rtol=0.03, atol=0.01 * ref.total)
    assert sim.component.sum() == pytest.approx(sim.total, rel=1e-12)


def test_simulation_var_converges_to_parametric():
    rng = np.random.default_rng(11)
    r = rng.multivariate_normal(np.zeros(3), COV, size=400_000)
    sim = simulation_attribution(r * W, 0.01, measure="var")
    ref = parametric_attribution(W, COV, 0.01, measure="var")
    np.testing.assert_allclose(sim.component, ref.component, atol=0.03 * ref.total)


def test_simulation_es_total_equals_empirical_es():
    from eq_var import expected_shortfall

    rng = np.random.default_rng(3)
    x = rng.standard_normal((1003, 2))
    sim = simulation_attribution(x, 0.025, measure="es")
    assert sim.total == pytest.approx(expected_shortfall(x.sum(axis=1), 0.025), rel=1e-12)


def test_simulation_incremental_zero_trade():
    x = np.random.default_rng(1).standard_normal((500, 2))
    assert simulation_incremental(x, [0.0, 0.0], 0.01) == (0.0, 0.0)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        parametric_attribution([0.0, 0.0], np.eye(2))
    with pytest.raises(ValueError):
        parametric_attribution([1.0], np.eye(2))
    with pytest.raises(ValueError):
        parametric_attribution([1.0, np.nan], np.eye(2))
    with pytest.raises(ValueError):
        simulation_attribution(np.ones((5, 2)), 0.01)
    with pytest.raises(ValueError):
        parametric_attribution([1.0, 1.0], np.eye(2), measure="cvar")
